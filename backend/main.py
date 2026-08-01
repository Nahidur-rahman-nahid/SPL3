"""
FastAPI backend — merged ML inference + business logic (architecture.md
section 7 revision: single Python service, no Spring Boot).

Run locally:
    pip install -r requirements.txt
    # download FraudGNN_main.pt, type_encoder.pkl, scaler.pkl into ./model/
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive testing (Swagger UI
has an "Authorize" button once you've logged in — easier than curl).

What /score does now: looks up recent transactions for both the sender and
receiver account from Redis (the live graph memory, architecture.md section
5), merges those with any explicitly-supplied `neighbours` in the request,
scores with the real-time subgraph, persists the decision, registers the
transaction back into Redis for future lookups, and broadcasts the result
over WebSocket (/ws/alerts) to every connected dashboard client.

curl smoke test:
    curl -X POST http://127.0.0.1:8000/auth/login \\
      -d "username=admin&password=admin123"
    curl -X POST http://127.0.0.1:8000/score \\
      -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \\
      -d '{"transaction_id":"t1","sender_account":"A1","receiver_account":"A2",
           "features":{"step":5,"type":"TRANSFER","amount":181000.0,
                       "oldbalanceOrg":181000.0,"newbalanceOrig":0.0,
                       "oldbalanceDest":0.0,"newbalanceDest":0.0},
           "neighbours":[]}'
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

import ml_service
import models
import redis_client
import schemas
from auth import authenticate_user, create_access_token, decode_user_from_token, get_current_user
from database import Base, engine, get_db
from websocket_manager import manager

app = FastAPI(title="Fraud Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # add your Vercel domain once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    ml_service.load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return schemas.Token(access_token=token, role=user["role"])


def _merge_redis_neighbours(request: schemas.ScoreRequest) -> None:
    """Fetches recent transactions for both accounts from Redis, dedupes
    against each other and the current transaction, and prepends them to
    request.neighbours (mutated in place) alongside whatever the caller
    already supplied."""
    seen_ids = {request.transaction_id}
    redis_features = []

    for account_id in (request.sender_account, request.receiver_account):
        for entry in redis_client.get_recent_transactions(account_id):
            txn_id = entry.get("transaction_id")
            if txn_id in seen_ids:
                continue
            seen_ids.add(txn_id)
            try:
                redis_features.append(schemas.TransactionFeatures(**{k: v for k, v in entry.items() if k != "transaction_id"}))
            except Exception:
                continue  # skip malformed/legacy entries rather than fail the whole request

    request.neighbours = redis_features + list(request.neighbours)


@app.post("/score", response_model=schemas.ScoreResponse)
async def score(
    request: schemas.ScoreRequest,
    db: Session = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    _merge_redis_neighbours(request)
    result = ml_service.score_transaction(request)

    existing = db.get(models.Transaction, request.transaction_id)
    if existing is None:
        f = request.features
        db.add(models.Transaction(
            transaction_id=request.transaction_id,
            sender_account=request.sender_account,
            receiver_account=request.receiver_account,
            amount=f.amount,
            transaction_type=f.type,
            step=int(f.step),
            old_balance_sender=f.oldbalanceOrg,
            new_balance_sender=f.newbalanceOrig,
            old_balance_receiver=f.oldbalanceDest,
            new_balance_receiver=f.newbalanceDest,
        ))

    decided_at = datetime.now(timezone.utc)
    decision_row = models.FraudDecision(
        transaction_id=request.transaction_id,
        fraud_probability=result["fraud_probability"],
        decision=result["decision"],
        risk_level=result["risk_level"],
        explanation=result["explanation"],
        neighbour_count=result["neighbour_count"],
        inference_latency_ms=result["inference_latency_ms"],
        decided_at=decided_at,
    )
    db.add(decision_row)
    db.flush()  # populate decision_row.decision_id before using it below

    alert_id = None
    if result["decision"] == "BLOCK":
        alert = models.Alert(decision_id=decision_row.decision_id, pushed_at=decided_at)
        db.add(alert)
        db.flush()
        alert_id = alert.alert_id

    db.commit()

    redis_client.store_transaction(
        request.transaction_id, request.sender_account, request.receiver_account, request.features.model_dump()
    )

    await manager.broadcast({
        "type": "new_decision",
        "decision_id": decision_row.decision_id,
        "transaction_id": request.transaction_id,
        "sender_account": request.sender_account,
        "receiver_account": request.receiver_account,
        "amount": request.features.amount,
        "fraud_probability": result["fraud_probability"],
        "decision": result["decision"],
        "risk_level": result["risk_level"],
        "decided_at": decided_at.isoformat(),
        "alert_id": alert_id,
        "acknowledged_by": None,
        "is_false_positive": False,
    })

    return schemas.ScoreResponse(**result)


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, token: str = ""):
    user = decode_user_from_token(token)
    if user is None:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # client sends nothing; just keep the connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/fraud/results", response_model=List[schemas.FraudResultOut])
def get_results(
    limit: int = 100,
    risk_level: Optional[str] = None,
    db: Session = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    query = (
        db.query(models.FraudDecision, models.Transaction, models.Alert)
        .join(models.Transaction, models.FraudDecision.transaction_id == models.Transaction.transaction_id)
        .outerjoin(models.Alert, models.Alert.decision_id == models.FraudDecision.decision_id)
        .order_by(models.FraudDecision.decided_at.desc())
    )
    if risk_level:
        query = query.filter(models.FraudDecision.risk_level == risk_level.upper())

    rows = query.limit(limit).all()
    return [
        schemas.FraudResultOut(
            decision_id=decision.decision_id,
            transaction_id=txn.transaction_id,
            sender_account=txn.sender_account,
            receiver_account=txn.receiver_account,
            amount=txn.amount,
            fraud_probability=decision.fraud_probability,
            decision=decision.decision,
            risk_level=decision.risk_level,
            decided_at=decision.decided_at,
            alert_id=alert.alert_id if alert else None,
            acknowledged_by=alert.acknowledged_by if alert else None,
            is_false_positive=alert.is_false_positive if alert else False,
        )
        for decision, txn, alert in rows
    ]


@app.get("/api/fraud/stats", response_model=schemas.StatsOut)
def get_stats(db: Session = Depends(get_db), _user: dict = Depends(get_current_user)):
    total = db.query(func.count(models.FraudDecision.decision_id)).scalar() or 0
    allow_count = db.query(func.count()).filter(models.FraudDecision.decision == "ALLOW").scalar() or 0
    review_count = db.query(func.count()).filter(models.FraudDecision.decision == "REVIEW").scalar() or 0
    block_count = db.query(func.count()).filter(models.FraudDecision.decision == "BLOCK").scalar() or 0
    avg_latency = db.query(func.avg(models.FraudDecision.inference_latency_ms)).scalar()

    return schemas.StatsOut(
        total_scored=total,
        allow_count=allow_count,
        review_count=review_count,
        block_count=block_count,
        fraud_rate_estimate=(block_count / total) if total else None,
        avg_inference_latency_ms=float(avg_latency) if avg_latency is not None else None,
    )


@app.post("/api/fraud/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    request: schemas.AcknowledgeAlertRequest,
    db: Session = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    alert = db.get(models.Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledged_by = request.acknowledged_by
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.is_false_positive = request.is_false_positive
    db.commit()

    await manager.broadcast({
        "type": "alert_acknowledged",
        "alert_id": alert_id,
        "decision_id": alert.decision_id,
        "acknowledged_by": alert.acknowledged_by,
        "is_false_positive": alert.is_false_positive,
    })

    return {"status": "acknowledged"}
