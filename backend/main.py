"""
FastAPI backend — merged ML inference + business logic (architecture.md
section 7 revision: single Python service, no Spring Boot).

Run locally:
    pip install -r requirements.txt
    # download FraudGNN_main.pt, type_encoder.pkl, scaler.pkl into ./model/
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive testing (Swagger UI
has an "Authorize" button once you've logged in — easier than curl).

The actual scoring work (Redis neighbour lookup, inference, persistence,
Redis self-registration, WebSocket broadcast) lives in one place —
scoring_pipeline.process_transaction() — used by BOTH:
  - POST /score, for manually/API-triggered scoring, and
  - stream_simulator.py's background loop, started at startup, which
    continuously generates realistic transactions so the dashboard has a
    live feed to watch without anyone needing to click anything. This
    stands in for a Kafka producer/consumer (see stream_simulator.py's
    docstring for why) — toggle it via POST /api/stream/toggle.

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

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

import escalation
import ml_service
import models
import schemas
import stream_simulator
from auth import authenticate_user, create_access_token, decode_user_from_token, get_current_user
from config import settings
from database import Base, engine, get_db
from scoring_pipeline import process_transaction
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
async def on_startup():
    Base.metadata.create_all(bind=engine)
    ml_service.load_model()
    asyncio.create_task(stream_simulator.run_stream())
    asyncio.create_task(escalation.run_escalation_watch())


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


@app.post("/score", response_model=schemas.ScoreResponse)
async def score(
    request: schemas.ScoreRequest,
    db: Session = Depends(get_db),
    _user: dict = Depends(get_current_user),
):
    result = await process_transaction(request, db, source="api")
    return schemas.ScoreResponse(**result)


@app.get("/api/stream/status")
def get_stream_status(_user: dict = Depends(get_current_user)):
    return {"enabled": stream_simulator.is_enabled(), "interval_seconds": settings.STREAM_INTERVAL_SECONDS}


@app.post("/api/stream/toggle")
def toggle_stream(_user: dict = Depends(get_current_user)):
    stream_simulator.set_enabled(not stream_simulator.is_enabled())
    return {"enabled": stream_simulator.is_enabled()}


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
    only_alerts: bool = False,
    unacknowledged_only: bool = False,
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
    if only_alerts:
        query = query.filter(models.Alert.alert_id.isnot(None))
    if unacknowledged_only:
        query = query.filter(models.Alert.acknowledged_at.is_(None))

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
            explanation=decision.explanation,
            decided_at=decision.decided_at,
            alert_id=alert.alert_id if alert else None,
            acknowledged_by=alert.acknowledged_by if alert else None,
            is_false_positive=alert.is_false_positive if alert else False,
            escalated=alert.escalated if alert else False,
            escalated_at=alert.escalated_at if alert else None,
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
