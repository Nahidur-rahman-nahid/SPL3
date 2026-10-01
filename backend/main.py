"""
FastAPI backend — merged ML inference + business logic (architecture.md
section 7 revision: single Python service, no Spring Boot).

Run locally:
    pip install -r requirements.txt
    # download FraudGNN_main.pt, type_encoder.pkl, scaler.pkl into ./model/
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive testing. Auth is now
cookie-based (see auth.py) rather than a Swagger "Authorize" bearer token —
log in via POST /auth/login from a browser (or curl -c/-b a cookie jar) to
exercise the other endpoints; see README.md for a curl walkthrough.

The actual scoring work (Redis neighbour lookup, inference, persistence,
Redis self-registration, WebSocket broadcast) lives in one place —
scoring_pipeline.process_transaction() — used by BOTH:
  - POST /score, for manually/API-triggered scoring, and
  - stream_simulator.py's background loop, started at startup, which
    continuously generates realistic transactions so the dashboard has a
    live feed to watch without anyone needing to click anything. This
    stands in for a Kafka producer/consumer (see stream_simulator.py's
    docstring for why) — toggle it via POST /api/stream/toggle.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

import escalation
import ml_service
import models
import schemas
import stream_simulator
import user_service
from auth import (
    RefreshTokenReuseError,
    clear_session_cookies,
    create_access_token,
    generate_csrf_token,
    get_current_user,
    issue_refresh_token,
    load_user_from_cookies,
    require_csrf,
    require_role,
    revoke_refresh_token,
    rotate_refresh_token,
    set_session_cookies,
)
from config import settings
from database import Base, SessionLocal, engine, get_db
from rate_limit import clear_attempts, is_locked_out, record_failed_attempt
from scoring_pipeline import process_transaction
from websocket_manager import manager

logger = logging.getLogger("main")

app = FastAPI(title="Fraud Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    if settings.COOKIE_SAMESITE.lower() == "none" and not settings.COOKIE_SECURE:
        raise RuntimeError(
            "COOKIE_SAMESITE=none requires COOKIE_SECURE=True (browsers reject "
            "SameSite=None cookies without Secure). Fix your .env before starting."
        )
    if settings.COOKIE_SECURE and settings.SECRET_KEY == "dev-only-secret-change-me":
        logger.critical(
            "COOKIE_SECURE=True (looks like a production deploy) but SECRET_KEY is "
            "still the development default. Set a real random SECRET_KEY."
        )

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        user_service.bootstrap_admin(db)
    finally:
        db.close()

    ml_service.load_model()
    asyncio.create_task(stream_simulator.run_stream())
    asyncio.create_task(escalation.run_escalation_watch())


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------- auth

@app.post("/auth/login", response_model=schemas.UserOut)
def login(payload: schemas.LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"

    if is_locked_out(payload.username, client_ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")

    user = user_service.authenticate(db, payload.username, payload.password)
    if user is None:
        record_failed_attempt(payload.username, client_ip)
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    clear_attempts(payload.username, client_ip)

    access_token = create_access_token(user)
    refresh_token, _ = issue_refresh_token(db, user)
    csrf_token = generate_csrf_token()
    set_session_cookies(response, access_token, refresh_token, csrf_token)
    return user


@app.post("/auth/refresh", response_model=schemas.UserOut, dependencies=[Depends(require_csrf)])
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw_refresh_token = request.cookies.get("refresh_token")
    if raw_refresh_token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        new_refresh_token, user = rotate_refresh_token(db, raw_refresh_token)
    except RefreshTokenReuseError:
        clear_session_cookies(response)
        raise HTTPException(status_code=401, detail="Session invalid — please log in again")

    access_token = create_access_token(user)
    csrf_token = generate_csrf_token()
    set_session_cookies(response, access_token, new_refresh_token, csrf_token)
    return user


@app.post("/auth/logout", dependencies=[Depends(require_csrf)])
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    raw_refresh_token = request.cookies.get("refresh_token")
    if raw_refresh_token:
        revoke_refresh_token(db, raw_refresh_token)
    clear_session_cookies(response)
    return {"status": "logged_out"}


@app.get("/auth/me", response_model=schemas.UserOut)
def get_me(user: models.User = Depends(get_current_user)):
    return user


@app.post("/auth/change-password", dependencies=[Depends(require_csrf)])
def change_password(
    payload: schemas.ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    ok = user_service.change_password(db, user, payload.current_password, payload.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    return {"status": "password_changed"}


@app.post("/auth/users", response_model=schemas.UserCreateOut, dependencies=[Depends(require_csrf)])
def create_user(
    payload: schemas.UserCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(require_role("ADMIN")),
):
    if user_service.get_user_by_username(db, payload.username) is not None:
        raise HTTPException(status_code=409, detail="Username already exists")
    user, temp_password = user_service.create_user(db, payload.username, payload.email, payload.role, admin.id)
    return schemas.UserCreateOut(user=user, temp_password=temp_password)


@app.get("/auth/users", response_model=List[schemas.UserOut])
def list_users(db: Session = Depends(get_db), _admin: models.User = Depends(require_role("ADMIN"))):
    return user_service.list_users(db)


@app.patch("/auth/users/{user_id}", response_model=schemas.UserOut, dependencies=[Depends(require_csrf)])
def update_user(
    user_id: int,
    payload: schemas.UserUpdate,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_role("ADMIN")),
):
    target = user_service.get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_service.update_user(db, target, role=payload.role, is_active=payload.is_active)


@app.post("/auth/users/{user_id}/reset-password", response_model=schemas.ResetPasswordResponse, dependencies=[Depends(require_csrf)])
def reset_password(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: models.User = Depends(require_role("ADMIN")),
):
    target = user_service.get_user_by_id(db, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    temp_password = user_service.reset_password(db, target)
    return schemas.ResetPasswordResponse(temp_password=temp_password)


# ------------------------------------------------------------------ scoring

@app.post("/score", response_model=schemas.ScoreResponse, dependencies=[Depends(require_csrf)])
async def score(
    request: schemas.ScoreRequest,
    db: Session = Depends(get_db),
    _user: models.User = Depends(get_current_user),
):
    result = await process_transaction(request, db, source="api")
    return schemas.ScoreResponse(**result)


@app.get("/api/stream/status")
def get_stream_status(_user: models.User = Depends(get_current_user)):
    return {"enabled": stream_simulator.is_enabled(), "interval_seconds": settings.STREAM_INTERVAL_SECONDS}


@app.post("/api/stream/toggle", dependencies=[Depends(require_csrf)])
def toggle_stream(_admin: models.User = Depends(require_role("ADMIN"))):
    stream_simulator.set_enabled(not stream_simulator.is_enabled())
    return {"enabled": stream_simulator.is_enabled()}


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    # CORSMiddleware does not apply to WebSocket routes, so the Origin check
    # that CORS normally provides has to be done by hand here.
    origin = websocket.headers.get("origin")
    if origin not in settings.cors_origins_list:
        await websocket.close(code=4403)
        return

    db = SessionLocal()
    try:
        user = load_user_from_cookies(websocket.cookies, db)
    finally:
        db.close()

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
    _user: models.User = Depends(get_current_user),
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
def get_stats(db: Session = Depends(get_db), _user: models.User = Depends(get_current_user)):
    # Single round-trip: all counts + avg in one query, each count pinned to
    # an explicit column so SQLAlchemy always has a FROM clause to work with.
    row = db.query(
        func.count(models.FraudDecision.decision_id),
        func.count(models.FraudDecision.decision_id).filter(models.FraudDecision.decision == "ALLOW"),
        func.count(models.FraudDecision.decision_id).filter(models.FraudDecision.decision == "REVIEW"),
        func.count(models.FraudDecision.decision_id).filter(models.FraudDecision.decision == "BLOCK"),
        func.avg(models.FraudDecision.inference_latency_ms),
    ).one()
    total, allow_count, review_count, block_count, avg_latency = row
    total = total or 0

    return schemas.StatsOut(
        total_scored=total,
        allow_count=allow_count or 0,
        review_count=review_count or 0,
        block_count=block_count or 0,
        fraud_rate_estimate=(block_count / total) if total else None,
        avg_inference_latency_ms=float(avg_latency) if avg_latency is not None else None,
    )


@app.post("/api/fraud/alerts/{alert_id}/acknowledge", dependencies=[Depends(require_csrf)])
async def acknowledge_alert(
    alert_id: int,
    request: schemas.AcknowledgeAlertRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    alert = db.get(models.Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    # acknowledged_by is derived from the authenticated session, not trusted
    # from the request body — otherwise any valid session could "acknowledge
    # as" an arbitrary name.
    alert.acknowledged_by = user.username
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
