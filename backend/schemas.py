from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class TransactionFeatures(BaseModel):
    """RAW transaction values, as they'd arrive from Kafka/a real system —
    NOT pre-scaled. ml_service.py applies the persisted LabelEncoder/
    StandardScaler (from 01_load_paysim.py) before these reach the model."""

    step: int
    # Constrained to what the model was trained on (PaySim: fraud only ever
    # occurs on these two types) — FastAPI rejects anything else with a
    # clean 422 instead of it reaching the LabelEncoder and crashing.
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class ScoreRequest(BaseModel):
    transaction_id: str
    sender_account: str
    receiver_account: str
    features: TransactionFeatures
    # Optional manually-supplied neighbours (e.g. for curl testing, or a demo
    # preset that wants to show richness before Redis has real history for
    # that account). Real neighbours are also looked up from Redis by
    # sender_account/receiver_account and merged in — see main.py's /score.
    neighbours: List[TransactionFeatures] = []


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    risk_level: str
    explanation: dict
    neighbour_count: int
    inference_latency_ms: int


class FraudResultOut(BaseModel):
    decision_id: int
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    fraud_probability: float
    decision: str
    risk_level: str
    explanation: Optional[dict] = None
    decided_at: datetime
    alert_id: Optional[int] = None
    acknowledged_by: Optional[str] = None
    is_false_positive: bool = False
    escalated: bool = False
    escalated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_scored: int
    allow_count: int
    review_count: int
    block_count: int
    fraud_rate_estimate: Optional[float]
    avg_inference_latency_ms: Optional[float]


class AcknowledgeAlertRequest(BaseModel):
    acknowledged_by: str
    is_false_positive: bool = False
