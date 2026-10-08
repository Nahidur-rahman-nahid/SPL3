from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional

from pydantic import AfterValidator, BaseModel, Field

Role = Literal["ADMIN", "ANALYST"]


def _as_utc(value: datetime) -> datetime:
    # The database stores UTC but hands timestamps back without a timezone
    # (SQLite always, Postgres for these columns). Sent as-is, the browser
    # would read "18:27" as local time instead of 00:27 in Dhaka.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# Response timestamps: always serialized with an explicit UTC offset.
UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    role: Role
    is_active: bool
    must_change_password: bool
    created_at: UTCDateTime
    last_login_at: Optional[UTCDateTime] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: Optional[str] = None
    role: Role


class UserCreateOut(BaseModel):
    user: UserOut
    temp_password: str


class UserUpdate(BaseModel):
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    # Where forgot-password links go; "" removes it.
    email: Optional[str] = Field(default=None, max_length=254, pattern=r"^$|^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ResetPasswordWithTokenRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    new_password: str = Field(min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class ResetPasswordResponse(BaseModel):
    temp_password: str


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
    # When the transaction actually happened, used by time-of-day rules
    # (decision_engine.apply_night_rule). Omitted = "now". A naive datetime
    # is treated as Bangladesh local time.
    transaction_time: Optional[datetime] = None


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: str
    risk_level: str
    explanation: dict
    neighbour_count: int
    inference_latency_ms: int
    # Sender's Bangladesh Bank daily limit usage for this transaction's type
    # ({type, period, used, limit, remaining}); None when the check didn't
    # run (disabled, stream source, or Redis unavailable).
    limit_usage: Optional[dict] = None


class FraudResultOut(BaseModel):
    decision_id: int
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: Optional[str] = None
    fraud_probability: float
    decision: str
    risk_level: str
    explanation: Optional[dict] = None
    decided_at: UTCDateTime
    alert_id: Optional[int] = None
    acknowledged_by: Optional[str] = None
    is_false_positive: bool = False
    escalated: bool = False
    escalated_at: Optional[UTCDateTime] = None

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_scored: int
    allow_count: int
    review_count: int
    block_count: int
    fraud_rate_estimate: Optional[float]
    avg_inference_latency_ms: Optional[float]
    # Alerts nobody has acknowledged yet — drives the navbar's Alerts badge.
    pending_alerts: int = 0


class AcknowledgeAlertRequest(BaseModel):
    # acknowledged_by is intentionally NOT here — it's derived server-side
    # from the authenticated user (see main.py's acknowledge_alert), not
    # trusted from the client.
    is_false_positive: bool = False


# ------------------------------------------------- admin: delete transactions

# "stream" = stream simulator output ("stream-…" IDs), "manual" = everything
# scored through POST /score, "all" = both. See transaction_admin.py.
PurgeSource = Literal["stream", "manual", "all"]


class DeleteTransactionsRequest(BaseModel):
    transaction_ids: List[str] = Field(min_length=1, max_length=500)


class PurgeTransactionsRequest(BaseModel):
    source: PurgeSource
    # Only delete transactions received before this instant (optional).
    before: Optional[datetime] = None


class DeleteTransactionsResult(BaseModel):
    deleted_transactions: int
    deleted_decisions: int
    deleted_alerts: int


class PurgePreview(BaseModel):
    count: int


# ------------------------------------------------- fraud patterns demo page

PatternVariant = Literal["fraud", "lookalike"]


class PatternInfo(BaseModel):
    id: str
    title: str
    summary: str
    signal: str
    lookalike_title: str
    lookalike_summary: str
    # Plain-language story for the demo page.
    how: List[str] = []
    why_hidden: str = ""
    why_caught: str = ""
    lookalike_why: str = ""


class PatternList(BaseModel):
    model: dict
    patterns: List[PatternInfo]


class PatternStep(BaseModel):
    index: int
    transaction_id: str
    sender: str
    receiver: str
    type: str
    amount: float
    share_of_balance: Optional[float]
    receiver_was_empty: bool
    note: str
    neighbour_count: int
    probability: float
    alone_probability: float
    decision: str
    risk_level: str


class PatternRunResult(BaseModel):
    pattern_id: str
    variant: PatternVariant
    run_id: str
    redis_available: bool
    steps: List[PatternStep]
    flagged: int
    mean_probability: float
    mean_alone_probability: float
