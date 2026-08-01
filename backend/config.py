"""
Central config. Everything here is overridable via environment variables /
a .env file, so the same code runs locally (SQLite, no setup) and in
production (Postgres, real secrets) without code changes.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth
    SECRET_KEY: str = "dev-only-secret-change-me"  # MUST be overridden in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Dev login accounts. Override via env vars before deploying anywhere real.
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    ANALYST_USERNAME: str = "analyst"
    ANALYST_PASSWORD: str = "analyst123"

    # Database — defaults to a local SQLite file so you can run this with zero
    # setup. Point DATABASE_URL at Postgres (e.g. from Render) for production:
    # postgresql://user:password@host:5432/dbname
    DATABASE_URL: str = "sqlite:///./fraud.db"

    # Redis — the live graph memory (architecture.md section 5). Defaults to
    # a local instance; falls back gracefully (fewer neighbours found, not a
    # crash) if unreachable. Point at a free Upstash instance for anything
    # beyond local testing — same "hosted, no Docker" pattern as Neon.
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML model + the fitted preprocessing artifacts from 01_load_paysim.py —
    # required to transform RAW incoming transactions the same way the
    # training data was transformed before it reaches the model.
    MODEL_PATH: str = "./model/FraudGNN_main.pt"
    TYPE_ENCODER_PATH: str = "./model/type_encoder.pkl"
    SCALER_PATH: str = "./model/scaler.pkl"

    # 4-tier decision policy (architecture.md section 4). Fallback values if
    # the checkpoint doesn't embed its own tiers.
    LOW_MAX: float = 0.40
    REVIEW_MAX: float = 0.75  # == BLOCK_THRESHOLD
    HIGH_MAX: float = 0.90

    # Live transaction stream simulator (stream_simulator.py) — stands in
    # for a Kafka producer/consumer (see that file's docstring for why).
    STREAM_ENABLED: bool = False
    STREAM_INTERVAL_SECONDS: float = 2.0

    # Alert policy (decision_engine.py) — deliberately separate from
    # LOW_MAX/REVIEW_MAX/HIGH_MAX above. Those three drive what tier a
    # caller is TOLD (ALLOW/REVIEW/BLOCK); this drives who actually lands
    # in the analyst review queue. Default (0.40 == LOW_MAX) means both
    # REVIEW and BLOCK tiers alert, but it can be tuned independently
    # (e.g. raised to only alert on high-confidence BLOCK) without
    # touching the tiering thresholds themselves.
    ALERT_THRESHOLD: float = 0.40

    # escalation.py: an unacknowledged alert older than this gets flagged
    # "escalated" and re-broadcast so it stands out in the UI. Stands in
    # for a real paging/email SLA timer — see escalation.py's docstring.
    ALERT_SLA_MINUTES: float = 10.0
    ALERT_ESCALATION_CHECK_SECONDS: float = 60.0

    class Config:
        env_file = ".env"


settings = Settings()
