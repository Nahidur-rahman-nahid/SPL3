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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Session cookies. Locally, frontend (localhost:3000) and backend
    # (localhost:8000) are "same-site" per RFC 6265 (cookie scoping never
    # includes port), so COOKIE_SECURE=False + COOKIE_SAMESITE=lax works over
    # plain http. A real deployment with frontend/backend on different
    # domains is cross-site and MUST set COOKIE_SECURE=True and
    # COOKIE_SAMESITE=none (browsers reject SameSite=None without Secure) —
    # see main.py's startup check, which refuses to start on a bad
    # combination. Note Safari/Chrome third-party-cookie restrictions may
    # still block a SameSite=None cross-domain cookie by default; the real
    # fix there is serving frontend+backend from the same parent domain.
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    # Comma-separated list of allowed frontend origins for CORS. Must be
    # explicit (no wildcard) since allow_credentials=True is required for
    # cookie-based auth to work cross-origin.
    CORS_ORIGINS: str = "http://localhost:3000"

    # One-time bootstrap admin, seeded only if the users table is empty (see
    # main.py startup). All other accounts are created via POST /auth/users
    # by an existing admin — there is no public signup.
    BOOTSTRAP_ADMIN_USERNAME: str = "admin"
    BOOTSTRAP_ADMIN_EMAIL: str = "admin@example.com"
    BOOTSTRAP_ADMIN_PASSWORD: str = "change-me-on-first-login"

    # Failed-login lockout (backend/rate_limit.py). Fails open (doesn't block
    # login) if Redis is unreachable — same degrade-gracefully policy as the
    # Redis-backed neighbour lookup in redis_client.py.
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: float = 15.0

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

    # Night-time high-value transfer rule (decision_engine.apply_night_rule):
    # a high-value transaction during these local hours is raised to at
    # least REVIEW regardless of the GNN score. Hours are evaluated in
    # Bangladesh time (UTC+6, no DST). Applies to POST /score only — the
    # stream simulator skips it so the live demo feed is unaffected.
    NIGHT_RULE_ENABLED: bool = True
    NIGHT_RULE_HOURS: str = "23,0,1,2,3"
    NIGHT_RULE_MIN_AMOUNT: float = 10_000
    LOCAL_UTC_OFFSET_HOURS: int = 6

    # Bangladesh Bank daily MFS limits (scoring_pipeline.check_regulatory_limits).
    # A POST /score transaction that would push the sender's same-type total
    # for the current Bangladesh calendar day past its limit is hard-BLOCKed
    # with risk_level="REGULATORY". TRANSFER stands in for BB's "P2P send
    # money" limit. Totals live in Redis (redis_client.reserve_daily_amount);
    # fails open if Redis is down. The stream simulator skips this check.
    REGULATORY_LIMITS_ENABLED: bool = True
    DAILY_LIMIT_CASH_OUT: float = 30_000
    DAILY_LIMIT_TRANSFER: float = 50_000

    # escalation.py: an unacknowledged alert older than this gets flagged
    # "escalated" and re-broadcast so it stands out in the UI. Stands in
    # for a real paging/email SLA timer — see escalation.py's docstring.
    ALERT_SLA_MINUTES: float = 10.0
    ALERT_ESCALATION_CHECK_SECONDS: float = 60.0

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def night_rule_hours_set(self) -> set[int]:
        return {int(h.strip()) for h in self.NIGHT_RULE_HOURS.split(",") if h.strip()}

    @property
    def daily_limits(self) -> dict[str, float]:
        return {"CASH_OUT": self.DAILY_LIMIT_CASH_OUT, "TRANSFER": self.DAILY_LIMIT_TRANSFER}

    class Config:
        env_file = ".env"


settings = Settings()
