"""
SQLAlchemy ORM models — matches architecture.md section 9. Uses generic
column types (JSON not JSONB, Integer autoincrement not BIGSERIAL) so the
exact same code works against SQLite (local dev) and PostgreSQL (production)
without changes.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False, unique=True, index=True)
    email = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # ADMIN, ANALYST — validated in schemas.py
    is_active = Column(Boolean, nullable=False, default=True)
    # Forces a password change on next login — set on account creation and on
    # any admin-issued password reset, since both hand the user a temp
    # password an admin has seen.
    must_change_password = Column(Boolean, nullable=False, default=True)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)
    last_login_at = Column(DateTime, nullable=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # Hash of the refresh token, never the raw value — same principle as
    # password storage. Looked up by hashing the presented token.
    token_hash = Column(String, nullable=False, unique=True, index=True)
    # Every token descended from one login shares a family_id. Reuse
    # detection (a revoked/expired token presented again = theft signal)
    # bulk-revokes the whole family instead of walking a replacement chain.
    family_id = Column(String, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utcnow)


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True)
    sender_account = Column(String, nullable=False, index=True)
    receiver_account = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String, nullable=False)  # TRANSFER, CASH_OUT
    step = Column(Integer, nullable=False)
    old_balance_sender = Column(Float)
    new_balance_sender = Column(Float)
    old_balance_receiver = Column(Float)
    new_balance_receiver = Column(Float)
    raw_payload = Column(JSON)
    received_at = Column(DateTime, nullable=False, default=utcnow)

    decisions = relationship("FraudDecision", back_populates="transaction")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    trained_at = Column(DateTime, nullable=False, default=utcnow)
    auc_roc = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    training_dataset = Column(String)  # PaySim, IEEE-CIS
    is_active = Column(Boolean, default=False)


class FraudDecision(Base):
    __tablename__ = "fraud_decisions"

    decision_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    decision = Column(String, nullable=False)  # ALLOW, REVIEW, BLOCK
    risk_level = Column(String, nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    explanation = Column(JSON)
    neighbour_count = Column(Integer)
    inference_latency_ms = Column(Integer)
    model_version_id = Column(Integer, ForeignKey("model_versions.version_id"), nullable=True)
    decided_at = Column(DateTime, nullable=False, default=utcnow)

    transaction = relationship("Transaction", back_populates="decisions")
    alerts = relationship("Alert", back_populates="decision")


class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, ForeignKey("fraud_decisions.decision_id"), nullable=False)
    pushed_at = Column(DateTime, nullable=False, default=utcnow)
    acknowledged_by = Column(String, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    is_false_positive = Column(Boolean, default=False)
    escalated = Column(Boolean, default=False)
    escalated_at = Column(DateTime, nullable=True)

    decision = relationship("FraudDecision", back_populates="alerts")


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    edge_id = Column(Integer, primary_key=True, autoincrement=True)
    source_transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=False)
    target_transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=False)
    edge_type = Column(String)  # SAME_SENDER, SAME_RECEIVER
    created_at = Column(DateTime, nullable=False, default=utcnow)
