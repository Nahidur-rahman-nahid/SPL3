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
