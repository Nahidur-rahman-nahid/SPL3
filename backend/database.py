from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    # Serverless Postgres (Neon) can drop idle connections server-side.
    # pool_pre_ping tests each connection before use and transparently
    # reconnects if it's dead, instead of surfacing a raw OperationalError.
    # pool_recycle proactively retires connections before that idle cutoff.
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
