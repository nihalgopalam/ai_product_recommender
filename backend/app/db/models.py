import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

engine = None
SessionLocal: sessionmaker | None = None


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    updated_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())


class UserProfileDB(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    feature_preferences = Column(Text, nullable=False, default="{}")
    liked_products = Column(Text, default="[]")
    disliked_products = Column(Text, default="[]")
    session_history = Column(Text, default="[]")
    updated_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())


def init_db() -> None:
    global engine, SessionLocal
    settings = get_settings()
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    logger.info("database: initialized at %s", db_path)


def get_session() -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    return SessionLocal()
