"""
Database session management.

One SQLite file (data/agent.db) holds data for ALL channels - rows are
distinguished by channel_id. This is deliberately simple for now (SQLite has
no server to run, no config); if this ever needs to run on a server with
concurrent writers, swapping the DB_URL for a Postgres connection string is
the only change needed, because everything else goes through SQLAlchemy.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from agent.db.models import Base

DB_PATH = Path("data/agent.db")
DB_URL = f"sqlite:///{DB_PATH}"

_engine = None
_SessionLocal = None


def init_db() -> None:
    """Create the database file and tables if they don't exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    global _engine, _SessionLocal
    _engine = create_engine(DB_URL)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


@contextmanager
def get_session() -> Session:
    """Usage: with get_session() as session: session.add(obj); session.commit()"""
    if _SessionLocal is None:
        init_db()
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
