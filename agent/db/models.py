"""
Database models.

Phase 1 only needs the Idea table. We're using SQLAlchemy (not raw SQL or
sqlite3) because Phase 2+ will add Script, Video, and PerformanceSnapshot
tables that reference each other (idea -> script -> video -> performance),
and an ORM makes those relationships much less error-prone than hand-written
JOINs. Starting with the right tool now saves a painful migration later.
"""

from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Idea(Base):
    __tablename__ = "ideas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[str] = mapped_column(String(64), index=True)

    title: Mapped[str] = mapped_column(String(255))
    hook: Mapped[str] = mapped_column(Text)
    angle: Mapped[str] = mapped_column(Text)
    why_now: Mapped[str] = mapped_column(Text)
    search_interest: Mapped[str] = mapped_column(String(16))   # High / Medium / Low
    competition: Mapped[str] = mapped_column(String(16))       # High / Medium / Low

    # pending_review -> approved -> scripted -> produced -> published -> rejected
    status: Mapped[str] = mapped_column(String(32), default="pending_review", index=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )

    def __repr__(self) -> str:
        return f"<Idea id={self.id} status={self.status!r} title={self.title!r}>"


class Script(Base):
    __tablename__ = "scripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idea_id: Mapped[int] = mapped_column(Integer, index=True)
    channel_id: Mapped[str] = mapped_column(String(64), index=True)

    hook: Mapped[str] = mapped_column(Text)
    # Scenes stored as a JSON string: [{"line_number", "text", "visual_cue",
    # "est_duration_seconds"}, ...]. SQLite has no native array/JSON column
    # type in SQLAlchemy's core set, and a full separate ScriptLine table is
    # more than Phase 2 needs - if scenes ever need independent querying
    # (e.g. "find all lines using visual cue X"), that's the signal to split
    # this into its own table.
    scenes_json: Mapped[str] = mapped_column(Text)
    cta: Mapped[str] = mapped_column(Text)

    estimated_duration_seconds: Mapped[float] = mapped_column()
    target_duration_seconds: Mapped[float] = mapped_column()

    # pending_review -> approved -> produced
    status: Mapped[str] = mapped_column(String(32), default="pending_review", index=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.UTC)
    )

    def __repr__(self) -> str:
        return f"<Script id={self.id} idea_id={self.idea_id} status={self.status!r}>"
