from collections.abc import AsyncIterator
from datetime import UTC, datetime

from sqlalchemy import JSON, MetaData, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from nodeloom.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


# Deterministic constraint names, so Alembic migrations are stable.
NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map = {dict: JSON, list: JSON}


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        url = get_settings().db_url
        # a proxy in front of PostgreSQL (HAProxy, pgbouncer) may drop idle connections: check a pooled connection
        # before use and renew it before a typical 30-minute timeout, or the next request gets 500
        _engine = (create_async_engine(url) if url.startswith("sqlite")
                   else create_async_engine(url, pool_pre_ping=True, pool_recycle=1500))
        if url.startswith("sqlite"):
            @event.listens_for(_engine.sync_engine, "connect")
            def _sqloom_pragmas(conn, _):
                cur = conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.close()
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


def is_sqlite() -> bool:
    return get_settings().db_url.startswith("sqlite")


async def init_db() -> None:
    """Create the tables that are missing (no migrations yet: the schema has not changed since the first release)."""
    from nodeloom import models  # noqa: F401  registers the tables

    if is_sqlite():
        get_settings().data_dir.mkdir(parents=True, exist_ok=True)
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker = None, None


async def get_session() -> AsyncIterator[AsyncSession]:
    async with sessionmaker()() as session:
        yield session
