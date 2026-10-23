"""Database connection (SQLite by default) for users and chat history."""

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

import config

# SQLite needs check_same_thread=False because FastAPI serves requests from several threads.
connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(config.DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def add_missing_columns() -> None:
    """Add columns introduced after a database was created.

    create_all() only creates missing tables, not missing columns, so databases
    from earlier versions need chat_history.sources added once.
    """
    columns = {column["name"] for column in inspect(engine).get_columns("chat_history")}
    if "sources" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE chat_history ADD COLUMN sources JSON"))


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
