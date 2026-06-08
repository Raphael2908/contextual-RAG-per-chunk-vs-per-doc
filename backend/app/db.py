"""Postgres connection pool + a tiny migration runner (applied on startup)."""

from __future__ import annotations

from pathlib import Path

from psycopg_pool import ConnectionPool

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def make_pool(database_url: str) -> ConnectionPool:
    pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=10, open=True)
    pool.wait()
    return pool


def run_migrations(pool: ConnectionPool) -> None:
    files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    with pool.connection() as conn:
        for path in files:
            conn.execute(path.read_text())
        conn.commit()
