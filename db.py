"""
db.py - SQLite persistence for saved analysis run history.

Note: login/user management now lives in Firebase (see firebase_auth.py).
This file handles run history and role assignment.
"""

import sqlite3
import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "app.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            created_by TEXT,
            fruit_count INTEGER,
            avg_confidence REAL,
            processing_time_ms REAL
        )
    """)

    # Migrate DBs created before created_by existed
    c.execute("PRAGMA table_info(runs)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "created_by" not in existing_cols:
        c.execute("ALTER TABLE runs ADD COLUMN created_by TEXT")

    c.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            email TEXT PRIMARY KEY,
            role TEXT NOT NULL DEFAULT 'client'
        )
    """)
    conn.commit()
    conn.close()


def get_role(email: str) -> str:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT role FROM user_roles WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else "client"


def set_role(email: str, role: str) -> None:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO user_roles (email, role) VALUES (?, ?) "
        "ON CONFLICT(email) DO UPDATE SET role = excluded.role",
        (email, role),
    )
    conn.commit()
    conn.close()


def add_run(created_by: str, fruit_count: int, avg_confidence: float, processing_time_ms: float) -> None:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO runs (timestamp, created_by, fruit_count, avg_confidence, processing_time_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            created_by,
            fruit_count,
            avg_confidence,
            processing_time_ms,
        ),
    )
    conn.commit()
    conn.close()


def get_run_history(created_by: str = None) -> pd.DataFrame:
    """
    Pass created_by to scope to one user (clients).
    Pass nothing to get every run (admins).
    """
    conn = get_connection()
    if created_by:
        df = pd.read_sql_query(
            "SELECT * FROM runs WHERE created_by = ? ORDER BY id DESC", conn, params=(created_by,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)
    conn.close()
    return df