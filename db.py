"""
db.py - SQLite persistence for saved analysis run history.

Note: login/user management now lives in Firebase (see firebase_auth.py).
This file handles run history, role assignment, and feedback.
"""

import sqlite3
import datetime
import json
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

    c.execute("PRAGMA table_info(runs)")
    existing_cols = [row[1] for row in c.fetchall()]

    migrations = [
        ("created_by", "TEXT"),
        ("is_blurry", "INTEGER DEFAULT 0"),
        ("calibrated", "INTEGER DEFAULT 0"),
        ("detections_json", "TEXT DEFAULT '[]'"),
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            c.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")

    c.execute("""
        CREATE TABLE IF NOT EXISTS user_roles (
            email TEXT PRIMARY KEY,
            role TEXT NOT NULL DEFAULT 'client'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            timestamp TEXT NOT NULL,
            created_by TEXT,
            rating TEXT NOT NULL,
            comment TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
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


def add_run(
    created_by: str,
    fruit_count: int,
    avg_confidence: float,
    processing_time_ms: float,
    is_blurry: bool = False,
    calibrated: bool = False,
    detections: list = None,
) -> int:
    """Returns the new run's id, so feedback can be linked to it."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO runs
            (timestamp, created_by, fruit_count, avg_confidence, processing_time_ms,
             is_blurry, calibrated, detections_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            created_by,
            fruit_count,
            avg_confidence,
            processing_time_ms,
            int(bool(is_blurry)),
            int(bool(calibrated)),
            json.dumps(detections or []),
        ),
    )
    conn.commit()
    run_id = c.lastrowid
    conn.close()
    return run_id


def add_feedback(run_id: int, created_by: str, rating: str, comment: str) -> None:
    """rating should be 'good' or 'bad'."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO feedback (run_id, timestamp, created_by, rating, comment)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            run_id,
            datetime.datetime.now().isoformat(timespec="seconds"),
            created_by,
            rating,
            comment,
        ),
    )
    conn.commit()
    conn.close()


def get_feedback_history() -> pd.DataFrame:
    """Returns all feedback, joined with basic run info, newest first."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            feedback.id, feedback.timestamp, feedback.created_by,
            feedback.rating, feedback.comment,
            runs.fruit_count, runs.avg_confidence
        FROM feedback
        LEFT JOIN runs ON feedback.run_id = runs.id
        ORDER BY feedback.id DESC
        """,
        conn,
    )
    conn.close()
    return df


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