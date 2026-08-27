"""
db.py - SQLite persistence for users (login) and run history (metrics).
"""

import sqlite3
import datetime
from pathlib import Path

import bcrypt
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "app.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            processing_time_ms REAL
        )
    """)
    conn.commit()

    # Seed a default user on first run only
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        default_hash = bcrypt.hashpw("changeme".encode(), bcrypt.gensalt()).decode()
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", default_hash),
        )
        conn.commit()

    conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return False
    return bcrypt.checkpw(password.encode(), row[0].encode())


def update_password(username: str, new_password: str) -> None:
    conn = get_connection()
    c = conn.cursor()
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    c.execute("UPDATE users SET password_hash = ? WHERE username = ?", (new_hash, username))
    conn.commit()
    conn.close()


def add_run(accuracy: float, precision_score: float, recall: float, processing_time_ms: float) -> None:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO runs (timestamp, accuracy, precision_score, recall, processing_time_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            accuracy,
            precision_score,
            recall,
            processing_time_ms,
        ),
    )
    conn.commit()
    conn.close()


def get_run_history() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY id ASC", conn)
    conn.close()
    return df