import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ctx_capture.schema import RunRecord

SCHEMA_VERSION = "1"

# Canonical sNrN run-id format, shared by ctx_evaluate and ctx so there's
# one parser for the format instead of a copy re-implemented per package.
TARGET_RE = re.compile(r"^s(\d+)r(\d+)$", re.IGNORECASE)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT,
    pipeline   TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    session_id  INTEGER NOT NULL REFERENCES sessions(session_id),
    run_seq     INTEGER NOT NULL,
    query       TEXT NOT NULL,
    pipeline    TEXT,
    created_at  TEXT NOT NULL,
    run_data    TEXT NOT NULL,
    PRIMARY KEY (session_id, run_seq)
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_query      ON runs(query);
CREATE INDEX IF NOT EXISTS idx_runs_pipeline   ON runs(pipeline);
"""


def _ctx_dir() -> Path:
    return Path.home() / ".ctx"


def _db_path() -> Path:
    return _ctx_dir() / "runs.db"


def _connect() -> sqlite3.Connection | None:
    """Row-factory connection for read/update call sites across ctx_evaluate
    and ctx. Returns None if the store doesn't exist yet -- callers treat
    a missing DB as "no data" rather than an error."""
    path = _db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def parse_target_id(target: str) -> tuple[int, int] | None:
    """Parse an sNrN run identifier into (session_id, run_seq), or None if
    `target` isn't in that format."""
    m = TARGET_RE.match(target)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def init_store() -> Path:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SCHEMA)
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
    return db_path


def get_or_create_session(pipeline, idle_gap_minutes=30) -> int:
    init_store()
    with sqlite3.connect(str(_db_path())) as conn:
        if pipeline is not None:
            row = conn.execute(
                "SELECT session_id, created_at FROM sessions "
                "WHERE pipeline = ? ORDER BY created_at DESC LIMIT 1",
                (pipeline,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT session_id, created_at FROM sessions "
                "WHERE pipeline IS NULL ORDER BY created_at DESC LIMIT 1",
            ).fetchone()

        if row is not None:
            session_id, session_created = row
            last_run = conn.execute(
                "SELECT created_at FROM runs WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            last_time = datetime.fromisoformat(
                last_run[0] if last_run else session_created
            )
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if (now - last_time).total_seconds() < idle_gap_minutes * 60:
                return session_id

        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            "INSERT INTO sessions (pipeline, created_at) VALUES (?, ?)",
            (pipeline, now_iso),
        )
        return cursor.lastrowid


def next_run_seq(session_id) -> int:
    with sqlite3.connect(str(_db_path())) as conn:
        row = conn.execute(
            "SELECT MAX(run_seq) FROM runs WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return (row[0] or 0) + 1


def write_run(session_id, run_seq, record: RunRecord, pipeline) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(_db_path())) as conn:
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                run_seq,
                record.query,
                pipeline,
                now,
                json.dumps(record.to_json()),
            ),
        )


def write_runs_batch(session_id: int, start_seq: int, records: list, pipeline: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (session_id, start_seq + i, record.query, pipeline, now,
         json.dumps(record.to_json()))
        for i, record in enumerate(records)
    ]
    with sqlite3.connect(str(_db_path())) as conn:
        conn.executemany(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
