import re
import sqlite3
import sys
from pathlib import Path

EXPECTED_SCHEMA_VERSION = "1"

_TARGET_RE = re.compile(r"^s(\d+)r(\d+)$", re.IGNORECASE)


def _ctx_dir() -> Path:
    return Path.home() / ".ctx"


def get_db_path() -> Path:
    return _ctx_dir() / "runs.db"


def _connect() -> sqlite3.Connection | None:
    path = get_db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def check_schema_version() -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            print(
                "Warning: no schema version found in ctx database.",
                file=sys.stderr,
            )
        elif int(row["value"]) < int(EXPECTED_SCHEMA_VERSION):
            print(
                f"Warning: ctx database schema version {row['value']} "
                f"is older than minimum supported version {EXPECTED_SCHEMA_VERSION}. "
                f"Some features may not work correctly.",
                file=sys.stderr,
            )
    finally:
        conn.close()


def list_sessions(pipeline: str | None = None) -> list[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        sql = (
            "SELECT s.session_id, s.title, s.pipeline, s.created_at, "
            "COUNT(r.run_seq) as run_count "
            "FROM sessions s LEFT JOIN runs r ON s.session_id = r.session_id "
        )
        params: list = []
        if pipeline is not None:
            sql += "WHERE s.pipeline = ? "
            params.append(pipeline)
        sql += "GROUP BY s.session_id ORDER BY s.created_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def list_runs(session_id: int) -> list[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at "
            "FROM runs WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _run_select_cols(conn: sqlite3.Connection) -> str:
    base = "session_id, run_seq, query, pipeline, created_at, run_data"
    if _has_column(conn, "runs", "eval_scores"):
        base += ", eval_scores, risk_score, evaluated_at"
    return base


def get_run(session_id: int, run_seq: int) -> dict | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        cols = _run_select_cols(conn)
        row = conn.execute(
            f"SELECT {cols} FROM runs WHERE session_id = ? AND run_seq = ?",
            (session_id, run_seq),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_latest_run() -> dict | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        cols = _run_select_cols(conn)
        row = conn.execute(
            f"SELECT {cols} FROM runs ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _has_fts5(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs_fts'"
    ).fetchone()
    return row is not None


def search_runs(
    hint: str | None = None,
    exact: bool = False,
    session_id: int | None = None,
    pipeline: str | None = None,
    from_dt: str | None = None,
    to_dt: str | None = None,
    recent_n: int | None = None,
) -> list[dict]:
    from ctx_cli.find.query_builder import build_search_query

    conn = _connect()
    if conn is None:
        return []
    try:
        fts5 = _has_fts5(conn) if hint is not None else False
        sql, params = build_search_query(
            hint=hint,
            exact=exact,
            session_id=session_id,
            pipeline=pipeline,
            from_dt=from_dt,
            to_dt=to_dt,
            recent_n=recent_n,
            fts5_available=fts5,
        )
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def resolve_target(target: str | None = None) -> dict | list[dict] | None:
    if target is None:
        return get_latest_run()

    m = _TARGET_RE.match(target)
    if m:
        return get_run(int(m.group(1)), int(m.group(2)))

    results = search_runs(hint=target)
    if len(results) == 1:
        return get_run(results[0]["session_id"], results[0]["run_seq"])
    if len(results) > 1:
        from ctx_cli.find.bm25 import score

        results.sort(key=lambda r: score(target, r["query"]), reverse=True)
        return results
    return None


def rename_session(session_id: int, title: str) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?",
            (title, session_id),
        )
        conn.commit()
    finally:
        conn.close()
