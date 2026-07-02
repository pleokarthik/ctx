import json
from datetime import datetime, timezone

from ctx_capture.store import _connect, _column_exists  # noqa: F401 (re-exported for callers)


def apply_migration() -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        version = row["value"] if row else None

        if version == "3":
            return

        if version not in ("1", "2"):
            raise RuntimeError(
                f"Unsupported schema version: {version!r}. "
                f"Expected '1', '2', or '3'. Cannot migrate."
            )

        if version == "1":
            for col, col_type in [
                ("eval_scores", "TEXT"),
                ("risk_score", "REAL"),
                ("evaluated_at", "TEXT"),
            ]:
                if not _column_exists(conn, "runs", col):
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")

            conn.execute(
                """CREATE TABLE IF NOT EXISTS benchmark (
                    pipeline      TEXT NOT NULL,
                    factor        TEXT NOT NULL,
                    threshold     REAL,
                    correlation   REAL,
                    sample_count  INTEGER NOT NULL DEFAULT 0,
                    updated_at    TEXT NOT NULL,
                    PRIMARY KEY (pipeline, factor)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS policies (
                    pipeline     TEXT PRIMARY KEY,
                    policy_data  TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )"""
            )
            conn.execute(
                "UPDATE meta SET value = '2' WHERE key = 'schema_version'"
            )
            conn.commit()
            version = "2"

        if version == "2":
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts
                USING fts5(
                    query,
                    content=runs,
                    content_rowid=rowid,
                    tokenize='unicode61 remove_diacritics 1'
                )"""
            )
            conn.execute("INSERT INTO runs_fts(runs_fts) VALUES('rebuild')")
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_ins
                AFTER INSERT ON runs BEGIN
                    INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_del
                AFTER DELETE ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
                END"""
            )
            conn.execute(
                """CREATE TRIGGER IF NOT EXISTS runs_fts_upd
                AFTER UPDATE OF query ON runs BEGIN
                    INSERT INTO runs_fts(runs_fts, rowid, query)
                    VALUES ('delete', old.rowid, old.query);
                    INSERT INTO runs_fts(rowid, query) VALUES (new.rowid, new.query);
                END"""
            )
            conn.execute("DROP INDEX IF EXISTS idx_runs_query")
            conn.execute(
                "UPDATE meta SET value = '3' WHERE key = 'schema_version'"
            )
            conn.commit()
    finally:
        conn.close()


def get_run(session_id: int, run_seq: int) -> dict | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at "
            "FROM runs WHERE session_id = ? AND run_seq = ?",
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
        row = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at "
            "FROM runs ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_runs_in_session(session_id: int) -> list[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at "
            "FROM runs WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def write_eval_scores(
    session_id: int,
    run_seq: int,
    eval_scores: dict,
    risk_score: float,
) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE runs SET eval_scores = ?, risk_score = ?, evaluated_at = ? "
            "WHERE session_id = ? AND run_seq = ?",
            (json.dumps(eval_scores), risk_score, now, session_id, run_seq),
        )
        conn.commit()
    finally:
        conn.close()


def write_eval_scores_batch(entries: list[tuple]) -> None:
    """Write eval scores for multiple runs in a single transaction.

    Each entry is (session_id, run_seq, eval_scores_dict, risk_score).
    """
    if not entries:
        return
    conn = _connect()
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (json.dumps(eval_scores), risk_score, now, session_id, run_seq)
            for session_id, run_seq, eval_scores, risk_score in entries
        ]
        conn.executemany(
            "UPDATE runs SET eval_scores = ?, risk_score = ?, evaluated_at = ? "
            "WHERE session_id = ? AND run_seq = ?",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_eval_scores(session_id: int, run_seq: int) -> dict | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT eval_scores, risk_score FROM runs "
            "WHERE session_id = ? AND run_seq = ?",
            (session_id, run_seq),
        ).fetchone()
        if row is None or row["eval_scores"] is None:
            return None
        result = json.loads(row["eval_scores"])
        result["risk_score"] = row["risk_score"]
        return result
    finally:
        conn.close()


def write_benchmark_entry(
    pipeline: str,
    factor: str,
    threshold: float,
    correlation: float,
    sample_count: int,
) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO benchmark "
            "(pipeline, factor, threshold, correlation, sample_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (pipeline, factor, threshold, correlation, sample_count, now),
        )
        conn.commit()
    finally:
        conn.close()


def write_benchmark_entries_batch(entries: list[tuple]) -> None:
    """Write multiple benchmark entries in a single transaction.

    Each entry is (pipeline, factor, threshold, correlation, sample_count).
    """
    if not entries:
        return
    conn = _connect()
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(p, f, t, c, s, now) for p, f, t, c, s in entries]
        conn.executemany(
            "INSERT OR REPLACE INTO benchmark "
            "(pipeline, factor, threshold, correlation, sample_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def get_benchmark(pipeline: str) -> list[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT factor, threshold, correlation, sample_count, updated_at "
            "FROM benchmark WHERE pipeline = ?",
            (pipeline,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_evaluated_runs(pipeline: str | None = None) -> list[dict]:
    conn = _connect()
    if conn is None:
        return []
    try:
        sql = (
            "SELECT session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at "
            "FROM runs WHERE eval_scores IS NOT NULL"
        )
        params: list = []
        if pipeline is not None:
            sql += " AND pipeline = ?"
            params.append(pipeline)
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def write_policy(pipeline: str, policy: dict) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO policies (pipeline, policy_data, updated_at) "
            "VALUES (?, ?, ?)",
            (pipeline, json.dumps(policy), now),
        )
        conn.commit()
    finally:
        conn.close()


def get_policy(pipeline: str) -> dict | None:
    conn = _connect()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT policy_data FROM policies WHERE pipeline = ?",
            (pipeline,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["policy_data"])
    finally:
        conn.close()
