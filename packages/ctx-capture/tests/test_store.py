import sqlite3
from datetime import datetime, timezone, timedelta

from ctx_capture import store
from ctx_capture.schema import RunRecord


class TestInitStore:
    def test_creates_db(self):
        path = store.init_store()
        assert path.exists()
        assert path.name == "runs.db"

    def test_schema_version_in_meta(self):
        store.init_store()
        with sqlite3.connect(str(store._db_path())) as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        assert row is not None
        assert row[0] == store.SCHEMA_VERSION

    def test_idempotent(self):
        store.init_store()
        store.init_store()
        with sqlite3.connect(str(store._db_path())) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM meta").fetchone()
        assert rows[0] == 1


class TestGetOrCreateSession:
    def test_creates_new_session(self):
        sid = store.get_or_create_session("test_pipe")
        assert sid >= 1

    def test_reuses_session_within_gap(self):
        s1 = store.get_or_create_session("test_pipe")
        s2 = store.get_or_create_session("test_pipe")
        assert s1 == s2

    def test_creates_new_session_after_gap(self):
        s1 = store.get_or_create_session("test_pipe")
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        with sqlite3.connect(str(store._db_path())) as conn:
            conn.execute(
                "UPDATE sessions SET created_at = ? WHERE session_id = ?",
                (old_time, s1),
            )
        s2 = store.get_or_create_session("test_pipe")
        assert s2 != s1

    def test_separate_pipelines_get_separate_sessions(self):
        s1 = store.get_or_create_session("pipe_a")
        s2 = store.get_or_create_session("pipe_b")
        assert s1 != s2

    def test_none_pipeline(self):
        s1 = store.get_or_create_session(None)
        s2 = store.get_or_create_session(None)
        assert s1 == s2


class TestWriteRun:
    def test_writes_and_is_retrievable(self):
        sid = store.get_or_create_session("test_pipe")
        seq = store.next_run_seq(sid)
        rec = RunRecord(query="test query", response="test response")
        store.write_run(sid, seq, rec, "test_pipe")

        with sqlite3.connect(str(store._db_path())) as conn:
            row = conn.execute(
                "SELECT query, pipeline, run_data FROM runs WHERE session_id = ? AND run_seq = ?",
                (sid, seq),
            ).fetchone()

        assert row is not None
        assert row[0] == "test query"
        assert row[1] == "test_pipe"
        assert '"test response"' in row[2]

    def test_next_run_seq_increments(self):
        sid = store.get_or_create_session("test_pipe")
        assert store.next_run_seq(sid) == 1
        rec = RunRecord(query="q", response="r")
        store.write_run(sid, 1, rec, "test_pipe")
        assert store.next_run_seq(sid) == 2
