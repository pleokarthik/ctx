import json
import sqlite3

import ctx_capture
from ctx_capture import store
from ctx_capture.thread_local import get_active_run, clear_active_run


class TestCapture:
    def test_minimal_capture(self):
        ctx_capture.capture("what is RRF?", "reciprocal rank fusion")
        with sqlite3.connect(str(store._db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        assert row is not None
        data = json.loads(row[0])
        assert data["query"] == "what is RRF?"
        assert data["response"] == "reciprocal rank fusion"

    def test_capture_with_all_fields(self):
        ctx_capture.capture(
            "test query",
            "test response",
            pipeline="pipe",
            chunks=[{
                "chunk_id": "c1",
                "source_doc_id": "d1",
                "content": "text",
                "token_count": 10,
            }],
            final_prompt="assembled prompt",
            token_budget={
                "total_limit": 4096,
                "chunks_allocated": 2000,
                "history_allocated": 500,
                "system_allocated": 800,
                "headroom": 796,
            },
            model="gpt-4",
            token_usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            cache_events=[{"chunk_id": "c1", "hit": True}],
        )
        with sqlite3.connect(str(store._db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        data = json.loads(row[0])
        assert data["model"] == "gpt-4"
        assert data["chunks"][0]["chunk_id"] == "c1"
        assert data["token_budget"]["headroom"] == 796


class TestStartAndRun:
    def setup_method(self):
        clear_active_run()

    def test_start_chunks_response_commits(self):
        run = ctx_capture.start("test query", pipeline="test")
        run.chunks([{
            "chunk_id": "c1",
            "source_doc_id": "d1",
            "content": "hello",
            "token_count": 5,
        }])
        run.response("test response")

        with sqlite3.connect(str(store._db_path())) as conn:
            row = conn.execute("SELECT run_data FROM runs").fetchone()
        data = json.loads(row[0])
        assert data["query"] == "test query"
        assert data["response"] == "test response"
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "c1"


class TestFailureSilence:
    def test_capture_failure_never_raises(self, monkeypatch):
        def fail(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr("ctx_capture.store.get_or_create_session", fail)
        ctx_capture.capture("q", "r")

    def test_run_method_failure_never_raises(self):
        run = ctx_capture.start("q", pipeline="test")
        run.chunks("not a list of dicts")
        run.context(None)
        run.history(None, None)
        run.cache("bad")

    def test_commit_failure_never_raises(self, monkeypatch):
        def fail(*a, **k):
            raise RuntimeError("db error")

        monkeypatch.setattr("ctx_capture.store.write_run", fail)
        run = ctx_capture.start("q", pipeline="test")
        run._record.response = "r"
        run.commit()


class TestThreadLocal:
    def setup_method(self):
        clear_active_run()

    def test_proxy_routes_to_active_run(self):
        run = ctx_capture.start("proxy test", pipeline="test")
        ctx_capture.chunks([{
            "chunk_id": "c1",
            "source_doc_id": "d1",
            "content": "hello",
            "token_count": 5,
        }])
        assert run._record.chunks is not None
        assert len(run._record.chunks) == 1
        assert run._record.chunks[0].chunk_id == "c1"

        ctx_capture.response("proxy response")
        assert run._committed

    def test_proxy_without_active_run_is_silent(self):
        ctx_capture.chunks([])
        ctx_capture.context("prompt")
        ctx_capture.response("r")
        ctx_capture.commit()

    def test_active_run_accessible(self):
        run = ctx_capture.start("q", pipeline="test")
        assert get_active_run() is run


class TestCommitIdempotent:
    def test_double_commit_writes_once(self):
        run = ctx_capture.start("q", pipeline="test")
        run._record.response = "r"
        run.commit()
        run.commit()

        with sqlite3.connect(str(store._db_path())) as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 1
