import json
import sqlite3

import pytest

from ctx_capture.schema import (
    RunRecord, ChunkRecord, TokenBudget, TokenUsage, Turn, CacheEvent,
)
from ctx_capture.store import SCHEMA


@pytest.fixture(autouse=True)
def ctx_home(tmp_path, monkeypatch):
    ctx_dir = tmp_path / ".ctx"
    # ctx_evaluate.store re-exports ctx_capture.store's _ctx_dir rather than
    # defining its own, so patching the one canonical home covers both.
    monkeypatch.setattr("ctx_capture.store._ctx_dir", lambda: ctx_dir)
    return tmp_path


def _full_record():
    return RunRecord(
        query="does RRF handle score scale differences",
        response="Yes, RRF normalizes scores via reciprocal rank.",
        chunks=[
            ChunkRecord(
                chunk_id="c1", source_doc_id="d1",
                content="RRF normalizes retrieval scores across methods",
                token_count=50, retrieval_score=0.9, rerank_score=0.85,
                retrieval_path="bm25", truncated=False, cache_hit=True,
            ),
            ChunkRecord(
                chunk_id="c2", source_doc_id="d2",
                content="Score fusion combines signals from multiple retrievers",
                token_count=30, retrieval_score=0.7, rerank_score=0.4,
                retrieval_path="ann", truncated=True, cache_hit=False,
            ),
        ],
        final_prompt="System: answer\nContext: ...\nQuery: RRF",
        token_budget=TokenBudget(
            total_limit=4096, chunks_allocated=2000,
            history_allocated=500, system_allocated=800, headroom=796,
        ),
        history_pre=[
            Turn(role="user", content="hello", tokens=3),
            Turn(role="assistant", content="hi there", tokens=5),
        ],
        history_post=[Turn(role="user", content="hello", tokens=3)],
        eviction_reason="token_budget",
        cache_events=[
            CacheEvent(chunk_id="c1", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c2", hit=False),
        ],
        model="gpt-4",
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, total_tokens=350),
    )


@pytest.fixture
def v1_db(ctx_home):
    """Create a v1 schema database with existing data."""
    db_path = ctx_home / ".ctx" / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rec = _full_record()
    rec_min = RunRecord(query="what is BM25", response="a ranking function")

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        conn.execute(
            "INSERT INTO sessions VALUES (1, NULL, 'pipe_a', '2026-06-08T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO sessions VALUES (2, 'RRF investigation', 'pipe_a', '2026-06-09T10:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (1, 1, ?, 'pipe_a', '2026-06-08T10:05:00+00:00', ?)",
            (rec_min.query, json.dumps(rec_min.to_json())),
        )
        conn.execute(
            "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, run_data) "
            "VALUES (2, 1, ?, 'pipe_a', '2026-06-09T10:05:00+00:00', ?)",
            (rec.query, json.dumps(rec.to_json())),
        )

    return db_path


@pytest.fixture
def migrated_db(v1_db):
    """v1 database after migration to v2."""
    from ctx_evaluate.store import apply_migration
    apply_migration()
    return v1_db


@pytest.fixture
def full_record():
    return _full_record()
