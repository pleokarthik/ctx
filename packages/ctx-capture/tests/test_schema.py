from ctx_capture.schema import (
    ChunkRecord,
    TokenBudget,
    TokenUsage,
    Turn,
    CacheEvent,
    RunRecord,
)


def _full_record():
    return RunRecord(
        query="what is RRF?",
        response="RRF is reciprocal rank fusion.",
        chunks=[
            ChunkRecord(
                chunk_id="c1",
                source_doc_id="doc1",
                content="RRF combines scores",
                token_count=50,
                retrieval_score=0.9,
                rerank_score=0.85,
                retrieval_path="hybrid",
                truncated=False,
                cache_hit=True,
            ),
            ChunkRecord(
                chunk_id="c2",
                source_doc_id="doc2",
                content="BM25 baseline",
                token_count=30,
                retrieval_score=0.7,
            ),
        ],
        final_prompt="System: ...\nContext: ...\nQuery: what is RRF?",
        token_budget=TokenBudget(
            total_limit=4096,
            chunks_allocated=2000,
            history_allocated=500,
            system_allocated=800,
            headroom=796,
        ),
        history_pre=[
            Turn(role="user", content="hello", tokens=3),
            Turn(role="assistant", content="hi there", tokens=5),
        ],
        history_post=[
            Turn(role="user", content="hello", tokens=3),
        ],
        eviction_reason="token_budget",
        cache_events=[
            CacheEvent(chunk_id="c1", hit=True, cache_source="disk"),
            CacheEvent(chunk_id="c2", hit=False),
        ],
        model="gpt-4",
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, total_tokens=350),
    )


class TestMinimalRunRecord:
    def test_serialises(self):
        rec = RunRecord(query="q", response="r")
        data = rec.to_json()
        assert data["query"] == "q"
        assert data["response"] == "r"
        assert data["chunks"] is None

    def test_deserialises(self):
        data = {"query": "q", "response": "r"}
        rec = RunRecord.from_json(data)
        assert rec.query == "q"
        assert rec.response == "r"
        assert rec.chunks is None

    def test_round_trip(self):
        original = RunRecord(query="q", response="r")
        restored = RunRecord.from_json(original.to_json())
        assert original.to_json() == restored.to_json()


class TestFullRunRecord:
    def test_serialises(self):
        rec = _full_record()
        data = rec.to_json()
        assert data["query"] == "what is RRF?"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "c1"
        assert data["token_budget"]["total_limit"] == 4096
        assert data["token_usage"]["total_tokens"] == 350
        assert len(data["history_pre"]) == 2
        assert len(data["history_post"]) == 1
        assert len(data["cache_events"]) == 2
        assert data["model"] == "gpt-4"
        assert data["eviction_reason"] == "token_budget"

    def test_deserialises(self):
        data = _full_record().to_json()
        rec = RunRecord.from_json(data)
        assert isinstance(rec.chunks[0], ChunkRecord)
        assert rec.chunks[0].retrieval_score == 0.9
        assert isinstance(rec.token_budget, TokenBudget)
        assert rec.token_budget.headroom == 796
        assert isinstance(rec.history_pre[0], Turn)
        assert rec.history_pre[0].tokens == 3
        assert isinstance(rec.cache_events[0], CacheEvent)
        assert rec.cache_events[0].cache_source == "disk"
        assert isinstance(rec.token_usage, TokenUsage)

    def test_round_trip(self):
        original = _full_record()
        restored = RunRecord.from_json(original.to_json())
        assert original.to_json() == restored.to_json()


class TestFlexibleInit:
    def test_unknown_kwargs_ignored(self):
        rec = RunRecord(query="q", response="r", unknown_field="x", another=42)
        assert rec.query == "q"
        assert not hasattr(rec, "unknown_field")

    def test_chunk_unknown_kwargs_ignored(self):
        c = ChunkRecord(
            chunk_id="c1",
            source_doc_id="d1",
            content="text",
            token_count=10,
            future_field="ignored",
        )
        assert c.chunk_id == "c1"
        assert not hasattr(c, "future_field")

    def test_from_json_ignores_unknown_fields(self):
        data = {"query": "q", "response": "r", "new_field": True}
        rec = RunRecord.from_json(data)
        assert rec.query == "q"
