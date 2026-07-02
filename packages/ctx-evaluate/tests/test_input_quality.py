from ctx_capture.schema import RunRecord, ChunkRecord, TokenBudget
from ctx_evaluate.layers.input_quality import score_input_quality, cosine_similarity
from ctx_evaluate.policy.schema import InputQualityPolicy


class TestEmptyRecord:
    def test_no_chunks(self):
        rec = RunRecord(query="q", response="r")
        assert score_input_quality(rec, InputQualityPolicy()) is None

    def test_empty_chunks(self):
        rec = RunRecord(query="q", response="r", chunks=[])
        assert score_input_quality(rec, InputQualityPolicy()) is None


class TestRelevance:
    def test_uses_rerank_score_when_no_embedding(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[ChunkRecord(
                chunk_id="c1", source_doc_id="d1", content="text",
                token_count=10, rerank_score=0.9,
            )],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert 0.9 in result["relevance_scores"]

    def test_uses_retrieval_score_as_fallback(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[ChunkRecord(
                chunk_id="c1", source_doc_id="d1", content="text",
                token_count=10, retrieval_score=0.7,
            )],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert 0.7 in result["relevance_scores"]

    def test_embedding_fn_used_when_provided(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[ChunkRecord(
                chunk_id="c1", source_doc_id="d1", content="text",
                token_count=10,
            )],
        )
        fake_embed = lambda text: [1.0, 0.0, 0.0]
        result = score_input_quality(rec, InputQualityPolicy(), embedding_fn=fake_embed)
        assert result["relevance_scores"][0] == 1.0


class TestDuplicates:
    def test_path_duplicate_detection(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[
                ChunkRecord(chunk_id="c1", source_doc_id="d1", content="text A",
                            token_count=10, retrieval_path="bm25"),
                ChunkRecord(chunk_id="c1", source_doc_id="d1", content="text A",
                            token_count=10, retrieval_path="ann"),
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["path_dup_count"] == 1
        assert result["duplicate_ratio"] > 0

    def test_window_duplicate_detection(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[
                ChunkRecord(chunk_id="c1", source_doc_id="d1",
                            content="the quick brown fox", token_count=10),
                ChunkRecord(chunk_id="c2", source_doc_id="d1",
                            content="the quick brown fox jumps over", token_count=15),
            ],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["window_dup_count"] == 1


class TestTruncation:
    def test_high_score_truncation(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[ChunkRecord(
                chunk_id="c1", source_doc_id="d1", content="text",
                token_count=10, rerank_score=0.85, truncated=True,
            )],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["high_score_truncations"] == 1
        assert result["truncation_severity"] == "high"

    def test_low_score_truncation(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[ChunkRecord(
                chunk_id="c1", source_doc_id="d1", content="text",
                token_count=10, rerank_score=0.3, truncated=True,
            )],
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["truncation_severity"] == "low"


class TestPolicy:
    def test_violation_detected(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[
                ChunkRecord(chunk_id="c1", source_doc_id="d1", content="a",
                            token_count=10, retrieval_path="bm25"),
                ChunkRecord(chunk_id="c1", source_doc_id="d1", content="a",
                            token_count=10, retrieval_path="ann"),
            ],
        )
        policy = InputQualityPolicy(max_duplicate_ratio=0.0)
        result = score_input_quality(rec, policy)
        assert "max_duplicate_ratio" in result["policy_violations"]
        assert not result["passes_policy"]

    def test_passes_on_clean_record(self):
        rec = RunRecord(
            query="q", response="r",
            chunks=[
                ChunkRecord(chunk_id="c1", source_doc_id="d1", content="good chunk",
                            token_count=10, rerank_score=0.9),
                ChunkRecord(chunk_id="c2", source_doc_id="d2", content="another good",
                            token_count=10, rerank_score=0.85),
            ],
            token_budget=TokenBudget(
                total_limit=4096, chunks_allocated=2000,
                history_allocated=500, system_allocated=800, headroom=796,
            ),
        )
        result = score_input_quality(rec, InputQualityPolicy())
        assert result["passes_policy"]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
