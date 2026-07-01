import math
from typing import Callable

from ctx_capture.schema import RunRecord
from ctx_evaluate.policy.schema import InputQualityPolicy

SEMANTIC_DUP_THRESHOLD = 0.92


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _detect_path_dups(chunks) -> int:
    chunk_paths: dict[str, list[str]] = {}
    for c in chunks:
        chunk_paths.setdefault(c.chunk_id, [])
        if c.retrieval_path:
            chunk_paths[c.chunk_id].append(c.retrieval_path)
    return sum(1 for paths in chunk_paths.values() if len(paths) > 1)


def _detect_window_dups(chunks) -> int:
    by_source: dict[str, list] = {}
    for c in chunks:
        by_source.setdefault(c.source_doc_id, []).append(c)

    count = 0
    for group in by_source.values():
        if len(group) < 2:
            continue
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                pair = tuple(sorted([a.chunk_id, b.chunk_id]))
                if pair in seen:
                    continue
                if a.content in b.content or b.content in a.content:
                    seen.add(pair)
                    count += 1
                    continue
                tokens_a = set(a.content.lower().split())
                tokens_b = set(b.content.lower().split())
                if tokens_a and tokens_b:
                    overlap = len(tokens_a & tokens_b)
                    total = max(len(tokens_a), len(tokens_b))
                    if overlap / total > 0.5:
                        seen.add(pair)
                        count += 1
    return count


def _detect_semantic_dups(chunks, embedding_fn) -> int:
    if embedding_fn is None:
        return 0
    embeddings = [(c, embedding_fn(c.content)) for c in chunks]
    count = 0
    for i, (ca, va) in enumerate(embeddings):
        for cb, vb in embeddings[i + 1 :]:
            if ca.source_doc_id == cb.source_doc_id:
                continue
            if cosine_similarity(va, vb) > SEMANTIC_DUP_THRESHOLD:
                count += 1
    return count


def score(
    record: RunRecord,
    policy: InputQualityPolicy,
    embedding_fn: Callable | None = None,
) -> dict | None:
    if not record.chunks:
        return None

    chunks = record.chunks
    total = len(chunks)

    # --- Relevance ---
    relevance_scores: list[float] = []
    if embedding_fn is not None:
        query_vec = embedding_fn(record.query)
        for c in chunks:
            chunk_vec = embedding_fn(c.content)
            relevance_scores.append(cosine_similarity(query_vec, chunk_vec))
    else:
        for c in chunks:
            if c.rerank_score is not None:
                relevance_scores.append(c.rerank_score)
            elif c.retrieval_score is not None:
                relevance_scores.append(c.retrieval_score)

    mean_relevance = (
        sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    )

    rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    top_chunk_score = max(rerank_scores) if rerank_scores else None

    # --- Duplicates ---
    path_dup_count = _detect_path_dups(chunks)
    window_dup_count = _detect_window_dups(chunks)
    semantic_dup_count = _detect_semantic_dups(chunks, embedding_fn)
    duplicate_ratio = (path_dup_count + window_dup_count) / total if total else 0.0

    # --- Truncation ---
    truncated = [c for c in chunks if c.truncated]
    truncated_count = len(truncated)
    high_score_truncations = sum(
        1
        for c in truncated
        if (c.retrieval_score or 0) > 0.7 or (c.rerank_score or 0) > 0.7
    )
    if not truncated:
        truncation_severity = "none"
    elif high_score_truncations > 0:
        truncation_severity = "high"
    else:
        truncation_severity = "low"

    # --- Token efficiency ---
    token_headroom_pct = 0.0
    if record.token_budget and record.token_budget.total_limit > 0:
        token_headroom_pct = record.token_budget.headroom / record.token_budget.total_limit

    low_score_chunks = sum(
        1
        for c in chunks
        if (c.rerank_score is not None and c.rerank_score < 0.5)
        or (
            c.rerank_score is None
            and c.retrieval_score is not None
            and c.retrieval_score < 0.5
        )
    )
    low_score_chunk_ratio = low_score_chunks / total if total else 0.0

    # --- Coherence ---
    source_domain_count = len({c.source_doc_id for c in chunks})

    score_variance = None
    if len(rerank_scores) > 1:
        mean = sum(rerank_scores) / len(rerank_scores)
        score_variance = round(
            sum((s - mean) ** 2 for s in rerank_scores) / len(rerank_scores), 4
        )

    # --- Policy violations ---
    violations: list[str] = []
    if duplicate_ratio > policy.max_duplicate_ratio:
        violations.append("max_duplicate_ratio")
    if top_chunk_score is not None and top_chunk_score < policy.min_top_chunk_score:
        violations.append("min_top_chunk_score")
    if high_score_truncations > policy.max_high_score_truncations:
        violations.append("max_high_score_truncations")
    if low_score_chunk_ratio > policy.max_low_score_chunk_ratio:
        violations.append("max_low_score_chunk_ratio")
    if record.token_budget and token_headroom_pct < policy.min_token_headroom:
        violations.append("min_token_headroom")
    if source_domain_count > policy.max_source_domains:
        violations.append("max_source_domains")
    if mean_relevance < policy.min_chunk_relevance_score and relevance_scores:
        violations.append("min_chunk_relevance_score")

    return {
        "relevance_scores": [round(s, 4) for s in relevance_scores],
        "mean_relevance": round(mean_relevance, 4),
        "top_chunk_score": top_chunk_score,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "path_dup_count": path_dup_count,
        "window_dup_count": window_dup_count,
        "semantic_dup_count": semantic_dup_count,
        "truncated_count": truncated_count,
        "high_score_truncations": high_score_truncations,
        "truncation_severity": truncation_severity,
        "token_headroom_pct": round(token_headroom_pct, 4),
        "low_score_chunk_ratio": round(low_score_chunk_ratio, 4),
        "source_domain_count": source_domain_count,
        "score_variance": score_variance,
        "policy_violations": violations,
        "passes_policy": len(violations) == 0,
    }
