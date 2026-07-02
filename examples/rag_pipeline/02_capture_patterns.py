"""
ctx-capture patterns beyond the quickstart. Each pattern_*() function is
runnable independently (`python -c "import importlib; ..."` or via a
REPL) or all together via __main__ below.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import ctxrun
from ctxrun import ChunkRecord, TokenBudget, TokenUsage, Turn, CacheEvent

PIPELINE = "rag_example"


def _sample_chunks():
    """Four chunks engineered to trigger ctx explain's window-dup, truncation, and low-score signals."""
    return [
        ChunkRecord(
            chunk_id="rrf_norm_1", source_doc_id="rrf_paper_2024",
            content="Reciprocal Rank Fusion normalizes scores from different retrieval systems",
            token_count=180, retrieval_score=0.85, rerank_score=0.92,
            retrieval_path="hybrid", cache_hit=True,
        ),
        ChunkRecord(
            chunk_id="rrf_norm_2", source_doc_id="rrf_paper_2024",
            content=(
                "Reciprocal Rank Fusion normalizes scores from different "
                "retrieval systems and ranks documents accordingly."
            ),
            token_count=160, retrieval_score=0.71, rerank_score=0.78,
            retrieval_path="bm25", cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="bm25_tf_idf", source_doc_id="ir_textbook_ch3",
            content="BM25 computes relevance using term frequency and inverse document frequency.",
            token_count=140, retrieval_score=0.82, rerank_score=0.88,
            retrieval_path="bm25", truncated=True, cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="ctx_window", source_doc_id="rag_patterns",
            content="Context window management determines which chunks survive token budget constraints.",
            token_count=145, retrieval_score=0.48, rerank_score=0.39,
            retrieval_path="bm25", cache_hit=False,
        ),
    ]


def pattern_full_fields():
    """Populate every optional RunRecord field -- chunks, context, history, cache, tool calls, response -- in one staged run."""
    run = ctxrun.start(query="what is RRF and how does it normalize scores?", pipeline=PIPELINE)

    chunks = _sample_chunks()
    run.chunks(chunks)

    prompt = (
        "System: answer using context.\n\nContext:\n"
        + "\n".join(f"[{i}] {c.content}" for i, c in enumerate(chunks, 1))
        + "\n\nQuery: what is RRF?"
    )
    run.context(prompt, TokenBudget(
        total_limit=4096, chunks_allocated=2800, history_allocated=600,
        system_allocated=500, headroom=196,
    ))

    run.history(
        pre=[
            Turn(role="user", content="Can you help me understand retrieval systems?", tokens=9),
            Turn(role="assistant", content="Of course!", tokens=4),
            Turn(role="user", content="Start with BM25.", tokens=5),
            Turn(role="assistant", content="BM25 ranks by term frequency.", tokens=8),
        ],
        post=[
            Turn(role="user", content="Can you help me understand retrieval systems?", tokens=9),
            Turn(role="assistant", content="Of course!", tokens=4),
        ],
        reason="token_budget",
    )

    run.cache([CacheEvent(chunk_id=c.chunk_id, hit=bool(c.cache_hit)) for c in chunks])

    run.tool_call({
        "tool_name": "rerank",
        "arguments": {"chunk_ids": [c.chunk_id for c in chunks]},
        "result": "reranked 4 chunks",
        "latency_ms": 42.0,
    })

    run.response(
        "RRF replaces raw retrieval scores with rank-based reciprocal values, "
        "making it robust to score-scale differences across retrievers.",
        token_usage=TokenUsage(input_tokens=1850, output_tokens=40, total_tokens=1890),
        model="gpt-4-turbo",
    )
    # run.commit() already called by run.response()


def _backdate_pipeline_runs(db_path: Path, pipeline: str, minutes: int) -> None:
    """test/demo-only: rewrites timestamps directly via raw SQL to simulate
    an idle gap. NOT part of the public ctx-capture API -- real pipelines
    never touch runs.db directly; session gaps happen naturally over
    wall-clock time between calls to ctxrun.start().
    """
    with sqlite3.connect(str(db_path)) as conn:
        for table, key_cols in [("sessions", ["session_id"]), ("runs", ["session_id", "run_seq"])]:
            rows = conn.execute(
                f"SELECT {', '.join(key_cols)}, created_at FROM {table} WHERE pipeline = ?",
                (pipeline,),
            ).fetchall()
            for row in rows:
                keys, old_ts = row[:-1], datetime.fromisoformat(row[-1])
                new_ts = (old_ts - timedelta(minutes=minutes)).isoformat()
                where = " AND ".join(f"{c} = ?" for c in key_cols)
                conn.execute(f"UPDATE {table} SET created_at = ? WHERE {where}", (new_ts, *keys))


def pattern_multi_session_gap():
    """Capture two query groups 31 minutes apart to trigger ctx-capture's auto session split."""
    for q in ["what is RRF?", "why does BM25 differ from vector similarity?"]:
        run = ctxrun.start(query=q, pipeline=PIPELINE)
        run.chunks(_sample_chunks())
        run.response(f"Answer to: {q}")

    db_path = Path.home() / ".ctx" / "runs.db"
    _backdate_pipeline_runs(db_path, PIPELINE, minutes=31)

    for q in ["what does a cross-encoder compute?", "when should reranking be skipped?"]:
        run = ctxrun.start(query=q, pipeline=PIPELINE)
        run.chunks(_sample_chunks())
        run.response(f"Answer to: {q}")


def pattern_thread_local_proxy():
    """Capture via module-level ctxrun.chunks()/response() -- no `run` object threaded through the call stack."""
    ctxrun.start(query="does rerank order affect final context assembly?", pipeline="proxy_demo")
    ctxrun.chunks([
        ChunkRecord(
            chunk_id="proxy_c1", source_doc_id="proxy_doc",
            content="Rerank order changes which chunks survive truncation.",
            token_count=20, retrieval_score=0.8, rerank_score=0.83,
        ),
    ])
    ctxrun.response("Yes -- rerank order determines what gets truncated when the budget is tight.")


if __name__ == "__main__":
    # pattern_full_fields() runs last so it ends up as the latest run --
    # `ctx explain` with no target shows the most recently captured run,
    # and this is the one that lights up every analysis factor.
    pattern_multi_session_gap()
    pattern_thread_local_proxy()
    pattern_full_fields()
    print("Captured capture-pattern demo runs. Try: ctx list")
