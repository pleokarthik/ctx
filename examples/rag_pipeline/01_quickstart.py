"""
ctx-capture quickstart -- the whole capture API surface, fast.

Run this, then: ctx list && ctx explain
"""

import ctxrun
from ctxrun import ChunkRecord

# One-liner: capture query + response, nothing else.
ctxrun.capture("what is 2+2?", "4")

# Staged: start a run, feed it stages as they happen, then respond.
run = ctxrun.start(query="what is RRF?", pipeline="quickstart")

run.chunks([
    ChunkRecord(
        chunk_id="c1", source_doc_id="rrf_paper",
        content="Reciprocal Rank Fusion combines rankings from multiple retrievers.",
        token_count=12, retrieval_score=0.9, rerank_score=0.95,
    ),
])

run.response("RRF combines rankings from multiple retrievers into one ranked list.")
# run.commit() already called automatically by run.response()

print("Captured 2 runs. Try: ctx list")
