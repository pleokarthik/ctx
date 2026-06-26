"""
Fake RAG pipeline instrumented with ctxrun.

No external dependencies beyond the ctx workspace packages.
Shows the staged capture pattern with realistic data that triggers
all ctx explain analysis factors.
"""

import ctxrun
from ctxrun import ChunkRecord, TokenBudget, TokenUsage, Turn, CacheEvent


def _build_chunks():
    """Return 7 hardcoded chunks designed to trigger analysis signals.

    Signals by design:
    - Window dup: c1 and c2 share source_doc_id "rrf_paper_2024"
      and c2's content is a substring-overlapping expansion of c1's topic
    - High-score truncation: c3 has rerank_score=0.88, truncated=True
    - Cache hit: c1 has cache_hit=True
    - Low-score chunks: c6 (0.41) and c7 (0.39) pull down the mean
    - Mixed retrieval paths: bm25, ann, hybrid
    - 6 source domains (exceeds default max_source_domains=3)
    """
    return [
        ChunkRecord(
            chunk_id="rrf_norm_1",
            source_doc_id="rrf_paper_2024",
            content=(
                "Reciprocal Rank Fusion normalizes scores from different "
                "retrieval systems. RRF converts raw scores to rank positions "
                "and computes 1/(k+rank) for each document, producing a fused "
                "score that is robust to score scale differences."
            ),
            token_count=180,
            retrieval_score=0.85,
            rerank_score=0.92,
            retrieval_path="hybrid",
            truncated=False,
            cache_hit=True,
        ),
        ChunkRecord(
            chunk_id="rrf_norm_2",
            source_doc_id="rrf_paper_2024",
            content=(
                "RRF converts raw scores to rank positions "
                "and computes 1/(k+rank) for each document"
            ),
            token_count=160,
            retrieval_score=0.71,
            rerank_score=0.78,
            retrieval_path="bm25",
            truncated=False,
            cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="bm25_tf_idf",
            source_doc_id="ir_textbook_ch3",
            content=(
                "BM25 computes relevance using term frequency and inverse "
                "document frequency with length normalization controlled by "
                "parameters k1 and b."
            ),
            token_count=140,
            retrieval_score=0.82,
            rerank_score=0.88,
            retrieval_path="bm25",
            truncated=True,
            cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="vec_cosine",
            source_doc_id="embedding_guide",
            content=(
                "Vector similarity search uses cosine distance between query "
                "and document embeddings in high-dimensional space to find "
                "semantically relevant passages."
            ),
            token_count=150,
            retrieval_score=0.72,
            rerank_score=0.65,
            retrieval_path="ann",
            truncated=False,
            cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="cross_enc",
            source_doc_id="reranker_survey",
            content=(
                "Cross-encoder rerankers process query-document pairs jointly "
                "through a transformer model, producing precise relevance "
                "scores at the cost of higher latency than bi-encoders."
            ),
            token_count=170,
            retrieval_score=0.76,
            rerank_score=0.84,
            retrieval_path="hybrid",
            truncated=False,
            cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="score_calib",
            source_doc_id="fusion_methods",
            content=(
                "Score calibration aligns heterogeneous retrieval scores "
                "before fusion to prevent high-variance systems from "
                "dominating the final ranking."
            ),
            token_count=130,
            retrieval_score=0.55,
            rerank_score=0.41,
            retrieval_path="ann",
            truncated=False,
            cache_hit=False,
        ),
        ChunkRecord(
            chunk_id="ctx_window",
            source_doc_id="rag_patterns",
            content=(
                "Context window management determines which retrieved chunks "
                "survive token budget constraints during prompt assembly in "
                "retrieval-augmented generation pipelines."
            ),
            token_count=145,
            retrieval_score=0.48,
            rerank_score=0.39,
            retrieval_path="bm25",
            truncated=False,
            cache_hit=False,
        ),
    ]


def _build_final_prompt(query, chunks):
    header = (
        "System: You are a helpful AI assistant specializing in information "
        "retrieval and RAG pipeline design. Answer questions accurately based "
        "on the provided context. If the context is insufficient, say so.\n\n"
    )
    context_block = "Context:\n"
    for i, c in enumerate(chunks, 1):
        context_block += (
            f"[{i}] ({c.source_doc_id} | rerank: {c.rerank_score:.2f}) "
            f"{c.content}\n"
        )
    history_block = (
        "\nConversation:\n"
        "User: Can you help me understand retrieval systems?\n"
        "Assistant: Of course! I can explain retrieval, scoring, and reranking.\n\n"
    )
    return header + context_block + history_block + f"Query: {query}\n"


def run_pipeline(query: str) -> str:
    """Run the fake RAG pipeline and capture with ctxrun.

    # --- Scaffold pattern (commented) ---
    # run = ctxrun.start(query=query, pipeline="rag_example")
    # chunks = your_retriever.retrieve(query)
    # run.chunks(chunks)
    # prompt, budget = your_assembler.assemble(chunks, history)
    # run.context(prompt, budget)
    # run.history(pre=history_before, post=history_after, reason="token_budget")
    # response = your_llm_client.call(prompt)
    # run.response(response)
    # run.commit() is called automatically after run.response()
    """

    # --- Stage 0: Start run ---
    run = ctxrun.start(query=query, pipeline="rag_example")

    # --- Stage 1: Retrieval ---
    chunks = _build_chunks()
    run.chunks(chunks)

    # --- Stage 2: Context assembly ---
    final_prompt = _build_final_prompt(query, chunks)
    budget = TokenBudget(
        total_limit=4096,
        chunks_allocated=2800,
        history_allocated=600,
        system_allocated=500,
        headroom=196,
    )
    run.context(final_prompt, budget)

    # --- Stage 3: History management ---
    history_pre = [
        Turn(role="user", content="Can you help me understand retrieval systems?", tokens=9),
        Turn(role="assistant", content="Of course! I can explain retrieval, scoring, and reranking.", tokens=12),
        Turn(role="user", content="Start with how BM25 works.", tokens=7),
        Turn(role="assistant", content="BM25 uses term frequency and inverse document frequency to rank documents by relevance.", tokens=16),
    ]
    history_post = [
        Turn(role="user", content="Can you help me understand retrieval systems?", tokens=9),
        Turn(role="assistant", content="Of course! I can explain retrieval, scoring, and reranking.", tokens=12),
    ]
    run.history(pre=history_pre, post=history_post, reason="token_budget")

    # --- Stage 4: Cache events ---
    cache_events = [
        CacheEvent(chunk_id="rrf_norm_1", hit=True, cache_source="disk"),
        CacheEvent(chunk_id="rrf_norm_2", hit=False),
        CacheEvent(chunk_id="bm25_tf_idf", hit=False),
        CacheEvent(chunk_id="vec_cosine", hit=False),
        CacheEvent(chunk_id="cross_enc", hit=False),
        CacheEvent(chunk_id="score_calib", hit=False),
        CacheEvent(chunk_id="ctx_window", hit=False),
    ]
    run.cache(cache_events)

    # --- Stage 5: LLM call ---
    response_text = (
        f"Regarding your question about {query.split()[0:4]}: "
        "Reciprocal Rank Fusion (RRF) addresses score normalization by "
        "replacing raw retrieval scores with rank-based reciprocal values. "
        "The formula 1/(k+rank) ensures that top-ranked documents receive "
        "consistently high fused scores regardless of the original score "
        "scale from each retrieval system. This makes RRF particularly "
        "robust when combining BM25 lexical scores with dense vector "
        "similarity scores, as it sidesteps the calibration problem entirely."
    )
    usage = TokenUsage(input_tokens=1850, output_tokens=95, total_tokens=1945)
    run.response(response_text, token_usage=usage, model="gpt-4-turbo")

    # run.commit() already called by run.response()

    return response_text
