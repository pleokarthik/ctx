# RAG Pipeline Example

End-to-end demonstration of ctx-capture, ctx, and ctx-evaluate
using a fake RAG pipeline with hardcoded retrieval data.

The pipeline produces runs that trigger all ctx explain analysis factors:
window duplicates, high-score truncation, low token headroom,
history eviction, and mixed chunk scores.

## Quick start

Install from workspace root:

```bash
cd <repo root>
uv sync
```

Run the pipeline (captures 8 runs across 2 sessions):

```bash
cd examples/rag_pipeline
python run_pipeline.py
```

## Browse runs with ctx

```bash
ctx list
ctx list s1
ctx explain
ctx explain s1r3
ctx explain s1r3 --full
ctx explain s1r3 --html
ctx find "reranking"
ctx diff s1r4 s1r3
ctx budget s1r1
ctx session rename s1 "Retrieval mechanics"
```

## Evaluate with ctx-evaluate

```bash
python evaluate.py
```

Or use the CLI directly:

```bash
ctx-evaluate run --input-only
ctx-evaluate run s1r1 --input-only
ctx-evaluate policy show
ctx-evaluate benchmark show
ctx-evaluate benchmark check s1r1
```

## What each ctx explain factor shows

| Factor | What you'll see |
|---|---|
| Token usage | Low headroom (4.8%) — budget is tight by design |
| Duplicates | Window dup from two chunks sharing `rrf_paper_2024` source |
| Chunk scores | Distribution across 7 chunks (0.39 to 0.92 rerank) |
| Truncation | Severity: high — `bm25_tf_idf` truncated with rerank 0.88 |
| Dropped history | 4 -> 2 turns, 2 dropped, reason: token_budget |
| Cache hits | 1/7 hit (rrf_norm_1 from disk cache) |
| Final prompt | Assembled system + context + history + query |

## Clean slate

To start fresh, remove the local store:

```bash
rm -rf ~/.ctx
```

Output files from evaluate.py are written to `output/` (gitignored).
