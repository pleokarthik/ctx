# ctx

Local-first observability for RAG and agentic LLM pipelines.

Most RAG failures are silent. A pipeline can return duplicate chunks,
truncate its best content, evict critical history, or blow the token
budget — and still produce a plausible-looking response. There is no
standard tool that surfaces what actually went into the context window
and flags what went wrong.

`ctx` fills that gap.

---

## The problem

```
Query → [Retrieve] → [Assemble] → CONTEXT WINDOW → [LLM] → Response
                                         ↑                        ↑
                               nothing here              everything here
```

Every existing evaluation tool fires after the LLM call — measuring
whether the response was good. None of them inspect the context window
before the expensive call. None of them flag mechanical failures:
duplicate chunks, high-score truncations, dropped history, token
misallocation.

`ctx` is the observability layer that lives before the LLM call.

---

## Three tools

```
ctxrun (ctx-capture)   →   instrument your pipeline, capture runs
ctx                    →   browse sessions, search runs, explain any run
ctx-evaluate           →   score input quality, build benchmarks over time
```

Each tool is independent. They share a single local SQLite store at
`~/.ctx/runs.db`. No server. No cloud. No configuration required.

---

## Quick start

```bash
# install
uv sync

# run the example pipeline — captures 8 runs across 2 sessions
python examples/rag_pipeline/run_pipeline.py

# browse what was captured
ctx list
ctx list s1

# inspect the latest run — all seven factors
ctx explain --full

# inspect a specific run
ctx explain s1r3
ctx explain s1r3 --full
ctx explain s1r3 --html        # snapshot to ~/.ctx/reports/

# search runs by query text
ctx find "reranking"
ctx find "RRF" --session s1

# compare two runs
ctx diff s1r4 s1r3

# evaluate input quality — no LLM required
ctx-evaluate run --input-only

# see benchmark thresholds built from accumulated runs
ctx-evaluate benchmark show

# check a specific run against benchmark
ctx-evaluate benchmark check s1r1

# see active quality policy
ctx-evaluate policy show
```

---

## What ctx explain shows

Seven analysis factors, computed deterministically from captured data.
Each factor is skipped silently if the required data was not captured.

```
Token usage       — per-section breakdown, headroom, model limit
Duplicate chunks  — path dups, window dups, semantic dups
Chunk scores      — retrieval + rerank score distribution
Truncation        — which chunks were trimmed, at what score
Dropped history   — what was evicted, why, what survived
Cache hits        — hit/miss ratio per chunk
Final prompt      — assembled prompt as-is
```

The example pipeline is designed to trigger all seven factors visibly:
low headroom (4.8%), window duplicates, one high-score truncation
(rerank 0.88, truncated=True), two evicted history turns, one cache hit.

---

## Instrumenting your own pipeline

```bash
pip install ctx-capture

# greenfield — generates a scaffold with capture calls pre-positioned
ctx-capture init
```

Minimum instrumentation — two lines:

```python
import ctxrun

ctxrun.capture(query, response)
```

Full staged instrumentation:

```python
import ctxrun
from ctxrun import ChunkRecord, TokenBudget, Turn, CacheEvent

run = ctxrun.start(query=query, pipeline="my_project")

run.chunks(chunks)                              # after retrieval
run.context(final_prompt, token_budget)         # after assembly
run.history(pre=history, post=trimmed,
            reason="token_budget")              # after history management
run.response(response, token_usage=usage)       # after LLM call
run.cache(cache_events)                         # cache hit/miss events

run.commit()                                    # auto-called on run.response()
```

Every field except `query` and `response` is optional. More
instrumentation unlocks more analysis. Nothing breaks at any level.

---

## Architecture

```
your pipeline
  └── ctxrun (ctx-capture)  →  ~/.ctx/runs.db
                                     ↑
                    ctx (analyst CLI) ┤
                    ctx-evaluate      ┘
```

```
ctx/
  packages/
    ctx-capture/      # instrumentation SDK — stdlib only, zero deps
    ctx/              # analyst CLI — rich, click
    ctx-evaluate/     # evaluation layer — ragas, scipy, sentence-transformers
  examples/
    rag_pipeline/     # end-to-end working example
  docs/
    internal/         # design doc, scope doc
```

---

## Tool reference

### ctxrun (ctx-capture)

```bash
pip install ctx-capture
ctx-capture init          # generate scaffold for a new pipeline
```

### ctx

```bash
pip install ctx

ctx list                          # list sessions
ctx list <session>                # list runs in session
ctx find <hint>                   # search by query text
ctx find <hint> --today           # with date filter
ctx find <hint> --pipeline <name> # with pipeline filter
ctx explain                       # latest run
ctx explain <target>              # e.g. s2r3
ctx explain <target> --full       # expanded
ctx explain <target> --html       # HTML snapshot
ctx diff <target> <target>        # compare two runs
ctx budget <target>               # token waterfall only
ctx session rename <id> <title>   # rename a session
```

Optional semantic search:

```bash
pip install ctx[semantic]         # enables embedding-based search
```

### ctx-evaluate

```bash
pip install ctx-evaluate

ctx-evaluate run <target>                    # both layers
ctx-evaluate run <target> --input-only       # no LLM required
ctx-evaluate run <target> --output-only      # RAGAS only
ctx-evaluate run --session <id>              # all runs in session

ctx-evaluate benchmark seed <pipeline>       # synthetic day-zero baseline
ctx-evaluate benchmark build                 # requires 10+ evaluated runs
ctx-evaluate benchmark show                  # per-factor thresholds
ctx-evaluate benchmark check <target>        # ok / warn / fail per factor
ctx-evaluate benchmark export                # RAGAS-compatible JSONL

ctx-evaluate policy show                     # active thresholds
ctx-evaluate policy set <field> <value>      # override a threshold
ctx-evaluate policy reset                    # restore defaults
```

---

## Evaluation layers

**Layer 1 — Input quality (deterministic, no LLM)**

```
Relevance score       chunk similarity vs query (uses existing rerank scores)
Duplicate ratio       path + window + semantic duplicates
Truncation severity   were high-score chunks cut?
Token efficiency      headroom, low-score chunk ratio
Coherence signal      source domain count, score variance
```

**Layer 2 — Output quality (RAGAS, LLM-as-judge)**

```
faithfulness          is the response grounded in retrieved context?
answer_relevancy      does the response address the query?
context_precision     how much retrieved content was actually used?
context_recall        was the necessary information present?
```

**Benchmark system**

After 10+ evaluated runs, `ctx-evaluate benchmark build` correlates
input quality factors against RAGAS scores. Discovered thresholds
replace policy defaults progressively. The system becomes pipeline-specific
over time — your data, your thresholds.

---

## Roadmap

```
v0.1.0   ctx-capture + ctx              ✓ shipped
v0.2.0   ctx-evaluate + examples        ✓ shipped
v0.3.0   ctx-improve (input quality     — planned
         improvement before LLM call)
```

---

## Development

```bash
git clone <repo>
cd ctx
uv sync

# run all tests
uv run pytest

# run per package
uv run pytest packages/ctx-capture/tests/
uv run pytest packages/ctx/tests/
uv run pytest packages/ctx-evaluate/tests/
```

181 tests across three packages. All pass.

---

## Why ctx

| | ctx | LangSmith | RAGAS | Print debugging |
|---|---|---|---|---|
| Pre-call inspection | ✓ | ✗ | ✗ | manual |
| Local / offline | ✓ | ✗ | partial | ✓ |
| Zero infrastructure | ✓ | ✗ | ✓ | ✓ |
| Pipeline agnostic | ✓ | partial | ✓ | ✓ |
| Persistent run store | ✓ | ✓ | ✗ | ✗ |
| Benchmark over time | ✓ | ✓ | ✗ | ✗ |
| No LLM for navigation | ✓ | ✓ | ✗ | ✓ |

ctx is not a replacement for LangSmith or RAGAS. It occupies the
pre-call mechanical observability position that neither covers.
The three tools compose: ctx captures, RAGAS scores, LangSmith traces.