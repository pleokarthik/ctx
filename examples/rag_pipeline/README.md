# RAG Pipeline Example

Three small scripts demonstrating ctx-capture, ctx, and ctx-evaluate
end to end, using fake retrieval data (no external services).

| File | Demonstrates |
|---|---|
| `01_quickstart.py` | The whole capture API surface in under 30 lines: the `ctxrun.capture()` one-liner and the staged `ctxrun.start()` → `run.chunks()` → `run.response()` pattern. |
| `02_capture_patterns.py` | Three named patterns: `pattern_full_fields()` (every optional `RunRecord` field populated), `pattern_multi_session_gap()` (auto session-splitting after a 30-minute idle gap), `pattern_thread_local_proxy()` (`ctxrun.chunks()`/`ctxrun.response()` without threading a `run` object through the call stack). |
| `03_evaluate.py` | The five eval steps (input eval, seed, build, check, export) via the `ctx_evaluate` facade — `evaluate_run()` and `benchmark_cycle()` — instead of importing scoring internals directly. |

## Quick start

Install from workspace root:

```bash
cd <repo root>
uv sync
```

Run the three scripts in order:

```bash
cd examples/rag_pipeline
python 01_quickstart.py
python 02_capture_patterns.py
python 03_evaluate.py
```

`01_quickstart.py` and `02_capture_patterns.py` only capture runs — the
last run captured (`pattern_full_fields()`, run last on purpose) is the
one engineered to trigger every `ctx explain` analysis factor. `03_evaluate.py`
evaluates every `rag_example` run captured by `02_capture_patterns.py`,
so run that one first.

## Browse runs with ctx

```bash
ctx list
ctx list s4               # session numbers will vary run to run
ctx explain                # latest run — all seven factors
ctx explain s4r3 --full
ctx explain s4r3 --html
ctx find "reranking"
ctx diff s4r1 s4r3
ctx budget s4r3
ctx session rename s4 "Retrieval mechanics"
```

Session/run numbers depend on what else has run against your local
`~/.ctx/runs.db` — use `ctx list` to see the actual IDs on your machine.

## Evaluate with ctx-evaluate

```bash
python 03_evaluate.py
```

Or use the CLI directly:

```bash
ctx-evaluate run --input-only
ctx-evaluate policy show
ctx-evaluate benchmark show
ctx-evaluate benchmark check s4r3 --pipeline rag_example__seeded
```

## What pattern_full_fields()'s run shows in ctx explain

`pattern_full_fields()` in `02_capture_patterns.py` is engineered to
trigger every factor:

| Factor | What you'll see |
|---|---|
| Token usage | Headroom 196/4096 — budget is tight by design |
| Duplicates | Window dup between `rrf_norm_1` and `rrf_norm_2`, which share the `rrf_paper_2024` source |
| Chunk scores | Distribution across 4 chunks (0.39–0.92 rerank) |
| Truncation | Severity: high — `bm25_tf_idf` truncated with rerank 0.88 |
| Dropped history | 4 → 2 turns, 2 dropped, reason: `token_budget` |
| Cache hits | 1/4 hit (`rrf_norm_1`) |
| Final prompt | Assembled system + context + history + query |

## Clean slate

To start fresh, remove the local store:

```bash
rm -rf ~/.ctx
```

Output files from `03_evaluate.py` are written to `output/` (gitignored).
