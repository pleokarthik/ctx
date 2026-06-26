# ctx

Local-first observability for RAG and agentic LLM pipelines.

## Tools

| Tool | Purpose | Status |
|---|---|---|
| ctx-capture | Pipeline instrumentation SDK | Phase 1 |
| ctx | Analyst CLI | Phase 1 |
| ctx-evaluate | Evaluation and benchmarking | Phase 2 |
| ctx-improve | Input quality improvement | Phase 3 |

## Quick start

```bash
# instrument your pipeline
pip install ctx-capture

# analyze runs
pip install ctx
```

## Workspace

```bash
uv sync
```

See `docs/internal/ctx-scope.md` for full scope and acceptance criteria.
