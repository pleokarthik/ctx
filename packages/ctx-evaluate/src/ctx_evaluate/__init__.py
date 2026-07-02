import json
from pathlib import Path

from ctx_capture.schema import RunRecord
from ctx_capture.store import parse_target_id
from ctx_evaluate import store
from ctx_evaluate.layers.input_quality import score_input_quality as score_input
from ctx_evaluate.layers.output_quality import score_output_quality as score_output
from ctx_evaluate.policy.schema import InputQualityPolicy
from ctx_evaluate.policy.persistence import load_policy
from ctx_evaluate.policy.risk import compute_risk_score
from ctx_evaluate.benchmark import seeder, builder, checker, exporter

__all__ = [
    "score_input",
    "score_output",
    "InputQualityPolicy",
    "compute_risk_score",
    "evaluate_run",
    "benchmark_cycle",
    "check_run",
    "export_benchmark",
    "get_evaluated_runs",
]


def evaluate_run(
    record: RunRecord,
    pipeline: str = "__default",
    ground_truth: str | None = None,
    input_only: bool = False,
    output_only: bool = False,
    policy: InputQualityPolicy | None = None,
) -> dict:
    """Score a RunRecord's input and output quality and compute its risk.

    Pure computation: loads the pipeline's policy (unless one is passed
    in via `policy`) and scores the record directly, performing no store
    writes. This is the single source of truth for the load_policy ->
    input_quality.score -> output_quality.score -> compute_risk_score
    sequence; ctx_evaluate.cli._compute_eval calls this instead of
    duplicating it.

    Pass `policy` when the caller already resolved it -- e.g. a
    per-pipeline cache batching many runs -- to skip a redundant
    load_policy() call per run.
    """
    if policy is None:
        policy = load_policy(pipeline)

    result: dict = {"input": None, "output": None, "risk_score": 0.0}

    if not output_only:
        result["input"] = score_input(record, policy)

    if not input_only:
        try:
            result["output"] = score_output(record, ground_truth)
        except ImportError as e:
            result["output"] = None
            result["output_error"] = str(e)

    if result["input"]:
        result["risk_score"] = compute_risk_score(result["input"], policy)

    return result


def benchmark_cycle(pipeline: str, seed_count: int = 20) -> dict:
    """Seed synthetic runs for `pipeline` and build a benchmark from them.

    Seeding writes to the store -- unavoidable, since that's the only way
    synthetic runs get created. Building then reads whatever evaluated
    runs already exist under the seeded pipeline name and writes
    threshold rows; it does not evaluate the freshly seeded runs itself
    (seeder.seed() never populates eval_scores), so this raises ValueError
    if fewer than 10 evaluated runs exist for the seeded pipeline.
    """
    seeded_pipeline = f"{pipeline}__seeded"
    seeder.seed(pipeline, count=seed_count)
    try:
        return builder.build(seeded_pipeline)
    except ValueError as e:
        raise ValueError(
            f"{e} seeder.seed() only creates unevaluated rows -- evaluation "
            f"is a separate step. Score the seeded pipeline's runs (e.g. via "
            f"evaluate_run() + a store write, or `ctx-evaluate run --session`) "
            f"so they have eval_scores populated before calling "
            f"benchmark_cycle() again."
        ) from e


def check_run(target: str, pipeline: str | None = None) -> dict:
    """Check a run against benchmark thresholds.

    Thin wrapper around checker.check(), resolving `target` the same way
    ctx-evaluate's `benchmark check` CLI command does: an exact sNrN run
    id. There's no "latest" fallback here because the CLI command doesn't
    have one either -- its `target` argument is required, unlike `run`'s.
    """
    parsed = parse_target_id(target)
    if parsed is None:
        raise ValueError(f"Target must be in sNrN format, got: {target!r}")
    return checker.check(*parsed, pipeline)


def export_benchmark(pipeline: str | None = None, output_path: str | Path | None = None) -> Path:
    """Export evaluated runs as a RAGAS-compatible JSONL dataset.

    Thin wrapper around exporter.export(). Leaving output_path unset uses
    exporter.export()'s own default location (~/.ctx/exports/) -- this
    function does not invent a separate default.
    """
    return exporter.export(pipeline, Path(output_path) if output_path else None)


def get_evaluated_runs(pipeline: str | None = None) -> list[RunRecord]:
    """Fetch every evaluated run for `pipeline` as RunRecord objects.

    Thin wrapper around store.get_all_evaluated_runs() plus RunRecord
    deserialization, so callers get RunRecord objects directly instead of
    hand-rolling json.loads(row["run_data"]) + RunRecord.from_json()
    themselves. Returns an empty list if there are no evaluated runs for
    `pipeline` -- never raises for that case.
    """
    return [
        RunRecord.from_json(json.loads(row["run_data"]))
        for row in store.get_all_evaluated_runs(pipeline)
    ]
