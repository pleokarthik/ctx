"""
Evaluate rag_example runs via the ctx_evaluate facade -- evaluate_run(),
benchmark_cycle(), check_run(), export_benchmark() -- instead of
importing scoring/threshold internals directly.

Run 02_capture_patterns.py first so there are real rag_example runs to
evaluate.

Steps:
  1. Input-only eval on every real run captured so far
  2. Seed a synthetic baseline (benchmark_cycle)
  3. Build benchmark thresholds from the seeded batch (benchmark_cycle)
  4. Check the latest real run against the fresh benchmark (check_run)
  5. Export a RAGAS-compatible dataset of everything evaluated (export_benchmark)
"""

import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ctx_capture.schema import RunRecord
from ctx_evaluate import store, evaluate_run, benchmark_cycle, check_run, export_benchmark

console = Console()
PIPELINE = "rag_example"
SEEDED_PIPELINE = f"{PIPELINE}__seeded"


def _get_runs_by_pipeline(pipeline):
    """Raw run rows (evaluated or not) for `pipeline`.

    Not swapped for ctx_evaluate.get_evaluated_runs(): that helper only
    returns runs that already have eval_scores populated, and every call
    site below needs runs *before* they're evaluated (that's the point
    of this script) -- plus it needs the session_id/run_seq identity
    get_evaluated_runs() intentionally drops in favor of plain RunRecord
    objects.
    """
    conn = store._connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at, run_data "
            "FROM runs WHERE pipeline = ? ORDER BY created_at",
            (pipeline,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _evaluate_seeded_batch(pipeline_seeded):
    """Score every seeded run's input quality via evaluate_run(), attach a
    synthetic output score standing in for a real RAGAS/LLM judge call (so
    this example runs offline, no API key needed), and persist. Needed
    because seeder.seed() only creates unevaluated rows -- evaluation is
    a separate step, and builder.build() requires >=10 evaluated runs.
    """
    for run_row in _get_runs_by_pipeline(pipeline_seeded):
        record = RunRecord.from_json(json.loads(run_row["run_data"]))
        result = evaluate_run(record, pipeline=PIPELINE, input_only=True)
        mean_rel = (result["input"] or {}).get("mean_relevance", 0.5)
        output = {
            "faithfulness": round(min(1.0, max(0.0, mean_rel + random.uniform(-0.1, 0.1))), 4),
            "answer_relevancy": round(min(1.0, max(0.0, mean_rel + random.uniform(-0.05, 0.05))), 4),
        }
        store.write_eval_scores(
            run_row["session_id"], run_row["run_seq"],
            {"input": result["input"], "output": output}, result["risk_score"],
        )


def main():
    store.apply_migration()

    real_runs = _get_runs_by_pipeline(PIPELINE)
    if not real_runs:
        print(f"No {PIPELINE} runs found. Run 02_capture_patterns.py first.")
        return

    # Step 1 -- input-only eval on every real run captured so far.
    real_results = {}
    for run_row in real_runs:
        record = RunRecord.from_json(json.loads(run_row["run_data"]))
        result = evaluate_run(record, pipeline=PIPELINE, input_only=True)
        store.write_eval_scores(
            run_row["session_id"], run_row["run_seq"],
            {"input": result["input"], "output": None}, result["risk_score"],
        )
        real_results[(run_row["session_id"], run_row["run_seq"])] = result

    # Steps 2+3 -- seed a synthetic baseline, then build benchmark
    # thresholds from it. benchmark_cycle() seeds first and builds
    # immediately after; on a pipeline with no prior evaluated history the
    # build half raises (the batch it just seeded isn't scored yet). Score
    # that batch, then retry with seed_count=0 -- no further seeding, just
    # the build -- via the same facade call.
    try:
        benchmark_result = benchmark_cycle(PIPELINE, seed_count=20)
    except ValueError:
        _evaluate_seeded_batch(SEEDED_PIPELINE)
        benchmark_result = benchmark_cycle(PIPELINE, seed_count=0)

    # Step 4 -- check the latest real run against the fresh benchmark.
    last = real_runs[-1]
    target = f"s{last['session_id']}r{last['run_seq']}"
    check_result = check_run(target, pipeline=SEEDED_PIPELINE)

    # Step 5 -- export a RAGAS-compatible dataset of everything evaluated.
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    export_path = export_benchmark(PIPELINE, output_dir / "benchmark_export.jsonl")

    _print_results(real_runs, real_results, benchmark_result, check_result, export_path)


def _print_results(real_runs, real_results, benchmark_result, check_result, export_path):
    console.print(f"\n[bold]Evaluated {len(real_runs)} real '{PIPELINE}' runs.[/bold]")

    tbl = Table(title="Risk Score Summary (real runs)")
    tbl.add_column("Run", style="cyan")
    tbl.add_column("Query")
    tbl.add_column("Risk", justify="right")
    tbl.add_column("Violations")
    for run_row in real_runs:
        result = real_results[(run_row["session_id"], run_row["run_seq"])]
        risk = result["risk_score"]
        violations = (result["input"] or {}).get("policy_violations", [])
        style = "green" if risk < 0.3 else "yellow" if risk < 0.7 else "red"
        tbl.add_row(
            f"s{run_row['session_id']}r{run_row['run_seq']}",
            run_row["query"][:45],
            f"[{style}]{risk:.2f}[/{style}]",
            ", ".join(violations) if violations else "-",
        )
    console.print(tbl)

    console.print(f"\n[bold]Benchmark[/bold] built from {benchmark_result['run_count']} evaluated runs")
    tbl = Table(title="Benchmark Factors")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Samples", justify="right")
    for factor, data in benchmark_result["factors"].items():
        tbl.add_row(factor, f"{data['suggested_threshold']:.4f}", str(data["sample_count"]))
    console.print(tbl)

    overall_style = {"ok": "green", "warn": "yellow", "fail": "red"}[check_result["overall"]]
    console.print(
        f"\n[bold]Check[/bold] {check_result['run_id']}: "
        f"[{overall_style}]{check_result['overall']}[/{overall_style}]  "
        f"risk: {check_result['risk_score']:.2f}"
    )

    console.print(f"\n[bold]Exported[/bold] to {export_path}")


if __name__ == "__main__":
    main()
