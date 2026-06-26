"""
Evaluate all rag_example runs through the ctx-evaluate pipeline.

Steps:
  1. Input-only evaluation on all 8 real runs
  2. Benchmark seed (synthetic baseline)
  3. Evaluate seeded runs with synthetic output scores
  4. Benchmark build (correlations from seeded data)
  5. Benchmark check on latest real run
  6. Benchmark export to examples/rag_pipeline/output/
"""

import json
import random
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ctx_capture.schema import RunRecord
from ctx_evaluate import store
from ctx_evaluate.layers import input_quality
from ctx_evaluate.policy.store import load_policy
from ctx_evaluate.policy.risk import compute_risk_score
from ctx_evaluate.benchmark import seeder, builder, checker, exporter

console = Console()
PIPELINE = "rag_example"
SEEDED_PIPELINE = f"{PIPELINE}__seeded"


def _get_runs_by_pipeline(pipeline):
    conn = store._connect()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT session_id, run_seq, query, pipeline, created_at, "
            "run_data, eval_scores, risk_score, evaluated_at "
            "FROM runs WHERE pipeline = ? ORDER BY created_at",
            (pipeline,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def main():
    store.apply_migration()
    policy = load_policy(PIPELINE)

    # ------------------------------------------------------------------
    # Step 1 — Input-only evaluation on real runs
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Step 1:[/bold cyan] Input-only evaluation on real runs\n")
    real_runs = _get_runs_by_pipeline(PIPELINE)
    if not real_runs:
        console.print("[red]No rag_example runs found. Run run_pipeline.py first.[/red]")
        return

    for run_row in real_runs:
        record = RunRecord.from_json(json.loads(run_row["run_data"]))
        input_scores = input_quality.score(record, policy)
        risk = compute_risk_score(input_scores, policy) if input_scores else 0.0
        store.write_eval_scores(
            run_row["session_id"], run_row["run_seq"],
            {"input": input_scores}, risk,
        )
        rid = f"s{run_row['session_id']}r{run_row['run_seq']}"
        style = "green" if risk < 0.3 else "yellow" if risk < 0.7 else "red"
        console.print(f"  {rid}  risk: [{style}]{risk:.2f}[/{style}]")

    console.print(f"\n  Evaluated {len(real_runs)} real runs.")

    # ------------------------------------------------------------------
    # Step 2 — Benchmark seed
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Step 2:[/bold cyan] Seeding synthetic baseline\n")
    n = seeder.seed(PIPELINE, count=20)
    console.print(f"  Seeded {n} synthetic runs (pipeline: {SEEDED_PIPELINE})")

    # Evaluate seeded runs with synthetic output scores so benchmark
    # build has input+output pairs for correlation analysis.
    console.print("  Evaluating seeded runs with synthetic output scores...")
    seeded_runs = _get_runs_by_pipeline(SEEDED_PIPELINE)
    for run_row in seeded_runs:
        record = RunRecord.from_json(json.loads(run_row["run_data"]))
        input_scores = input_quality.score(record, policy)
        risk = compute_risk_score(input_scores, policy) if input_scores else 0.0
        mean_rel = input_scores["mean_relevance"] if input_scores else 0.5
        output_scores = {
            "faithfulness": round(min(1.0, max(0.0, mean_rel + random.uniform(-0.1, 0.1))), 4),
            "answer_relevancy": round(min(1.0, max(0.0, mean_rel + random.uniform(-0.05, 0.05))), 4),
        }
        store.write_eval_scores(
            run_row["session_id"], run_row["run_seq"],
            {"input": input_scores, "output": output_scores}, risk,
        )
    console.print(f"  Evaluated {len(seeded_runs)} seeded runs.")

    # ------------------------------------------------------------------
    # Step 3 — Benchmark build
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Step 3:[/bold cyan] Building benchmark correlations\n")
    result = builder.build(SEEDED_PIPELINE)
    console.print(f"  Built from {result['run_count']} evaluated runs")

    tbl = Table(title="Benchmark Factors")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Samples", justify="right")
    for factor, data in result["factors"].items():
        tbl.add_row(factor, f"{data['suggested_threshold']:.4f}", str(data["sample_count"]))
    console.print(tbl)

    # ------------------------------------------------------------------
    # Step 4 — Benchmark check on latest real run
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Step 4:[/bold cyan] Checking latest real run against benchmark\n")
    last = real_runs[-1]
    check_result = checker.check(
        last["session_id"], last["run_seq"], SEEDED_PIPELINE,
    )
    overall_style = {"ok": "green", "warn": "yellow", "fail": "red"}[check_result["overall"]]
    console.print(
        f"  {check_result['run_id']}: "
        f"[{overall_style}]{check_result['overall']}[/{overall_style}]  "
        f"risk: {check_result['risk_score']:.2f}"
    )

    # ------------------------------------------------------------------
    # Step 5 — Benchmark export
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Step 5:[/bold cyan] Exporting RAGAS-compatible dataset\n")
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    path = exporter.export(PIPELINE, output_dir / "benchmark_export.jsonl")
    console.print(f"  Exported to {path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.print()
    tbl = Table(title="Risk Score Summary (real runs)")
    tbl.add_column("Run", style="cyan")
    tbl.add_column("Query")
    tbl.add_column("Risk", justify="right")
    tbl.add_column("Violations")

    real_runs = _get_runs_by_pipeline(PIPELINE)
    for run_row in real_runs:
        eval_data = store.get_eval_scores(run_row["session_id"], run_row["run_seq"])
        risk = eval_data.get("risk_score", 0.0) if eval_data else 0.0
        violations = []
        if eval_data:
            inp = eval_data.get("input") or {}
            violations = inp.get("policy_violations", [])
        style = "green" if risk < 0.3 else "yellow" if risk < 0.7 else "red"
        tbl.add_row(
            f"s{run_row['session_id']}r{run_row['run_seq']}",
            run_row["query"][:45],
            f"[{style}]{risk:.2f}[/{style}]",
            ", ".join(violations) if violations else "-",
        )
    console.print(tbl)


if __name__ == "__main__":
    main()
