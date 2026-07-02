import json
import re
from dataclasses import fields
from typing import get_type_hints

import click
from rich.console import Console
from rich.table import Table

from ctx_capture.schema import RunRecord
from ctx_capture.store import TARGET_RE, parse_target_id
from ctx_evaluate import store, evaluate_run, check_run, export_benchmark
from ctx_evaluate.policy.persistence import load_policy, save_policy, reset_policy
from ctx_evaluate.policy.schema import InputQualityPolicy
from ctx_evaluate.benchmark import builder, seeder

console = Console()
_SESSION_RE = re.compile(r"^s(\d+)$", re.IGNORECASE)


def _parse_session_id(value: str) -> int:
    m = _SESSION_RE.match(value)
    if m:
        return int(m.group(1))
    return int(value)


def _resolve_target(target: str | None = None) -> dict | None:
    if target is None:
        return store.get_latest_run()
    parsed = parse_target_id(target)
    if parsed:
        return store.get_run(*parsed)
    return None


def _compute_eval(run_row, input_only, output_only, ground_truth, pipeline_override,
                   policy=None):
    """Compute eval scores for a single run without writing to the DB.

    Delegates the actual scoring sequence (policy load -> input quality ->
    output quality -> risk) to ctx_evaluate.evaluate_run(), the package's
    public facade -- this function's job is resolving the run row into a
    RunRecord + pipeline key, then reshaping the result into the
    {"eval_scores": {...}, "risk_score": ...} contract the rest of this
    module and store.write_eval_scores[_batch] expect.

    Pass policy to avoid a redundant load_policy() call when the caller
    already resolved it (e.g. from a per-pipeline cache in the session loop).
    """
    record = RunRecord.from_json(json.loads(run_row["run_data"]))
    pipeline = pipeline_override or run_row["pipeline"] or "__default"

    result = evaluate_run(
        record,
        pipeline=pipeline,
        ground_truth=ground_truth,
        input_only=input_only,
        output_only=output_only,
        policy=policy,
    )

    if result.get("output_error"):
        console.print(f"[yellow]{result['output_error']}[/yellow]")

    return {
        "eval_scores": {"input": result.get("input"), "output": result.get("output")},
        "risk_score": result.get("risk_score", 0.0),
    }


def _evaluate_run_and_persist(run_row, input_only, output_only, ground_truth, pipeline_override):
    result = _compute_eval(run_row, input_only, output_only, ground_truth, pipeline_override)
    store.write_eval_scores(
        session_id=run_row["session_id"],
        run_seq=run_row["run_seq"],
        eval_scores=result["eval_scores"],
        risk_score=result["risk_score"],
    )
    return result


def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _render_eval_result(run_row, result):
    run_id = f"s{run_row['session_id']}r{run_row['run_seq']}"
    risk = result["risk_score"]
    risk_style = "green" if risk < 0.3 else "yellow" if risk < 0.7 else "red"

    console.print(f"\n[bold]{run_id}[/bold] -- {run_row['query'][:60]}")
    console.print(f"Risk score: [{risk_style}]{risk:.2f}[/{risk_style}]")

    inp = result["eval_scores"].get("input")
    if inp:
        passes = inp.get("passes_policy", True)
        header_style = "green" if passes else "red"

        tbl = Table(title=f"[{header_style}]Input Quality[/{header_style}]")
        tbl.add_column("Factor")
        tbl.add_column("Value", justify="right")

        for key in [
            "mean_relevance", "top_chunk_score", "duplicate_ratio",
            "low_score_chunk_ratio", "token_headroom_pct",
            "source_domain_count", "truncation_severity",
            "high_score_truncations",
        ]:
            val = inp.get(key)
            if val is not None:
                tbl.add_row(key, _fmt(val))
        console.print(tbl)

        violations = inp.get("policy_violations", [])
        if violations:
            console.print(f"[red]Policy violations: {', '.join(sorted(violations))}[/red]")
        else:
            console.print("[green]All policy checks passed.[/green]")

    out = result["eval_scores"].get("output")
    if out and out.get("error") is None:
        tbl = Table(title="Output Quality (RAGAS)")
        tbl.add_column("Metric")
        tbl.add_column("Score", justify="right")
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = out.get(key)
            if val is None:
                continue
            style = "green" if val > 0.7 else "yellow" if val > 0.5 else "red"
            tbl.add_row(key, f"[{style}]{val:.4f}[/{style}]")
        console.print(tbl)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@click.group()
def main():
    """ctx-evaluate -- evaluation layer for ctx observability system."""
    store.apply_migration()


@main.command("run")
@click.argument("target", required=False)
@click.option("--input-only", is_flag=True)
@click.option("--output-only", is_flag=True)
@click.option("--session", "session_filter", default=None)
@click.option("--ground-truth", default=None)
@click.option("--pipeline", default=None)
def run_cmd(target, input_only, output_only, session_filter, ground_truth, pipeline):
    """Evaluate a run or all runs in a session."""
    if session_filter:
        sid = _parse_session_id(session_filter)
        runs = store.get_runs_in_session(sid)
        if not runs:
            console.print(f"No runs found in session {sid}.")
            return
        policy_cache: dict = {}
        computed = []
        for run_row in runs:
            pipeline_key = pipeline or run_row["pipeline"] or "__default"
            if pipeline_key not in policy_cache:
                policy_cache[pipeline_key] = load_policy(pipeline_key)
            result = _compute_eval(
                run_row, input_only, output_only, ground_truth, pipeline,
                policy=policy_cache[pipeline_key],
            )
            computed.append((run_row, result))
        store.write_eval_scores_batch([
            (run_row["session_id"], run_row["run_seq"],
             result["eval_scores"], result["risk_score"])
            for run_row, result in computed
        ])
        for run_row, result in computed:
            _render_eval_result(run_row, result)
    else:
        run_row = _resolve_target(target)
        if run_row is None:
            console.print("No runs found.")
            return
        result = _evaluate_run_and_persist(run_row, input_only, output_only, ground_truth, pipeline)
        _render_eval_result(run_row, result)


# ---------------------------------------------------------------------------
# Benchmark subgroup
# ---------------------------------------------------------------------------

@main.group()
def benchmark():
    """Benchmark commands."""


@benchmark.command("build")
@click.option("--pipeline", default=None)
def benchmark_build(pipeline):
    """Build correlation model from evaluated runs."""
    try:
        result = builder.build(pipeline)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    tbl = Table(title=f"Benchmark ({result['run_count']} runs)")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Correlation", justify="right")
    tbl.add_column("Samples", justify="right")

    for factor, data in result["factors"].items():
        corr_vals = [v for k, v in data.items() if "correlation" in k and v is not None]
        best_corr = max(corr_vals, key=abs) if corr_vals else None
        tbl.add_row(
            factor,
            _fmt(data["suggested_threshold"]),
            _fmt(best_corr),
            str(data["sample_count"]),
        )
    console.print(tbl)


@benchmark.command("show")
@click.option("--pipeline", default=None)
def benchmark_show(pipeline):
    """Show benchmark thresholds and correlations."""
    pipeline_key = pipeline or "__default"
    entries = store.get_benchmark(pipeline_key)
    if not entries:
        console.print("No benchmark data found.")
        return

    tbl = Table(title=f"Benchmark: {pipeline_key}")
    tbl.add_column("Factor")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Correlation", justify="right")
    tbl.add_column("Samples", justify="right")
    tbl.add_column("Updated")

    for e in entries:
        tbl.add_row(
            e["factor"], _fmt(e["threshold"]), _fmt(e["correlation"]),
            str(e["sample_count"]), e["updated_at"][:10],
        )
    console.print(tbl)


@benchmark.command("check")
@click.argument("target")
@click.option("--pipeline", default=None)
def benchmark_check(target, pipeline):
    """Check a run against benchmark thresholds."""
    if not TARGET_RE.match(target):
        console.print("[red]Target must be in sNrN format.[/red]")
        raise SystemExit(1)

    try:
        result = check_run(target, pipeline)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    overall_style = {"ok": "green", "warn": "yellow", "fail": "red"}[result["overall"]]
    console.print(f"\n[bold]{result['run_id']}[/bold] -- overall: [{overall_style}]{result['overall']}[/{overall_style}]")
    console.print(f"Risk score: {result['risk_score']:.2f}")

    tbl = Table(title="Factor Check")
    tbl.add_column("Factor")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Threshold", justify="right")
    tbl.add_column("Status")

    for factor, data in result["factors"].items():
        style = {"ok": "green", "warn": "yellow", "fail": "red"}[data["status"]]
        tbl.add_row(
            factor, _fmt(data["value"]), _fmt(data["benchmark_threshold"]),
            f"[{style}]{data['status']}[/{style}]",
        )
    console.print(tbl)


@benchmark.command("seed")
@click.argument("pipeline")
@click.option("--count", default=20, type=int)
def benchmark_seed(pipeline, count):
    """Generate synthetic runs as day-zero baseline."""
    n = seeder.seed(pipeline, count)
    console.print(f"Seeded {n} synthetic runs for pipeline '{pipeline}'.")


@benchmark.command("export")
@click.option("--pipeline", default=None)
@click.option("--output", default=None)
def benchmark_export(pipeline, output):
    """Export evaluated runs as RAGAS-compatible JSONL."""
    path = export_benchmark(pipeline, output)
    console.print(f"Exported to {path}")


# ---------------------------------------------------------------------------
# Policy subgroup
# ---------------------------------------------------------------------------

@main.group()
def policy():
    """Policy management commands."""


@policy.command("show")
@click.option("--pipeline", default=None)
def policy_show(pipeline):
    """Show current policy values."""
    pipeline_key = pipeline or "__default"
    pol = load_policy(pipeline_key)
    default = InputQualityPolicy.default()

    tbl = Table(title=f"Policy: {pipeline_key}")
    tbl.add_column("Field")
    tbl.add_column("Value", justify="right")
    tbl.add_column("Default", justify="right")

    for f in fields(InputQualityPolicy):
        val = getattr(pol, f.name)
        dval = getattr(default, f.name)
        style = "" if val == dval else "[bold]"
        end = "" if val == dval else "[/bold]"
        tbl.add_row(f.name, f"{style}{val}{end}", str(dval))
    console.print(tbl)


@policy.command("set")
@click.argument("field")
@click.argument("value")
@click.option("--pipeline", default=None)
def policy_set(field, value, pipeline):
    """Set a policy field value."""
    pipeline_key = pipeline or "__default"

    valid_fields = {f.name: f for f in fields(InputQualityPolicy)}
    if field not in valid_fields:
        console.print(f"[red]Unknown field: {field}[/red]")
        console.print(f"Valid fields: {', '.join(sorted(valid_fields))}")
        raise SystemExit(1)

    f = valid_fields[field]
    field_type = get_type_hints(InputQualityPolicy)[f.name]
    try:
        typed_value = field_type(value)
    except (ValueError, TypeError):
        console.print(f"[red]Invalid value for {field} (expected {field_type.__name__}): {value}[/red]")
        raise SystemExit(1)

    pol = load_policy(pipeline_key)
    setattr(pol, field, typed_value)
    save_policy(pipeline_key, pol)
    console.print(f"Set {field} = {typed_value} for pipeline '{pipeline_key}'.")


@policy.command("reset")
@click.option("--pipeline", default=None)
def policy_reset(pipeline):
    """Reset policy to defaults."""
    pipeline_key = pipeline or "__default"
    reset_policy(pipeline_key)
    console.print(f"Policy reset to defaults for pipeline '{pipeline_key}'.")
