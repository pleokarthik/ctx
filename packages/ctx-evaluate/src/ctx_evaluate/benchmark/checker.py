import json

from ctx_capture.schema import RunRecord
from ctx_evaluate import store
from ctx_evaluate.layers import input_quality
from ctx_evaluate.policy.persistence import load_policy


def check(
    session_id: int,
    run_seq: int,
    pipeline: str | None = None,
) -> dict:
    run_row = store.get_run(session_id, run_seq)
    if run_row is None:
        raise ValueError(f"Run s{session_id}r{run_seq} not found.")

    record = RunRecord.from_json(json.loads(run_row["run_data"]))
    pipeline = pipeline or run_row["pipeline"] or "__default"
    policy = load_policy(pipeline)

    input_scores = input_quality.score_input_quality(record, policy)
    benchmark = store.get_benchmark(pipeline)
    benchmark_map = {b["factor"]: b for b in benchmark}

    factors = {}
    fail_count = 0

    check_factors = [
        ("duplicate_ratio", "higher_bad"),
        ("top_chunk_score", "lower_bad"),
        ("high_score_truncations", "higher_bad"),
        ("token_headroom_pct", "lower_bad"),
        ("source_domain_count", "higher_bad"),
        ("low_score_chunk_ratio", "higher_bad"),
    ]

    for factor, direction in check_factors:
        value = input_scores.get(factor) if input_scores else None
        bench = benchmark_map.get(factor)
        threshold = bench["threshold"] if bench else None

        if value is None or threshold is None:
            status = "ok"
        elif direction == "lower_bad":
            status = "fail" if value < threshold else "ok"
        else:
            status = "fail" if value > threshold else "ok"

        if status == "fail":
            fail_count += 1

        factors[factor] = {
            "value": value,
            "benchmark_threshold": threshold,
            "status": status,
        }

    eval_data = store.get_eval_scores(session_id, run_seq)
    risk = eval_data.get("risk_score", 0.0) if eval_data else 0.0

    if risk > 0.7 or fail_count >= 3:
        overall = "fail"
    elif fail_count >= 1:
        overall = "warn"
    else:
        overall = "ok"

    return {
        "run_id": f"s{session_id}r{run_seq}",
        "risk_score": risk,
        "benchmark_available": len(benchmark) > 0,
        "factors": factors,
        "overall": overall,
    }
