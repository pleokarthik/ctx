import json
import sqlite3
from unittest import mock

import pytest
from click.testing import CliRunner

from ctx_capture.schema import RunRecord
from ctx_evaluate import (
    evaluate_run,
    benchmark_cycle,
    check_run,
    export_benchmark,
    get_evaluated_runs,
    store,
)
from ctx_evaluate.cli import main
from ctx_evaluate.benchmark import checker
from ctx_evaluate.policy.schema import InputQualityPolicy
from ctx_evaluate.policy.store import load_policy as _real_load_policy


def _seed_evaluated_runs(db_path, pipeline, count=10, session_id=500):
    """Insert `count` already-evaluated runs directly, bypassing seeder.seed()
    (which never populates eval_scores) so builder.build()'s >=10-evaluated-
    runs gate is satisfied without needing a real RAGAS/LLM call.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions VALUES (?, NULL, ?, '2026-06-10T10:00:00+00:00')",
            (session_id, pipeline),
        )
        for i in range(count):
            rec = RunRecord(query=f"seed q{i}", response=f"seed r{i}")
            score_val = 0.5 + (i % 5) * 0.05
            eval_scores = {
                "input": {"mean_relevance": score_val, "top_chunk_score": score_val},
                "output": {"faithfulness": score_val, "answer_relevancy": score_val},
            }
            conn.execute(
                "INSERT INTO runs (session_id, run_seq, query, pipeline, created_at, "
                "run_data, eval_scores, risk_score, evaluated_at) "
                "VALUES (?, ?, ?, ?, '2026-06-10T10:00:00+00:00', ?, ?, 0.1, "
                "'2026-06-10T10:00:00+00:00')",
                (
                    session_id, i + 1, rec.query, pipeline,
                    json.dumps(rec.to_json()), json.dumps(eval_scores),
                ),
            )


class TestEvaluateRunMatchesCli:
    def test_matches_cli_run_output(self, migrated_db, full_record):
        # Real RAGAS calls need network/API keys and aren't exercised by
        # any existing test in this suite -- stick to --input-only /
        # input_only=True on both sides, same as the rest of test_cli.py.
        CliRunner().invoke(main, ["run", "s2r1", "--input-only"])
        cli_scores = store.get_eval_scores(2, 1)

        facade_result = evaluate_run(full_record, pipeline="pipe_a", input_only=True)

        assert facade_result["input"] == cli_scores["input"]
        assert facade_result["output"] is None
        assert cli_scores["output"] is None
        assert facade_result["risk_score"] == cli_scores["risk_score"]

    def test_return_shape(self, full_record):
        result = evaluate_run(full_record, input_only=True)
        assert set(result.keys()) >= {"input", "output", "risk_score"}
        assert result["output"] is None
        assert isinstance(result["risk_score"], float)
        assert result["input"] is not None


class TestEvaluateRunPolicyOverride:
    def test_skips_load_policy_when_policy_given(self, full_record):
        custom_policy = InputQualityPolicy(min_top_chunk_score=0.99)
        with mock.patch("ctx_evaluate.load_policy") as mock_load:
            result = evaluate_run(full_record, input_only=True, policy=custom_policy)

        mock_load.assert_not_called()
        assert result["input"] is not None
        # min_top_chunk_score=0.99 should be violated by the fixture's
        # rerank scores (max 0.85) -- proves the passed-in policy is the
        # one actually used, not silently ignored in favor of a default.
        assert "min_top_chunk_score" in result["input"]["policy_violations"]

    def test_calls_load_policy_when_policy_omitted(self, full_record):
        with mock.patch(
            "ctx_evaluate.load_policy", wraps=_real_load_policy
        ) as mock_load:
            evaluate_run(full_record, input_only=True)

        mock_load.assert_called_once_with("__default")


class TestEvaluateRunNoSideEffects:
    def test_does_not_write_to_runs_table(self, migrated_db, full_record):
        evaluate_run(full_record, pipeline="pipe_a", input_only=True)

        with sqlite3.connect(str(migrated_db)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT eval_scores, risk_score, evaluated_at FROM runs "
                "WHERE session_id = 2 AND run_seq = 1"
            ).fetchone()

        assert row["eval_scores"] is None
        assert row["risk_score"] is None
        assert row["evaluated_at"] is None


class TestBenchmarkCycle:
    def test_seeds_and_builds(self, migrated_db):
        pipeline = "cycle_pipe"
        seeded_pipeline = f"{pipeline}__seeded"
        _seed_evaluated_runs(migrated_db, seeded_pipeline, count=10)

        result = benchmark_cycle(pipeline, seed_count=4)

        assert result["run_count"] == 10
        assert len(result["factors"]) > 0

        with sqlite3.connect(str(migrated_db)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT COUNT(*) as n FROM runs WHERE pipeline = ? AND eval_scores IS NULL",
                (seeded_pipeline,),
            ).fetchone()
        assert row["n"] == 4

    def test_raises_without_enough_evaluated_runs(self, migrated_db):
        # seeder.seed() alone never populates eval_scores, so a bare
        # seed-then-build cycle on a pipeline with no prior evaluated
        # history always fails this gate -- documented behavior, not a bug.
        # The message should explain *why* and what to do about it, not
        # just repeat builder.build()'s generic "need at least 10" text.
        with pytest.raises(
            ValueError,
            match=r"at least 10.*seeder\.seed\(\) only creates unevaluated rows",
        ):
            benchmark_cycle("empty_pipe", seed_count=4)


class TestCheckRun:
    def test_matches_checker_check(self, migrated_db):
        store.write_benchmark_entry(
            "pipe_a", "top_chunk_score", threshold=0.95, correlation=0.5, sample_count=12,
        )
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.4)

        direct = checker.check(2, 1, "pipe_a")
        via_facade = check_run("s2r1", pipeline="pipe_a")

        assert via_facade == direct
        assert via_facade["factors"]["top_chunk_score"]["status"] == "fail"

    def test_no_benchmark_entries_yet(self, migrated_db):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.1)

        result = check_run("s2r1", pipeline="pipe_a")

        assert result["benchmark_available"] is False
        assert all(f["status"] == "ok" for f in result["factors"].values())

    def test_invalid_target_format_raises(self):
        with pytest.raises(ValueError, match="sNrN format"):
            check_run("not-a-target")


class TestExportBenchmark:
    def test_explicit_output_path(self, migrated_db, tmp_path):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.2)

        out = tmp_path / "custom.jsonl"
        path = export_benchmark("pipe_a", out)

        assert path == out
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["question"] == "does RRF handle score scale differences"

    def test_default_output_path_matches_exporter_default(self, migrated_db):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.2)

        path = export_benchmark("pipe_a")

        assert path.parent == store._ctx_dir() / "exports"
        assert path.exists()


class TestGetEvaluatedRuns:
    def test_returns_run_records_for_evaluated_pipeline(self, migrated_db, full_record):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.3)

        results = get_evaluated_runs("pipe_a")

        assert len(results) == 1
        assert isinstance(results[0], RunRecord)
        assert results[0].query == full_record.query

    def test_empty_pipeline_returns_empty_list(self, migrated_db):
        results = get_evaluated_runs("no_such_pipeline")
        assert results == []
