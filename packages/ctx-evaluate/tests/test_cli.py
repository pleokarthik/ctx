import sqlite3

from click.testing import CliRunner

from ctx_evaluate.cli import main
from ctx_evaluate import store


class TestMigrationOnCommand:
    def test_migration_runs(self, v1_db):
        runner = CliRunner()
        runner.invoke(main, ["policy", "show"])
        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            ver = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        assert ver["value"] == "2"


class TestRunCommand:
    def test_input_only(self, migrated_db):
        result = CliRunner().invoke(main, ["run", "--input-only"])
        assert result.exit_code == 0

    def test_specific_target(self, migrated_db):
        result = CliRunner().invoke(main, ["run", "s2r1", "--input-only"])
        assert result.exit_code == 0
        assert "Input Quality" in result.output

    def test_no_runs(self, ctx_home):
        result = CliRunner().invoke(main, ["run"])
        assert "No runs found" in result.output


class TestPolicyCommands:
    def test_show_uses_defaults(self, migrated_db):
        result = CliRunner().invoke(main, ["policy", "show"])
        assert result.exit_code == 0
        assert "0.7" in result.output
        assert "0.2" in result.output

    def test_set_valid_field(self, migrated_db):
        result = CliRunner().invoke(
            main, ["policy", "set", "min_top_chunk_score", "0.8"]
        )
        assert result.exit_code == 0
        assert "0.8" in result.output

    def test_set_invalid_field(self, migrated_db):
        result = CliRunner().invoke(
            main, ["policy", "set", "unknown_field", "0.8"]
        )
        assert result.exit_code != 0

    def test_reset(self, migrated_db):
        CliRunner().invoke(main, ["policy", "set", "min_top_chunk_score", "0.99"])
        result = CliRunner().invoke(main, ["policy", "reset"])
        assert result.exit_code == 0
        assert "reset" in result.output.lower()


class TestBenchmarkCommands:
    def test_seed_then_show(self, migrated_db):
        result = CliRunner().invoke(main, ["benchmark", "seed", "test_pipe"])
        assert result.exit_code == 0
        assert "Seeded" in result.output

    def test_build_fails_under_10(self, migrated_db):
        result = CliRunner().invoke(main, ["benchmark", "build"])
        assert result.exit_code != 0

    def test_export(self, migrated_db):
        store.write_eval_scores(2, 1, {"input": {}, "output": {}}, 0.1)
        result = CliRunner().invoke(main, ["benchmark", "export"])
        assert result.exit_code == 0
        assert "Exported" in result.output
