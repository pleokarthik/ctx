from click.testing import CliRunner

from ctx_cli.cli import main


class TestList:
    def test_sessions_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["list"])
        assert result.exit_code == 0
        assert "s2" in result.output
        assert "s1" in result.output

    def test_runs_in_session(self, populated_db):
        result = CliRunner().invoke(main, ["list", "s2"])
        assert result.exit_code == 0
        assert "r1" in result.output
        assert "r2" in result.output

    def test_empty_db(self):
        result = CliRunner().invoke(main, ["list"])
        assert result.exit_code == 0
        assert "No sessions found" in result.output


class TestFind:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["find", "score"])
        assert result.exit_code == 0

    def test_shows_results(self, populated_db):
        result = CliRunner().invoke(main, ["find", "RRF"])
        assert result.exit_code == 0
        assert "RRF" in result.output

    def test_no_match(self, populated_db):
        result = CliRunner().invoke(main, ["find", "zzzznonexistent"])
        assert result.exit_code == 0
        assert "No matching runs found" in result.output

    def test_recent(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--recent", "1"])
        assert result.exit_code == 0

    def test_pipeline_filter(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--pipeline", "pipe_a"])
        assert result.exit_code == 0

    def test_session_filter(self, populated_db):
        result = CliRunner().invoke(main, ["find", "--session", "s2", "score"])
        assert result.exit_code == 0


class TestExplain:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["explain"])
        assert result.exit_code == 0

    def test_specific_target(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s2r1"])
        assert result.exit_code == 0
        assert "RRF" in result.output

    def test_full_mode(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s2r1", "--full"])
        assert result.exit_code == 0

    def test_html_output(self, populated_db, ctx_home):
        result = CliRunner().invoke(main, ["explain", "s2r1", "--html"])
        assert result.exit_code == 0
        assert "Report written to" in result.output
        reports_dir = ctx_home / ".ctx" / "reports"
        assert reports_dir.exists()
        html_files = list(reports_dir.glob("*.html"))
        assert len(html_files) == 1

    def test_no_runs(self):
        result = CliRunner().invoke(main, ["explain"])
        assert "No runs found" in result.output

    def test_factors_skip_on_empty_record(self, populated_db):
        result = CliRunner().invoke(main, ["explain", "s1r1"])
        assert result.exit_code == 0
        assert "BM25" in result.output


class TestDiff:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["diff", "s2r1", "s2r2"])
        assert result.exit_code == 0
        assert "Comparing" in result.output

    def test_nonexistent_target(self, populated_db):
        result = CliRunner().invoke(main, ["diff", "s99r99", "s2r1"])
        assert "Could not resolve" in result.output


class TestBudget:
    def test_exits_0(self, populated_db):
        result = CliRunner().invoke(main, ["budget", "s2r1"])
        assert result.exit_code == 0
        assert "Token Usage" in result.output

    def test_no_budget_data(self, populated_db):
        result = CliRunner().invoke(main, ["budget", "s1r1"])
        assert "No token budget data" in result.output


class TestSessionRename:
    def test_renames(self, populated_db):
        result = CliRunner().invoke(main, ["session", "rename", "s1", "My Title"])
        assert result.exit_code == 0
        assert "My Title" in result.output

        result = CliRunner().invoke(main, ["list"])
        assert "My Title" in result.output
