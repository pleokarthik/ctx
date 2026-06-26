import json
import sqlite3

from ctx_evaluate.store import apply_migration, _db_path


class TestMigration:
    def test_migration_from_v1(self, v1_db):
        apply_migration()

        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
            assert "eval_scores" in cols
            assert "risk_score" in cols
            assert "evaluated_at" in cols

            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            assert "benchmark" in tables
            assert "policies" in tables

            ver = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            assert ver["value"] == "2"

    def test_existing_data_intact(self, v1_db):
        with sqlite3.connect(str(v1_db)) as conn:
            before = conn.execute("SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1").fetchone()[0]

        apply_migration()

        with sqlite3.connect(str(v1_db)) as conn:
            after = conn.execute("SELECT run_data FROM runs WHERE session_id = 2 AND run_seq = 1").fetchone()[0]

        assert json.loads(before) == json.loads(after)

    def test_idempotent(self, v1_db):
        apply_migration()
        apply_migration()

        with sqlite3.connect(str(v1_db)) as conn:
            conn.row_factory = sqlite3.Row
            ver = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            assert ver["value"] == "2"

    def test_v2_is_noop(self, migrated_db):
        apply_migration()

    def test_unsupported_version_raises(self, v1_db):
        with sqlite3.connect(str(v1_db)) as conn:
            conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")

        try:
            apply_migration()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "99" in str(e)

    def test_new_columns_nullable(self, migrated_db):
        with sqlite3.connect(str(migrated_db)) as conn:
            row = conn.execute(
                "SELECT eval_scores, risk_score, evaluated_at FROM runs WHERE session_id = 1 AND run_seq = 1"
            ).fetchone()
            assert row[0] is None
            assert row[1] is None
            assert row[2] is None
