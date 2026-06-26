"""
Run the fake RAG pipeline with 8 queries across two topic groups.

Inserts a 31-minute time gap between groups to trigger auto-session creation.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from pipeline import run_pipeline

GROUP_1 = [
    "what is RRF and how does it normalize scores",
    "why does BM25 score differ from vector similarity score",
    "does RRF handle score scale differences across retrievers",
    "what is the role of k in RRF formula",
]

GROUP_2 = [
    "what does a cross-encoder actually compute",
    "why is cross-encoder slower than bi-encoder",
    "does rerank order affect final context assembly",
    "when should I skip reranking in a RAG pipeline",
]

PIPELINE = "rag_example"


def _backdate_pipeline_runs(db_path: Path, minutes: int) -> None:
    """Shift all rag_example timestamps back to simulate idle gap."""
    with sqlite3.connect(str(db_path)) as conn:
        for table, key_cols in [
            ("sessions", ["session_id"]),
            ("runs", ["session_id", "run_seq"]),
        ]:
            rows = conn.execute(
                f"SELECT {', '.join(key_cols)}, created_at "
                f"FROM {table} WHERE pipeline = ?",
                (PIPELINE,),
            ).fetchall()
            for row in rows:
                keys = row[:-1]
                old_ts = datetime.fromisoformat(row[-1])
                new_ts = (old_ts - timedelta(minutes=minutes)).isoformat()
                where = " AND ".join(f"{c} = ?" for c in key_cols)
                conn.execute(
                    f"UPDATE {table} SET created_at = ? WHERE {where}",
                    (new_ts, *keys),
                )


def main():
    print("Group 1: retrieval mechanics (4 queries)...")
    for q in GROUP_1:
        run_pipeline(q)
        print(f"  captured: {q[:50]}")

    print("\nSimulating 31-minute idle gap...")
    db_path = Path.home() / ".ctx" / "runs.db"
    _backdate_pipeline_runs(db_path, minutes=31)

    print("\nGroup 2: reranking (4 queries)...")
    for q in GROUP_2:
        run_pipeline(q)
        print(f"  captured: {q[:50]}")

    print("\nCaptured 8 runs across 2 sessions.")
    print("Run: ctx list")
    print("Run: ctx explain")
    print("Run: ctx-evaluate run --input-only")


if __name__ == "__main__":
    main()
