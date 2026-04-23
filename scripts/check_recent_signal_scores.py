from __future__ import annotations

import argparse
import sqlite3
from typing import Iterable


DEFAULT_DB_PATH = "out/assistant.db"


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{(numerator / denominator) * 100:.2f}%"


def _get_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(signals)").fetchall()
    return {row[1] for row in rows}


def _count_non_null(conn: sqlite3.Connection, column: str, limit: int, available: set[str]) -> int:
    if column not in available:
        return 0
    query = f"""
        SELECT COUNT(*)
        FROM (
            SELECT {column}
            FROM signals
            ORDER BY id DESC
            LIMIT ?
        )
        WHERE {column} IS NOT NULL
    """
    cur = conn.execute(query, (limit,))
    row = cur.fetchone()
    return int(row[0] if row else 0)


def _sample_rows(conn: sqlite3.Connection, limit: int, available: set[str]) -> Iterable[tuple]:
    desired = [
        "timestamp",
        "score",
        "heuristic_score",
        "strategy_score",
        "raw_hybrid_score",
        "calibrated_hybrid_score",
        "scoring_mode",
    ]
    select_cols = [c for c in desired if c in available]
    if not select_cols:
        return []

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM signals
        ORDER BY id DESC
        LIMIT ?
    """
    return conn.execute(query, (limit,)).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect recent signal score field coverage.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path)
    try:
        available = _get_columns(conn)
        total = int(conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0])
        recent_limit = max(1, int(args.limit))
        heuristic_non_null = _count_non_null(conn, "heuristic_score", recent_limit, available)
        raw_non_null = _count_non_null(conn, "raw_hybrid_score", recent_limit, available)
        calibrated_non_null = _count_non_null(conn, "calibrated_hybrid_score", recent_limit, available)
        scoring_mode_non_null = _count_non_null(conn, "scoring_mode", recent_limit, available)

        print(f"db_path={args.db_path}")
        print(f"recent_window={recent_limit}")
        print(f"total_signals={total}")
        print(f"available_score_columns={sorted(c for c in available if 'score' in c or c == 'scoring_mode')}")
        print(f"heuristic_score_non_null={heuristic_non_null} ({_format_pct(heuristic_non_null, recent_limit)})")
        print(f"raw_hybrid_score_non_null={raw_non_null} ({_format_pct(raw_non_null, recent_limit)})")
        print(f"calibrated_hybrid_score_non_null={calibrated_non_null} ({_format_pct(calibrated_non_null, recent_limit)})")
        print(f"scoring_mode_non_null={scoring_mode_non_null} ({_format_pct(scoring_mode_non_null, recent_limit)})")
        print("latest_rows=")
        for row in _sample_rows(conn, min(10, recent_limit), available):
            print(row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
