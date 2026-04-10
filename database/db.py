import os
import sqlite3
from typing import Dict, Any, List, Optional


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class DB:
    def __init__(self, db_path: str, schema_path: str) -> None:
        ensure_parent_dir(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

        self._migrate_signals_table()
        self._migrate_cycle_logs_table()
        self.conn.commit()

    def _get_table_columns(self, table_name: str) -> List[str]:
        cur = self.conn.execute(f"PRAGMA table_info({table_name})")
        rows = cur.fetchall()
        return [row["name"] for row in rows]

    def _add_column_if_missing(self, table_name: str, column_name: str, column_type: str) -> None:
        existing = self._get_table_columns(table_name)
        if column_name not in existing:
            self.conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )

    def _migrate_signals_table(self) -> None:
        existing_tables = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        ).fetchone()

        if not existing_tables:
            return

        migrations = [
            ("event_type", "TEXT DEFAULT 'signal'"),
            ("context", "TEXT"),
            ("action", "TEXT"),
            ("why", "TEXT"),
            ("entry", "REAL"),
            ("sl", "REAL"),
            ("tp1", "REAL"),
            ("tp2", "REAL"),
            ("rr_estimated", "REAL"),
            ("score", "REAL"),
            ("ob_imbalance", "REAL"),
            ("ob_raw", "REAL"),
            ("ob_age_ms", "INTEGER"),
            ("funding_rate", "REAL"),
            ("oi_now", "REAL"),
            ("oi_change_pct", "REAL"),
            ("crowding", "TEXT"),
            ("strategy_mode", "TEXT"),
            ("strategy_score", "INTEGER"),
            ("news_bias", "TEXT"),
            ("news_sentiment", "REAL"),
            ("news_impact", "REAL"),
            ("news_score", "INTEGER"),
            ("quantum_state", "TEXT"),
            ("quantum_coherence", "REAL"),
            ("quantum_phase_bias", "REAL"),
            ("quantum_interference", "REAL"),
            ("quantum_tunneling", "REAL"),
            ("quantum_score", "INTEGER"),
            ("snapshot_path", "TEXT"),
            ("ticket_path", "TEXT"),
        ]

        for col_name, col_type in migrations:
            self._add_column_if_missing("signals", col_name, col_type)

    def _migrate_cycle_logs_table(self) -> None:
        existing_tables = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cycle_logs'"
        ).fetchone()
        if not existing_tables:
            return

        migrations = [
            ("symbol", "TEXT"),
            ("rr_estimated", "REAL"),
            ("trigger", "TEXT"),
            ("decision", "TEXT"),
            ("setup", "TEXT"),
            ("context", "TEXT"),
            ("action", "TEXT"),
            ("score", "REAL"),
        ]
        for col_name, col_type in migrations:
            self._add_column_if_missing("cycle_logs", col_name, col_type)

    def insert_orderbook_snapshot(self, row: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO orderbook_snapshots
            (timestamp, symbol, imbalance_avg, imbalance_raw, age_ms,
             bid_notional, ask_notional, top_bid, top_ask, spread, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"],
                row["symbol"],
                row.get("imbalance_avg"),
                row.get("imbalance_raw"),
                row.get("age_ms"),
                row.get("bid_notional"),
                row.get("ask_notional"),
                row.get("top_bid"),
                row.get("top_ask"),
                row.get("spread"),
                row.get("source", "depth20@100ms"),
            ),
        )
        self.conn.commit()

    def insert_decision_log(self, row: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO decision_logs
            (timestamp, symbol, trigger, event_type, decision, setup, context, action, score, ticket_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"],
                row["symbol"],
                row.get("trigger"),
                row.get("event_type"),
                row.get("decision"),
                row.get("setup"),
                row.get("context"),
                row.get("action"),
                row.get("score"),
                row.get("ticket_path"),
            ),
        )
        self.conn.commit()

    def insert_cycle_log(self, row: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO cycle_logs
            (timestamp, symbol, trigger, decision, setup, context, action, score, rr_estimated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["timestamp"],
                row["symbol"],
                row.get("trigger"),
                row.get("decision"),
                row.get("setup"),
                row.get("context"),
                row.get("action"),
                row.get("score"),
                row.get("rr_estimated"),
            ),
        )
        self.conn.commit()

    def upsert_candle(self, symbol: str, timeframe: str, candle: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO candles
            (symbol, timeframe, open_time, close_time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                timeframe,
                int(candle["open_time"]),
                int(candle["close_time"]),
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
                float(candle["volume"]),
            ),
        )
        self.conn.commit()

    def insert_signal(self, row: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO signals
            (signal_id, timestamp, symbol, event_type, decision, setup, context, action, why,
             entry, sl, tp1, tp2, rr_estimated, score,
             ob_imbalance, ob_raw, ob_age_ms,
             funding_rate, oi_now, oi_change_pct, crowding,
             strategy_mode, strategy_score, news_bias, news_sentiment, news_impact, news_score,
             quantum_state, quantum_coherence, quantum_phase_bias, quantum_interference, quantum_tunneling, quantum_score,
             snapshot_path, ticket_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["signal_id"],
                row["timestamp"],
                row["symbol"],
                row.get("event_type", "signal"),
                row["decision"],
                row["setup"],
                row.get("context"),
                row.get("action"),
                row.get("why"),
                row.get("entry"),
                row.get("sl"),
                row.get("tp1"),
                row.get("tp2"),
                row.get("rr_estimated"),
                row.get("score"),
                row.get("ob_imbalance"),
                row.get("ob_raw"),
                row.get("ob_age_ms"),
                row.get("funding_rate"),
                row.get("oi_now"),
                row.get("oi_change_pct"),
                row.get("crowding"),
                row.get("strategy_mode"),
                row.get("strategy_score"),
                row.get("news_bias"),
                row.get("news_sentiment"),
                row.get("news_impact"),
                row.get("news_score"),
                row.get("quantum_state"),
                row.get("quantum_coherence"),
                row.get("quantum_phase_bias"),
                row.get("quantum_interference"),
                row.get("quantum_tunneling"),
                row.get("quantum_score"),
                row.get("snapshot_path"),
                row.get("ticket_path"),
            ),
        )
        self.conn.commit()

    def list_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT signal_id, timestamp, symbol, event_type, decision, setup, context, action, why,
                   entry, sl, tp1, tp2, rr_estimated, score,
                   ob_imbalance, ob_raw, ob_age_ms,
                   funding_rate, oi_now, oi_change_pct, crowding,
                   strategy_mode, strategy_score, news_bias, news_sentiment, news_impact, news_score,
                   quantum_state, quantum_coherence, quantum_phase_bias, quantum_interference, quantum_tunneling, quantum_score,
                   ticket_path
            FROM signals
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_last_event(self) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT event_type, decision, setup, context, action, why
            FROM signals
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None
