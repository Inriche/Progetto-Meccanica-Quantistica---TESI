from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

import pandas as pd
from dateutil import tz

from config import CONFIG
from data.data_store import Candle, DataStore
from data.orderbook_store import OrderBookStore
from database.db import DB
from engine.pipeline import TradingEngine
from risk.risk_governor import RiskGovernor
from runtime.runtime_config import load_runtime_config


ROME_TZ = tz.gettz("Europe/Rome")


@dataclass
class SimulatedClock:
    timezone: Any
    current: Optional[datetime] = None

    def set_from_ms(self, ts_ms: int) -> None:
        self.current = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).astimezone(self.timezone)

    def now(self) -> datetime:
        if self.current is not None:
            return self.current
        return datetime.now(self.timezone)


def _parse_ts_ms(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return int(ts.timestamp() * 1000)


def _next_or_none(iterator: Iterator[Candle]) -> Optional[Candle]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _iter_candles(
    *,
    db: DB,
    symbol: str,
    timeframe: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    fetch_size: int = 2000,
) -> Iterator[Candle]:
    where = ["timeframe = ?", "UPPER(symbol) = UPPER(?)"]
    params: list[Any] = [timeframe, symbol]
    if start_ms is not None:
        where.append("close_time >= ?")
        params.append(int(start_ms))
    if end_ms is not None:
        where.append("close_time <= ?")
        params.append(int(end_ms))

    query = f"""
        SELECT open_time, close_time, open, high, low, close, volume
        FROM candles
        WHERE {" AND ".join(where)}
        ORDER BY close_time ASC, open_time ASC
    """
    cur = db.conn.execute(query, tuple(params))
    while True:
        rows = cur.fetchmany(int(fetch_size))
        if not rows:
            break
        for row in rows:
            yield Candle(
                open_time=int(row["open_time"]),
                close_time=int(row["close_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )


def _count_m15_rows(
    *,
    db: DB,
    symbol: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> int:
    where = ["timeframe = '15m'", "UPPER(symbol) = UPPER(?)"]
    params: list[Any] = [symbol]
    if start_ms is not None:
        where.append("close_time >= ?")
        params.append(int(start_ms))
    if end_ms is not None:
        where.append("close_time <= ?")
        params.append(int(end_ms))
    query = f"SELECT COUNT(*) AS n FROM candles WHERE {' AND '.join(where)}"
    row = db.conn.execute(query, tuple(params)).fetchone()
    return int(row["n"]) if row is not None else 0


def _orderbook_snapshot_stats(
    db: DB,
    symbol: str,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> dict[str, Any]:
    where = ["UPPER(symbol) = UPPER(?)"]
    params: list[Any] = [symbol]
    if start_ms is not None:
        where.append("(CAST(strftime('%s', timestamp) AS INTEGER) * 1000) >= ?")
        params.append(int(start_ms))
    if end_ms is not None:
        where.append("(CAST(strftime('%s', timestamp) AS INTEGER) * 1000) <= ?")
        params.append(int(end_ms))
    query = f"""
        SELECT COUNT(*) AS n, MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
        FROM orderbook_snapshots
        WHERE {" AND ".join(where)}
    """
    row = db.conn.execute(query, tuple(params)).fetchone()
    if row is None:
        return {"n": 0, "min_ts": None, "max_ts": None}
    return {
        "n": int(row["n"] or 0),
        "min_ts": row["min_ts"],
        "max_ts": row["max_ts"],
    }


def _build_backfill_runtime_config_loader() -> Any:
    def _loader() -> Dict[str, Any]:
        cfg = load_runtime_config().copy()
        cfg["alerts_enabled"] = False
        cfg["news_enabled"] = False
        cfg["use_external_context"] = False
        # Backfill runs without live orderbook/derivatives streams; persist neutral
        # microstructure fallbacks to avoid mostly-empty training columns.
        cfg["persist_neutral_microstructure"] = True
        return cfg

    return _loader


async def run_backfill(args: argparse.Namespace) -> None:
    logger = logging.getLogger("scripts.run_historical_backfill")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(handler)

    os.makedirs("out", exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    os.makedirs(args.ticket_dir, exist_ok=True)

    db = DB(args.db_path, "database/schema.sql")
    store = DataStore(CONFIG.max_candles_m15, CONFIG.max_candles_h1, CONFIG.max_candles_h4)
    orderbook_store = OrderBookStore()

    runtime_cfg = load_runtime_config()
    risk = RiskGovernor(
        max_signals_per_day=int(runtime_cfg["max_signals_per_day"]),
        cooldown_minutes=int(runtime_cfg["cooldown_minutes"]),
        state_path=args.risk_state_path,
    )
    if args.reset_risk_state and os.path.exists(args.risk_state_path):
        os.remove(args.risk_state_path)
        risk = RiskGovernor(
            max_signals_per_day=int(runtime_cfg["max_signals_per_day"]),
            cooldown_minutes=int(runtime_cfg["cooldown_minutes"]),
            state_path=args.risk_state_path,
        )

    clock = SimulatedClock(timezone=ROME_TZ)
    trading_engine = TradingEngine(
        symbol=str(args.symbol).upper(),
        store=store,
        orderbook_store=orderbook_store,
        db=db,
        risk_governor=risk,
        runtime_config_loader=_build_backfill_runtime_config_loader(),
        snapshot_dir=args.snapshot_dir,
        ticket_dir=args.ticket_dir,
        timezone=ROME_TZ,
        time_provider=clock.now,
        enable_artifacts=False,
    )

    start_ms = _parse_ts_ms(args.start)
    end_ms = _parse_ts_ms(args.end)
    if start_ms is not None and end_ms is not None and end_ms < start_ms:
        raise ValueError("end must be >= start")

    warmup_start_ms = None
    if start_ms is not None:
        warmup_start_ms = max(0, int(start_ms - int(args.warmup_hours) * 60 * 60 * 1000))

    m15_total = _count_m15_rows(
        db=db,
        symbol=str(args.symbol).upper(),
        start_ms=warmup_start_ms,
        end_ms=end_ms,
    )
    if m15_total == 0:
        logger.info("No M15 candles found for symbol=%s in selected range.", str(args.symbol).upper())
        return

    logger.info(
        "Backfill started symbol=%s start=%s end=%s warmup_hours=%s m15_rows=%s",
        str(args.symbol).upper(),
        args.start,
        args.end,
        args.warmup_hours,
        m15_total,
    )
    ob_stats = _orderbook_snapshot_stats(
        db,
        str(args.symbol).upper(),
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if ob_stats["n"] <= 0:
        logger.warning(
            "No orderbook_snapshots in selected backfill window for symbol=%s. "
            "Historical microstructure features are not reconstructible from candles alone.",
            str(args.symbol).upper(),
        )
    else:
        logger.info(
            "Orderbook snapshot coverage in window symbol=%s rows=%s min_ts=%s max_ts=%s",
            str(args.symbol).upper(),
            ob_stats["n"],
            ob_stats["min_ts"],
            ob_stats["max_ts"],
        )
    logger.info(
        "Derivatives historical context is not replayed in backfill. "
        "Funding/OI fields will use neutral fallback persistence when enabled."
    )

    m15_iter = _iter_candles(
        db=db,
        symbol=str(args.symbol).upper(),
        timeframe="15m",
        start_ms=warmup_start_ms,
        end_ms=end_ms,
    )
    h1_iter = _iter_candles(
        db=db,
        symbol=str(args.symbol).upper(),
        timeframe="1h",
        start_ms=warmup_start_ms,
        end_ms=end_ms,
    )
    h4_iter = _iter_candles(
        db=db,
        symbol=str(args.symbol).upper(),
        timeframe="4h",
        start_ms=warmup_start_ms,
        end_ms=end_ms,
    )

    next_h1 = _next_or_none(h1_iter)
    next_h4 = _next_or_none(h4_iter)

    rows_seen = 0
    rows_in_test_window = 0
    emitted_signals = 0
    status_events = 0

    for m15 in m15_iter:
        rows_seen += 1
        store.add_candle("15m", m15)

        while next_h1 is not None and next_h1.close_time <= m15.close_time:
            store.add_candle("1h", next_h1)
            next_h1 = _next_or_none(h1_iter)

        while next_h4 is not None and next_h4.close_time <= m15.close_time:
            store.add_candle("4h", next_h4)
            next_h4 = _next_or_none(h4_iter)

        if start_ms is not None and m15.close_time < start_ms:
            continue

        rows_in_test_window += 1
        clock.set_from_ms(m15.close_time)
        result = await trading_engine.run_cycle(trigger="BACKFILL")
        if result.event_type == "signal":
            emitted_signals += 1
        elif result.event_type == "status":
            status_events += 1

        db.insert_cycle_log(
            {
                "timestamp": clock.now().isoformat(),
                "symbol": str(args.symbol).upper(),
                "trigger": "BACKFILL",
                "decision": result.decision,
                "setup": result.setup,
                "context": result.context,
                "action": result.action,
                "score": float(result.score),
                "rr_estimated": result.rr_estimated,
            }
        )

        if rows_in_test_window % int(args.progress_every) == 0:
            logger.info(
                "Progress processed=%s/%s emitted_signals=%s status_events=%s",
                rows_seen,
                m15_total,
                emitted_signals,
                status_events,
            )

    logger.info(
        "Backfill completed bars_processed=%s bars_in_window=%s signals=%s status=%s risk_state=%s",
        rows_seen,
        rows_in_test_window,
        emitted_signals,
        status_events,
        args.risk_state_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chronological historical backfill with TradingEngine.")
    parser.add_argument("--db-path", default=CONFIG.db_path)
    parser.add_argument("--symbol", default=CONFIG.symbol.upper())
    parser.add_argument("--start", default=None, help="ISO UTC timestamp. Example: 2026-03-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="ISO UTC timestamp. Example: 2026-04-01T00:00:00Z")
    parser.add_argument("--warmup-hours", type=int, default=240)
    parser.add_argument("--risk-state-path", default="out/risk_backfill.json")
    parser.add_argument("--reset-risk-state", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--snapshot-dir", default="out/backfill_snapshots")
    parser.add_argument("--ticket-dir", default="out/backfill_tickets")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
