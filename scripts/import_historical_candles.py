from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import CONFIG
from database.db import DB


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SUPPORTED_TIMEFRAMES = ("15m", "1h", "4h")
INTERVAL_MS: dict[str, int] = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


def _parse_ts_ms(value: str | None, *, default_now: bool = False) -> int | None:
    if value is None or str(value).strip() == "":
        if not default_now:
            return None
        return int(pd.Timestamp.utcnow().timestamp() * 1000)
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return int(ts.timestamp() * 1000)


def _parse_timeframes(raw: str) -> list[str]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return list(SUPPORTED_TIMEFRAMES)
    invalid = [p for p in parts if p not in SUPPORTED_TIMEFRAMES]
    if invalid:
        raise ValueError(f"Unsupported timeframes: {invalid}. Allowed: {list(SUPPORTED_TIMEFRAMES)}")
    return parts


def _fetch_klines_page(
    *,
    symbol: str,
    interval: str,
    start_ms: int | None,
    end_ms: int | None,
    limit: int,
    timeout_s: float,
) -> list[list[Any]]:
    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit),
    }
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=float(timeout_s))
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Binance response type: {type(data)}")
    return data


def _normalize_rows(symbol: str, timeframe: str, raw_page: list[list[Any]]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for k in raw_page:
        if not isinstance(k, list) or len(k) < 7:
            continue
        rows.append(
            (
                symbol.upper(),
                timeframe,
                int(k[0]),  # open_time
                int(k[6]),  # close_time
                float(k[1]),  # open
                float(k[2]),  # high
                float(k[3]),  # low
                float(k[4]),  # close
                float(k[5]),  # volume
            )
        )
    return rows


def _insert_candle_rows(db: DB, rows: list[tuple[Any, ...]]) -> tuple[int, int]:
    if not rows:
        return 0, 0
    before = int(db.conn.total_changes)
    db.conn.executemany(
        """
        INSERT OR IGNORE INTO candles
        (symbol, timeframe, open_time, close_time, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    db.conn.commit()
    inserted = int(db.conn.total_changes) - before
    ignored = len(rows) - inserted
    return inserted, ignored


def import_historical_candles(args: argparse.Namespace) -> None:
    logger = logging.getLogger("scripts.import_historical_candles")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(handler)

    schema_path = Path(__file__).resolve().parents[1] / "database" / "schema.sql"
    db = DB(args.db_path, str(schema_path))

    start_ms = _parse_ts_ms(args.start, default_now=False)
    end_ms = _parse_ts_ms(args.end, default_now=True)
    if start_ms is None:
        raise ValueError("--start is required (ISO UTC timestamp).")
    if end_ms is not None and end_ms < start_ms:
        raise ValueError("end must be >= start")

    timeframes = _parse_timeframes(args.timeframes)
    limit = max(1, min(int(args.limit), 1000))

    logger.info(
        "Import started symbol=%s timeframes=%s start=%s end=%s limit=%s",
        args.symbol.upper(),
        ",".join(timeframes),
        args.start,
        args.end or "now",
        limit,
    )

    for timeframe in timeframes:
        interval_step = INTERVAL_MS[timeframe]
        cursor_ms: int | None = int(start_ms)
        total_fetched = 0
        total_inserted = 0
        total_ignored = 0
        page_idx = 0

        while True:
            page_idx += 1
            raw_page = _fetch_klines_page(
                symbol=args.symbol,
                interval=timeframe,
                start_ms=cursor_ms,
                end_ms=end_ms,
                limit=limit,
                timeout_s=float(args.timeout_s),
            )
            if not raw_page:
                break

            rows = _normalize_rows(args.symbol, timeframe, raw_page)
            if end_ms is not None:
                rows = [r for r in rows if int(r[2]) <= int(end_ms)]
            if not rows:
                break

            inserted, ignored = _insert_candle_rows(db, rows)
            fetched = len(rows)
            total_fetched += fetched
            total_inserted += inserted
            total_ignored += ignored

            first_open = int(rows[0][2])
            last_open = int(rows[-1][2])
            logger.info(
                "timeframe=%s page=%s fetched=%s inserted=%s ignored=%s open_time=[%s..%s]",
                timeframe,
                page_idx,
                fetched,
                inserted,
                ignored,
                first_open,
                last_open,
            )

            if fetched < limit:
                break
            next_cursor = int(last_open + interval_step)
            if cursor_ms is not None and next_cursor <= cursor_ms:
                logger.warning("Stopping on non-increasing cursor timeframe=%s cursor=%s next=%s", timeframe, cursor_ms, next_cursor)
                break
            if end_ms is not None and next_cursor > end_ms:
                break

            cursor_ms = next_cursor
            if float(args.sleep_ms) > 0:
                time.sleep(float(args.sleep_ms) / 1000.0)

        logger.info(
            "timeframe=%s completed fetched=%s inserted=%s ignored=%s",
            timeframe,
            total_fetched,
            total_inserted,
            total_ignored,
        )

    logger.info("Import completed symbol=%s", args.symbol.upper())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import historical Binance candles into SQLite candles table.")
    parser.add_argument("--db-path", default=CONFIG.db_path, help="SQLite DB path.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol, e.g. BTCUSDT.")
    parser.add_argument(
        "--timeframes",
        default="15m,1h,4h",
        help="Comma-separated timeframes: 15m,1h,4h",
    )
    parser.add_argument("--start", required=True, help="Start timestamp (ISO UTC), e.g. 2024-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="End timestamp (ISO UTC), default=now")
    parser.add_argument("--limit", type=int, default=1000, help="Binance klines page size (max 1000).")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="HTTP timeout per request.")
    parser.add_argument("--sleep-ms", type=float, default=120.0, help="Delay between paginated calls.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import_historical_candles(args)


if __name__ == "__main__":
    main()
