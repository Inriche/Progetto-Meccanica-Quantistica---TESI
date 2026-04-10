from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import pandas as pd

from backtest.metrics import BacktestMetrics, compute_backtest_metrics
from backtest.portfolio import (
    PortfolioConfig,
    apply_slippage,
    compute_fees,
    compute_gross_pnl,
    compute_position_size,
)
from execution.outcome_simulator import simulate_outcome_from_candles


InputMode = Literal["signals", "scores"]


@dataclass(frozen=True)
class BacktestConfig:
    db_path: str = "out/assistant.db"
    symbol: Optional[str] = None
    input_mode: InputMode = "signals"
    min_score: Optional[float] = None
    timeframe: str = "15m"
    horizon_bars: int = 24
    close_open_positions_at_horizon: bool = True
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


@dataclass
class BacktestResult:
    config: BacktestConfig
    signals_df: pd.DataFrame
    trades_df: pd.DataFrame
    equity_curve_df: pd.DataFrame
    metrics: BacktestMetrics

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "trades": int(len(self.trades_df)),
            **self.metrics.to_dict(),
        }


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("backtest.engine")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def _infer_decision(row: Dict[str, Any]) -> Optional[str]:
    raw = str(row.get("decision", "")).upper().strip()
    if raw in ("BUY", "SELL"):
        return raw

    entry = _coerce_float(row.get("entry"))
    tp1 = _coerce_float(row.get("tp1"))
    if entry is None or tp1 is None:
        return None
    if tp1 > entry:
        return "BUY"
    if tp1 < entry:
        return "SELL"
    return None


def _normalize_inputs(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["decision"] = [
        _infer_decision(row)
        for row in out.to_dict("records")
    ]
    out = out[out["decision"].isin(["BUY", "SELL"])].copy()

    required = ["timestamp", "entry", "sl", "tp1"]
    for col in required:
        out = out[out[col].notna()].copy()

    if config.min_score is not None and "score" in out.columns:
        out = out[pd.to_numeric(out["score"], errors="coerce") >= float(config.min_score)].copy()

    if "signal_id" in out.columns:
        out = out.drop_duplicates(subset=["signal_id"], keep="first")

    out = out.sort_values(["timestamp", "id"] if "id" in out.columns else ["timestamp"]).reset_index(drop=True)
    return out


def load_historical_inputs(config: BacktestConfig) -> pd.DataFrame:
    if not os.path.exists(config.db_path):
        return pd.DataFrame()

    conn = _get_conn(config.db_path)
    where = [
        "event_type = 'signal'",
        "timestamp IS NOT NULL",
        "entry IS NOT NULL",
        "sl IS NOT NULL",
        "tp1 IS NOT NULL",
    ]
    params: list[Any] = []

    if config.symbol:
        where.append("UPPER(symbol) = ?")
        params.append(str(config.symbol).upper())

    if config.input_mode == "signals":
        where.append("decision IN ('BUY', 'SELL')")
    else:
        where.append("score IS NOT NULL")

    query = f"""
        SELECT
            id,
            signal_id,
            timestamp,
            symbol,
            decision,
            setup,
            context,
            action,
            score,
            rr_estimated,
            entry,
            sl,
            tp1,
            tp2,
            snapshot_path,
            ticket_path
        FROM signals
        WHERE {" AND ".join(where)}
        ORDER BY timestamp ASC, id ASC
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return _normalize_inputs(df, config)


def _load_future_candles(
    *,
    db_path: str,
    timestamp_iso: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        event_ms = int(pd.Timestamp(timestamp_iso).timestamp() * 1000)
    except Exception:
        return pd.DataFrame()

    conn = _get_conn(db_path)
    query = """
        SELECT open_time, close_time, open, high, low, close
        FROM candles
        WHERE timeframe = ?
          AND open_time >= ?
        ORDER BY open_time ASC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(timeframe, event_ms, limit))
    conn.close()
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def _iso_to_ms(ts: Any) -> Optional[int]:
    try:
        if ts is None:
            return None
        return int(pd.Timestamp(ts).timestamp() * 1000)
    except Exception:
        return None


def _compute_buy_hold_return(
    *,
    db_path: str,
    timeframe: str,
    symbol: Optional[str],
    start_ts: Any,
    end_ts: Any,
) -> Optional[float]:
    start_ms = _iso_to_ms(start_ts)
    end_ms = _iso_to_ms(end_ts)
    if start_ms is None or end_ms is None or end_ms < start_ms:
        return None

    if not os.path.exists(db_path):
        return None

    conn = _get_conn(db_path)
    where = [
        "timeframe = ?",
        "open_time >= ?",
        "open_time <= ?",
    ]
    params: list[Any] = [timeframe, start_ms, end_ms]
    if symbol:
        where.append("UPPER(symbol) = ?")
        params.append(str(symbol).upper())

    query = f"""
        SELECT open_time, close
        FROM candles
        WHERE {" AND ".join(where)}
        ORDER BY open_time ASC
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty or len(df) < 2:
        return None

    first_close = _coerce_float(df["close"].iloc[0])
    last_close = _coerce_float(df["close"].iloc[-1])
    if first_close is None or last_close is None or first_close <= 0:
        return None

    return float((last_close / first_close) - 1.0)


def _simulate_single_trade(
    *,
    signal: Dict[str, Any],
    equity_before: float,
    config: BacktestConfig,
) -> Optional[Dict[str, Any]]:
    decision = str(signal["decision"]).upper()
    entry = _coerce_float(signal.get("entry"))
    sl = _coerce_float(signal.get("sl"))
    tp1 = _coerce_float(signal.get("tp1"))
    tp2 = _coerce_float(signal.get("tp2"))
    if decision not in ("BUY", "SELL") or entry is None or sl is None or tp1 is None:
        return None

    qty = compute_position_size(
        equity=equity_before,
        entry_price=entry,
        stop_loss=sl,
        risk_per_trade=config.portfolio.risk_per_trade,
        max_notional_fraction=config.portfolio.max_notional_fraction,
    )
    if qty <= 0:
        return None

    future_df = _load_future_candles(
        db_path=config.db_path,
        timestamp_iso=str(signal["timestamp"]),
        timeframe=config.timeframe,
        limit=config.horizon_bars,
    )
    if future_df.empty:
        return None

    outcome = simulate_outcome_from_candles(
        decision=decision,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        future_df=future_df,
    )

    status = str(outcome.get("status", "no_data"))
    if status == "no_data":
        return None

    if status == "open":
        if not config.close_open_positions_at_horizon:
            return None
        exit_ref = float(future_df["close"].iloc[-1])
        exit_reason = "horizon_close"
        exit_time = str(future_df["time"].iloc[-1])
    else:
        hit_price = _coerce_float(outcome.get("hit_price"))
        if hit_price is None:
            return None
        exit_ref = hit_price
        exit_reason = status
        exit_time = str(outcome.get("hit_time"))

    entry_exec = apply_slippage(entry, decision, "entry", config.portfolio.slippage_bps)
    exit_exec = apply_slippage(exit_ref, decision, "exit", config.portfolio.slippage_bps)

    gross_pnl = compute_gross_pnl(decision, entry_exec, exit_exec, qty)
    fees = compute_fees(entry_exec, exit_exec, qty, config.portfolio.fee_rate)
    net_pnl = gross_pnl - fees
    equity_after = equity_before + net_pnl
    ret_pct = 0.0 if equity_before == 0 else net_pnl / equity_before

    bars_held = int(outcome.get("bars_checked", len(future_df)))
    return {
        "signal_id": signal.get("signal_id"),
        "timestamp": signal.get("timestamp"),
        "exit_time": exit_time,
        "symbol": signal.get("symbol"),
        "decision": decision,
        "setup": signal.get("setup"),
        "context": signal.get("context"),
        "action": signal.get("action"),
        "score": signal.get("score"),
        "rr_estimated": signal.get("rr_estimated"),
        "entry_price": entry_exec,
        "exit_price": exit_exec,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "quantity": qty,
        "notional": entry_exec * qty,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "return_pct": ret_pct,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "outcome_status": status,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "snapshot_path": signal.get("snapshot_path"),
        "ticket_path": signal.get("ticket_path"),
    }


def run_backtest(config: BacktestConfig, inputs_df: Optional[pd.DataFrame] = None) -> BacktestResult:
    logger = _get_logger()
    logger.info("Backtest started input_mode=%s timeframe=%s horizon=%s", config.input_mode, config.timeframe, config.horizon_bars)

    signals_df = _normalize_inputs(inputs_df, config) if inputs_df is not None else load_historical_inputs(config)
    if signals_df.empty:
        empty_trades = pd.DataFrame()
        equity_curve_df = pd.DataFrame([{"timestamp": None, "equity": config.portfolio.initial_capital}])
        metrics = compute_backtest_metrics(
            empty_trades,
            equity_curve_df,
            initial_capital=config.portfolio.initial_capital,
            benchmark_buy_hold_return=None,
        )
        return BacktestResult(
            config=config,
            signals_df=signals_df,
            trades_df=empty_trades,
            equity_curve_df=equity_curve_df,
            metrics=metrics,
        )

    trades: list[Dict[str, Any]] = []
    equity_points: list[Dict[str, Any]] = []
    equity = float(config.portfolio.initial_capital)
    first_ts = str(signals_df.iloc[0]["timestamp"])
    equity_points.append({"timestamp": first_ts, "equity": equity})

    for signal in signals_df.to_dict("records"):
        trade = _simulate_single_trade(
            signal=signal,
            equity_before=equity,
            config=config,
        )
        if trade is None:
            continue
        trades.append(trade)
        equity = float(trade["equity_after"])
        equity_points.append({"timestamp": trade["exit_time"], "equity": equity})

    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_points)

    start_ts = signals_df.iloc[0]["timestamp"] if not signals_df.empty else None
    end_ts = None
    if not trades_df.empty and "exit_time" in trades_df.columns:
        exits = pd.to_datetime(trades_df["exit_time"], errors="coerce").dropna()
        if not exits.empty:
            end_ts = exits.max()
    if end_ts is None and not signals_df.empty:
        end_ts = signals_df.iloc[-1]["timestamp"]

    symbol_for_benchmark = config.symbol
    if symbol_for_benchmark is None and "symbol" in signals_df.columns and not signals_df.empty:
        symbol_for_benchmark = str(signals_df.iloc[0]["symbol"])

    benchmark_buy_hold_return = None
    if start_ts is not None and end_ts is not None:
        benchmark_buy_hold_return = _compute_buy_hold_return(
            db_path=config.db_path,
            timeframe=config.timeframe,
            symbol=symbol_for_benchmark,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    metrics = compute_backtest_metrics(
        trades_df,
        equity_curve_df,
        initial_capital=config.portfolio.initial_capital,
        benchmark_buy_hold_return=benchmark_buy_hold_return,
    )

    logger.info(
        "Backtest completed trades=%s total_return=%.4f benchmark=%.4f alpha=%.4f",
        len(trades_df),
        metrics.total_return,
        metrics.benchmark_buy_hold_return if metrics.benchmark_buy_hold_return is not None else float("nan"),
        metrics.alpha_vs_benchmark if metrics.alpha_vs_benchmark is not None else float("nan"),
    )
    return BacktestResult(
        config=config,
        signals_df=signals_df,
        trades_df=trades_df,
        equity_curve_df=equity_curve_df,
        metrics=metrics,
    )


def run_example_backtest(db_path: str = "out/assistant.db") -> BacktestResult:
    config = BacktestConfig(
        db_path=db_path,
        input_mode="signals",
        min_score=70,
        timeframe="15m",
        horizon_bars=24,
        portfolio=PortfolioConfig(
            initial_capital=10_000.0,
            risk_per_trade=0.01,
            max_notional_fraction=1.0,
            fee_rate=0.0004,
            slippage_bps=2.0,
        ),
    )
    return run_backtest(config)


if __name__ == "__main__":
    result = run_example_backtest()
    print(result.summary_dict())
