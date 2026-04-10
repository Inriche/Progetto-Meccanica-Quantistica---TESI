from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max.replace(0, pd.NA)
    dd = drawdown.dropna()
    return 0.0 if dd.empty else float(dd.min())


def _compute_sharpe(returns: pd.Series, annualization_factor: float = 1.0) -> float:
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=1))
    if std <= 0:
        return 0.0
    return float((returns.mean() / std) * annualization_factor)


def _compute_sortino(returns: pd.Series, annualization_factor: float = 1.0) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf") if float(returns.mean()) > 0 else 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std <= 0:
        return 0.0
    return float((returns.mean() / downside_std) * annualization_factor)


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    *,
    initial_capital: float,
    annualization_factor: float = 1.0,
) -> BacktestMetrics:
    net_pnl = _safe_series(trades_df, "net_pnl")
    returns = _safe_series(trades_df, "return_pct")
    equity = _safe_series(equity_curve_df, "equity")

    final_equity = float(equity.iloc[-1]) if not equity.empty else float(initial_capital)
    total_return = 0.0 if initial_capital <= 0 else (final_equity / initial_capital) - 1.0

    wins = int((net_pnl > 0).sum())
    losses = int((net_pnl < 0).sum())
    trades = int(len(net_pnl))

    win_rate = 0.0 if trades == 0 else wins / trades

    gross_profit = float(net_pnl[net_pnl > 0].sum()) if trades > 0 else 0.0
    gross_loss = float(net_pnl[net_pnl < 0].sum()) if trades > 0 else 0.0
    profit_factor = 0.0
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0 and gross_loss == 0:
        profit_factor = float("inf")

    expectancy = float(net_pnl.mean()) if trades > 0 else 0.0
    max_drawdown = _compute_max_drawdown(equity)
    sharpe_ratio = _compute_sharpe(returns, annualization_factor=annualization_factor)
    sortino_ratio = _compute_sortino(returns, annualization_factor=annualization_factor)

    return BacktestMetrics(
        total_return=total_return,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        trades=trades,
    )
