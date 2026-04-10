from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Optional

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
    returns_t_statistic: Optional[float]
    returns_p_value: Optional[float]
    benchmark_buy_hold_return: Optional[float]
    alpha_vs_benchmark: Optional[float]
    trades: int

    def to_dict(self) -> Dict[str, Optional[float]]:
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


def _student_t_pdf(x: float, df: int) -> float:
    if df <= 0:
        return 0.0
    num = math.gamma((df + 1.0) / 2.0)
    den = math.sqrt(df * math.pi) * math.gamma(df / 2.0)
    base = 1.0 + (x * x) / df
    return (num / den) * (base ** (-(df + 1.0) / 2.0))


def _simpson_integral_t_pdf(a: float, b: float, df: int, steps: int = 2048) -> float:
    if b <= a:
        return 0.0
    n = max(64, int(steps))
    if n % 2 == 1:
        n += 1
    h = (b - a) / n

    s = _student_t_pdf(a, df) + _student_t_pdf(b, df)
    for i in range(1, n):
        x = a + i * h
        coeff = 4 if i % 2 == 1 else 2
        s += coeff * _student_t_pdf(x, df)
    return (h / 3.0) * s


def _student_t_cdf(x: float, df: int) -> float:
    if df <= 0:
        return 0.5
    if x == 0:
        return 0.5

    sign = 1.0 if x > 0 else -1.0
    upper = abs(float(x))
    area = _simpson_integral_t_pdf(0.0, upper, df=df)
    cdf = 0.5 + sign * area
    return float(min(1.0, max(0.0, cdf)))


def _one_sample_t_test_pvalue(returns: pd.Series) -> tuple[Optional[float], Optional[float]]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = int(len(r))
    if n < 2:
        return None, None

    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if std <= 0:
        return None, None

    t_stat = mean / (std / math.sqrt(n))
    df = n - 1
    cdf_abs_t = _student_t_cdf(abs(t_stat), df)
    p_value = 2.0 * (1.0 - cdf_abs_t)
    p_value = float(min(1.0, max(0.0, p_value)))
    return float(t_stat), p_value


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    *,
    initial_capital: float,
    annualization_factor: float = 1.0,
    benchmark_buy_hold_return: Optional[float] = None,
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
    t_stat, p_value = _one_sample_t_test_pvalue(returns)
    alpha = None if benchmark_buy_hold_return is None else float(total_return - float(benchmark_buy_hold_return))

    return BacktestMetrics(
        total_return=total_return,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        returns_t_statistic=t_stat,
        returns_p_value=p_value,
        benchmark_buy_hold_return=benchmark_buy_hold_return,
        alpha_vs_benchmark=alpha,
        trades=trades,
    )
