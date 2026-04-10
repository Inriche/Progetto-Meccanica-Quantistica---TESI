from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["BUY", "SELL"]
ExecutionSide = Literal["entry", "exit"]


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01
    max_notional_fraction: float = 1.0
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0


def apply_slippage(price: float, direction: Direction, side: ExecutionSide, slippage_bps: float) -> float:
    if price <= 0:
        return price

    slip = (slippage_bps / 10_000.0) * price
    if direction == "BUY":
        return price + slip if side == "entry" else price - slip
    return price - slip if side == "entry" else price + slip


def compute_position_size(
    *,
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade: float,
    max_notional_fraction: float,
) -> float:
    if equity <= 0 or entry_price <= 0:
        return 0.0

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return 0.0

    risk_budget = max(0.0, equity * risk_per_trade)
    qty_by_risk = risk_budget / stop_distance

    notional_cap = max(0.0, equity * max_notional_fraction)
    qty_by_notional = notional_cap / entry_price if entry_price > 0 else 0.0

    return max(0.0, min(qty_by_risk, qty_by_notional))


def compute_gross_pnl(direction: Direction, entry_price: float, exit_price: float, quantity: float) -> float:
    if quantity <= 0:
        return 0.0

    if direction == "BUY":
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def compute_fees(entry_price: float, exit_price: float, quantity: float, fee_rate: float) -> float:
    if quantity <= 0:
        return 0.0
    notional_in = abs(entry_price * quantity)
    notional_out = abs(exit_price * quantity)
    return (notional_in + notional_out) * max(0.0, fee_rate)
