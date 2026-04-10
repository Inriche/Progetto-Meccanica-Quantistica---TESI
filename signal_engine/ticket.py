import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from features.indicators import rr


@dataclass
class Ticket:
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, indent=2)

    def save(self, ticket_dir: str) -> str:
        os.makedirs(ticket_dir, exist_ok=True)
        signal_id = self.payload["id"]
        path = os.path.join(ticket_dir, f"{signal_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path


def make_signal_id(symbol: str, ts: datetime) -> str:
    return f"sig_{ts.strftime('%Y%m%d_%H%M%S')}_{symbol.upper()}"


def build_ticket(
    symbol: str,
    timestamp: datetime,
    decision: str,
    setup: str,
    bias_h1: str,
    bias_h4: str,
    combined_bias: str,
    volatility: str,
    entry: Optional[float],
    sl: Optional[float],
    tp1: Optional[float],
    tp2: Optional[float],
    score: int,
    grade: str,
    components: list,
    reasons: list,
    snapshot_path: Optional[str],
    rr_min_required: float = 2.0,
    min_score_for_signal: int = 70,
    max_signals_per_day: int = 2,
    cooldown_minutes: int = 60,
    context: str = "transition",
    action: str = "STAND_BY",
    liquidation_cluster: Optional[float] = None,
    strategy_snapshot: Optional[Dict[str, Any]] = None,
    news_snapshot: Optional[Dict[str, Any]] = None,
    quantum_snapshot: Optional[Dict[str, Any]] = None,
    event_snapshot: Optional[Dict[str, Any]] = None,
    risk_gate_status: Optional[Dict[str, Any]] = None,
) -> Ticket:

    signal_id = make_signal_id(symbol, timestamp)

    rr_est = None
    if entry is not None and sl is not None and tp1 is not None:
        rr_est = rr(entry, sl, tp1)

    payload = {
        "id": signal_id,
        "timestamp": timestamp.isoformat(),
        "symbol": symbol.upper(),
        "mode": "B",
        "decision": decision,
        "setup": setup,
        "timeframes": {
            "context": ["H4", "H1"],
            "entry": "M15"
        },
        "market_state": {
            "regime": context,
            "volatility": volatility,
            "session": "unknown"
        },
        "bias": {
            "H4": bias_h4,
            "H1": bias_h1,
            "combined": combined_bias
        },
        "levels": {
            "entry": {
                "type": "limit",
                "price": entry,
                "valid_for_minutes": 180
            },
            "stop_loss": {
                "price": sl,
                "reason": "Invalidation + ATR buffer"
            },
            "take_profits": [
                {"price": tp1, "reason": "TP1 - structure / liquidity"},
                {"price": tp2, "reason": "TP2 - extended target"}
            ]
        },
        "risk": {
            "rr_min_required": rr_min_required,
            "min_score_for_signal": min_score_for_signal,
            "rr_estimated": rr_est,
            "max_signals_per_day": max_signals_per_day,
            "cooldown_minutes": cooldown_minutes,
            "gate_status": risk_gate_status or {},
        },
        "confidence": {
            "score": score,
            "grade": grade,
            "components": [{"name": n, "points": p} for (n, p) in components],
            "note": "Score-based confidence (not calibrated probability)."
        },
        "advice": {
            "context": context,
            "action": action
        },
        "liquidity": {
            "nearest_liquidation_cluster": liquidation_cluster
        },
        "strategy": strategy_snapshot or {},
        "news": news_snapshot or {},
        "quantum": quantum_snapshot or {},
        "event_snapshot": event_snapshot or {},
        "evidence": {
            "snapshot_path": snapshot_path,
            "key_reasons": reasons
        },
        "operator_actions": {
            "status": "NEW",
            "copied_to_clipboard": False,
            "executed": False,
            "ignored": False
        }
    }

    return Ticket(payload=payload)
