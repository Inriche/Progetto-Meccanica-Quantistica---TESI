import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional


DEFAULT_RISK_STATE_PATH = "out/risk_governor_state.json"


@dataclass
class RiskGovernorState:
    day: date
    signals_today: int
    last_signal_time: Optional[datetime]


class RiskGovernor:
    def __init__(
        self,
        max_signals_per_day: int,
        cooldown_minutes: int,
        state_path: str = DEFAULT_RISK_STATE_PATH,
    ) -> None:
        self.max_signals_per_day = int(max_signals_per_day)
        self.cooldown_minutes = int(cooldown_minutes)
        self.state_path = state_path
        self.state = self._load_state()

    def _default_state(self) -> RiskGovernorState:
        return RiskGovernorState(day=date.today(), signals_today=0, last_signal_time=None)

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _serialize_state(self) -> Dict[str, Any]:
        return {
            "day": self.state.day.isoformat(),
            "signals_today": int(self.state.signals_today),
            "last_signal_time": (
                self.state.last_signal_time.isoformat()
                if self.state.last_signal_time is not None
                else None
            ),
        }

    def _load_state(self) -> RiskGovernorState:
        if not os.path.exists(self.state_path):
            return self._default_state()

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if not isinstance(raw, dict):
                return self._default_state()

            day_raw = raw.get("day")
            last_signal_raw = raw.get("last_signal_time")

            return RiskGovernorState(
                day=date.fromisoformat(str(day_raw)) if day_raw else date.today(),
                signals_today=max(0, int(raw.get("signals_today", 0))),
                last_signal_time=(
                    datetime.fromisoformat(str(last_signal_raw))
                    if last_signal_raw
                    else None
                ),
            )
        except Exception:
            return self._default_state()

    def _persist_state(self) -> None:
        self._ensure_parent_dir()
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self._serialize_state(), f, indent=2)

    def _minutes_since_last_signal(self, now: datetime) -> Optional[float]:
        if self.state.last_signal_time is None:
            return None

        last_signal_time = self.state.last_signal_time

        if now.tzinfo is not None and last_signal_time.tzinfo is None:
            last_signal_time = last_signal_time.replace(tzinfo=now.tzinfo)
        elif now.tzinfo is None and last_signal_time.tzinfo is not None:
            now = now.replace(tzinfo=last_signal_time.tzinfo)

        delta = (now - last_signal_time).total_seconds() / 60.0
        return max(0.0, delta)

    def sync_limits(self, max_signals_per_day: int, cooldown_minutes: int) -> None:
        self.max_signals_per_day = int(max_signals_per_day)
        self.cooldown_minutes = int(cooldown_minutes)

    def _roll_day_if_needed(self, now: datetime) -> None:
        if now.date() != self.state.day:
            self.state = RiskGovernorState(
                day=now.date(),
                signals_today=0,
                last_signal_time=None,
            )
            self._persist_state()

    def status_snapshot(self, now: datetime) -> Dict[str, Any]:
        self._roll_day_if_needed(now)

        minutes_since_last_signal = self._minutes_since_last_signal(now)
        cooldown_remaining = None

        if minutes_since_last_signal is not None:
            cooldown_remaining = max(
                0.0,
                round(self.cooldown_minutes - minutes_since_last_signal, 2),
            )

        block_reason = None
        if self.state.signals_today >= self.max_signals_per_day:
            block_reason = "daily_limit"
        elif cooldown_remaining is not None and cooldown_remaining > 0:
            block_reason = "cooldown"

        return {
            "day": self.state.day.isoformat(),
            "signals_today": int(self.state.signals_today),
            "max_signals_per_day": int(self.max_signals_per_day),
            "cooldown_minutes": int(self.cooldown_minutes),
            "last_signal_time": (
                self.state.last_signal_time.isoformat()
                if self.state.last_signal_time is not None
                else None
            ),
            "minutes_since_last_signal": (
                round(minutes_since_last_signal, 2)
                if minutes_since_last_signal is not None
                else None
            ),
            "cooldown_remaining_minutes": cooldown_remaining,
            "can_emit": block_reason is None,
            "block_reason": block_reason,
        }

    def can_emit(self, now: datetime) -> bool:
        return bool(self.status_snapshot(now)["can_emit"])

    def mark_emitted(self, now: datetime) -> None:
        self._roll_day_if_needed(now)
        self.state.signals_today += 1
        self.state.last_signal_time = now
        self._persist_state()


def get_risk_governor_status(
    max_signals_per_day: int,
    cooldown_minutes: int,
    now: datetime,
    state_path: str = DEFAULT_RISK_STATE_PATH,
) -> Dict[str, Any]:
    governor = RiskGovernor(
        max_signals_per_day=max_signals_per_day,
        cooldown_minutes=cooldown_minutes,
        state_path=state_path,
    )
    return governor.status_snapshot(now)
