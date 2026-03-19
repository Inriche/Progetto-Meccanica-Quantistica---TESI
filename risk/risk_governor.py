from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional

@dataclass
class RiskGovernorState:
    day: date
    signals_today: int
    last_signal_time: Optional[datetime]

class RiskGovernor:
    def __init__(self, max_signals_per_day: int, cooldown_minutes: int) -> None:
        self.max_signals_per_day = max_signals_per_day
        self.cooldown_minutes = cooldown_minutes
        self.state = RiskGovernorState(day=date.today(), signals_today=0, last_signal_time=None)

    def _roll_day_if_needed(self, now: datetime) -> None:
        if now.date() != self.state.day:
            self.state = RiskGovernorState(day=now.date(), signals_today=0, last_signal_time=None)

    def can_emit(self, now: datetime) -> bool:
        self._roll_day_if_needed(now)

        if self.state.signals_today >= self.max_signals_per_day:
            return False

        if self.state.last_signal_time is None:
            return True

        delta = (now - self.state.last_signal_time).total_seconds() / 60.0
        return delta >= self.cooldown_minutes

    def mark_emitted(self, now: datetime) -> None:
        self._roll_day_if_needed(now)
        self.state.signals_today += 1
        self.state.last_signal_time = now