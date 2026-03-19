from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    symbol: str = "btcusdt"
    db_path: str = "out/assistant.db"
    coinglass_api_key: str = ""

    # Candle history kept in memory for each timeframe
    max_candles_m15: int = 400
    max_candles_h1: int = 400
    max_candles_h4: int = 400

    # Snapshot
    snapshot_dir: str = "out/snapshots"
    ticket_dir: str = "out/tickets"

    # Trading rules
    rr_min: float = 1.5
    min_score_for_signal: int = 70  # A-only by default
    cooldown_minutes: int = 60
    max_signals_per_day: int = 2

CONFIG = Config()