CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    close_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    UNIQUE(symbol, timeframe, open_time)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT DEFAULT 'signal',
    decision TEXT NOT NULL,
    setup TEXT NOT NULL,
    context TEXT,
    action TEXT,
    why TEXT,
    entry REAL,
    sl REAL,
    tp1 REAL,
    tp2 REAL,
    rr_estimated REAL,
    score REAL,
    ob_imbalance REAL,
    ob_raw REAL,
    ob_age_ms INTEGER,
    funding_rate REAL,
    oi_now REAL,
    oi_change_pct REAL,
    crowding TEXT,
    strategy_mode TEXT,
    strategy_score INTEGER,
    news_bias TEXT,
    news_sentiment REAL,
    news_impact REAL,
    news_score INTEGER,
    quantum_state TEXT,
    quantum_coherence REAL,
    quantum_phase_bias REAL,
    quantum_interference REAL,
    quantum_tunneling REAL,
    quantum_energy REAL,
    quantum_decoherence_rate REAL,
    quantum_transition_rate REAL,
    quantum_dominant_mode TEXT,
    quantum_score INTEGER,
    snapshot_path TEXT,
    ticket_path TEXT
);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    imbalance_avg REAL,
    imbalance_raw REAL,
    age_ms INTEGER,
    bid_notional REAL,
    ask_notional REAL,
    top_bid REAL,
    top_ask REAL,
    spread REAL,
    source TEXT DEFAULT 'depth20@100ms'
);

CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_symbol_ts
ON orderbook_snapshots(symbol, timestamp);

CREATE TABLE IF NOT EXISTS decision_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    trigger TEXT,
    event_type TEXT,
    decision TEXT,
    setup TEXT,
    context TEXT,
    action TEXT,
    score REAL,
    ticket_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_decision_logs_symbol_ts
ON decision_logs(symbol, timestamp);

CREATE TABLE IF NOT EXISTS cycle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    trigger TEXT,
    decision TEXT,
    setup TEXT,
    context TEXT,
    action TEXT,
    score REAL,
    rr_estimated REAL
);

CREATE INDEX IF NOT EXISTS idx_cycle_logs_symbol_ts
ON cycle_logs(symbol, timestamp);
