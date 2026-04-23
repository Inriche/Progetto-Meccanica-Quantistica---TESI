from datetime import datetime

import streamlit as st
from dateutil import tz

from risk.risk_governor import get_risk_governor_status
from runtime.runtime_config import (
    load_runtime_config,
    save_runtime_config,
    reset_runtime_config,
    DEFAULT_RUNTIME_CONFIG,
    PRESET_CONFIGS,
    apply_preset,
    describe_runtime_profile,
)
from signal_engine.strategy_profile import STRATEGY_PROFILES

ROME_TZ = tz.gettz("Europe/Rome")

st.set_page_config(
    page_title="Trading Assistant - Tune Engine",
    layout="wide"
)

st.title("Trading Assistant - Tune Engine")

cfg = load_runtime_config()
profile_info = describe_runtime_profile(cfg)
risk_status = get_risk_governor_status(
    max_signals_per_day=int(cfg["max_signals_per_day"]),
    cooldown_minutes=int(cfg["cooldown_minutes"]),
    now=datetime.now(ROME_TZ),
)

st.write("Modify the live runtime thresholds used by the engine.")
st.write("These values are saved in `out/config_runtime.json`.")

st.subheader("Expected Behavior")

p1, p2 = st.columns([1, 2])

with p1:
    st.metric("Engine Profile", profile_info["profile"])

with p2:
    st.info(profile_info["description"])

st.divider()

st.subheader("Current Risk Governor State")

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric("Can Emit Now", "YES" if risk_status["can_emit"] else "NO")

with r2:
    st.metric(
        "Signals Today",
        f"{risk_status['signals_today']}/{risk_status['max_signals_per_day']}",
    )

with r3:
    cooldown_remaining = risk_status.get("cooldown_remaining_minutes")
    st.metric(
        "Cooldown Remaining",
        "0m" if cooldown_remaining is None else f"{cooldown_remaining:.2f}m",
    )

with r4:
    st.metric("Block Reason", risk_status.get("block_reason") or "none")

st.caption(
    "Last signal time: "
    + str(risk_status.get("last_signal_time") or "none")
    + " | State day: "
    + str(risk_status.get("day"))
)

st.divider()

st.subheader("Quick Presets")

p1, p2, p3 = st.columns(3)

with p1:
    if st.button("Apply Conservative"):
        apply_preset("Conservative")
        st.success("Conservative preset applied.")
        st.rerun()

with p2:
    if st.button("Apply Balanced"):
        apply_preset("Balanced")
        st.success("Balanced preset applied.")
        st.rerun()

with p3:
    if st.button("Apply Aggressive"):
        apply_preset("Aggressive")
        st.success("Aggressive preset applied.")
        st.rerun()

st.divider()

with st.form("tune_engine_form"):
    st.subheader("Core Filters")

    strategy_codes = list(STRATEGY_PROFILES.keys())
    current_strategy = str(cfg.get("strategy_mode", strategy_codes[0]))
    if current_strategy not in strategy_codes:
        current_strategy = strategy_codes[0]

    strategy_mode = st.selectbox(
        "Strategy Mode",
        options=strategy_codes,
        index=strategy_codes.index(current_strategy),
        format_func=lambda code: f"{STRATEGY_PROFILES[code].label} ({code})",
    )

    st.caption(STRATEGY_PROFILES[strategy_mode].description)

    rr_min = st.number_input(
        "RR Minimum",
        min_value=0.5,
        max_value=5.0,
        value=float(cfg["rr_min"]),
        step=0.05,
    )

    min_score_for_signal = st.number_input(
        "Minimum Score For Signal",
        min_value=0,
        max_value=100,
        value=int(cfg["min_score_for_signal"]),
        step=1,
    )

    max_signals_per_day = st.number_input(
        "Max Signals Per Day",
        min_value=1,
        max_value=20,
        value=int(cfg["max_signals_per_day"]),
        step=1,
    )

    cooldown_minutes = st.number_input(
        "Cooldown Minutes",
        min_value=0,
        max_value=1440,
        value=int(cfg["cooldown_minutes"]),
        step=5,
    )

    st.divider()
    st.subheader("Order Book Thresholds")

    orderbook_neutral_threshold = st.number_input(
        "OrderBook Neutral Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg["orderbook_neutral_threshold"]),
        step=0.01,
    )

    orderbook_full_score_threshold = st.number_input(
        "OrderBook Full Score Threshold",
        min_value=0.01,
        max_value=1.0,
        value=float(cfg["orderbook_full_score_threshold"]),
        step=0.01,
    )

    st.divider()
    st.subheader("Squeeze Risk Thresholds")

    squeeze_risk_high_pct = st.number_input(
        "Squeeze Risk High %",
        min_value=0.0001,
        max_value=0.05,
        value=float(cfg["squeeze_risk_high_pct"]),
        step=0.0001,
        format="%.4f",
    )

    squeeze_risk_medium_pct = st.number_input(
        "Squeeze Risk Medium %",
        min_value=0.0001,
        max_value=0.05,
        value=float(cfg["squeeze_risk_medium_pct"]),
        step=0.0001,
        format="%.4f",
    )

    st.divider()
    st.subheader("Derivatives Thresholds")

    derivatives_extreme_oi_pct = st.number_input(
        "Derivatives Extreme OI %",
        min_value=0.001,
        max_value=0.20,
        value=float(cfg["derivatives_extreme_oi_pct"]),
        step=0.001,
        format="%.3f",
    )

    derivatives_mild_oi_pct = st.number_input(
        "Derivatives Mild OI %",
        min_value=0.001,
        max_value=0.20,
        value=float(cfg["derivatives_mild_oi_pct"]),
        step=0.001,
        format="%.3f",
    )

    st.divider()
    st.subheader("Quantum-Inspired Thresholds")

    quantum_coherence_threshold = st.number_input(
        "Quantum Coherence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=float(cfg["quantum_coherence_threshold"]),
        step=0.01,
        format="%.2f",
    )

    quantum_tunneling_threshold = st.number_input(
        "Quantum Tunneling Threshold",
        min_value=0.10,
        max_value=1.00,
        value=float(cfg["quantum_tunneling_threshold"]),
        step=0.01,
        format="%.2f",
    )

    st.divider()
    st.subheader("News Engine")

    news_enabled = st.checkbox(
        "Enable News Context",
        value=bool(cfg["news_enabled"]),
    )

    news_cache_minutes = st.number_input(
        "News Cache Minutes",
        min_value=1,
        max_value=240,
        value=int(cfg["news_cache_minutes"]),
        step=1,
    )

    news_headline_limit = st.number_input(
        "News Headline Limit",
        min_value=1,
        max_value=20,
        value=int(cfg["news_headline_limit"]),
        step=1,
    )

    st.divider()
    st.subheader("Alerts")

    alerts_enabled = st.checkbox(
        "Enable Alerts",
        value=bool(cfg.get("alerts_enabled", True)),
    )

    alert_min_score = st.number_input(
        "Live Signal Alert Min Score",
        min_value=0,
        max_value=100,
        value=int(cfg.get("alert_min_score", 74)),
        step=1,
    )

    blocked_alert_min_score = st.number_input(
        "Blocked Candidate Alert Min Score",
        min_value=0,
        max_value=100,
        value=int(cfg.get("blocked_alert_min_score", 68)),
        step=1,
    )

    news_alert_min_impact = st.number_input(
        "News Alert Min Impact",
        min_value=0.0,
        max_value=1.0,
        value=float(cfg.get("news_alert_min_impact", 0.70)),
        step=0.01,
        format="%.2f",
    )

    alert_cooldown_minutes = st.number_input(
        "Alert Cooldown Minutes",
        min_value=1,
        max_value=1440,
        value=int(cfg.get("alert_cooldown_minutes", 30)),
        step=1,
    )

    st.divider()
    st.subheader("Telegram Alerts")

    telegram_alerts_enabled = st.checkbox(
        "Enable Telegram Alerts",
        value=bool(cfg.get("telegram_alerts_enabled", False)),
    )

    telegram_alert_min_score = st.number_input(
        "Telegram Alert Min Score",
        min_value=0,
        max_value=100,
        value=int(cfg.get("telegram_alert_min_score", cfg.get("alert_min_score", 74))),
        step=1,
    )

    telegram_alert_min_calibrated_score = st.number_input(
        "Telegram Calibrated Min Score",
        min_value=0,
        max_value=100,
        value=int(cfg.get("telegram_alert_min_calibrated_score", cfg.get("alert_min_score", 74))),
        step=1,
    )

    telegram_alert_cooldown_minutes = st.number_input(
        "Telegram Cooldown Minutes",
        min_value=1,
        max_value=1440,
        value=int(cfg.get("telegram_alert_cooldown_minutes", cfg.get("alert_cooldown_minutes", 30))),
        step=1,
    )

    st.caption("Telegram bot token and chat ID are read from environment variables: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    st.divider()
    st.subheader("Market Read Validation")

    validation_horizon_bars = st.number_input(
        "Validation Horizon Bars (M15)",
        min_value=2,
        max_value=200,
        value=int(cfg.get("validation_horizon_bars", 16)),
        step=1,
    )

    validation_min_follow_through_pct = st.number_input(
        "Validation Min Follow Through %",
        min_value=0.0005,
        max_value=0.05,
        value=float(cfg.get("validation_min_follow_through_pct", 0.0035)),
        step=0.0005,
        format="%.4f",
    )

    validation_max_adverse_pct = st.number_input(
        "Validation Max Adverse %",
        min_value=0.0005,
        max_value=0.05,
        value=float(cfg.get("validation_max_adverse_pct", 0.0025)),
        step=0.0005,
        format="%.4f",
    )

    submitted = st.form_submit_button("Save Runtime Config")

    if submitted:
        save_runtime_config(
            {
                "strategy_mode": str(strategy_mode),
                "rr_min": float(rr_min),
                "min_score_for_signal": int(min_score_for_signal),
                "max_signals_per_day": int(max_signals_per_day),
                "cooldown_minutes": int(cooldown_minutes),
                "orderbook_neutral_threshold": float(orderbook_neutral_threshold),
                "orderbook_full_score_threshold": float(orderbook_full_score_threshold),
                "squeeze_risk_high_pct": float(squeeze_risk_high_pct),
                "squeeze_risk_medium_pct": float(squeeze_risk_medium_pct),
                "derivatives_extreme_oi_pct": float(derivatives_extreme_oi_pct),
                "derivatives_mild_oi_pct": float(derivatives_mild_oi_pct),
                "quantum_coherence_threshold": float(quantum_coherence_threshold),
                "quantum_tunneling_threshold": float(quantum_tunneling_threshold),
                "news_enabled": bool(news_enabled),
                "news_cache_minutes": int(news_cache_minutes),
                "news_headline_limit": int(news_headline_limit),
                "alerts_enabled": bool(alerts_enabled),
                "alert_min_score": int(alert_min_score),
                "blocked_alert_min_score": int(blocked_alert_min_score),
                "news_alert_min_impact": float(news_alert_min_impact),
                "alert_cooldown_minutes": int(alert_cooldown_minutes),
                "telegram_alerts_enabled": bool(telegram_alerts_enabled),
                "telegram_alert_min_score": int(telegram_alert_min_score),
                "telegram_alert_min_calibrated_score": int(telegram_alert_min_calibrated_score),
                "telegram_alert_cooldown_minutes": int(telegram_alert_cooldown_minutes),
                "validation_horizon_bars": int(validation_horizon_bars),
                "validation_min_follow_through_pct": float(validation_min_follow_through_pct),
                "validation_max_adverse_pct": float(validation_max_adverse_pct),
            }
        )
        st.success("Runtime config saved.")
        st.rerun()

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Runtime Config")
    st.json(load_runtime_config())

with col2:
    st.subheader("Available Presets")
    st.json(PRESET_CONFIGS)

st.divider()

if st.button("Reset To Defaults"):
    reset_runtime_config()
    st.success("Runtime config reset to defaults.")
    st.rerun()

st.caption("Tune Engine v3 - presets + thresholds")
