import streamlit as st

from runtime.runtime_config import (
    load_runtime_config,
    save_runtime_config,
    reset_runtime_config,
    DEFAULT_RUNTIME_CONFIG,
    PRESET_CONFIGS,
    apply_preset,
    describe_runtime_profile,
)


st.set_page_config(
    page_title="Trading Assistant - Tune Engine",
    layout="wide"
)

st.title("Trading Assistant - Tune Engine")

cfg = load_runtime_config()
profile_info = describe_runtime_profile(cfg)

st.write("Modify the live runtime thresholds used by the engine.")
st.write("These values are saved in `out/config_runtime.json`.")

st.subheader("Expected Behavior")

p1, p2 = st.columns([1, 2])

with p1:
    st.metric("Engine Profile", profile_info["profile"])

with p2:
    st.info(profile_info["description"])

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

    submitted = st.form_submit_button("Save Runtime Config")

    if submitted:
        save_runtime_config(
            {
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