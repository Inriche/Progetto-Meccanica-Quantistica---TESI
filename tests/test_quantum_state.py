from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.quantum_state import build_quantum_state


def _make_price_frame(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    prices = [float(start)]
    for r in returns:
        prices.append(prices[-1] * (1.0 + float(r)))
    return pd.DataFrame({"close": prices})


def _make_returns(length: int, drift: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return drift + rng.normal(0.0, noise, size=length)


def _assert_state_ranges(state) -> None:
    assert isinstance(state.state, str)
    assert 0.0 <= state.coherence <= 1.0
    assert -1.0 <= state.phase_bias <= 1.0
    assert -1.0 <= state.interference <= 1.0
    assert 0.0 <= state.tunneling_probability <= 1.0
    assert 0.0 <= state.amplitude <= 1.0
    assert 0.0 <= state.state_confidence <= 1.0
    assert 0.0 <= state.energy <= 1.0
    assert 0.0 <= state.decoherence_rate <= 1.0
    assert 0.0 <= state.transition_rate <= 1.0
    assert state.dominant_mode in {"m15", "h1", "h4"}

    for value in vars(state).values():
        if isinstance(value, (float, np.floating)):
            assert math.isfinite(float(value))


def test_quantum_state_warming_up_on_insufficient_data() -> None:
    df_short = _make_price_frame(np.array([0.001, 0.002, -0.001]))
    state = build_quantum_state(df_short, df_short, df_short)

    assert state.state == "WARMING_UP"
    _assert_state_ranges(state)
    assert state.energy == 0.0
    assert state.decoherence_rate == 0.0
    assert state.transition_rate == 0.0


def test_quantum_state_ranges_and_stable_structure() -> None:
    m15 = _make_price_frame(_make_returns(220, drift=0.0008, noise=0.0007, seed=1))
    h1 = _make_price_frame(_make_returns(160, drift=0.0006, noise=0.0005, seed=2))
    h4 = _make_price_frame(_make_returns(100, drift=0.0004, noise=0.0004, seed=3))

    state = build_quantum_state(m15, h1, h4, runtime_cfg={"quantum_phase_sensitivity": 1.0})

    _assert_state_ranges(state)
    assert state.state != "WARMING_UP"


def test_aligned_series_are_more_coherent_than_misaligned_series() -> None:
    aligned_m15 = _make_price_frame(_make_returns(240, drift=0.0010, noise=0.0003, seed=11))
    aligned_h1 = _make_price_frame(_make_returns(180, drift=0.0009, noise=0.0003, seed=12))
    aligned_h4 = _make_price_frame(_make_returns(120, drift=0.0008, noise=0.0003, seed=13))

    misaligned_m15 = _make_price_frame(_make_returns(240, drift=0.0010, noise=0.0003, seed=21))
    misaligned_h1 = _make_price_frame(_make_returns(180, drift=-0.0010, noise=0.0003, seed=22))
    misaligned_h4 = _make_price_frame(np.where(np.arange(120) % 2 == 0, 0.0012, -0.0012))

    aligned = build_quantum_state(aligned_m15, aligned_h1, aligned_h4)
    misaligned = build_quantum_state(misaligned_m15, misaligned_h1, misaligned_h4)

    _assert_state_ranges(aligned)
    _assert_state_ranges(misaligned)

    assert aligned.coherence >= misaligned.coherence
    assert aligned.decoherence_rate <= misaligned.decoherence_rate
    assert aligned.energy <= 1.0
    assert misaligned.energy <= 1.0


def test_energy_computation_does_not_explode() -> None:
    rng = np.random.default_rng(99)
    m15 = _make_price_frame(rng.normal(0.0003, 0.0020, size=260))
    h1 = _make_price_frame(rng.normal(0.0001, 0.0015, size=200))
    h4 = _make_price_frame(rng.normal(0.0002, 0.0010, size=140))

    state = build_quantum_state(m15, h1, h4, runtime_cfg={"quantum_coupling_strength": 0.75})

    _assert_state_ranges(state)
    assert state.energy <= 1.0
    assert state.state_confidence >= 0.0
