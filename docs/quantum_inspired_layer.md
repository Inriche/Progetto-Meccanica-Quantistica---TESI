# Quantum-Inspired Layer

This layer is **quantum-inspired**, not quantum computing.
It uses classical market data and interprets the market as a 3-mode effective system over:

- `m15`
- `h1`
- `h4`

## State interpretation

The market state is represented as:

`psi_t = [a_15 exp(i phi_15), a_1h exp(i phi_1h), a_4h exp(i phi_4h)]^T`

where:

- amplitudes `a_i` come from volatility, persistence, and directional intensity proxies
- phases `phi_i` come from standardized directional drift
- pairwise couplings encode cross-timeframe alignment

## Effective mechanics

An effective Hamiltonian `H_t` is built from:

- diagonal terms: instability / effective potential
- off-diagonal terms: coupling strengths between timeframes

From this we compute:

- effective energy `E = <psi|H|psi>`
- density matrix `rho = |psi><psi|`
- coherence from normalized off-diagonal magnitude of `rho`
- decoherence rate from disagreement and loss of alignment
- transition rate from instability and regime-shift likelihood
- dominant mode from the largest amplitude contribution

## Why this framing is useful

The goal is not physical quantum advantage.
The goal is a more structured statistical-physics view of market regime behavior:

- coherent states = aligned multi-timeframe structure
- decoherent states = conflicting horizons and unstable regime reading
- low energy states = more ordered configurations
- transition states = controlled breakouts or tunnel-like expansions

## How it enters scoring and validation

The upgraded quantum-inspired fields feed:

- heuristic scoring
- ML feature sets
- signal persistence and review logs
- database persistence for analysis and ablation

This keeps the implementation interpretable while making the mechanics layer more explicit and testable.
