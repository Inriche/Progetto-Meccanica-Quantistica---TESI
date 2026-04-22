# Calibrated Confirmation Filter: March/April OOS Validation

## 1. Objective
Evaluate whether the calibrated hybrid score is useful as a narrow confirmation filter, rather than as a general-purpose ranking score, when combined with the existing legacy ranking.

## 2. Experimental Setup
Two out-of-sample windows were tested with a fixed training cutoff:

- March 2026 OOS: `2026-03-01T00:00:00+00:00` to `2026-03-31T23:59:59+00:00`
- April 2026 OOS: `2026-04-01T00:00:00+00:00` to `2026-04-22T23:59:59+00:00`

The comparison used three variants:

- `legacy`
- `calibrated`
- `legacy + confirmation`

The confirmation rule was:

- `calibrated_hybrid_score >= 65`
- `setup = BLOCKED`
- `quantum_state = LOW_ENERGY`
- `decision = BUY`
- `context = trend_clean`

## 3. Compared Variants
The baseline `legacy` variant served as the general-purpose ranking reference. The `calibrated` variant tested whether the calibrated score improved threshold selectivity. The `legacy + confirmation` variant preserved legacy ranking while requiring the calibrated confirmation rule as an additional gate.

## 4. Main Findings
The calibrated confirmation subgroup was highly regime-sensitive.

- March:
  - `23` signals in the confirmation subgroup
  - validation rate: `13.0%`
  - invalidation rate: `43.5%`
  - average read score: `39.48`
  - context: entirely `trend_clean`
  - dominant modes: `h4` `13`, `h1` `5`, `m15` `5`

- April:
  - `10` signals in the confirmation subgroup
  - validation rate: `60.0%`
  - invalidation rate: `20.0%`
  - average read score: `64.68`
  - context: entirely `trend_clean`
  - dominant modes: `h4` `5`, `h1` `4`, `m15` `1`

At threshold `65`, the April confirmation-filtered subset outperformed the March subset materially, but it remained small:

- March `legacy + confirmation`:
  - `17` signals
  - win rate: `17.6%`
  - profit factor: `0.1635`

- April `legacy + confirmation`:
  - `4` signals
  - win rate: `100.0%`
  - profit factor: `inf`

The April result is encouraging, but the sample is too sparse for a stable production claim.

## 5. Why the Confirmation Filter Remains Analysis-Only
The filter is not robust across the two OOS windows.

- It did not reproduce in March.
- It produced only a handful of April trades.
- Its apparent edge is concentrated in a single context regime.
- The tail sample is too small for reliable operational thresholding.

For these reasons, it should remain analysis-only rather than being embedded in the main scoring pipeline.

## 6. Scientific Interpretation
The result is consistent with a regime-specific confirmation effect rather than a stable standalone ranking signal.

The calibrated score appears most useful as a conditional gate when the market is already in:

- `BLOCKED` setup conditions
- `LOW_ENERGY` quantum-inspired state
- `trend_clean` context

This suggests the calibrated score is acting more like a specialized confirmation statistic than a general ranking metric.

## 7. Next Validation Step
Test the same confirmation rule on the next contiguous OOS window after April, using the same frozen training cutoff:

- `2026-04-23T00:00:00+00:00` to `2026-05-20T23:59:59+00:00`

The key question is whether the April effect survives in a disjoint regime, not whether it looks favorable in a single window.
