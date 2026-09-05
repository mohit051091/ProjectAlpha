# Mover Auto-Research Plan Snapshot
- Generated: 2026-07-05T02:55:10.220127Z
- Mode: EXECUTE
- Project root: `C:\Users\MOHIT\.gemini\antigravity\playground\ProjectAlpha`
- Sweep source: `C:\Users\MOHIT\.gemini\antigravity\playground\ProjectAlpha\results\performance_sweeps_label_30m_1pct.xlsx`

## Current best config (mover objective)
- Target label family: `label_30m_1pct` (`early_momentum_30m_1pct`)
- Label: `L25_S30`
- LONG_TH: `0.25`
- SHORT_TH: `0.3`
- Executed trades: `172`
- Trades/day: `8.19`
- Win rate: `51.74%`
- Mover hit-rate (MFE>=4%): `23.26%`
- Universe symbol-days: `9328`
- Actual movers / non-movers: `1114` / `8214`
- Predicted movers: `168`
- TP / FP / FN / TN: `146` / `22` / `968` / `8192`
- Mover precision: `86.90%`
- Mover recall: `13.11%`
- Mover F1: `22.78%`
- Classification accuracy: `89.39%`
- Non-mover specificity: `99.73%`
- Non-mover false-positive rate: `0.27%`
- Mean daily mover precision: `87.75%`
- Mean daily mover recall: `17.68%`
- Directional precision LONG/SHORT: `70.29%` / `64.71%`
- Cumulative return: `+103.52%`
- Score: `0.6140`

## Proposed next experiments
1. `L25_S30` (LONG=0.25, SHORT=0.30) — Test LONG=0.25, SHORT=0.30 for better mover capture while keeping trade frequency in target band.
2. `L27_S30` (LONG=0.27, SHORT=0.30) — Test LONG=0.27, SHORT=0.30 for better mover capture while keeping trade frequency in target band.
3. `L23_S30` (LONG=0.23, SHORT=0.30) — Test LONG=0.23, SHORT=0.30 for better mover capture while keeping trade frequency in target band.
4. `L25_S32` (LONG=0.25, SHORT=0.32) — Test LONG=0.25, SHORT=0.32 for better mover capture while keeping trade frequency in target band.
5. `L25_S28` (LONG=0.25, SHORT=0.28) — Test LONG=0.25, SHORT=0.28 for better mover capture while keeping trade frequency in target band.
6. `L27_S28` (LONG=0.27, SHORT=0.28) — Test LONG=0.27, SHORT=0.28 for better mover capture while keeping trade frequency in target band.

## Notes
- This scaffold does not modify baseline code automatically.
- Universe-level mover classification uses symbol-day highs/lows vs open: up-mover if `(high-open)/open >= mover_threshold_pct`, down-mover if `(low-open)/open <= -mover_threshold_pct`.
- Predicted movers are symbol-days where at least one direction reached `EXECUTED` in the sweep sheet.
- Threshold candidates are generated here; wiring these into sweep execution should be done in sandbox copy first.
