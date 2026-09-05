# Model Channels

This folder now separates model artifacts into explicit channels so you can distinguish research outputs from production references.

## Folder layout
- `models/auto_research/releases/2026-07-05_jul03_refresh/`
  - Versioned snapshot of the Jul-05 auto-research refresh (data coverage through Jul-03).
- `models/live_production/current/`
  - Last-known live production reference snapshot used for rollout coordination.

## Current best from auto research
- Target label: `label_30m_1pct`
- Pair: `L25_S30`
- Recommended model file: `lgbm_model_30m_1pct_final.txt`
- Source report: `research/mover_auto_researcher_v1/reports/live_recommended_settings.json`

## Last-known live production reference
- Model family: `label_60m_1pct`
- Model file: `lgbm_model_60m_1pct_final.txt`
- Thresholds: LONG `0.25`, SHORT `0.25` (from last Railway snapshot context on Jul-04)

## Handoff guidance for another LLM
- Give access to **ProjectAlpha root** (not only `auto_research` subfolders), because:
  - Production model paths, scripts, docs, sweeps, and release manifests are spread across `models/`, `results/`, `research/`, and `docs/`.
  - Restricting to only an auto-research subfolder hides deployment-critical context.
