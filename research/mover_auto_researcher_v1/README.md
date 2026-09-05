# Mover Auto-Researcher (Isolated Workspace)
This folder is an isolated scaffold for a Karpathy-style auto-research loop focused on mover detection.
It is intentionally separated from the production/live code paths.

## Safety Boundary
- This workspace does **not** edit `ProjectAlpha` source files by default.
- `auto_research_loop.py` runs in **dry-run** mode unless `--execute` is explicitly passed.
- To avoid touching current artifacts, run experiments on a cloned/sandbox root using `--safe-copy-root`.

## What it does
- Reads latest sweep workbook (`results/performance_sweeps.xlsx`).
- Can evaluate multiple curated target-label families (configured in `config.json`) such as:
  - `balanced_60m_1pct` (`label_60m_1pct`)
  - `early_momentum_30m_1pct` (`label_30m_1pct`)
  - `conviction_60m_3pct` (`label_60m_3pct`)
  - `trend_follow_120m_3pct` (`label_120m_3pct`)
- Intentionally excludes `EOD` label families from this auto-research loop.
- Computes mover-centric metrics:
  - trade-level mover hit-rate (`mfe_pct >= 4%` on executed trades)
  - universe-level mover precision/recall/specificity using symbol-day `open/high/low`
  - non-mover false-positive tracking and directional precision (LONG/SHORT)
- Selects the current best configuration.
- Proposes next threshold-pair experiments around the current best pair.
- Optionally executes your configured command pipeline and logs outputs into this workspace.

## Evaluation window policy
- Controlled via `objective.eval_start_date` and `objective.eval_end_date` in `config.json`.
- Current default is set to `2026-06-03` through `2026-07-03` for forward-eval readiness as newer sweep days are regenerated.
- When newer sweep days are regenerated, move this window forward deliberately (or remove it) for forward-eval studies.

## Quick start
```powershell
python research/mover_auto_researcher_v1/auto_research_loop.py --dry-run
```

## Execute on isolated copy (recommended)
1. Create a separate copy of ProjectAlpha (example): `C:\work\ProjectAlpha_sandbox`
2. Run:
```powershell
python research/mover_auto_researcher_v1/auto_research_loop.py `
  --execute `
  --safe-copy-root C:\work\ProjectAlpha_sandbox
```

## Outputs
- `proposals/auto_generated_candidates.json` — candidate experiments
- `reports/latest_research_plan.md` — current plan and candidate list
- `reports/latest_result_snapshot.json` — parsed best-known metrics
- `runs/<timestamp>/...` — execution logs and metadata when `--execute` is used
- `results/performance_sweeps_<target_label>.xlsx` — target-specific sweep workbook for each configured label family
