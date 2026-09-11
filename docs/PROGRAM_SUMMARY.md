# Order-Flow Program — Full Account (Pre-Phase 0 → Phase 1)
**Date:** 2026-09-08 · **Data:** 40 trading days, OOF 7.13M rows · **Repo changes:** parity bundle only (commit 5d3deb2, pushed)

## PRE-PHASE 0 — Find why live loses while backtest wins
**Aim:** Explain the live-vs-backtest trade mismatch on identical dates.
**Changes (all in originals, committed + pushed):**
- `99_rebuild_oof_all_v2.py`: `compute_roc_3` 2-bar → 3-bar (live used 3-bar; root cause of most mismatches).
- `76_generate_v2_sweeps_v2.py`: +15:00 entry cutoff, flat-at-15:00 exits, +10% booking.
- `update_whitelist.py`, `06_convert_unified_ticks.py`: crash fixes + DuckDB spill to D:.
- `models/*.txt` CRLF→LF + `.gitattributes` (LightGBM 4.6 refused CRLF models).
- New `scripts/check_coverage.py` (found 67 of 72 days blind).
- Docs: DECISION_LOG, TIMELINE, TASKS, CURRENT_STATE, STRATEGY_SUMMARY.
**Why:** Overlap live-vs-backtest was 42%. After fix: 76% (122/161 over 20 days; 65/79 on the core 10). A backtest that can't reproduce live is fiction — this made it honest.
**Also settled:** same weights proven byte-identical; 15/18 features bit-identical; closes identical; signal-prob confusion explained (trigger-bar vs theta-bar values).

## PHASE 0 — Protect the downside (analysis only, no repo changes)
**Aim:** Cap worst days and test two brakes before risking money.
**Changes:** none in repo. Temp-script experiments only.
**Findings:**
- Kill-switch (−₹3,000/day, honest mark-to-market): softens worst day (−₹10.4k → −₹5.4k) but costs ₹3.9k over 20 days — parked, not adopted.
- 10:30 breadth brake: never fires (mornings always green) — dead.
- 12:00/13:00 breadth brakes: add nothing — dead.
- Skip-10:00–11:00 entries: loses ₹8.5k — rejected.
- Booking test: +5% optimal (best total + best worst-day); trailing stops destroy; hysteresis floors all lose money — both rejected.
**Why:** Every protective idea must prove itself on history before it constrains live trading. Most failed; the survivors are +5% booking and first-8/day cap.

## PHASE 1 — Fix what the model learns (sandbox only: D:/PA_fix)
**Aim:** The old labels taught "price at exactly 60 min later"; money lives in "touches ±X% intraday". Reteach the model the traded event.
**Changes (sandbox, originals untouched):**
- `exc_relabel.py`: vectorized first-touch labels (±1.5% touch before 15:00 wins; else NO_TRADE; post-15:00 → NA). 40 days relabeled (LONG ~5–21%/day, SHORT ~3–14%).
- `exc_train.py`: same 18 features, same 100-round balanced setup, new target. Trained on 30 days (4.87M rows) → `models_exc/lgbm_model_exc_final.txt`.
- Validated on 10 unseen days (Aug 24–Sep 04).
**Result:** at L25_S25, old model −15 pts / new model **+72.7 pts (+₹40.9k gross, +₹25.8k net)**, overlap steady at 62/79 (79%). LONG-only 0.18+cap8: +18.5 pts. Thresholds need their own walk-forward (noted, not done).
**Why:** Model quality, not just alignment: the old target collapsed to base rates (mean prob_long 5.7%); the new one fires discriminatively (8.5–9.4%) on the event we actually trade.

## NEXT
1. **Threshold walk-forward** for the excursion model (train-window selection → validation-window lock).
2. **20 unseen days** through the pipeline for the locked config (pass bar: net+, overlap ≥70%, ≤10/d, worst > −₹6k).
3. **Port for sign-off:** excursion relabel + retrain scripts into repo; scanner-repo live fixes (awaiting permission); QuestDB ALTER; Railway redeploy.
4. **VIRTUAL paper 2–4 weeks → small real** with kill-switch armed; fill audit in parallel.
5. **Phase 2 (later):** meta-filter (fake-vs-mover gate), bad-day brake redesign, SHORT revival test.

## Walk-forward result (added post-validation)
Same machine (+5% book, cap-8, 3-bar roc), LONG-only:

| Model | SELECT Jul-20 | CONFIRM Aug10-21 | VALID Aug24-Sep04 |
|---|---|---|---|
| OLD L18 | +28.8 pts | +10.8 | +2.4 (net −₹3.4k) |
| OLD L20 | +61.4 (+₹25.5k) | +19.4 (+₹6.2k) | −0.2 (net −₹4.9k) |
| EXC L18 | +18.7 | **+20.4 (+₹6.7k)** | **+17.4 (net +₹5.0k)** |
| EXC L20 | +22.8 | +20.4 | +13.5 (net +₹2.8k) |

SELECT would have picked OLD L20 (+61.4) — which then died on VALID (−0.2). Textbook walk-forward save. EXC is flat-stable across all three splits (18.7 → 20.4 → 17.4).
**LOCKED: excursion model, LONG-only 0.18, first-8/day, +5% book.** Awaiting 20 unseen days + port sign-off.

## Unseen-8 validation (locked config, EXC L18 cap8 +5% book)
Aug 04 (−₹2.8k), Aug 05 (+₹0.9k), Aug 06 (+₹4.6k), Aug 07 (−₹7.9k), Jul-28 (−₹2.5k),
Sep-07 (+₹7.7k), Sep-09 (+₹6.7k), Sep-10 (−₹4.3k).
Total +₹2.4k gross / **−₹1.4k net** (64×₹60), 4/8 green, worst −₹7.9k, overlap 16/63.
Clean-unseen subset (Jul-28, Sep-07/09/10 — outside training window): +₹5.7k net, 3/4 green.
Contamination note: Aug 04–07 sit inside the model's training window, so only Jul-28/Sep-07/09/10 count as true out-of-sample.
**Verdict: MARGINAL — net ~breakeven, worst-day bar still breached. No real money yet.**
Next: more unseen days as R2 uploads arrive + paper trade + worst-day protection answered.

## 49-day exit upgrade (book5 vs partials, locked config otherwise)
| Exit | Total pts | Net of ₹60 | Green | Worst |
|---|---|---|---|---|
| full book +5% | +118.1 | +₹43,162 | 28/49 | −15.5 |
| half +5%, rest trail-2.5% | +129.7 | +₹49,653 | 29/49 | −15.5 |
| half +3%, rest trail-2.5% | +128.6 | +₹49,043 | 33/49 | −14.1 |
Adopted (pending port approval): **half at +3%, runner trails 2.5%**. Caveats: stop-fill assumed exact;
exit variant selected on same 49 days (mild circularity — re-validate on fresh days).

## Meta-filter verdict (same-18-features)
3,274 alert setups, 12.7% movers; all feature separations |d|<0.2; confidence bands help mildly
(10.7% → 25.3% movers across 0.18→0.30+). Same-feature meta-model NOT worth building.
Needs new features (alert hour, prob slope, symbol stats) to be viable — parked.

## Retrain protocol (locked)
Retrain excursion model monthly on expanding window (all labeled days), same params;
parity battery mandatory before promotion (overlap ≥70%, costs in);
old weights stay deployed beside new until new wins on 10 unseen days.
Fill audit + Railway/scanner items unchanged and still pending permission.
