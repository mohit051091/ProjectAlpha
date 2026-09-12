# FULL SESSION HANDOFF — Order-Flow Project (for continuing AI)
**Written:** 2026-09-12 (Sat) · **Status:** everything below is fact, verified by execution.

## 0. People, machines, repos
- User: Rahul (iamrk583@gmail.com; GitHub iamrk07; Railway trio93342@gmail.com).
- This machine (Windows): project `C:\ProjectAlpha` (research/backtest, git origin mohit051091/ProjectAlpha, branch master, pushed through `37d5d97`).
- Live repo clone/workdir: `D:\dhan-scanner` (origin mohit051091/dhan-orderflow-scanner, branch main, pushed through `ebee2ac`).
- Data: `D:\FriendData` (72 unified tick parquets + `raw/` per-symbol; moved from C: which filled to 0 bytes).
- Isolated run workspace: `D:\PA_Sep04` (raw subset per date). Sandbox: `D:\PA_fix` (copies, exc models, fixed OOFs). Live R2 pulls: `D:\LiveData` (ml_signals_*, ml_predictions_*). Temp scripts (mostly wiped, recreatable): `C:\Users\Rahul\AppData\Local\Temp\opencode\`.
- R2 bucket `orderflow-historical-archive` via rclone remote `r2:` (configured on machine). Tables: ticks, ml_predictions, ml_signals, ml_state, dom_snapshots, historical_closes, delta_bars.
- Railway: project adventurous-communication (2320ac24…), env production (ace6c36e…). Services: dhan-collector-rahul (24164ecb…), dhan-collector-mohit (382536f2…), questdb (945d442c…), 2 crons. Railway CLI installed (npm). No gh CLI.
- QuestDB reachable ONLY inside Railway network (private domain questdbquestdb.railway.internal; public TCP proxy dead; HTTP console 404). QuestDB deliberately STOPPED for weekend; rahul collector STOPPED too. Mohit has a deployment stuck NEEDS_APPROVAL.

## 1. Pipeline built (49 dates, OOF 8.71M rows)
Per day: 06_convert → isolate+scrub (whitelist!) → 10_clean → 20_trades → 30_features → 40_labels → 99_rebuild → 76_sweep. Dates: Jul 06–31 (19, ex 28-broken-then-fixed? Jul-28 thin 348 syms included), Aug 03–07, 10–14, 17–21, 24–28, 31, Sep 01–04, 07, 09, 10, 11. Labeled counts ~500/day (Sep-01/02 467-468, Jul-10 154 thin-DOM, Jul-16 433, Jul-28 348).
- Gotchas solved: C: disk-full (moved to D:), DuckDB OOM (temp→D:/tmp_duckdb, 6GB cap), Stage-05 unfiltered extras (scrub non-whitelist per date), Aug-07 missing dom files (force-reconverted), Stage1/2 --force runs everything (never use globally), 99 OOM at 7M rows (chunked per-date in repo code now), LightGBM 4.6 cannot load CRLF models (git autocrlf damage; fixed + .gitattributes in both repos).

## 2. Gap analysis: backtest won, live lost — 10 gaps found
Same weights proven byte-identical; 15/18 features bit-identical; closes identical; machine logic identical. Divergences: (1) roc_3 2-bar backtest vs 3-bar live — ROOT CAUSE, fixed; (2) daily-open source differs; (3) large-trade median window 2k vs 10k; (4) NaN vs 0.0 first bars; (5) sweep lacked 15:00 cutoff/exits; (6) ml_signals stored trigger-bar probs (look "wrong" vs 0.25 — by design) + missing theta-prob/entry-time cols; (7) model CRLF; (8) entry avg-vs-close ~0.03%; (9) blind symbols (HEG/HFCL etc.); (10) fills/costs/exits unrecorded. Overlap L25_S25: 42% → 76% (122/161 over 20d), 89% symbol coverage for 0.18 candidate.

## 3. Money (identical exits everywhere; costs now ₹60/trade at ₹56k tickets)
- Live actuals 0.25/0.25, 20d: 158 trades, −55.01%. Fixed L25_S25 10d: −15.4%.
- Old-board best (2-bar): L15_S15 +69.6–220% (over cap). In-cap (≤10/d) everything red after costs.
- Funnel (363 mover-days ≥5% range): 56% never alert <0.25 / 20% roc breaks / 4% vetoed / 20% entered. At 0.20 alerts catch 75%, at 0.15 94%.
- Movers: 133 events (36 unique), 125 LONG / 8 SHORT. Fingerprint real in means, hard gate keeps 22% → rejected as rule.
- Config lab verdicts: hysteresis (all floors lose — dead), trailing stops (destroy — dead), breadth brakes (never fire — dead), skip-10xx (loses ₹8.5k — dead), kill-switch −₹3k (mixed, parked), booking +5% adopted; partial half@+3%+trail wins on 49d but locked def stays full-book +5% (simpler, validated).
- Costs: ₹208/round trip at ₹4.5L tickets; ₹60 at ₹56k tickets (user capital: ₹90-100k, ₹11.25k margin ×8 slots ×5x = ₹56k positions). STT dominates. Concurrency solved by the 8-way split.

## 4. Phase 1 retrain (excursion labels) — DONE, sandbox only
Spec: LONG = first touch +1.5% before 15:00, SHORT −1.5% first, else NO_TRADE, post-15:00 → NA. 40d relabeled (LONG 5–21%/d). Retrained same-setup LightGBM on 30d (4.87M rows) → `D:\PA_fix\models_exc\lgbm_model_exc_final.txt`. Walk-forward (select Jul-20 / confirm Aug10-21 / judge Aug24-Sep04): OLD star (+61) died on unseen (−0.2); EXC flat-stable (+19/+20/+17). LOCKED: EXC, LONG-only 0.18, first-8/day, +5% book. Unseen-8: +₹2.4k gross/−₹1.4k net, 4/8 green (marginal — no real money yet).
- Killed idea log: retrain-on-old-labels, horizon fiddling (30m test dead-ended as predicted).

## 5. Deploy state (do NOT redo)
- ProjectAlpha pushed (`37d5d97`): roc fix, sweep clock/exits, whitelist+06 fixes, model LF, coverage script, 99 chunking, docs (DECISION_LOG/TIMELINE/TASKS/CURRENT_STATE/PROGRAM_SUMMARY/STRATEGY_SUMMARY).
- Scanner repo pushed (`ebee2ac` on main): 4 live fixes, exc model, STRATEGY_ENV.locked.md, gitattributes, self-migrating signal columns.
- Railway rahul collector: redeployed SUCCESS with VIRTUAL + exc model + 0.18/999 + TP5 + cap8 env (then STOPPED for weekend). Mohit untouched (stuck deploy needs human approve).
- QuestDB: ALTER (entry_time_ist, theta_prob) NOT yet run — code self-migrates on boot instead. Must verify in logs post-restart.
- Live env deltas vs backtest assumptions to remember: FIXED_QTY=1 (tiny sizing!), SL_PCT=3 (no backtest stop), pmocata caps enforced server-side, SHORT_TH=999 (LONG-only), first-8 = pmocata maxStocks (ordering assumption).

## 6. Agreed next steps (in order)
1. Tuesday pre-market: QuestDB up → approve mohit → set same 6 env on BOTH collectors BUT flip to previous strategy? NO — latest decision: trade NEW strategy live, 1 trade/day (ML_MAX_STOCKS_PER_DAY=1), FIXED_QTY stays 1 (pocket-change verification week). Then restart both, verify logs (model load, thresholds, VIRTUAL? NO — user chose LIVE money, 1 stock).
2. Daily watch: overlap, fills vs signals, booking/SL behavior, diary completeness.
3. Kill-switch −₹3k armed manually (no code). Halt rules: −10% month, overlap <70% a week.
4. Then: unseen-day accumulation, fill audit, threshold walk-forward for exc model (0.18 came from old-model analysis!), meta-filter Phase 2, SHORT revival, bad-day brake redesign, retrain protocol (monthly expanding window + parity gate).
5. Open questions log: SHORT keep/kill; cap 8 vs 10; booking +5% vs +3%; brake designs; regime coverage (all data one season); scaling/impact beyond ₹56k; tax (CA).

## 7. Hard rules from user (never break)
- No original-code changes without explicit permission (sandbox D:\PA_fix\ + temp scripts only until approved).
- Scanner repo: permission granted late; pushes allowed there now (verify each time).
- Universes "don't matter" per user for comparisons, but whitelist both sides going forward.
- Costs (₹60/ticket) inside every number. Slippage assumed zero — flagged everywhere.
- Sample discipline: selection on train, lock, judge on unseen. Never tune on validation.
- Token budget: user on free tier (~74% used at one point) — keep answers tight.
