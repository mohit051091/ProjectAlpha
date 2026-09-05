"""
Alert-Trigger Backtest Engine — 60_alert_trigger_backtest.py
===========================================================
Approved methodology incorporating the 9-period ROC confirmation filter.

Design:
  1. Train models out-of-fold using the baseline 5-fold cross-validation setup.
  2. Compute 9-period ROC on 1-minute Close price (LTP).
  3. Simulate 3 strategies side-by-side:
     - ML Alone: Enter immediately at the model alert.
     - ROC Trigger Alone: Enter whenever the 9-period ROC crosses the threshold.
     - ML + ROC (Alert-Trigger): Enter when model alert is confirmed by ROC trigger.
  4. Sweep hold horizons and expiration windows.
  5. Compute and report comprehensive performance metrics.
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────

TARGET_LABEL = "label_60m_1pct"

FEATURES = [
    "large_trade_ratio",
    "delta_1m",
    "volume_burst",
    "aggressor_ratio",
    "trade_count_burst",
    "imbalance_top5",
    "spread",
    "depth_drop_bid",
    "depth_drop_ask",
    "vwap_distance",
    "volatility_5m",
    "price_acceleration",
    "iceberg_score",
    "bid_replenishment_rate",
    "absorption_buyer_1m",
    "absorption_buyer_5m",
    "absorption_seller_1m",
    "absorption_seller_5m",
]

N_FOLDS = 5
RANDOM_STATE = 42

# Sweep space
HORIZONS = [30, 60, 120, 180]
EXP_WINDOWS = [60, 120, 180, 240, 300, 360]
ROC_THRESH = 0.01  # 1%

PROC = Path("02_processed")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

results_path = RESULTS / "backtest_results.json"
fold_path = RESULTS / "baseline_fold_assignments.json"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ─── Data Discovery & Loading ───────────────────────────────────────────────

def get_both_slugs(proc_dir: Path):
    dom_slugs = {f.stem.replace("cleaned_dom_", "") for f in proc_dir.glob("cleaned_dom_*.parquet")}
    tick_slugs = {f.stem.replace("cleaned_ticks_", "") for f in proc_dir.glob("cleaned_ticks_*.parquet")}
    inf_slugs = {
        f.stem.replace("inferred_trades_", "")
        for f in proc_dir.glob("inferred_trades_*.parquet")
        if f.stem.replace("inferred_trades_", "").count("_") >= 2
    }
    both = sorted(dom_slugs & tick_slugs & inf_slugs)
    return both

def load_master_dataframe(slugs: list, proc_dir: Path, columns: list, label: str):
    """Load and combine features and metadata into a single DataFrame."""
    all_dfs = []
    load_cols = columns + [label, "ltp", "ts"]

    for slug in slugs:
        fpath = proc_dir / f"labeled_features_1m_{slug}.parquet"
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath, columns=load_cols)
            df = df.dropna(subset=[label] + columns + ["ltp"])
            if len(df) == 0:
                continue
            df["slug"] = slug
            df["symbol"] = slug.split("_")[0].upper()
            all_dfs.append(df)
        except Exception as e:
            log(f"Error loading {slug}: {e}")

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# ─── Out-of-Fold Prediction Engine ──────────────────────────────────────────

def generate_oof_predictions(df_all: pd.DataFrame, fold_assignments: dict):
    """Train baseline models and store out-of-fold predictions in df_all."""
    # Map key (which has date suffix like _29_May_26) to base symbol
    symbol_to_fold = {}
    for key, fold in fold_assignments.items():
        sym = key.split("_")[0].upper()
        symbol_to_fold[sym] = fold

    # Assign fold to each row based on base symbol
    def get_fold(slug):
        sym = slug.split("_")[0].upper()
        if sym in symbol_to_fold:
            return symbol_to_fold[sym]
        else:
            # Fallback for new symbols: deterministic MD5 hash mapping to 0..N_FOLDS-1
            import hashlib
            h = int(hashlib.md5(sym.encode("utf-8")).hexdigest(), 16)
            return h % N_FOLDS

    df_all["fold"] = df_all["slug"].apply(get_fold)
    df_all["fold"] = df_all["fold"].astype(int)

    # Initialize prediction columns
    df_all["pred_lr"] = "NO_TRADE"
    df_all["pred_rf"] = "NO_TRADE"

    X = df_all[FEATURES].values
    y = df_all[TARGET_LABEL].values

    for fold in range(N_FOLDS):
        log(f"Training models for fold {fold+1}/{N_FOLDS}...")
        
        train_mask = df_all["fold"] != fold
        val_mask = df_all["fold"] == fold

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_val) == 0:
            continue

        # Subsample training data systematically (every 5th row) for 5x training speedup
        X_train_sub = X_train[::5]
        y_train_sub = y_train[::5]

        # ── Model A: Logistic Regression (L2, fast L-BFGS) ──
        scaler = RobustScaler()
        X_train_sub_lr = scaler.fit_transform(X_train_sub)
        X_val_lr = scaler.transform(X_val)

        lr = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            max_iter=100,
            random_state=RANDOM_STATE,
        )
        lr.fit(X_train_sub_lr, y_train_sub)
        pred_lr = lr.predict(X_val_lr)
        df_all.loc[val_mask, "pred_lr"] = pred_lr
        mcc_lr = matthews_corrcoef(y_val, pred_lr)

        # ── Model B: Random Forest (100 trees, single-threaded to prevent Windows deadlocks) ──
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        rf.fit(X_train_sub, y_train_sub)
        pred_rf = rf.predict(X_val)
        df_all.loc[val_mask, "pred_rf"] = pred_rf
        mcc_rf = matthews_corrcoef(y_val, pred_rf)

        log(f"  Fold {fold+1} complete: LR MCC={mcc_lr:.4f}, RF MCC={mcc_rf:.4f}")

    # Verify overall OOF MCCs
    mcc_lr_total = matthews_corrcoef(df_all[TARGET_LABEL], df_all["pred_lr"])
    mcc_rf_total = matthews_corrcoef(df_all[TARGET_LABEL], df_all["pred_rf"])
    log(f"Overall OOF MCC: LR={mcc_lr_total:.4f}, RF={mcc_rf_total:.4f}")

    return df_all

# ─── Simulation Engine ───────────────────────────────────────────────────────

def simulate_symbol_day(df_symbol: pd.DataFrame, model_col: str, horizon: int, exp_window: int, roc_thresh: float):
    """
    Simulates trades for a single symbol-day.
    df_symbol is sorted chronologically.
    """
    ltp = df_symbol["ltp"].values
    ts = df_symbol["ts"].values
    preds = df_symbol[model_col].values
    
    # 9-period ROC on LTP
    roc = df_symbol["ltp"].pct_change(9).values
    n = len(df_symbol)
    
    trades_ml = []
    trades_roc = []
    trades_ml_roc = []
    
    # --- 1. ML Alone ---
    holding_until = -1
    for t in range(n):
        if t <= holding_until:
            continue
        pred = preds[t]
        if pred in ["LONG", "SHORT"]:
            entry_price = ltp[t]
            exit_idx = min(t + horizon, n - 1)
            exit_price = ltp[exit_idx]
            
            pnl = (exit_price - entry_price) / entry_price if pred == "LONG" else (entry_price - exit_price) / entry_price
            trades_ml.append({
                "entry_time": ts[t],
                "exit_time": ts[exit_idx],
                "direction": pred,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl
            })
            holding_until = exit_idx

    # --- 2. ROC Alone ---
    holding_until = -1
    for t in range(9, n):
        if t <= holding_until:
            continue
        
        long_trigger = roc[t] >= roc_thresh
        short_trigger = roc[t] <= -roc_thresh
        
        direction = None
        if long_trigger:
            direction = "LONG"
        elif short_trigger:
            direction = "SHORT"
            
        if direction:
            entry_price = ltp[t]
            exit_idx = min(t + horizon, n - 1)
            exit_price = ltp[exit_idx]
            pnl = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price
            
            trades_roc.append({
                "entry_time": ts[t],
                "exit_time": ts[exit_idx],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl
            })
            holding_until = exit_idx

    # --- 3. ML + ROC (Alert-Trigger) ---
    holding_until = -1
    alert_long_active = False
    alert_long_time = -1
    alert_short_active = False
    alert_short_time = -1
    
    for t in range(9, n):
        # Update alert state (expiration)
        if alert_long_active and (t - alert_long_time > exp_window):
            alert_long_active = False
        if alert_short_active and (t - alert_short_time > exp_window):
            alert_short_active = False
            
        # Check for new ML alerts (only if not holding a position)
        if t > holding_until:
            pred = preds[t]
            if pred == "LONG":
                alert_long_active = True
                alert_long_time = t
                alert_short_active = False
            elif pred == "SHORT":
                alert_short_active = True
                alert_short_time = t
                alert_long_active = False
                
        # Check for trigger confirmation (only if not holding a position)
        if t > holding_until:
            direction = None
            if alert_long_active:
                if roc[t] >= roc_thresh:
                    direction = "LONG"
                    alert_long_active = False
            elif alert_short_active:
                if roc[t] <= -roc_thresh:
                    direction = "SHORT"
                    alert_short_active = False
                    
            if direction:
                entry_price = ltp[t]
                exit_idx = min(t + horizon, n - 1)
                exit_price = ltp[exit_idx]
                pnl = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price
                
                trades_ml_roc.append({
                    "entry_time": ts[t],
                    "exit_time": ts[exit_idx],
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl
                })
                holding_until = exit_idx
                
    return trades_ml, trades_roc, trades_ml_roc

# ─── Metrics Aggregation ─────────────────────────────────────────────────────

def compute_metrics(trades):
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "mean_pnl": 0.0,
            "median_pnl": 0.0,
            "profit_factor": 0.0,
            "cumulative_pnl": 0.0
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else (sum_wins if sum_wins > 0 else 1.0)
    
    return {
        "num_trades": len(pnls),
        "win_rate": round(win_rate, 4),
        "mean_pnl": round(float(np.mean(pnls)), 6),
        "median_pnl": round(float(np.median(pnls)), 6),
        "profit_factor": round(float(profit_factor), 4),
        "cumulative_pnl": round(float(sum(pnls)), 6)
    }

# ─── Main Orchestrator ───────────────────────────────────────────────────────

def main():
    t_start = time.time()
    log("Starting Alert-Trigger Backtest Engine...")

    # Load slugs
    slugs = get_both_slugs(PROC)
    log(f"Found {len(slugs)} BOTH slugs in {PROC}")

    if not slugs:
        log("ERROR: No clean parquet files found in 02_processed. Run pipeline first.")
        return

    # Load Master DataFrame
    log("Loading Master DataFrame...")
    df_all = load_master_dataframe(slugs, PROC, FEATURES, TARGET_LABEL)
    if df_all.empty:
        log("ERROR: Master DataFrame is empty.")
        return
    log(f"Master DataFrame loaded: {len(df_all)} rows across {df_all['slug'].nunique()} stocks.")

    # Load Fold Assignments
    if not fold_path.exists():
        log(f"ERROR: Fold assignments file not found at {fold_path}. Run 50_baseline_model.py first.")
        return
    with open(fold_path, "r") as f:
        fold_assignments = json.load(f)

    # Generate Out-of-Fold Predictions
    df_all = generate_oof_predictions(df_all, fold_assignments)

    # Sort each symbol-day chronologically before simulation
    log("Sorting symbol-days...")
    df_all["ts"] = pd.to_datetime(df_all["ts"])
    df_all = df_all.sort_values(by=["slug", "ts"]).reset_index(drop=True)

    # Group by slug for time-series backtests
    symbol_groups = [group for _, group in df_all.groupby("slug")]
    log(f"Grouped into {len(symbol_groups)} symbol-day series.")

    # Sweep loops
    sweep_results = []

    log("Beginning parameter sweeps...")
    for model_name, model_col in [("Logistic Regression", "pred_lr"), ("Random Forest", "pred_rf")]:
        for horizon in HORIZONS:
            # Baseline: ML Alone & ROC Alone (no exp_window needed)
            # We run these once per horizon
            trades_ml_all = []
            trades_roc_all = []
            
            for df_symbol in symbol_groups:
                trades_ml, trades_roc, _ = simulate_symbol_day(df_symbol, model_col, horizon, exp_window=60, roc_thresh=ROC_THRESH)
                trades_ml_all.extend(trades_ml)
                trades_roc_all.extend(trades_roc)
                
            metrics_ml = compute_metrics(trades_ml_all)
            metrics_roc = compute_metrics(trades_roc_all)
            
            sweep_results.append({
                "model": model_name,
                "strategy": "ML Alone",
                "horizon": horizon,
                "exp_window": None,
                "metrics": metrics_ml
            })
            sweep_results.append({
                "model": model_name,
                "strategy": "ROC Alone",
                "horizon": horizon,
                "exp_window": None,
                "metrics": metrics_roc
            })
            
            # Sweeping exp_window for ML + ROC
            for exp_window in EXP_WINDOWS:
                trades_ml_roc_all = []
                for df_symbol in symbol_groups:
                    _, _, trades_ml_roc = simulate_symbol_day(df_symbol, model_col, horizon, exp_window, ROC_THRESH)
                    trades_ml_roc_all.extend(trades_ml_roc)
                    
                metrics_ml_roc = compute_metrics(trades_ml_roc_all)
                sweep_results.append({
                    "model": model_name,
                    "strategy": "ML + ROC",
                    "horizon": horizon,
                    "exp_window": exp_window,
                    "metrics": metrics_ml_roc
                })
                log(f"Sweep: {model_name} | Horizon {horizon}m | ExpWindow {exp_window}m | Trades={metrics_ml_roc['num_trades']} | WinRate={metrics_ml_roc['win_rate']:.4%}")

    # Save results to JSON
    with open(results_path, "w") as f:
        json.dump({
            "target_label": TARGET_LABEL,
            "roc_threshold": ROC_THRESH,
            "runtime_s": round(time.time() - t_start, 2),
            "sweep_results": sweep_results
        }, f, indent=2)
    log(f"Backtest results saved to {results_path}")

    # Display comparison table for Default Horizon = 60m, ExpWindow = 60m
    print("\n" + "=" * 90)
    print("BACKTEST COMPARISON: HORIZON = 60m, EXPIRATION WINDOW = 60m")
    print("=" * 90)
    print(f"{'Model':<22} | {'Strategy':<15} | {'Trades':<8} | {'Win Rate':<10} | {'Mean Return':<12} | {'Profit Factor':<14} | {'Cum Return':<12}")
    print("-" * 105)
    
    for r in sweep_results:
        if r["horizon"] == 60 and (r["exp_window"] is None or r["exp_window"] == 60):
            m = r["metrics"]
            print(f"{r['model']:<22} | {r['strategy']:<15} | {m['num_trades']:<8} | {m['win_rate']:<10.2%} | {m['mean_pnl']:<12.4%} | {m['profit_factor']:<14.4f} | {m['cumulative_pnl']:<12.4%}")
            
    print("=" * 90)

    # Find best configuration by Mean Return
    valid_ml_roc = [r for r in sweep_results if r["strategy"] == "ML + ROC" and r["metrics"]["num_trades"] >= 20]
    if valid_ml_roc:
        best_r = max(valid_ml_roc, key=lambda x: x["metrics"]["mean_pnl"])
        print("\n" + "=" * 90)
        print("BEST ML + ROC CONFIGURATION FOUND (Min 20 trades)")
        print("=" * 90)
        print(f"Model          : {best_r['model']}")
        print(f"Horizon        : {best_r['horizon']} minutes")
        print(f"Exp Window     : {best_r['exp_window']} minutes")
        print(f"Trades         : {best_r['metrics']['num_trades']}")
        print(f"Win Rate       : {best_r['metrics']['win_rate']:.2%}")
        print(f"Mean Return    : {best_r['metrics']['mean_pnl']:.4%}")
        print(f"Profit Factor  : {best_r['metrics']['profit_factor']:.4f}")
        print(f"Cumulative P&L : {best_r['metrics']['cumulative_pnl']:.4%}")
        print("=" * 90)
        
    log(f"Total run time: {time.time() - t_start:.1f} seconds.")

if __name__ == "__main__":
    main()
