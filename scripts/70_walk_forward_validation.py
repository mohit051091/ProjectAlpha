"""
Walk-Forward Validation & Model Upgrade Experiment — 70_walk_forward_validation.py
==================================================================================
Implements:
  1. Expanding window walk-forward validation (purging/embargoing temporal split).
  2. Baseline LR, baseline Random Forest, and LightGBM upgrade.
  3. Daily Trend Filters (Open Price and Daily VWAP).
  4. Regime Analysis and SHAP Feature Importance.
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import matthews_corrcoef, classification_report
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────

TARGET_LABEL = "label_60m_1pct"
HORIZON_MIN = 60

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

RANDOM_STATE = 42
ROC_THRESH = 1.0  # 1%
HOLD_HORIZON = 180
EXP_WINDOW = 60

# ─── Paths ──────────────────────────────────────────────────────────────────

PROC = Path("02_processed")
RESULTS = Path("results")
LOGS = Path("logs")
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

log_path = LOGS / "walk_forward.out.txt"
results_path = RESULTS / "walk_forward_results.json"
shap_path = RESULTS / "walk_forward_shap.csv"


# ─── Logging ────────────────────────────────────────────────────────────────

def log(msg, file=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if file:
        print(line, file=file, flush=True)


# ─── Date Extraction & Parsing ──────────────────────────────────────────────

def get_chronological_days(proc_dir: Path):
    """Scan labeled files, extract distinct dates, and sort them chronologically."""
    labeled_files = list(proc_dir.glob("labeled_features_1m_*.parquet"))
    date_strs = set()
    for f in labeled_files:
        parts = f.stem.split("_")
        if len(parts) >= 7:
            date_strs.add("_".join(parts[-3:]))
            
    # Parse to datetime objects to sort correctly
    parsed = []
    for d_str in date_strs:
        try:
            dt = datetime.strptime(d_str, "%d_%b_%y")
            parsed.append((dt, d_str))
        except Exception:
            pass
            
    parsed.sort()
    return [d_str for _, d_str in parsed]


# ─── DuckDB Connection (lazy parquet scan) ──────────────────────────────────

def create_duckdb_connection(proc_dir: Path):
    """Create DuckDB connection with a view over all labeled parquet files."""
    con = duckdb.connect()
    parquet_glob = str(proc_dir / "labeled_features_1m_*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW all_labeled AS
        SELECT *,
               regexp_extract(filename, '_(\d{{1,2}}_[A-Z][a-z]{{2}}_\d{{2}})\.parquet$', 1) AS date_str
        FROM read_parquet('{parquet_glob}', filename=true, union_by_name=true)
    """)
    return con


# ─── Purge and Embargo Split Engine ─────────────────────────────────────────

def get_train_val_split_purged(
    df_all: pd.DataFrame, 
    train_dates: list, 
    val_date: str, 
    horizon_min: int = 60
):
    """
    Split df_all into train and validation sets, applying purging.
    Purges any training sample whose forward return window overlaps with 
    the validation set start.
    """
    df_train_raw = df_all[df_all["date"].isin(train_dates)].copy()
    df_val = df_all[df_all["date"] == val_date].copy()
    
    if df_train_raw.empty or df_val.empty:
        return df_train_raw, df_val
        
    val_start = df_val["ts"].min()
    
    # Purging condition: training sample timestamp + horizon_min must be < validation start
    cutoff_time = val_start - timedelta(minutes=horizon_min)
    df_train_purged = df_train_raw[df_train_raw["ts"] < cutoff_time]
    
    return df_train_purged, df_val


# ─── Backtest Simulation Engine ─────────────────────────────────────────────

def simulate_trades(df_symbol: pd.DataFrame, pred_col: str, model_name: str, filter_type: str = "none"):
    """
    Simulates trades on a single symbol-day dataframe.
    filter_type can be:
      - 'none': no trend filter
      - 'open': filter by daily open price (allow long if ltp > open, short if ltp < open)
      - 'vwap': filter by running VWAP (allow long if ltp > running_vwap, short if ltp < running_vwap)
    """
    df_sorted = df_symbol.sort_values("ts").reset_index(drop=True)
    ltp = df_sorted["ltp"].values
    preds = df_sorted[pred_col].values
    roc = df_sorted["roc_9"].values if "roc_9" in df_sorted.columns else (df_sorted["ltp"].pct_change(9).values * 100.0)
    n = len(df_sorted)
    
    daily_open = ltp[0] if n > 0 else 0.0
    
    # Running VWAP
    vwap_1m = df_sorted["vwap_1m"].values if "vwap_1m" in df_sorted.columns else np.zeros(n)
    volume_1m = df_sorted["volume_1m"].values if "volume_1m" in df_sorted.columns else np.zeros(n)
    cum_value = (vwap_1m * volume_1m).cumsum()
    cum_vol = volume_1m.cumsum()
    running_vwap = np.where(cum_vol > 0, cum_value / cum_vol, daily_open)
    
    alert_active = False
    alert_direction = None
    first_alert_idx = -1
    latest_alert_idx = -1
    
    trades = []
    
    if n > 0:
        ts_ist_str = df_sorted["ts_ist_str"].values if "ts_ist_str" in df_sorted.columns else []
    else:
        ts_ist_str = []
        
    for t in range(n):
        pred = preds[t]
        if pred == "LONG":
            if alert_active and alert_direction == "LONG":
                latest_alert_idx = t
            else:
                alert_active = True
                alert_direction = "LONG"
                first_alert_idx = t
                latest_alert_idx = t
            # Cancel opposite alert
            if alert_active and alert_direction == "SHORT":
                alert_active = True
                alert_direction = "LONG"
                first_alert_idx = t
                latest_alert_idx = t
                
        elif pred == "SHORT":
            if alert_active and alert_direction == "SHORT":
                latest_alert_idx = t
            else:
                alert_active = True
                alert_direction = "SHORT"
                first_alert_idx = t
                latest_alert_idx = t
            # Cancel opposite alert
            if alert_active and alert_direction == "LONG":
                alert_active = True
                alert_direction = "SHORT"
                first_alert_idx = t
                latest_alert_idx = t
                
        # Expiration Check
        if alert_active:
            if t - latest_alert_idx > 360:
                alert_active = False
                alert_direction = None
                first_alert_idx = -1
                latest_alert_idx = -1
                
        # Trigger Check
        if alert_active:
            direction = None
            if alert_direction == "LONG" and roc[t] >= 1.0:
                # Apply daily trend filters
                if filter_type == "none":
                    direction = "LONG"
                elif filter_type == "open" and ltp[t] > daily_open:
                    direction = "LONG"
                elif filter_type == "vwap" and ltp[t] > running_vwap[t]:
                    direction = "LONG"
                    
            elif alert_direction == "SHORT" and roc[t] <= -1.0:
                # Apply daily trend filters
                if filter_type == "none":
                    direction = "SHORT"
                elif filter_type == "open" and ltp[t] < daily_open:
                    direction = "SHORT"
                elif filter_type == "vwap" and ltp[t] < running_vwap[t]:
                    direction = "SHORT"
                    
            if direction:
                entry_price = ltp[t]
                exit_idx = n - 1  # Always exit at EOD
                exit_price = ltp[exit_idx]
                pnl = (exit_price - entry_price) / entry_price if direction == "LONG" else (entry_price - exit_price) / entry_price
                
                # Excursion till EOD
                future_prices = ltp[t:]
                if direction == "LONG":
                    mfe = (np.max(future_prices) - entry_price) / entry_price
                    mae = (np.min(future_prices) - entry_price) / entry_price
                else:
                    mfe = (entry_price - np.min(future_prices)) / entry_price
                    mae = (entry_price - np.max(future_prices)) / entry_price
                    
                # Excursion within 60 minutes
                horizon_prices = ltp[t : min(t + 60 + 1, n)]
                if direction == "LONG":
                    mfe_60m = (np.max(horizon_prices) - entry_price) / entry_price
                    mae_60m = (np.min(horizon_prices) - entry_price) / entry_price
                else:
                    mfe_60m = (entry_price - np.min(horizon_prices)) / entry_price
                    mae_60m = (entry_price - np.max(horizon_prices)) / entry_price
                    
                trades.append({
                    "symbol": df_sorted["symbol"].iloc[0] if "symbol" in df_sorted.columns else "",
                    "date": df_sorted["date"].iloc[0] if "date" in df_sorted.columns else "",
                    "entry_time": ts_ist_str[t] if len(ts_ist_str) > t else f"bar_{t}",
                    "direction": direction,
                    "entry_price": float(entry_price),
                    "pnl": float(pnl),
                    "mfe": float(mfe),
                    "mae": float(mae),
                    "mfe_60m": float(mfe_60m),
                    "mae_60m": float(mae_60m)
                })
                
                # Consume alert state
                alert_active = False
                alert_direction = None
                first_alert_idx = -1
                latest_alert_idx = -1
                
    return trades



def evaluate_backtest(trades):
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "mean_pnl": 0.0,
            "cumulative_pnl": 0.0
        }
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    return {
        "num_trades": len(pnls),
        "win_rate": round(win_rate, 4),
        "mean_pnl": round(float(np.mean(pnls)), 6),
        "cumulative_pnl": round(float(np.sum(pnls)), 6)
    }


# ─── Main Execution ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="label_60m_1pct", help="Target label column")
    args, unknown = parser.parse_known_args()
    
    global TARGET_LABEL
    TARGET_LABEL = args.target
    
    with open(log_path, "w") as log_fh:
        log("=" * 80, file=log_fh)
        log(f"WALK-FORWARD VALIDATION & LIGHTGBM UPGRADE EXPERIMENT (Target: {TARGET_LABEL})", file=log_fh)
        log(f"Started: {datetime.now().isoformat()}", file=log_fh)
        log("=" * 80, file=log_fh)
        
        # 1. Date discovery
        log("Discovering unique day dates...", file=log_fh)
        days = get_chronological_days(PROC)
        log(f"Found {len(days)} chronological days: {days}", file=log_fh)
        
        if len(days) < 6:
            log("ERROR: Need at least 6 days of data for walk-forward validation.", file=log_fh)
            return
            
        # Create DuckDB connection over all labeled parquet files
        log("Opening DuckDB connection (lazy parquet scan, no RAM load)...", file=log_fh)
        con = create_duckdb_connection(PROC)
        n_total = con.execute("SELECT COUNT(*) FROM all_labeled").fetchone()[0]
        log(f"DuckDB view created: {n_total:,} rows across all symbols/dates", file=log_fh)
        
        min_train_days = 5
        val_days = days[min_train_days:]
        log(f"Validation dates ({len(val_days)} folds): {val_days}", file=log_fh)
        
        fold_results = []
        
        for f_idx, val_date in enumerate(val_days):
            train_dates = days[:min_train_days + f_idx]
            log(f"\n--- Running Fold {f_idx + 1}/{len(val_days)}: Train={train_dates} -> Val={val_date} ---", file=log_fh)
            
            # Load per-fold data from DuckDB (only relevant dates, no RAM waste)
            date_filter_str = ", ".join(f"'{d}'" for d in train_dates + [val_date])
            col_list = ", ".join(f'"{c}"' for c in FEATURES + [TARGET_LABEL, "ltp", "ts", "close_price", "vwap_1m", "volume_1m", "filename"])
            raw = con.execute(f"""
                SELECT {col_list}, date_str
                FROM all_labeled
                WHERE date_str IN ({date_filter_str})
                  AND ltp IS NOT NULL AND ts IS NOT NULL
            """).fetchdf()
            if len(raw) == 0:
                log("  Skipping fold: empty data from DuckDB", file=log_fh)
                continue
            # Extract slug/symbol from filename
            prefix = "labeled_features_1m_"
            raw["slug"] = raw["filename"].str.extract(prefix + r"(.+?)_\d{1,2}_[A-Z][a-z]{2}_\d{2}\.parquet$")[0]
            raw["symbol"] = raw["slug"].str.split("_").str[0].str.upper()
            raw["date"] = raw["date_str"]
            raw["ts"] = pd.to_datetime(raw["ts"])
            df_fold = raw.sort_values(["symbol", "ts"]).reset_index(drop=True)
            df_fold["roc_9"] = df_fold.groupby("symbol")["close_price"].pct_change(9) * 100.0
            
            # Split train/val
            df_train_raw = df_fold[df_fold["date"].isin(train_dates)].copy()
            df_val = df_fold[df_fold["date"] == val_date].copy()
            
            # Purge training samples overlapping validation window
            if not df_val.empty:
                val_start = df_val["ts"].min()
                cutoff = val_start - timedelta(minutes=HORIZON_MIN)
                df_train = df_train_raw[df_train_raw["ts"] < cutoff]
            else:
                df_train = df_train_raw
            
            log(f"  Train set: {len(df_train)} rows | Purged validation overlap.", file=log_fh)
            log(f"  Val set  : {len(df_val)} rows", file=log_fh)
            
            if len(df_train) == 0 or len(df_val) == 0:
                log("  Skipping fold: empty train or validation set", file=log_fh)
                continue
                
            df_train_clean = df_train.dropna(subset=[TARGET_LABEL])
            X_train, y_train = df_train_clean[FEATURES].values, df_train_clean[TARGET_LABEL].values
            X_val = df_val[FEATURES].values
            
            # --- Model A: Logistic Regression (L2) ---
            scaler = RobustScaler()
            X_train_lr = scaler.fit_transform(X_train)
            X_val_lr = scaler.transform(X_val)
            
            lr = LogisticRegression(
                solver="lbfgs",
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=100
            )
            lr.fit(X_train_lr, y_train)
            pred_lr = lr.predict(X_val_lr)
            
            # --- Model B: Random Forest (Subsampled by 5 for CPU speed) ---
            X_train_sub = X_train[::5]
            y_train_sub = y_train[::5]
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            rf.fit(X_train_sub, y_train_sub)
            pred_rf = rf.predict(X_val)
            
            # --- Model C: LightGBM Upgrade (Fast leaf-wise tree boosting) ---
            lgb_train = lgb.Dataset(X_train, label=pd.Series(y_train).map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}))
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "n_jobs": -1
            }
            lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
            preds_lgb_probs = lgb_model.predict(X_val)
            pred_lgb_idx = np.argmax(preds_lgb_probs, axis=1)
            mapping = {0: "LONG", 1: "SHORT", 2: "NO_TRADE"}
            pred_lgb = np.vectorize(mapping.get)(pred_lgb_idx)
            
            # Store predictions back in validation dataframe
            df_val["pred_lr"] = pred_lr
            df_val["pred_rf"] = pred_rf
            df_val["pred_lgb"] = pred_lgb
            
            # MCC calculations on valid labels only
            val_clean_mask = df_val[TARGET_LABEL].notna()
            y_val_clean = df_val.loc[val_clean_mask, TARGET_LABEL].values
            mcc_lr = matthews_corrcoef(y_val_clean, df_val.loc[val_clean_mask, "pred_lr"].values)
            mcc_rf = matthews_corrcoef(y_val_clean, df_val.loc[val_clean_mask, "pred_rf"].values)
            mcc_lgb = matthews_corrcoef(y_val_clean, df_val.loc[val_clean_mask, "pred_lgb"].values)
            
            log(f"  Validation MCC results: LR={mcc_lr:.4f} | RF={mcc_rf:.4f} | LightGBM={mcc_lgb:.4f}", file=log_fh)
            
            # Run backtest simulation for RF and LightGBM across all trend filter variants
            backtests = {}
            for model_name, col in [("RF", "pred_rf"), ("LightGBM", "pred_lgb")]:
                for filt in ["none", "open", "vwap"]:
                    all_trades = []
                    for symbol, df_sym in df_val.groupby("symbol"):
                        all_trades.extend(simulate_trades(df_sym, col, model_name, filter_type=filt))
                    backtests[f"{model_name}_{filt}"] = evaluate_backtest(all_trades)
                    
            log(f"  Backtest (RF, no filter): Trades={backtests['RF_none']['num_trades']}, Cum P&L={backtests['RF_none']['cumulative_pnl']:.2%}", file=log_fh)
            log(f"  Backtest (RF, open filter): Trades={backtests['RF_open']['num_trades']}, Cum P&L={backtests['RF_open']['cumulative_pnl']:.2%}", file=log_fh)
            log(f"  Backtest (LGB, no filter): Trades={backtests['LightGBM_none']['num_trades']}, Cum P&L={backtests['LightGBM_none']['cumulative_pnl']:.2%}", file=log_fh)
            log(f"  Backtest (LGB, open filter): Trades={backtests['LightGBM_open']['num_trades']}, Cum P&L={backtests['LightGBM_open']['cumulative_pnl']:.2%}", file=log_fh)
            log(f"  Backtest (LGB, vwap filter): Trades={backtests['LightGBM_vwap']['num_trades']}, Cum P&L={backtests['LightGBM_vwap']['cumulative_pnl']:.2%}", file=log_fh)
            
            fold_results.append({
                "val_date": val_date,
                "n_train": len(df_train),
                "n_val": len(df_val),
                "mcc_lr": round(float(mcc_lr), 6),
                "mcc_rf": round(float(mcc_rf), 6),
                "mcc_lgb": round(float(mcc_lgb), 6),
                "backtests": backtests
            })
            
        # ─── Aggregate Summarization & Regime Analysis ───────────────────────────
        log("\n=== Aggregating Walk-Forward Performance ===", file=log_fh)
        
        mcc_lr_vals = [f["mcc_lr"] for f in fold_results]
        mcc_rf_vals = [f["mcc_rf"] for f in fold_results]
        mcc_lgb_vals = [f["mcc_lgb"] for f in fold_results]
        
        avg_mcc_lr = np.mean(mcc_lr_vals)
        avg_mcc_rf = np.mean(mcc_rf_vals)
        avg_mcc_lgb = np.mean(mcc_lgb_vals)
        
        log(f"Mean Walk-Forward MCC:", file=log_fh)
        log(f"  Logistic Regression: {avg_mcc_lr:.4f} ± {np.std(mcc_lr_vals):.4f}", file=log_fh)
        log(f"  Random Forest      : {avg_mcc_rf:.4f} ± {np.std(mcc_rf_vals):.4f}", file=log_fh)
        log(f"  LightGBM Upgrade   : {avg_mcc_lgb:.4f} ± {np.std(mcc_lgb_vals):.4f}", file=log_fh)
        
        # Aggregate Backtest Performance
        filter_strategies = ["none", "open", "vwap"]
        backtest_summary = {}
        for model in ["RF", "LightGBM"]:
            for filt in filter_strategies:
                key = f"{model}_{filt}"
                tot_trades = sum(f["backtests"][key]["num_trades"] for f in fold_results)
                tot_pnl = sum(f["backtests"][key]["cumulative_pnl"] for f in fold_results)
                mean_win = np.mean([f["backtests"][key]["win_rate"] for f in fold_results if f["backtests"][key]["num_trades"] > 0]) if tot_trades > 0 else 0.0
                backtest_summary[key] = {
                    "total_trades": int(tot_trades),
                    "total_pnl": round(float(tot_pnl), 6),
                    "mean_win_rate": round(float(mean_win), 4)
                }
                log(f"Strategy {key:<18}: Total Trades={tot_trades:<4} | Total P&L={tot_pnl:+.2%} | Win Rate={mean_win:.2%}", file=log_fh)
                
        # ─── SHAP Feature Importance Analysis (Final LGBM Fold) ────────────────
        log("\n=== Computing SHAP Feature Importance ===", file=log_fh)
        # Train explainer on final fold training set subset to run fast
        X_explain_sample = X_train[::20]
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_explain_sample)
        
        # Calculate mean absolute SHAP value per feature
        # For shap>=0.45.0, multiclass shap_values is a list of arrays [class_0, class_1, class_2] or single array [n_samples, n_features, n_classes]
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.mean(np.abs(shap_values[c]), axis=0) for c in range(len(shap_values))], axis=0)
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            mean_abs_shap = np.mean(np.mean(np.abs(shap_values), axis=0), axis=1)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
        shap_df = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        
        shap_df.to_csv(shap_path, index=False)
        log(f"SHAP feature importances saved to {shap_path}", file=log_fh)
        for i, r in shap_df.head(10).iterrows():
            log(f"  Rank {i+1}: {r['feature']:<25} (SHAP={r['mean_abs_shap']:.6f})", file=log_fh)
            
        # ─── Model Weight Export & Final Model Training ────────────────────────
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save Fold 7 weights
        lgb_model.save_model(str(models_dir / "lgbm_model_fold_7.txt"))
        with open(models_dir / "lgbm_model_fold_7.json", "w") as f:
            json.dump(lgb_model.dump_model(), f, indent=2)
        log("Saved Fold 7 LightGBM model weights to models/lgbm_model_fold_7.txt / .json", file=log_fh)
        
        # Train and save final model on all data
        log("\n=== Training Final Model on Entire Dataset ===", file=log_fh)
        col_list = ", ".join(f'"{c}"' for c in FEATURES + [TARGET_LABEL])
        df_all = con.execute(f"SELECT {col_list} FROM all_labeled").fetchdf().dropna()
        X_all, y_all = df_all[FEATURES].values, df_all[TARGET_LABEL].values
        lgb_all_train = lgb.Dataset(X_all, label=pd.Series(y_all).map({"LONG": 0, "SHORT": 1, "NO_TRADE": 2}))
        final_lgb_model = lgb.train(params, lgb_all_train, num_boost_round=100)
        
        final_lgb_model.save_model(str(models_dir / "lgbm_model_final.txt"))
        with open(models_dir / "lgbm_model_final.json", "w") as f:
            json.dump(final_lgb_model.dump_model(), f, indent=2)
        log("Saved Final LightGBM model weights to models/lgbm_model_final.txt / .json", file=log_fh)
            
        # Save output results JSON
        summary = {
            "experiment": "walk_forward_validation_v1",
            "date": datetime.now().isoformat(),
            "target_label": TARGET_LABEL,
            "mcc_summary": {
                "lr": {"mean": round(float(avg_mcc_lr), 6), "std": round(float(np.std(mcc_lr_vals)), 6)},
                "rf": {"mean": round(float(avg_mcc_rf), 6), "std": round(float(np.std(mcc_rf_vals)), 6)},
                "lgb": {"mean": round(float(avg_mcc_lgb), 6), "std": round(float(np.std(mcc_lgb_vals)), 6)}
            },
            "backtest_summary": backtest_summary,
            "fold_results": fold_results,
            "runtime_s": round(time.time() - t_start, 2)
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        log("=" * 80, file=log_fh)
        log("WALK-FORWARD EXPERIMENT COMPLETE", file=log_fh)
        log(f"Total Runtime: {time.time() - t_start:.1f}s", file=log_fh)
        log("=" * 80, file=log_fh)
        
    print(f"\nExperiment complete. Results: {results_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
