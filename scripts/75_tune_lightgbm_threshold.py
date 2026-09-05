"""
LightGBM Threshold Tuning & Exception Simulation — 75_tune_lightgbm_threshold.py
=============================================================================
Sweeps decision thresholds for LightGBM's class probabilities under:
  - Standard Mode (Daily Open Trend Filter + 9-period ROC confirmation)
  - Exception Mode (adds Morning Bypass 09:15-09:30 IST and High-Confidence Bypass >= 0.35)
Supports label_60m_1pct target.
Always exits at EOD. Calculates MFE/MAE excursions and MFE success rates (3% and 5%).
Saves all triggered trades across all days/symbols to results/all_triggered_trades.csv.
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Force paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ─── Configuration ──────────────────────────────────────────────────────────

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
ROC_PERIOD = 3       # 3-period ROC (was 9)
ROC_THRESH = 1.0     # 1%
EXP_WINDOW = 60
HORIZON_MIN = 60

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "02_processed"
RESULTS = PROJECT_ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Thresholds to sweep
THRESHOLDS = [0.20, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.45]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_chronological_days(proc_dir: Path):
    labeled_files = list(proc_dir.glob("labeled_features_1m_*.parquet"))
    date_strs = set()
    for f in labeled_files:
        parts = f.stem.split("_")
        if len(parts) >= 7:
            date_strs.add("_".join(parts[-3:]))
            
    parsed = []
    for d_str in date_strs:
        try:
            dt = datetime.strptime(d_str, "%d_%b_%y")
            parsed.append((dt, d_str))
        except Exception:
            pass
    parsed.sort()
    return [d_str for _, d_str in parsed]


def load_day_data(proc_dir: Path, date_str: str, columns: list, label: str):
    all_dfs = []
    load_cols = columns + [label, "ltp", "ts", "vwap_1m", "volume_1m"]
    import pyarrow.parquet as pq
    labeled_files = list(proc_dir.glob(f"labeled_features_1m_*_{date_str}.parquet"))
    for f in labeled_files:
        slug = f.stem.replace("labeled_features_1m_", "")
        symbol = slug.split("_")[0].upper()
        try:
            schema = pq.read_schema(str(f))
            current_cols = load_cols.copy()
            if "close_price" in schema.names:
                current_cols.append("close_price")
            df = pd.read_parquet(f, columns=current_cols)
            if "close_price" not in df.columns:
                df["close_price"] = df["ltp"]
            df = df.dropna(subset=columns + ["ltp", "ts"])
            if len(df) == 0:
                continue
            df["slug"] = slug
            df["symbol"] = symbol
            df["date"] = date_str
            all_dfs.append(df)
        except Exception:
            pass
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def simulate_trades_with_exceptions(
    df_symbol: pd.DataFrame,
    pred_col: str,
    prob_col_long: str,
    prob_col_short: str,
    theta: float
):
    """
    Simulate EOD trades with:
      - Optional Daily Open filter (logged, not blocking)
      - ROC confirmation trigger (9-period, 1.0%, 360m expiration)
      - Multiple sequential trade entries permitted
      - Track alert-to-trigger distance
      - Track distance from daily low/high
      - Track hypothetical daily open filter trigger time
    """
    df_sorted = df_symbol.sort_values("ts").reset_index(drop=True)
    ltp = df_sorted["ltp"].values
    preds = df_sorted[pred_col].values
    prob_long = df_sorted[prob_col_long].values
    prob_short = df_sorted[prob_col_short].values
    roc = df_sorted["roc_3"].values if "roc_3" in df_sorted.columns else (df_sorted["ltp"].pct_change(ROC_PERIOD).values * 100.0)
    n = len(df_sorted)
    
    daily_open = ltp[0] if n > 0 else 0.0
    
    alert_active = False
    alert_direction = None
    first_alert_idx = -1
    latest_alert_idx = -1
    
    long_alerts = []
    short_alerts = []
    
    trades = []
    
    if n > 0:
        ts_ist_str = df_sorted["ts_ist_str"].values
    else:
        ts_ist_str = []
        
    for t in range(n):
        pred = preds[t]
        if pred == "LONG":
            long_alerts.append(t)
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
            short_alerts.append(t)
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
                
        # Trigger Check (daily open filter is NOT blocking)
        if alert_active:
            direction = None
            if alert_direction == "LONG" and roc[t] >= 1.0:
                direction = "LONG"
            elif alert_direction == "SHORT" and roc[t] <= -1.0:
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
                    
                # Alert-to-Trigger Distance (mins)
                alert_distance = t - latest_alert_idx
                
                # Running low/high distance up to trigger index t
                running_prices = ltp[:t + 1]
                if direction == "LONG":
                    daily_low = np.min(running_prices)
                    dist_low_high = (entry_price - daily_low) / daily_low * 100.0 if daily_low > 0 else 0.0
                else:
                    daily_high = np.max(running_prices)
                    dist_low_high = (daily_high - entry_price) / daily_high * 100.0 if daily_high > 0 else 0.0
                    
                # Daily Open Filter Status & Timing
                if direction == "LONG":
                    daily_open_passed = (entry_price > daily_open)
                    
                    # Search for first trigger time if open filter were active
                    open_trigger_time = None
                    for ti in range(first_alert_idx, n):
                        alerts_before = [la for la in long_alerts if la <= ti]
                        if not alerts_before:
                            continue
                        latest_la = max(alerts_before)
                        if ti - latest_la <= 360 and roc[ti] >= 1.0 and ltp[ti] > daily_open:
                            open_trigger_time = ts_ist_str[ti]
                            break
                else:
                    daily_open_passed = (entry_price < daily_open)
                    
                    # Search for first trigger time if open filter were active
                    open_trigger_time = None
                    for ti in range(first_alert_idx, n):
                        alerts_before = [sa for sa in short_alerts if sa <= ti]
                        if not alerts_before:
                            continue
                        latest_sa = max(alerts_before)
                        if ti - latest_sa <= 360 and roc[ti] <= -1.0 and ltp[ti] < daily_open:
                            open_trigger_time = ts_ist_str[ti]
                            break
                            
                trades.append({
                    "symbol": df_sorted["symbol"].iloc[0],
                    "date": df_sorted["date"].iloc[0],
                    "entry_time_ist": ts_ist_str[t],
                    "direction": direction,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "mfe": float(mfe),
                    "mae": float(mae),
                    "mfe_60m": float(mfe_60m),
                    "mae_60m": float(mae_60m),
                    "trigger_reason": "Standard ROC",
                    "prob_long": float(prob_long[t]),
                    "prob_short": float(prob_short[t]),
                    "roc_9": float(roc[t]) if not np.isnan(roc[t]) else None,
                    "daily_open": float(daily_open),
                    "alert_distance": int(alert_distance),
                    "dist_low_high": float(dist_low_high),
                    "daily_open_passed": bool(daily_open_passed),
                    "open_trigger_time": open_trigger_time
                })
                
                # Consume alert state
                alert_active = False
                alert_direction = None
                first_alert_idx = -1
                latest_alert_idx = -1
                
    return trades




def evaluate_metrics(trades, theta):
    num_trades = len(trades)
    if num_trades > 0:
        pnls = [t["pnl"] for t in trades]
        mfes = [t["mfe"] for t in trades]
        maes = [t["mae"] for t in trades]
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls)
        mean_pnl = np.mean(pnls)
        cum_pnl = np.sum(pnls)
        
        mean_mfe = np.mean(mfes)
        mean_mae = np.mean(maes)
        
        # MFE Success rates at 3% and 5%
        success_3pct = [m for m in mfes if m >= 0.03]
        success_5pct = [m for m in mfes if m >= 0.05]
        success_rate_3 = len(success_3pct) / len(mfes)
        success_rate_5 = len(success_5pct) / len(mfes)
    else:
        win_rate = 0.0
        mean_pnl = 0.0
        cum_pnl = 0.0
        mean_mfe = 0.0
        mean_mae = 0.0
        success_rate_3 = 0.0
        success_rate_5 = 0.0
        
    return {
        "Threshold": theta,
        "Total Trades": num_trades,
        "Win Rate": f"{win_rate:.2%}",
        "Mean Return": f"{mean_pnl:.4%}",
        "Cumulative Return": f"{cum_pnl:+.2%}",
        "Mean MFE": f"{mean_mfe:.2%}",
        "Mean MAE": f"{mean_mae:.2%}",
        "MFE Acc (>=3%)": f"{success_rate_3:.2%}",
        "MFE Acc (>=5%)": f"{success_rate_5:.2%}",
        "raw_cum_pnl": cum_pnl,
        "raw_success_5": success_rate_5
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="label_60m_1pct", choices=["label_60m_1pct"], help="Target label column to run")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip threshold sweep, only save OOF parquet")
    args, unknown = parser.parse_known_args()
    
    # Date discovery
    days = get_chronological_days(PROC)
    log(f"Found {len(days)} chronological days.")
    
    if len(days) < 6:
        log("ERROR: Need at least 6 days of data.")
        return
        
    min_train_days = 5
    val_days = days[min_train_days:]
    
    targets_to_run = [args.target]
    
    # Store all trade details across sweeps for CSV export
    all_triggered_trades_records = []
    
    # Store comparative summary results
    comparative_summary = []
    
    for target in targets_to_run:
        log(f"\n=======================================================")
        log(f"RUNNING EXPANDING WALK-FORWARD FOR TARGET: {target}")
        log(f"=======================================================")
        
        # Load dataset
        all_dfs = []
        for d in days:
            df_d = load_day_data(PROC, d, FEATURES, target)
            if not df_d.empty:
                all_dfs.append(df_d)
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all["ts"] = pd.to_datetime(df_all["ts"])
        
        log("Pre-calculating IST timestamps...")
        if df_all["ts"].dt.tz is None:
            ts_ist = df_all["ts"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        else:
            ts_ist = df_all["ts"].dt.tz_convert("Asia/Kolkata")
        df_all["ts_ist_time"] = ts_ist.dt.time
        df_all["ts_ist_str"] = ts_ist.dt.strftime("%H:%M:%S")
        
        # Calculate continuous ROC(3) for each symbol across all days
        log("Calculating continuous 3-period ROC...")
        df_all = df_all.sort_values(["symbol", "ts"]).reset_index(drop=True)
        df_all["roc_3"] = df_all.groupby("symbol")["close_price"].pct_change(ROC_PERIOD) * 100.0
        
        val_dfs = []
        
        for f_idx, val_date in enumerate(val_days):
            train_dates = days[:min_train_days + f_idx]
            log(f"Training fold {f_idx + 1}/{len(val_days)}: Train={train_dates[-1]} -> Val={val_date}")
            
            # Split
            df_train = df_all[df_all["date"].isin(train_dates)].copy()
            df_val = df_all[df_all["date"] == val_date].copy()
            
            val_start = df_val["ts"].min()
            cutoff_time = val_start - timedelta(minutes=HORIZON_MIN)
            df_train_purged = df_train[df_train["ts"] < cutoff_time]
            
            df_train_clean = df_train_purged.dropna(subset=[target])
            X_train, y_train = df_train_clean[FEATURES].values, df_train_clean[target].values
            X_val = df_val[FEATURES].values
            
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
            
            # Predict probabilities
            probs = lgb_model.predict(X_val)
            df_val["prob_long"] = probs[:, 0]
            df_val["prob_short"] = probs[:, 1]
            df_val["prob_no_trade"] = probs[:, 2]
            
            val_dfs.append(df_val)
            
        df_oof = pd.concat(val_dfs, ignore_index=True)
        log(f"Accumulated {len(df_oof)} OOF validation rows for {target}.")
        oof_path = RESULTS / f"oof_predictions_lgbm_{target}.parquet"
        df_oof.to_parquet(oof_path, index=False)
        log(f"Saved OOF predictions parquet to: {oof_path}")
        
        # ──── OOF generation complete. Skip sweeps if requested ────
        if args.skip_sweep:
            log("--skip-sweep: OOF saved. Skipping all threshold sweeps and summary exports.")
            # Skip to next target (or exit)
            continue
        
        # We sweep only Standard Mode now
        for mode_name, enable_ex in [("Standard", False)]:
            log(f"\n--- Running sweeps for Mode: {mode_name} ---")
            
            mode_results = []
            for theta in THRESHOLDS:
                # Assign predictions based on theta
                preds = []
                prob_l = df_oof["prob_long"].values
                prob_s = df_oof["prob_short"].values
                
                for idx in range(len(df_oof)):
                    p_l = prob_l[idx]
                    p_s = prob_s[idx]
                    if p_l >= theta and p_l > p_s:
                        preds.append("LONG")
                    elif p_s >= theta and p_s > p_l:
                        preds.append("SHORT")
                    else:
                        preds.append("NO_TRADE")
                        
                pred_col = f"pred_lgb_t{int(theta*100)}_{mode_name}"
                df_oof[pred_col] = preds
                
                # Simulate
                trades = []
                for slug, df_sym in df_oof.groupby("slug"):
                    symbol_trades = simulate_trades_with_exceptions(
                        df_sym, pred_col, "prob_long", "prob_short", theta
                    )
                    trades.extend(symbol_trades)
                    
                    # Record detailed trades for export
                    for t in symbol_trades:
                        criteria_60m_met = (t["mfe_60m"] >= 0.01)
                        criteria_eod_met = (t["mfe"] >= 0.05)
                        
                        all_triggered_trades_records.append({
                            "Target Label": target,
                            "Execution Mode": mode_name,
                            "Probability Threshold": theta,
                            "Date": t["date"],
                            "Symbol": t["symbol"],
                            "Entry Time (IST)": t["entry_time_ist"],
                            "Direction": t["direction"],
                            "Entry Price": round(t["entry_price"], 2),
                            "Exit Price": round(t["exit_price"], 2),
                            "Daily Open": round(t["daily_open"], 2),
                            "Model Prob LONG": round(t["prob_long"], 4),
                            "Model Prob SHORT": round(t["prob_short"], 4),
                            "ROC 9-Period": round(t["roc_9"], 4) if t["roc_9"] is not None else "NaN",
                            "Alert-to-Trigger Distance (mins)": t["alert_distance"],
                            "Distance from Day Low/High (%)": f"{round(t['dist_low_high'], 2)}%",
                            "Daily Open Filter Passed": "True" if t["daily_open_passed"] else "False",
                            "Daily Open Filter Trigger Time": t["open_trigger_time"] if t["open_trigger_time"] is not None else "",
                            "60-Min Horizon MFE": f"+{round(t['mfe_60m']*100, 2)}%",
                            "60-Min Horizon MAE": f"{round(t['mae_60m']*100, 2)}%",
                            "EOD MFE": f"+{round(t['mfe']*100, 2)}%",
                            "EOD MAE": f"{round(t['mae']*100, 2)}%",
                            "EOD Return": f"{round(t['pnl']*100, 2):+.2f}%",
                            "1% in 60m Satisfied": "True" if criteria_60m_met else "False",
                            "5% EOD Satisfied": "True" if criteria_eod_met else "False",
                            "Trigger Reason": t["trigger_reason"],
                            "Status": "Triggered"
                        })
                        
                metrics = evaluate_metrics(trades, theta)
                mode_results.append(metrics)
                
                comparative_summary.append({
                    "Target": target,
                    "Mode": mode_name,
                    "Threshold": theta,
                    "Total Trades": metrics["Total Trades"],
                    "Win Rate": metrics["Win Rate"],
                    "MFE Acc (>=3%)": metrics["MFE Acc (>=3%)"],
                    "MFE Acc (>=5%)": metrics["MFE Acc (>=5%)"],
                    "Mean MFE": metrics["Mean MFE"],
                    "Mean MAE": metrics["Mean MAE"],
                    "Cumulative P&L": metrics["Cumulative Return"]
                })
                
            df_mode_res = pd.DataFrame(mode_results)
            print(f"\nLIGHTGBM THRESHOLD SWEEP - COMBINED | Target: {target} | Mode: {mode_name}")
            print("="*100)
            print(df_mode_res.drop(columns=["raw_cum_pnl", "raw_success_5"]).to_string(index=False))
            print("="*100)
            
    if not all_triggered_trades_records and not comparative_summary:
        log("No sweep results to export. Exiting.")
        return

    # Save triggered trades CSV
    df_trades_all = pd.DataFrame(all_triggered_trades_records)
    out_trades_csv = RESULTS / "all_triggered_trades.csv"
    try:
        df_trades_all.to_csv(out_trades_csv, index=False)
        log(f"\n[SUCCESS] Saved details of all triggered trades to: {out_trades_csv}")
    except PermissionError:
        fallback_path = RESULTS / "all_triggered_trades_fallback.csv"
        df_trades_all.to_csv(fallback_path, index=False)
        log(f"\n[WARNING] Permission Denied on {out_trades_csv} (likely open in Excel). Saved trades to fallback: {fallback_path}")
    
    # Save comparative summary CSV
    df_comp = pd.DataFrame(comparative_summary)
    out_comp_csv = RESULTS / "lgbm_scenarios_comparison_summary.csv"
    try:
        df_comp.to_csv(out_comp_csv, index=False)
        log(f"[SUCCESS] Saved comparative scenario summary to: {out_comp_csv}")
    except PermissionError:
        fallback_path = RESULTS / "lgbm_scenarios_comparison_summary_fallback.csv"
        df_comp.to_csv(fallback_path, index=False)
        log(f"[WARNING] Permission Denied on {out_comp_csv} (likely open in Excel). Saved summary to fallback: {fallback_path}")
    
    # Print the absolute best configuration
    # We sort by Selection Accuracy (MFE Acc >=5%) and require at least 20 trades to avoid overfitting to a tiny sample size.
    df_filtered = df_comp[df_comp["Total Trades"] >= 20]
    if df_filtered.empty:
        df_filtered = df_comp
        
    df_filtered["raw_acc_5"] = df_filtered["MFE Acc (>=5%)"].str.rstrip("%").astype(float)
    df_best = df_filtered.sort_values(by="raw_acc_5", ascending=False).iloc[0]
    
    print("\n" + "="*80)
    print("OPTIMAL SCENARIO CONFIGURATION REPORT")
    print("="*80)
    print(f"  Best Target Label   : {df_best['Target']}")
    print(f"  Best Mode           : {df_best['Mode']}")
    print(f"  Best Threshold      : {df_best['Threshold']}")
    print(f"  Total Trades        : {df_best['Total Trades']}")
    print(f"  Win Rate            : {df_best['Win Rate']}")
    print(f"  MFE Acc (>= 3%)     : {df_best['MFE Acc (>=3%)']}")
    print(f"  MFE Acc (>= 5%)     : {df_best['MFE Acc (>=5%)']} (Selection Accuracy)")
    print(f"  Mean Favorable MFE  : {df_best['Mean MFE']}")
    print(f"  Mean Adverse MAE    : {df_best['Mean MAE']}")
    print(f"  Cumulative P&L      : {df_best['Cumulative P&L']}")
    print("="*80)


if __name__ == "__main__":
    main()
