"""
Export Symbol Minute Details — export_symbol_minute_details.py
==============================================================
Loads the OOF predictions parquet and exports the complete minute-by-minute
predictions, probabilities, price, ROC, and all 20 features for a list of symbols
and a specific date to readable CSVs.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

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

def main():
    parser = argparse.ArgumentParser(description="Export minute-by-minute probabilities and features for specific symbols and date.")
    parser.add_argument("--target", type=str, default="label_60m_1pct", choices=["label_60m_1pct"])
    parser.add_argument("--date", type=str, required=True, help="Date to export, e.g. 11_Jun_26")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol name or comma-separated list of symbols, e.g. ABCAPITAL,AEGISLOG")
    args = parser.parse_args()

    parquet_file = RESULTS / f"oof_predictions_lgbm_{args.target}.parquet"
    if not parquet_file.exists():
        print(f"ERROR: OOF Parquet file {parquet_file} not found. Run sweeps first.")
        sys.exit(1)

    print(f"Loading predictions from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    date_str = args.date

    for symbol in symbols:
        print(f"\n=========================================")
        print(f"PROCESSING SYMBOL: {symbol} on {date_str}")
        print(f"=========================================")
        
        # Filter all dates for this symbol so we can calculate continuous ROC across day boundaries
        df_sym = df[df["symbol"] == symbol].copy()
        if df_sym.empty:
            print(f"WARNING: No data found for symbol {symbol} in predictions. Skipping.")
            continue
            
        df_sym["ts"] = pd.to_datetime(df_sym["ts"])
        df_sym = df_sym.sort_values("ts").reset_index(drop=True)
        
        # Scan 02_processed for ALL labeled feature files of this symbol to construct the COMPLETE timeline
        proc_dir = RESULTS.parent / "02_processed"
        feat_files = list(proc_dir.glob(f"labeled_features_1m_{symbol}_*.parquet"))
        
        all_unfiltered_dfs = []
        for f in feat_files:
            # Load only timestamps to get the complete, unfiltered 1-minute grid
            df_f = pd.read_parquet(f, columns=["ts"])
            df_f["ts"] = pd.to_datetime(df_f["ts"])
            # Extract date from file name
            parts = f.stem.split("_")
            df_f["date"] = "_".join(parts[-3:])
            all_unfiltered_dfs.append(df_f)
            
        if not all_unfiltered_dfs:
            print(f"ERROR: No processed features found in 02_processed for {symbol}.")
            continue
            
        df_timeline = pd.concat(all_unfiltered_dfs, ignore_index=True)
        df_timeline = df_timeline.sort_values("ts").reset_index(drop=True)
        
        # Load ticks and extract close prices for all those dates
        tick_dfs = []
        for d in df_timeline["date"].unique():
            tick_file = proc_dir / f"cleaned_ticks_{symbol}_{d}.parquet"
            if tick_file.exists():
                df_ticks = pd.read_parquet(tick_file, columns=["ts", "ltp"])
                df_ticks["ts"] = pd.to_datetime(df_ticks["ts"])
                df_ticks = df_ticks.sort_values("ts").reset_index(drop=True)
                df_ticks["ts_1m"] = df_ticks["ts"].dt.floor("1min")
                df_close = df_ticks.groupby("ts_1m")["ltp"].last().reset_index()
                df_close = df_close.rename(columns={"ts_1m": "ts", "ltp": "close_price"})
                tick_dfs.append(df_close)
                
        if tick_dfs:
            df_all_close = pd.concat(tick_dfs, ignore_index=True)
            if df_all_close["ts"].dt.tz is None:
                df_all_close["ts"] = df_all_close["ts"].dt.tz_localize("UTC")
            if df_timeline["ts"].dt.tz is None:
                df_timeline["ts"] = df_timeline["ts"].dt.tz_localize("UTC")
                
            df_timeline = pd.merge(df_timeline, df_all_close, on="ts", how="left")
            # Load ltp from labeled features for any minutes with missing ticks
            df_timeline["close_price"] = df_timeline["close_price"].ffill()
        else:
            # Fallback
            df_timeline["close_price"] = df_sym["ltp"]
            
        # Compute the correct continuous 9-period ROC on the complete timeline
        df_timeline["roc_9"] = df_timeline["close_price"].pct_change(9) * 100.0
        
        # Drop existing old columns from prediction df before merging to prevent suffixes
        for col in ["roc_9", "close_price"]:
            if col in df_sym.columns:
                df_sym = df_sym.drop(columns=[col])
                
        if df_sym["ts"].dt.tz is None:
            df_sym["ts"] = df_sym["ts"].dt.tz_localize("UTC")
        if df_timeline["ts"].dt.tz is None:
            df_timeline["ts"] = df_timeline["ts"].dt.tz_localize("UTC")
            
        df_sym = pd.merge(
            df_sym, 
            df_timeline[["ts", "close_price", "roc_9"]], 
            on="ts", 
            how="left"
        )

        # Now filter to target date
        df_filtered = df_sym[df_sym["date"] == date_str].copy()
        if df_filtered.empty:
            print(f"WARNING: No data found for symbol {symbol} on date {date_str} after merge. Skipping.")
            continue

        # Sort chronologically
        df_filtered = df_filtered.sort_values("ts").reset_index(drop=True)

        # Format IST times if needed
        if "ts_ist_str" not in df_filtered.columns:
            if df_filtered["ts"].dt.tz is None:
                ts_ist = df_filtered["ts"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
            else:
                ts_ist = df_filtered["ts"].dt.tz_convert("Asia/Kolkata")
            df_filtered["ts_ist_str"] = ts_ist.dt.strftime("%H:%M:%S")

        # Select audit columns (core + all 20 features)
        cols = [
            "ts_ist_str", 
            "ltp",
            "close_price",
            "prob_long", 
            "prob_short", 
            "prob_no_trade",
            "roc_9"
        ] + FEATURES

        df_out = df_filtered[cols].copy()
        
        # Round float values for readability
        df_out["ltp"] = df_out["ltp"].round(2)
        df_out["close_price"] = df_out["close_price"].round(2)
        df_out["prob_long"] = df_out["prob_long"].round(4)
        df_out["prob_short"] = df_out["prob_short"].round(4)
        df_out["prob_no_trade"] = df_out["prob_no_trade"].round(4)
        df_out["roc_9"] = df_out["roc_9"].round(4)
        for feat in FEATURES:
            if feat in df_out.columns:
                df_out[feat] = df_out[feat].round(4)

        # Rename core columns for clarity
        df_out = df_out.rename(columns={
            "ts_ist_str": "Time (IST)",
            "ltp": "LTP",
            "close_price": "Close Price",
            "prob_long": "Prob LONG",
            "prob_short": "Prob SHORT",
            "prob_no_trade": "Prob NO_TRADE",
            "roc_9": "ROC (9-Period)"
        })

        # Save to CSV
        out_csv = RESULTS / f"{symbol}_{date_str}_minute_probabilities.csv"
        try:
            df_out.to_csv(out_csv, index=False)
            print(f"[SUCCESS] Saved minute-level details to: {out_csv}")
        except PermissionError:
            fallback = RESULTS / f"{symbol}_{date_str}_minute_probabilities_fallback.csv"
            df_out.to_csv(fallback, index=False)
            print(f"[WARNING] Permission denied on {out_csv}. Saved to: {fallback}")

        # Print a preview of the first 10 rows
        print(f"\nPREVIEW FOR {symbol} ON {date_str} (Target: {args.target})")
        print("=" * 80)
        pd.set_option("display.max_columns", 15)
        print(df_out.head(10).to_string(index=False))
        print("=" * 80)

if __name__ == "__main__":
    main()
