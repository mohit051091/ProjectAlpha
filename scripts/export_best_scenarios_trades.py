"""
Export Consolidated Best Scenario Trades — export_best_scenarios_trades.py
==========================================================================
Loads results/all_triggered_trades.csv, filters it to exactly two key configurations:
  1. label_60m_1pct | Exception-Enabled Mode | Threshold 0.20
  2. label_60m_1pct | Exception-Enabled Mode | Threshold 0.25
Adds a 'Variation' column identifying the configuration and saves the output
consolidated to results/consolidated_best_trades.csv.
"""

from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

def main():
    in_csv = RESULTS / "all_triggered_trades.csv"
    if not in_csv.exists():
        print(f"ERROR: File {in_csv} does not exist. Please run sweeps first.")
        return
        
    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df)} total sweep trades from {in_csv}.")
    
    # Define targets
    cond1 = (df["Target Label"] == "label_60m_1pct") & (df["Execution Mode"] == "Standard") & (df["Probability Threshold"] == 0.20)
    cond2 = (df["Target Label"] == "label_60m_1pct") & (df["Execution Mode"] == "Standard") & (df["Probability Threshold"] == 0.25)
    
    # Filter
    df1 = df[cond1].copy()
    df1["Variation"] = "60m_1pct_Standard_0.20"
    
    df2 = df[cond2].copy()
    df2["Variation"] = "60m_1pct_Standard_0.25"
    
    # Combine
    df_best = pd.concat([df1, df2], ignore_index=True)
    
    # Move 'Variation' to first column
    cols = ["Variation"] + [c for c in df_best.columns if c != "Variation"]
    df_best = df_best[cols]
    
    out_csv = RESULTS / "consolidated_best_trades.csv"
    try:
        df_best.to_csv(out_csv, index=False)
        print(f"\n[SUCCESS] Consolidated {len(df_best)} best trades to: {out_csv}")
    except PermissionError:
        fallback_path = RESULTS / "consolidated_best_trades_fallback.csv"
        df_best.to_csv(fallback_path, index=False)
        print(f"\n[WARNING] Permission Denied on {out_csv} (likely open in Excel). Saved consolidated trades to fallback: {fallback_path}")
    print(df_best["Variation"].value_counts().to_string())

if __name__ == "__main__":
    main()
