"""
Update Whitelist — scripts/update_whitelist.py
=============================================
Downloads the latest Dhan API scrip master dynamically, reads the manual
Nifty 500 watchlist from Clean_Scanner, filters for NSE Segment 'E' (Equity)
and Series 'EQ' (Equity Series) only, and overwrites config/equity_symbols.parquet.
"""

import os
import sys
import urllib.request
import csv
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_ROOT / "config"
WATCHLIST_PATH = PROJECT_ROOT.parent / "OrderFlow" / "Clean_Scanner" / "ind_nifty500list.csv"
OUTPUT_PATH = CONFIG_DIR / "equity_symbols.parquet"
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

def main():
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Starting dynamic whitelist update...")
    
    # 1. Load manual watchlist from ind_nifty500list.csv (optional)
    watchlist_symbols = None
    if WATCHLIST_PATH.exists():
        print(f"Loading watchlist from: {WATCHLIST_PATH}")
        watchlist_df = pd.read_csv(WATCHLIST_PATH)
        col = "Symbol" if "Symbol" in watchlist_df.columns else watchlist_df.columns[2]
        watchlist_symbols = set(watchlist_df[col].dropna().astype(str).str.strip().str.upper().tolist())
        print(f"Loaded {len(watchlist_symbols)} symbols from manual watchlist.")
    else:
        print(f"Watchlist not found at {WATCHLIST_PATH}. Using all NSE EQ symbols from Dhan.")
        
    print(f"Loading watchlist from: {WATCHLIST_PATH}")
    watchlist_df = pd.read_csv(WATCHLIST_PATH)
    col = "Symbol" if "Symbol" in watchlist_df.columns else watchlist_df.columns[2] # Fallback to 3rd column
    watchlist_symbols = set(watchlist_df[col].dropna().astype(str).str.strip().str.upper().tolist())
    print(f"Loaded {len(watchlist_symbols)} symbols from manual watchlist.")

    # 2. Download Dhan scrip master online
    print(f"Downloading latest scrip master from: {SCRIP_MASTER_URL}")
    temp_csv = CONFIG_DIR / "temp_scrip_master.csv"
    try:
        urllib.request.urlretrieve(SCRIP_MASTER_URL, temp_csv)
    except Exception as e:
        print(f"ERROR: Failed to download scrip master: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Parse and filter rows (NSE, Segment E, Series EQ, inside Watchlist)
    print("Filtering scrip master for NSE cash equities (series EQ) matching watchlist...")
    matched_symbols = set()
    
    with open(temp_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exch = (row.get("SEM_EXM_EXCH_ID") or "").strip().upper()
            seg = (row.get("SEM_SEGMENT") or "").strip().upper()
            series = (row.get("SEM_SERIES") or "").strip().upper()
            trading_sym = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
            
            if exch == "NSE" and seg == "E" and series == "EQ":
                if watchlist_symbols is None or trading_sym in watchlist_symbols:
                    matched_symbols.add(trading_sym)
                    
    # Clean up temp file
    if temp_csv.exists():
        os.remove(temp_csv)
        
    print(f"Matched {len(matched_symbols)} eligible 'EQ' series symbols.")
    
    # Print status of demerged Vedanta entities
    for demerged_sym in ["VOGL", "VAML", "VISL", "VEDPOWER"]:
        if demerged_sym in matched_symbols:
            print(f"[INFO] Demerged symbol {demerged_sym} matches as EQ series (legitimate transition).")
            
    # 4. Save to config/equity_symbols.parquet
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(sorted(list(matched_symbols)), columns=["symbol"])
    df_out.to_parquet(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Wrote {len(df_out)} symbols to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
