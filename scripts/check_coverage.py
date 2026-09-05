"""Coverage alarm (NEW file, sandbox): which source dates/symbols never made it to OOF.
Usage: set PROJECTALPHA_DATA to the data dir, run: python check_coverage.py
Exit 2 + loud report when blind spots exist. Read-only; writes nothing to the repo."""
import os
import re
import sys
from pathlib import Path

DATA_DIR = Path(os.environ.get("PROJECTALPHA_DATA", r"D:\FriendData"))
PROCESSED = Path(os.environ.get("PROJECTALPHA_PROCESSED", r"C:\ProjectAlpha\02_processed"))
WL = Path(r"C:\ProjectAlpha\config\equity_symbols.parquet")

MONTH = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
         "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}


def main() -> int:
    import pandas as pd
    issues = 0
    wl = set(pd.read_parquet(WL)["symbol"]) if WL.exists() else set()
    print(f"whitelist symbols: {len(wl)}")

    src_days = sorted(DATA_DIR.glob("ticks_year=*_month=*_day=*_ticks.parquet"))
    print(f"source days: {len(src_days)}")
    for src in src_days:
        m = re.search(r"year=(\d+)_month=(\d+)_day=(\d+)", src.name)
        y, mo, d = m.group(1), m.group(2), m.group(3)
        slug = f"{d}_{[k for k, v in MONTH.items() if v == mo][0]}_{y[2:]}"
        oof = list(PROCESSED.glob(f"labeled_features_1m_*_{slug}.parquet"))
        raw = list((DATA_DIR / "raw").glob(f"*_{slug}.parquet")) if (DATA_DIR / "raw").exists() else []
        n_sym_src = None
        try:
            import duckdb
            n_sym_src = duckdb.connect().execute(
                "SELECT COUNT(DISTINCT symbol) FROM read_parquet(?)", [str(src)]).fetchone()[0]
        except Exception as e:
            print(f"  {slug}: symbol count unreadable ({e})")
        status = "OK " if (oof and raw) else "BLIND"
        if status == "BLIND":
            issues += 1
        print(f"  [{status}] {slug}: source_symbols={n_sym_src} raw_files={len(raw)} oof_files={len(oof)}")

    if issues:
        print(f"\nALARM: {issues} source day(s) with no usable backtest output.")
        return 2
    print("\nOK: every source day has raw + labeled output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
