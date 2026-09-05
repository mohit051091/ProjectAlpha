"""
SCRIPT 06: CONVERT UNIFIED TICK PARQUETS -> PER-SYMBOL DOM+TICK FILES

Reads unified hive-style tick parquets from Data/
(ticks_year=YYYY_month=MM_day=DD_ticks.parquet) which contain both
depth-5 book data and trade data in a single table, splits them into
per-symbol dom_<SYM>_<DATE>.parquet and tick_<SYM>_<DATE>.parquet
files in Data/raw/ — same format as 05_split_day_level.py produces
for the legacy two-source days.

Design:
  - Uses DuckDB to stream one symbol at a time (memory-safe for 10M rows)
  - DOM files get depth-5 columns only (bid1-5/bqty1-5/ask1-5/aqty1-5)
  - TICK files get L1 + trade columns (matching existing raw tick format)
  - `imbalance` computed as (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
  - Filters IST trading hours (03:30 UTC = 09:00 IST start, 10:00 UTC = 15:30 IST end)
  - Filters to exchange symbols only (no sentinels)

Usage:
    python scripts/06_convert_unified_ticks.py --day 15
    python scripts/06_convert_unified_ticks.py --day 16
    python scripts/06_convert_unified_ticks.py --all
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
from utils.logger import setup_logger
from utils.constants import DATA_DIR as ROOT_DATA_DIR

logger = setup_logger(__name__)

DATA_DIR = Path(ROOT_DATA_DIR)
RAW_DIR = DATA_DIR / "raw"
CONFIG_DIR = Path(__file__).parent.parent / "config"

UNIFIED_PATTERN = DATA_DIR / "ticks_year=*_month=06_day={day:02d}_ticks.parquet"

# Columns for per-symbol DOM files (depth 5 only)
DOM_COLUMNS = [
    "ts", "symbol", "ltp",
    "total_bid_qty", "total_ask_qty",
    "bid1", "bqty1", "bid2", "bqty2", "bid3", "bqty3", "bid4", "bqty4", "bid5", "bqty5",
    "ask1", "aqty1", "ask2", "aqty2", "ask3", "aqty3", "ask4", "aqty4", "ask5", "aqty5",
]

# Columns for per-symbol TICK files (L1 + trades)
TICK_COLUMNS = [
    "ts", "symbol", "ltp", "volume", "oi",
    "aggressor", "trade_qty", "bid1", "bqty1", "ask1", "aqty1", "cum_delta",
]

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def make_slug(symbol: str, day: int, month: int, year: int) -> str:
    return f"{str(symbol).upper()}_{day:02d}_{MONTH_ABBR[month]}_{year % 100:02d}"


def convert_day(day: int, month: int, year: int, force: bool = False) -> dict:
    """Convert one unified tick file to per-symbol DOM+TICK files."""
    con = duckdb.connect()
    con.execute("SET TIME ZONE 'UTC'")

    src_pattern = str(DATA_DIR / f"ticks_year={year}_month={month:02d}_day={day:02d}_ticks.parquet")
    src_files = list(DATA_DIR.glob(f"ticks_year={year}_month={month:02d}_day={day:02d}_ticks.parquet"))

    if not src_files:
        logger.error(f"No unified tick file found for {year}-{month:02d}-{day:02d}")
        con.close()
        return {"day": day, "symbols": 0, "dom_written": 0, "tick_written": 0, "skipped": 0}

    src_path = str(src_files[0])
    logger.info(f"Source: {src_files[0].name}")

    # Load equity symbol whitelist
    equity_path = CONFIG_DIR / "equity_symbols.parquet"
    if not equity_path.exists():
        logger.error(f"Equity symbol whitelist not found at {equity_path}. Create it first.")
        con.close()
        return {"day": day, "symbols": 0, "dom_written": 0, "tick_written": 0, "skipped": 0}
    con.execute(f"CREATE TABLE equity_syms AS SELECT symbol FROM read_parquet('{equity_path}')")
    eq_count = con.execute("SELECT COUNT(*) FROM equity_syms").fetchone()[0]
    logger.info(f"Equity whitelist: {eq_count} symbols")

    # Get list of equity symbols present in this file
    symbols = con.execute(f"""
        SELECT DISTINCT u.symbol FROM read_parquet('{src_path}') u
        INNER JOIN equity_syms e ON u.symbol = e.symbol
        ORDER BY u.symbol
    """).fetchall()
    symbols = [r[0] for r in symbols]
    logger.info(f"Equity symbols in file: {len(symbols)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"day": day, "symbols": len(symbols), "dom_written": 0, "tick_written": 0, "skipped": 0, "errors": []}

    # Load only equity rows into DuckDB table (memory-mapped, ~1-pass scan)
    logger.info(f"Loading equity rows from {src_files[0].name} into DuckDB...")
    con.execute(f"""
        CREATE TABLE unified AS
        SELECT u.* FROM read_parquet('{src_path}') u
        INNER JOIN equity_syms e ON u.symbol = e.symbol
    """)
    row_count = con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
    logger.info(f"Loaded {row_count:,} equity rows")
    con.execute("DROP TABLE IF EXISTS equity_syms")

    # Build DOM query template against the in-memory table
    dom_cols = ", ".join(DOM_COLUMNS)
    dom_query = f"""
        SELECT {dom_cols},
               CAST(total_bid_qty AS DOUBLE) / NULLIF(CAST(total_bid_qty AS DOUBLE) + CAST(total_ask_qty AS DOUBLE), 0)
               - CAST(total_ask_qty AS DOUBLE) / NULLIF(CAST(total_bid_qty AS DOUBLE) + CAST(total_ask_qty AS DOUBLE), 0) AS imbalance
        FROM unified
        WHERE symbol = ?
        ORDER BY ts
    """

    tick_query = f"""
        SELECT {", ".join(TICK_COLUMNS)}
        FROM unified
        WHERE symbol = ?
        ORDER BY ts
    """

    t0 = time.time()
    for idx, sym in enumerate(symbols):
        slug = make_slug(sym, day, month, year)

        # Check if already exists
        dom_path = RAW_DIR / f"dom_{slug}.parquet"
        tick_path = RAW_DIR / f"tick_{slug}.parquet"
        if not force and dom_path.exists() and tick_path.exists():
            stats["skipped"] += 1
            continue

        try:
            # Write DOM
            con.execute(f"""
                COPY (
                    {dom_query}
                ) TO '{dom_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
            """, [sym])
            stats["dom_written"] += 1

            # Write TICK
            con.execute(f"""
                COPY (
                    {tick_query}
                ) TO '{tick_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
            """, [sym])
            stats["tick_written"] += 1

        except Exception as e:
            stats["errors"].append((sym, str(e)))
            logger.warning(f"  FAILED {sym}: {e}")

        if (idx + 1) % 100 == 0 or idx == len(symbols) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(symbols) - idx - 1) / rate if rate > 0 else 0
            logger.info(f"  [{idx+1}/{len(symbols)}] dom={stats['dom_written']} tick={stats['tick_written']} skipped={stats['skipped']} | {elapsed:.0f}s elapsed, {eta:.0f}s eta")

    con.execute("DROP TABLE IF EXISTS unified")

    con.close()

    elapsed = time.time() - t0
    logger.info(f"Day {day}: {stats['dom_written']} DOM + {stats['tick_written']} TICK files written, "
                f"{stats['skipped']} skipped, {len(stats['errors'])} errors in {elapsed:.0f}s")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert unified tick parquets to per-symbol DOM+TICK files")
    parser.add_argument("--day", type=int, help="Day of month to convert (15 or 16)")
    parser.add_argument("--all", action="store_true", help="Convert all available unified tick days")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--month", type=int, default=6, help="Month (default: 6)")
    parser.add_argument("--year", type=int, default=2026, help="Year (default: 2026)")
    args = parser.parse_args()

    if args.all:
        day_specs = []
        for f in sorted(DATA_DIR.glob("ticks_year=*_month=*_day=*_ticks.parquet")):
            parts = f.stem.split("_")
            yp = [p for p in parts if p.startswith("year=")]
            mp = [p for p in parts if p.startswith("month=")]
            dp = [p for p in parts if p.startswith("day=")]
            if yp and mp and dp:
                day_specs.append((int(yp[0].split("=")[1]), int(mp[0].split("=")[1]), int(dp[0].split("=")[1])))
        day_specs = sorted(set(day_specs))
    elif args.day:
        day_specs = [(args.year, args.month, args.day)]
    else:
        parser.print_help()
        return

    logger.info("=" * 80)
    logger.info("SCRIPT 06: CONVERT UNIFIED TICKS -> PER-SYMBOL DOM+TICK")
    logger.info("=" * 80)

    total = {"dom": 0, "tick": 0, "skipped": 0, "errors": 0}
    for year, month, day in day_specs:
        logger.info(f"\n--- {year}-{month:02d}-{day:02d} ---")
        s = convert_day(day, month, year, force=args.force)
        total["dom"] += s["dom_written"]
        total["tick"] += s["tick_written"]
        total["skipped"] += s["skipped"]
        total["errors"] += len(s["errors"])

    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL: {total['dom']} DOM + {total['tick']} TICK files, "
                f"{total['skipped']} skipped, {total['errors']} errors")
    logger.info("Next step: run scripts/10_prepare_data.py")


if __name__ == "__main__":
    raise SystemExit(main())
