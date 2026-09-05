"""
SCRIPT 10: PREPARE DATA (REFACTORED — one-file-at-a-time)
Load raw per-symbol data (DOM and ticks), clean it, and save processed
version.  Streams one file at a time to keep peak memory under 200 MB.

Two loops after Stage 0 ingest:
  - DOM loop:  Data/dom_<slug>.parquet        -> 02_processed/cleaned_dom_<slug>.parquet
  - Tick loop: Data/raw/tick_<slug>.parquet  -> 02_processed/cleaned_ticks_<slug>.parquet

Resumable — skips already‑cleaned outputs on rerun.

Usage:
    python scripts/10_prepare_data.py
"""

import gc
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from raw.data_cleaner import DataCleaner
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR, DATA_DIR as ROOT_DATA_DIR

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug_from_filename(filename: str, prefix: str) -> str:
    """Strip the ``<prefix>_`` prefix from a filename to get the slug."""
    stem = Path(filename).stem
    needle = f"{prefix}_"
    if stem.startswith(needle):
        return stem[len(needle):]
    return stem


def _glob_raw_files(base_dir: Path, prefix: str) -> list[Path]:
    """Walk *base_dir* and *base_dir/raw/*, return sorted unique file paths.

    Files in *base_dir* take priority over files in *base_dir/raw/* with the
    same name (mimics the ``seen``-set dedup in ``DataLoader.load_all_*``).
    """
    seen: set[str] = set()
    paths: list[Path] = []

    for d in (base_dir, base_dir / "raw"):
        if not d.exists():
            continue
        for f in sorted(d.glob(f"{prefix}_*.parquet")):
            if f.name in seen:
                continue
            # Exclude day-level source files from processing as symbol files
            if "_year=" in f.name or "snapshots" in f.name or f.name.startswith("doms_") or f.name.startswith("ticks_"):
                continue
            seen.add(f.name)
            paths.append(f)

    paths.sort(key=lambda p: p.name)
    return paths


def _process_dom_file(filepath: Path, processed_dir: Path, force: bool = False) -> tuple[str, int, int]:
    """Load, clean, save one DOM file.  Returns ``(slug, raw_rows, clean_rows)``."""
    slug = slug_from_filename(filepath.name, "dom")
    output_path = processed_dir / f"cleaned_dom_{slug}.parquet"

    if not force and output_path.exists():
        logger.info(f"[DOM] Skipping {filepath.name} (slug={slug}) — already exists")
        return slug, 0, 0

    logger.info(f"\n[DOM] Processing {filepath.name}  (slug={slug})")
    try:
        raw_rows, clean_rows = DataCleaner.clean_dom_file_duckdb(filepath, output_path, label=filepath.name)
    except Exception as e:
        logger.warning(f"    FAILED to clean {filepath.name}: {e}. Skipping.")
        return slug, 0, 0
    logger.info(f"    Raw rows: {raw_rows:,}")
    logger.info(f"    Clean rows: {clean_rows:,}")
    logger.info(f"    Saved to: {output_path}")

    try:
        import duckdb
        con = duckdb.connect()
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{output_path}')").fetchall()
        has_ltp = any(r[0] == 'ltp' for r in desc)
        has_symbol = any(r[0] == 'symbol' for r in desc)
        num_cols = len(desc)

        stats_select = ["CAST(MIN(ts) AS VARCHAR) as min_ts", "CAST(MAX(ts) AS VARCHAR) as max_ts"]
        if has_ltp:
            stats_select.extend(["MIN(ltp) as min_ltp", "MAX(ltp) as max_ltp"])
        
        stats_res = con.execute(f"SELECT {', '.join(stats_select)} FROM read_parquet('{output_path}')").fetchone()
        min_ts, max_ts = stats_res[0], stats_res[1]

        logger.info(f"    Sample Statistics:")
        logger.info(f"    - Rows: {clean_rows:,}")
        logger.info(f"    - Columns: {num_cols}")
        logger.info(f"    - Time range: {min_ts} to {max_ts}")
        
        if has_symbol:
            syms_res = con.execute(f"SELECT DISTINCT symbol FROM read_parquet('{output_path}') LIMIT 4").fetchall()
            syms = [r[0] for r in syms_res]
            total_syms = con.execute(f"SELECT COUNT(DISTINCT symbol) FROM read_parquet('{output_path}')").fetchone()[0]
            logger.info(
                f"    - Symbols: {syms[:3]}{'...' if len(syms) > 3 else ''} "
                f"({total_syms} total)"
            )
        if has_ltp:
            min_ltp, max_ltp = stats_res[2], stats_res[3]
            logger.info(f"    - Price range: {min_ltp:.2f} to {max_ltp:.2f}")
    except Exception as e:
        logger.warning(f"    Failed to print stats for {filepath.name}: {e}")

    return slug, raw_rows, clean_rows


def _process_tick_file(filepath: Path, processed_dir: Path, force: bool = False) -> tuple[str, int, int]:
    """Load, clean, save one tick file.  Returns ``(slug, raw_rows, clean_rows)``."""
    slug = slug_from_filename(filepath.name, "tick")
    output_path = processed_dir / f"cleaned_ticks_{slug}.parquet"

    if not force and output_path.exists():
        logger.info(f"[TICK] Skipping {filepath.name} (slug={slug}) — already exists")
        return slug, 0, 0

    logger.info(f"\n[TICK] Processing {filepath.name}  (slug={slug})")
    try:
        raw_rows, clean_rows = DataCleaner.clean_ticks_file_duckdb(filepath, output_path, label=filepath.name)
    except Exception as e:
        logger.warning(f"    FAILED to clean {filepath.name}: {e}. Skipping.")
        return slug, 0, 0
    logger.info(f"    Raw rows: {raw_rows:,}")
    logger.info(f"    Clean rows: {clean_rows:,}")
    logger.info(f"    Saved to: {output_path}")

    try:
        import duckdb
        con = duckdb.connect()
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{output_path}')").fetchall()
        has_ltp = any(r[0] == 'ltp' for r in desc)
        has_aggressor = any(r[0] == 'aggressor' for r in desc)
        num_cols = len(desc)

        stats_select = ["CAST(MIN(ts) AS VARCHAR) as min_ts", "CAST(MAX(ts) AS VARCHAR) as max_ts"]
        if has_ltp:
            stats_select.extend(["MIN(ltp) as min_ltp", "MAX(ltp) as max_ltp"])
        
        stats_res = con.execute(f"SELECT {', '.join(stats_select)} FROM read_parquet('{output_path}')").fetchone()
        min_ts, max_ts = stats_res[0], stats_res[1]

        logger.info(f"    Sample Statistics:")
        logger.info(f"    - Rows: {clean_rows:,}")
        logger.info(f"    - Columns: {num_cols}")
        logger.info(f"    - Time range: {min_ts} to {max_ts}")
        
        if has_ltp:
            min_ltp, max_ltp = stats_res[2], stats_res[3]
            logger.info(f"    - LTP range: {min_ltp:.2f} to {max_ltp:.2f}")
        if has_aggressor:
            agg_res = con.execute(f"SELECT aggressor, COUNT(*) FROM read_parquet('{output_path}') GROUP BY aggressor").fetchall()
            counts = {r[0]: r[1] for r in agg_res}
            logger.info(f"    - Aggressor distribution: {counts}")
    except Exception as e:
        logger.warning(f"    Failed to print stats for {filepath.name}: {e}")

    return slug, raw_rows, clean_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Data Cleaner")
    parser.add_argument("--force", action="store_true", help="Force data cleaning regeneration.")
    args, unknown = parser.parse_known_args()

    logger.info("=" * 80)
    logger.info("STEP 10: PREPARE DATA - LOAD & CLEAN (DOM + TICKS) [REFACTORED]")
    logger.info("=" * 80)

    base_dir = Path(ROOT_DATA_DIR)
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DOM loop — one file at a time
    # ------------------------------------------------------------------
    dom_files = _glob_raw_files(base_dir, "dom")
    if not dom_files:
        raise FileNotFoundError(
            "No DOM files found in Data/ or Data/raw/. "
            "Check data path before rerunning Stage 1."
        )
    logger.info(f"\n[1] DOM files found: {len(dom_files)}")

    t_dom0 = time.time()
    n_dom_in, n_dom_out = 0, 0
    dom_skipped = 0

    for filepath in dom_files:
        slug, raw_rows, clean_rows = _process_dom_file(filepath, processed_dir, force=args.force)
        n_dom_in += raw_rows
        n_dom_out += clean_rows
        if raw_rows == 0:
            dom_skipped += 1

    t_dom = time.time() - t_dom0
    processed_count = len(dom_files) - dom_skipped
    logger.info(
        f"\n[DOM loop] {processed_count} processed, {dom_skipped} skipped | "
        f"{n_dom_in:,} -> {n_dom_out:,} rows | {t_dom:.1f}s"
    )

    # ------------------------------------------------------------------
    # Tick loop — one file at a time
    # ------------------------------------------------------------------
    tick_files = _glob_raw_files(base_dir, "tick")
    logger.info(f"\n[2] Tick files found: {len(tick_files)}")

    if not tick_files:
        logger.warning("No tick files found. Tick-clean step is a no-op.")
    else:
        t_tick0 = time.time()
        n_tick_in, n_tick_out = 0, 0
        tick_skipped = 0

        for filepath in tick_files:
            slug, raw_rows, clean_rows = _process_tick_file(filepath, processed_dir, force=args.force)
            n_tick_in += raw_rows
            n_tick_out += clean_rows
            if raw_rows == 0:
                tick_skipped += 1

        t_tick = time.time() - t_tick0
        processed_count_ = len(tick_files) - tick_skipped
        logger.info(
            f"\n[TICK loop] {processed_count_} processed, {tick_skipped} skipped | "
            f"{n_tick_in:,} -> {n_tick_out:,} rows | {t_tick:.1f}s"
        )

    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION COMPLETE [OK]")
    logger.info("=" * 80)
    logger.info(f"\nOutput files saved to: {processed_dir}/")
    logger.info("Next step: Run 20_infer_trades.py to generate trade records")


if __name__ == "__main__":
    main()
