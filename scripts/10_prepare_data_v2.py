"""
SCRIPT 10 v2: PREPARE DATA (PARALLEL)
Same as original but uses ProcessPoolExecutor for parallel DOM + Tick cleaning.
"""
import gc
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from raw.data_cleaner import DataCleaner
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR, DATA_DIR as ROOT_DATA_DIR

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(ROOT_DATA_DIR)


def slug_from_filename(filename: str, prefix: str) -> str:
    stem = Path(filename).stem
    needle = f"{prefix}_"
    if stem.startswith(needle):
        return stem[len(needle):]
    return stem


def _glob_raw_files(base_dir: Path, prefix: str) -> list[Path]:
    seen = set()
    paths = []
    for d in (base_dir, base_dir / "raw"):
        if not d.exists():
            continue
        for f in sorted(d.glob(f"{prefix}_*.parquet")):
            if f.name in seen:
                continue
            if "_year=" in f.name or "snapshots" in f.name or f.name.startswith("doms_") or f.name.startswith("ticks_"):
                continue
            seen.add(f.name)
            paths.append(f)
    paths.sort(key=lambda p: p.name)
    return paths


def _process_dom_worker(args):
    filepath, processed_dir_str, force = args
    processed_dir = Path(processed_dir_str)
    slug = slug_from_filename(filepath.name, "dom")
    output_path = processed_dir / f"cleaned_dom_{slug}.parquet"
    if not force and output_path.exists():
        return ("skip", slug, filepath.name, 0, 0)
    try:
        raw_rows, clean_rows = DataCleaner.clean_dom_file_duckdb(filepath, output_path, label=filepath.name)
        return ("ok", slug, filepath.name, raw_rows, clean_rows)
    except Exception as e:
        return ("fail", slug, filepath.name, str(e))


def _process_tick_worker(args):
    filepath, processed_dir_str, force = args
    processed_dir = Path(processed_dir_str)
    slug = slug_from_filename(filepath.name, "tick")
    output_path = processed_dir / f"cleaned_ticks_{slug}.parquet"
    if not force and output_path.exists():
        return ("skip", slug, filepath.name, 0, 0)
    try:
        raw_rows, clean_rows = DataCleaner.clean_ticks_file_duckdb(filepath, output_path, label=filepath.name)
        return ("ok", slug, filepath.name, raw_rows, clean_rows)
    except Exception as e:
        return ("fail", slug, filepath.name, str(e))


def _process_files_in_parallel(files, worker_fn, label, workers, force=False):
    if not files:
        logger.info(f"  No {label} files to process.")
        return 0, 0, 0
    logger.info(f"  Processing {len(files)} {label} files with {workers} workers...")
    args_list = [(f, str(PROCESSED_DIR), force) for f in files]
    n_in, n_out = 0, 0
    skipped = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_fn, a): a[0].name for a in args_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            fname = futures[future]
            try:
                result = future.result()
                status = result[0]
                if status == "skip":
                    skipped += 1
                    if done % 200 == 0 or done == len(files):
                        elapsed = time.time() - t0
                        logger.info(f"    [{done}/{len(files)}] {fname}: skipped | {elapsed:.1f}s")
                elif status == "ok":
                    _, slug, _, raw, clean = result
                    n_in += raw
                    n_out += clean
                    if done % 200 == 0 or done == len(files):
                        elapsed = time.time() - t0
                        logger.info(f"    [{done}/{len(files)}] {fname}: {raw:,}->{clean:,} rows | {elapsed:.1f}s")
                else:
                    _, slug, fname, err = result
                    logger.warning(f"    [{done}/{len(files)}] {fname}: FAILED - {err}")
            except Exception as e:
                logger.warning(f"    [{done}/{len(files)}] {fname}: worker error - {e}")
                skipped += 1
    return n_in, n_out, skipped


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Data Cleaner (Parallel)")
    parser.add_argument("--force", action="store_true", help="Force regeneration.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args, _ = parser.parse_known_args()
    workers = args.workers or os.cpu_count()

    logger.info("=" * 80)
    logger.info("STEP 10 (v2): PREPARE DATA - LOAD & CLEAN (DOM + TICKS) [PARALLEL]")
    logger.info("=" * 80)

    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dom_files = _glob_raw_files(DATA_DIR, "dom")
    if not dom_files:
        raise FileNotFoundError("No DOM files found in Data/ or Data/raw/.")
    logger.info(f"\n[1] DOM files found: {len(dom_files)}")

    t_dom0 = time.time()
    n_dom_in, n_dom_out, dom_skipped = _process_files_in_parallel(
        dom_files, _process_dom_worker, "DOM", workers, force=args.force
    )
    t_dom = time.time() - t_dom0
    processed_count = len(dom_files) - dom_skipped
    logger.info(
        f"\n[DOM loop] {processed_count} processed, {dom_skipped} skipped | "
        f"{n_dom_in:,} -> {n_dom_out:,} rows | {t_dom:.1f}s"
    )

    tick_files = _glob_raw_files(DATA_DIR, "tick")
    logger.info(f"\n[2] Tick files found: {len(tick_files)}")

    if not tick_files:
        logger.warning("No tick files found. Tick-clean step is a no-op.")
    else:
        t_tick0 = time.time()
        n_tick_in, n_tick_out, tick_skipped = _process_files_in_parallel(
            tick_files, _process_tick_worker, "TICK", workers, force=args.force
        )
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


if __name__ == "__main__":
    main()
