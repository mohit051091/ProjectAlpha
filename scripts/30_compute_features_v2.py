"""
SCRIPT 30 v2: COMPUTE FEATURES (PARALLEL)
Uses ProcessPoolExecutor to process symbols in parallel.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR

logger = setup_logger(__name__)

STAGE1_FEATURES = ["delta_1m", "delta_5m"]
STAGE2_FEATURES = ["volume_burst", "aggressor_ratio", "trade_count_burst", "large_trade_ratio"]
MICROSTRUCTURE_FEATURES = ["iceberg_score", "order_cancel_rate", "bid_replenishment_rate"]


def _process_symbol_worker(args):
    slug, processed_dir_str = args
    processed = Path(processed_dir_str)
    dom_file = processed / f"cleaned_dom_{slug}.parquet"
    trade_file = processed / f"inferred_trades_{slug}.parquet"
    tick_file = processed / f"cleaned_ticks_{slug}.parquet"
    out_file = processed / f"features_1m_{slug}.parquet"

    if not dom_file.exists():
        return ("skip", slug, "no_dom")

    try:
        df_dom = pd.read_parquet(dom_file)
    except Exception as e:
        return ("fail", slug, f"dom_read: {e}")

    try:
        df_trades = pd.read_parquet(trade_file) if trade_file.exists() else pd.DataFrame()
    except Exception as e:
        df_trades = pd.DataFrame()

    if df_trades.empty:
        fallback = processed / f"inferred_trades_{slug.split('_')[0]}.parquet"
        if fallback.exists():
            try:
                df_trades = pd.read_parquet(fallback)
            except Exception:
                pass

    df_tick = None
    if tick_file.exists():
        try:
            df_tick = pd.read_parquet(tick_file)
        except Exception:
            df_tick = None

    from features.feature_factory import FeatureFactory

    if df_tick is not None and len(df_tick) > 0:
        df_features = FeatureFactory.build_feature_matrix(
            df_dom, df_tick=df_tick, df_trades=df_trades,
        )
    else:
        df_features = FeatureFactory.build_feature_matrix(df_dom, df_trades)

    if df_features.empty:
        return ("skip", slug, "empty_features")

    df_features.to_parquet(out_file, engine="pyarrow", compression="snappy")
    return ("ok", slug, len(df_features), df_features['ts'].min(), df_features['ts'].max())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Feature Factory (Parallel)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args, _ = parser.parse_known_args()
    workers = args.workers or os.cpu_count()

    logger.info("=" * 80)
    logger.info("STEP 30 (v2): COMPUTE MICROSTRUCTURE FEATURES [PARALLEL]")
    logger.info("=" * 80)

    processed = Path(PROCESSED_DIR)
    dom_files = sorted(processed.glob("cleaned_dom_*.parquet"))
    if not dom_files:
        logger.error(f"No cleaned DOM files in {PROCESSED_DIR}. Run 10_prepare_data.py first.")
        return

    import pyarrow.parquet as pq
    to_process = []
    for f in dom_files:
        slug = f.stem.replace("cleaned_dom_", "")
        out_file = processed / f"features_1m_{slug}.parquet"
        if out_file.exists():
            try:
                schema = pq.read_schema(out_file)
                if "absorption_buyer_1m" in schema.names:
                    continue
            except Exception:
                pass
        to_process.append((slug, str(processed)))

    logger.info(f"Found {len(dom_files)} DOM files. Processing {len(to_process)} remaining.")

    if not to_process:
        logger.info("All feature files already exist.")
        return

    t0 = time.time()
    total = len(to_process)
    ok_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_symbol_worker, a): a[0] for a in to_process}
        done = 0
        for future in as_completed(futures):
            done += 1
            slug = futures[future]
            try:
                result = future.result()
                status = result[0]
                if status == "ok":
                    _, _, n_rows, ts_min, ts_max = result
                    ok_count += 1
                    if done % 200 == 0 or done == total:
                        elapsed = time.time() - t0
                        logger.info(f"[{done}/{total}] {slug}: {n_rows} bars | {elapsed:.1f}s")
                elif status == "skip":
                    reason = result[2] if len(result) > 2 else ""
                    if done % 200 == 0 or done == total:
                        logger.info(f"[{done}/{total}] {slug}: skipped ({reason})")
                else:
                    _, _, err = result
                    logger.warning(f"[{done}/{total}] {slug}: FAILED - {err}")
            except Exception as e:
                logger.warning(f"[{done}/{total}] {slug}: worker error - {e}")

    elapsed = time.time() - t0
    logger.info(f"\nProcessed {ok_count}/{total} symbols in {elapsed:.1f}s ({elapsed/max(ok_count,1):.2f}s per symbol)")
    logger.info("=" * 80)
    logger.info("FEATURE COMPUTATION COMPLETE [OK]")


if __name__ == "__main__":
    main()
