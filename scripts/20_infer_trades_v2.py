"""
SCRIPT 20 v2: INFER TRADES (PARALLEL)
Uses ProcessPoolExecutor to process symbols in parallel.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from features.trade_inference import TradeInferenceEngine
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR

logger = setup_logger(__name__)


def _infer_trades_worker(args):
    dom_path_str, output_path_str = args
    dom_path = Path(dom_path_str)
    output_path = Path(output_path_str)
    slug = dom_path.stem.replace("cleaned_dom_", "")
    try:
        df_dom = pd.read_parquet(dom_path)
        if len(df_dom) == 0:
            return ("empty", slug)
        symbol = df_dom['symbol'].iloc[0] if 'symbol' in df_dom.columns else slug.split('_')[0].upper()
        df_trades = TradeInferenceEngine.infer_trades(df_dom)
        df_trades.to_parquet(output_path, engine='pyarrow', compression='snappy')
        return ("ok", slug, symbol, len(df_trades),
                len(df_trades[df_trades['direction'] == 'BUY']),
                len(df_trades[df_trades['direction'] == 'SELL']),
                int(df_trades['trade_qty'].sum()),
                float(df_trades['trade_qty'].mean()),
                int(df_trades['trade_qty'].max()),
                float(df_trades['trade_price'].min()),
                float(df_trades['trade_price'].max()))
    except Exception as e:
        return ("fail", slug, str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Trade Inference (Parallel)")
    parser.add_argument("--force", action="store_true", help="Force regeneration.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args, _ = parser.parse_known_args()
    workers = args.workers or os.cpu_count()

    logger.info("=" * 80)
    logger.info("STEP 20 (v2): INFER TRADES FROM ORDER BOOK LEVEL CHANGES [PARALLEL]")
    logger.info("=" * 80)

    processed_path = Path(PROCESSED_DIR)
    cleaned_files = sorted(processed_path.glob("cleaned_dom_*.parquet"))
    if not cleaned_files:
        logger.error(f"No cleaned DOM files in {PROCESSED_DIR}. Run 10_prepare_data.py first.")
        return

    to_process = []
    for f in cleaned_files:
        slug = f.stem.replace("cleaned_dom_", "")
        output_file = processed_path / f"inferred_trades_{slug}.parquet"
        if output_file.exists() and not args.force:
            continue
        to_process.append((str(f), str(output_file)))

    logger.info(f"Found {len(cleaned_files)} DOM files. Processing {len(to_process)} remaining.")
    if not to_process:
        logger.info("All inferred trades files already exist.")
        return

    t0 = time.time()
    total = len(to_process)
    ok_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_infer_trades_worker, a): Path(a[0]).name for a in to_process}
        done = 0
        for future in as_completed(futures):
            done += 1
            fname = futures[future]
            try:
                result = future.result()
                status = result[0]
                if status == "ok":
                    _, slug, symbol, n_trades, n_buys, n_sells, vol, mean_sz, max_sz, pmin, pmax = result
                    ok_count += 1
                    if done % 200 == 0 or done == total:
                        elapsed = time.time() - t0
                        logger.info(f"[{done}/{total}] {slug}: {n_trades:,} trades | {elapsed:.1f}s")
                elif status == "empty":
                    logger.warning(f"[{done}/{total}] {fname}: empty DOM, skipped")
                else:
                    _, slug, err = result
                    logger.warning(f"[{done}/{total}] {slug}: FAILED - {err}")
            except Exception as e:
                logger.warning(f"[{done}/{total}] {fname}: worker error - {e}")

    elapsed = time.time() - t0
    logger.info(f"\nProcessed {ok_count}/{total} files in {elapsed:.1f}s ({elapsed/max(ok_count,1):.2f}s per symbol)")
    logger.info("=" * 80)
    logger.info("TRADE INFERENCE COMPLETE [OK]")


if __name__ == "__main__":
    main()
