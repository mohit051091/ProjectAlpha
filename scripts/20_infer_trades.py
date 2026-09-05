"""
SCRIPT 20: INFER TRADES
Load cleaned DOM data, reconstruct trade events using the Trade Inference Engine,
and save the inferred trade histories.

Usage:
    python scripts/20_infer_trades.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pandas as pd
from features.trade_inference import TradeInferenceEngine
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR

logger = setup_logger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Trade Inference")
    parser.add_argument("--force", action="store_true", help="Force trade inference regeneration.")
    args, unknown = parser.parse_known_args()

    """Main execution block"""
    logger.info("="*80)
    logger.info("STEP 20: INFER TRADES FROM ORDER BOOK LEVEL CHANGES")
    logger.info("="*80)
    
    processed_path = Path(PROCESSED_DIR)
    
    # Find all cleaned DOM files
    cleaned_files = sorted(processed_path.glob("cleaned_dom_*.parquet"))
    
    if not cleaned_files:
        logger.error(f"No cleaned files found in {PROCESSED_DIR}. Run 10_prepare_data.py first.")
        return
        
    existing_count = 0
    to_process = []
    for f in cleaned_files:
        slug = f.stem.replace("cleaned_dom_", "")
        output_file = processed_path / f"inferred_trades_{slug}.parquet"
        if output_file.exists() and not args.force:
            existing_count += 1
        else:
            to_process.append((f, slug, output_file))

    logger.info(f"Found {len(cleaned_files)} DOM files. {existing_count} inferred trades files already exist.")
    logger.info(f"Processing remaining {len(to_process)} file(s).")
    
    total = len(to_process)
    start_time = time.time()
    for idx, (f, slug, output_file) in enumerate(to_process, 1):
        if idx % 100 == 0 or idx == total or idx == 1:
            elapsed = time.time() - start_time
            logger.info(f"[PROGRESS] {idx}/{total} symbols | Elapsed: {elapsed:.1f}s ({elapsed/idx:.2f}s per symbol)")
        logger.info(f"\n[1] Processing {f.name}...")
        
        # Load cleaned DOM data
        try:
            df_dom = pd.read_parquet(f)
        except Exception as e:
            logger.error(f"Error reading cleaned DOM file '{f.name}': {e}")
            continue
        if len(df_dom) == 0:
            logger.warning(f"Cleaned DOM file '{f.name}' is empty. Skipping trade inference.")
            continue
        symbol = df_dom['symbol'].iloc[0] if 'symbol' in df_dom.columns else slug.split('_')[0].upper()
        logger.info(f"    Loaded {len(df_dom):,} order book snapshots for symbol: {symbol} (slug: {slug})")
        
        # Infer trades
        df_trades = TradeInferenceEngine.infer_trades(df_dom)
        
        df_trades.to_parquet(output_file, engine='pyarrow', compression='snappy')
        logger.info(f"    Saved inferred trades to: {output_file}")
        
        # Compute and log statistics
        total_trades = len(df_trades)
        if total_trades > 0:
            buys = df_trades[df_trades['direction'] == 'BUY']
            sells = df_trades[df_trades['direction'] == 'SELL']
            
            logger.info(f"    Inferred Trade Statistics:")
            logger.info(f"    - Total Trades: {total_trades:,}")
            logger.info(f"    - Buys: {len(buys):,} ({len(buys)/total_trades*100:.1f}%)")
            logger.info(f"    - Sells: {len(sells):,} ({len(sells)/total_trades*100:.1f}%)")
            logger.info(f"    - Total Volume: {df_trades['trade_qty'].sum():,} shares")
            logger.info(f"    - Mean Trade Size: {df_trades['trade_qty'].mean():.1f} shares")
            logger.info(f"    - Max Trade Size: {df_trades['trade_qty'].max():,} shares")
            logger.info(f"    - Price Range: {df_trades['trade_price'].min():.2f} to {df_trades['trade_price'].max():.2f}")
        else:
            logger.warning("    No trades were inferred!")
            
    logger.info("\n" + "="*80)
    logger.info("TRADE INFERENCE COMPLETE [OK]")
    logger.info("="*80)


if __name__ == "__main__":
    main()
