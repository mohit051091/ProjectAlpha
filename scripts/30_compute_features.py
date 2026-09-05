"""
SCRIPT 30: COMPUTE FEATURES
Load cleaned DOM data and inferred trades, compute Stage 1 & 2 microstructure
features aggregated to 1-minute bars, validate outputs, and save results.

Usage:
    python scripts/30_compute_features.py

Inputs:
    02_processed/cleaned_dom_<symbol>.parquet       (from script 10)
    02_processed/inferred_trades_<symbol>.parquet   (from script 20)

Outputs:
    02_processed/features_1m_<symbol>.parquet

Features Computed:
    Stage 1: delta_1m, delta_5m
    Stage 2: volume_burst, aggressor_ratio, trade_count_burst, large_trade_ratio

Author: Senior Quant Researcher & Lead Developer
Date  : June 6, 2026
"""

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from features.feature_factory import FeatureFactory
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR

logger = setup_logger(__name__)

STAGE1_FEATURES = ["delta_1m", "delta_5m"]
STAGE2_FEATURES = ["volume_burst", "aggressor_ratio", "trade_count_burst", "large_trade_ratio"]
MICROSTRUCTURE_FEATURES = ["iceberg_score", "order_cancel_rate", "bid_replenishment_rate"]
ALL_FEATURES    = STAGE1_FEATURES + STAGE2_FEATURES + MICROSTRUCTURE_FEATURES


def load_pair(symbol_slug: str) -> tuple:
    """
    Load cleaned DOM + inferred trades + (optional) cleaned ticks for a
    given symbol slug (e.g. 'ifci_3_Jun_26' or 'IFCI_29_May_26').

    Returns
    -------
    (df_dom, df_trades, df_tick)
        df_dom     : cleaned DOM snapshots (always present)
        df_trades  : inferred trades (or empty DataFrame if Stage 2 has
                     not been run for this slug)
        df_tick    : real cleaned ticks, or None if not available
                     (R8 backwards-compat: the 2 pre-existing stock-days
                     have no tick file; the existing LTP-inference path
                     is used unchanged)
    """
    processed = Path(PROCESSED_DIR)

    dom_file = processed / f"cleaned_dom_{symbol_slug}.parquet"
    trade_file = processed / f"inferred_trades_{symbol_slug}.parquet"
    tick_file = processed / f"cleaned_ticks_{symbol_slug}.parquet"

    if not dom_file.exists():
        logger.warning(f"No cleaned DOM file found for slug '{symbol_slug}'. Skipping.")
        return None, None, None

    try:
        df_dom = pd.read_parquet(dom_file)
    except Exception as e:
        logger.error(f"Error reading cleaned DOM file '{dom_file.name}': {e}")
        return None, None, None

    try:
        df_trades = pd.read_parquet(trade_file) if trade_file.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error reading inferred trades file '{trade_file.name}': {e}")
        df_trades = pd.DataFrame()

    if not df_trades.empty:
        logger.info(f"  Trades: {trade_file.name}  ({len(df_trades):,} rows)")
    else:
        symbol_short = symbol_slug.split("_")[0]
        fallback = processed / f"inferred_trades_{symbol_short}.parquet"
        if fallback.exists():
            try:
                df_trades = pd.read_parquet(fallback)
                logger.warning(f"  Falling back to symbol-level trades: {fallback.name}")
            except Exception as e:
                logger.warning(f"Error reading fallback trades file '{fallback.name}': {e}")
        else:
            logger.warning("  No inferred trades file found — trade-flow features will be zero-filled.")

    # New: load real ticks if Stage 1 produced them. R8 backwards-compat:
    # for the 2 pre-existing slugs, no tick file exists and we keep the
    # old LTP-inference path (df_trades will be passed as df_tick below).
    df_tick = None
    if tick_file.exists():
        try:
            df_tick = pd.read_parquet(tick_file)
            logger.info(f"  Ticks : {tick_file.name}  ({len(df_tick):,} rows)  [REAL TICKS]")
        except Exception as e:
            logger.warning(f"Error reading ticks file '{tick_file.name}': {e}")
            df_tick = None
    else:
        logger.info("  Ticks : none — will use inferred trades as df_tick (R8 backwards-compat).")

    logger.info(f"  DOM   : {dom_file.name}  ({len(df_dom):,} rows)")
    return df_dom, df_trades, df_tick


def print_feature_sample(df: pd.DataFrame, symbol: str) -> None:
    """Print a human-readable summary of the feature matrix."""
    logger.info(f"\n{'='*70}")
    logger.info(f"FEATURE MATRIX SUMMARY — {symbol}")
    logger.info(f"{'='*70}")
    logger.info(f"  Shape          : {df.shape}")
    logger.info(f"  Time range     : {df['ts'].min()}  ->  {df['ts'].max()}")
    logger.info(f"  1-min bars     : {len(df)}")
    logger.info(f"\n  Stage 1 features (net order flow direction):")
    for col in STAGE1_FEATURES:
        if col in df.columns:
            s = df[col]
            logger.info(f"    {col:18s}: min={s.min():>12,.1f}  max={s.max():>12,.1f}  mean={s.mean():>10,.1f}")
    logger.info(f"\n  Stage 2 features (activity intensity):")
    for col in STAGE2_FEATURES:
        if col in df.columns:
            s = df[col]
            logger.info(f"    {col:22s}: min={s.min():>7.4f}  max={s.max():>7.4f}  mean={s.mean():>7.4f}")

    logger.info(f"\n  Microstructure features:")
    for col in MICROSTRUCTURE_FEATURES:
        if col in df.columns:
            s = df[col]
            logger.info(f"    {col:22s}: min={s.min():>7.4f}  max={s.max():>7.4f}  mean={s.mean():>7.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Feature Factory")
    parser.add_argument("--force", action="store_true", help="Force feature regeneration.")
    args, unknown = parser.parse_known_args()

    logger.info("="*80)
    logger.info("STEP 30: COMPUTE MICROSTRUCTURE FEATURES (Stage 1 & 2)")
    logger.info("="*80)

    processed = Path(PROCESSED_DIR)

    # Discover all cleaned DOM files and derive symbol slugs
    dom_files = sorted(processed.glob("cleaned_dom_*.parquet"))
    if not dom_files:
        logger.error(f"No cleaned DOM files in {PROCESSED_DIR}. Run 10_prepare_data.py first.")
        return

    import pyarrow.parquet as pq

    existing_count = 0
    to_process = []
    for f in dom_files:
        slug = f.stem.replace("cleaned_dom_", "")
        out_file = processed / f"features_1m_{slug}.parquet"
        if out_file.exists() and not args.force:
            try:
                schema = pq.read_schema(out_file)
                if "absorption_buyer_1m" in schema.names:
                    existing_count += 1
                    continue
            except Exception as e:
                pass
        to_process.append((f, slug, out_file))

    logger.info(f"Found {len(dom_files)} DOM files. {existing_count} feature files already exist.")
    logger.info(f"Processing remaining {len(to_process)} file(s).")

    import time
    total = len(to_process)
    start_time = time.time()
    for idx, (dom_file, slug, out_file) in enumerate(to_process, 1):
        elapsed = time.time() - start_time
        if idx % 100 == 0 or idx == total or idx == 1:
            logger.info(f"[PROGRESS] {idx}/{total} symbols | Elapsed: {elapsed:.1f}s ({elapsed/idx:.2f}s per symbol)")

        symbol_short = slug.split("_")[0]
        logger.info(f"\n[{idx}/{total}] Processing: {dom_file.name} (slug={slug})")

        df_dom, df_trades, df_tick = load_pair(slug)
        if df_dom is None:
            continue

        # Build feature matrix.
        # R8 backwards-compat: if no real ticks file exists (the 2
        # pre-existing slugs), pass df_trades as df_tick. The alignment
        # engine only needs `ts, symbol, trade_qty`; inferred trades
        # have the same schema.
        if df_tick is not None and len(df_tick) > 0:
            df_features = FeatureFactory.build_feature_matrix(
                df_dom, df_tick=df_tick, df_trades=df_trades,
            )
        else:
            df_features = FeatureFactory.build_feature_matrix(df_dom, df_trades)

        if df_features.empty:
            logger.error(f"Feature matrix is empty for {symbol_short}. Skipping save.")
            continue

        # Save
        out_file = processed / f"features_1m_{slug}.parquet"
        df_features.to_parquet(out_file, engine="pyarrow", compression="snappy")
        logger.info(f"  Saved feature matrix to: {out_file}")

        # Print summary
        print_feature_sample(df_features, symbol_short.upper())

    logger.info("\n" + "="*80)
    logger.info("FEATURE COMPUTATION COMPLETE [OK]")
    logger.info("="*80)
    logger.info(f"Next step: Run 40_generate_labels.py to assign LONG/SHORT/NO_TRADE labels")


if __name__ == "__main__":
    main()
