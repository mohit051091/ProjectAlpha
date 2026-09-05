"""Process a single trading day through stages 1-4."""
import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from raw.data_cleaner import DataCleaner
from features.trade_inference import TradeInferenceEngine
from features.feature_factory import FeatureFactory
from labels import build_label_matrix, summarize_all_label_configs
from utils.logger import setup_logger
from utils.constants import PROCESSED_DIR, DATA_DIR, LABEL_HORIZONS, LABEL_THRESHOLDS

logger = setup_logger(__name__)
con = duckdb.connect()

def parse_date_label(filepath: Path) -> str:
    stem = filepath.stem
    for prefix in ("dom_", "tick_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem

def stage_1(day_label: str):
    logger.info("="*80)
    logger.info(f"STAGE 1: Cleaning raw files for {day_label}")
    logger.info("="*80)
    proc = Path(PROCESSED_DIR)
    raw_dir = Path(DATA_DIR) / "raw"

    dom_out, tick_out = 0, 0
    for prefix, cleaner_method, out_prefix in [
        ("dom", DataCleaner.clean_dom_file_duckdb, "cleaned_dom"),
        ("tick", DataCleaner.clean_ticks_file_duckdb, "cleaned_ticks"),
    ]:
        files = sorted(raw_dir.glob(f"{prefix}_*_{day_label}.parquet"))
        logger.info(f"  {prefix.upper()}: {len(files)} files to process")
        for fp in files:
            slug = parse_date_label(fp)
            out_path = proc / f"{out_prefix}_{slug}.parquet"
            if out_path.exists():
                logger.info(f"  Skipping {fp.name} — already exists")
                continue
            try:
                r, c = cleaner_method(fp, out_path, label=fp.name)
                if prefix == "dom":
                    dom_out += 1
                else:
                    tick_out += 1
                logger.info(f"  {fp.name}: {r} raw -> {c} clean")
            except Exception as e:
                logger.warning(f"  FAILED {fp.name}: {e}")
    logger.info(f"Stage 1 done: {dom_out} dom + {tick_out} tick cleaned")

def stage_2(day_label: str):
    logger.info("="*80)
    logger.info(f"STAGE 2: Inferring trades for {day_label}")
    logger.info("="*80)
    proc = Path(PROCESSED_DIR)
    dom_files = sorted(proc.glob(f"cleaned_dom_*_{day_label}.parquet"))
    if not dom_files:
        logger.warning("No DOM files found for day")
        return
    logger.info(f"  {len(dom_files)} DOM files")
    engine = TradeInferenceEngine()
    processed = 0
    for fp in dom_files:
        slug = fp.stem.replace("cleaned_dom_", "")
        out_path = proc / f"inferred_trades_{slug}.parquet"
        if out_path.exists() and "--force" not in sys.argv:
            continue
        df = pd.read_parquet(fp)
        trades = engine.infer_trades(df)
        trades.to_parquet(out_path, engine="pyarrow", compression="snappy")
        processed += 1
        if processed % 100 == 0:
            logger.info(f"  {processed}/{len(dom_files)} done")
    logger.info(f"Stage 2 done: {processed} inferred")

def stage_3(day_label: str):
    logger.info("="*80)
    logger.info(f"STAGE 3: Computing features for {day_label}")
    logger.info("="*80)
    proc = Path(PROCESSED_DIR)
    dom_files = sorted(proc.glob(f"cleaned_dom_*_{day_label}.parquet"))
    logger.info(f"  {len(dom_files)} symbols")
    factory = FeatureFactory()
    processed = 0
    for fp in dom_files:
        slug = fp.stem.replace("cleaned_dom_", "")
        out_path = proc / f"features_1m_{slug}.parquet"
        if out_path.exists() and "--force" not in sys.argv:
            continue
        trades_path = proc / f"inferred_trades_{slug}.parquet"
        tick_path = proc / f"cleaned_ticks_{slug}.parquet"
        df_dom = pd.read_parquet(fp)
        df_trades = pd.read_parquet(trades_path) if trades_path.exists() else None
        df_tick = pd.read_parquet(tick_path) if tick_path.exists() else None
        df_feat = factory.build_feature_matrix(df_dom, df_tick, df_trades)
        df_feat.to_parquet(out_path, engine="pyarrow", compression="snappy")
        processed += 1
        if processed % 100 == 0:
            logger.info(f"  {processed}/{len(dom_files)} done")
    logger.info(f"Stage 3 done: {processed} feature files")

def stage_4(day_label: str):
    logger.info("="*80)
    logger.info(f"STAGE 4: Generating labels for {day_label}")
    logger.info("="*80)
    proc = Path(PROCESSED_DIR)
    feat_files = sorted(proc.glob(f"features_1m_*_{day_label}.parquet"))
    logger.info(f"  {len(feat_files)} feature files")
    horizons = LABEL_HORIZONS
    thresholds = LABEL_THRESHOLDS
    processed = 0
    for fp in feat_files:
        slug = fp.stem.replace("features_1m_", "")
        out_path = proc / f"labeled_features_1m_{slug}.parquet"
        if out_path.exists() and "--force" not in sys.argv:
            continue
        df = pd.read_parquet(fp)
        if len(df) == 0 or 'ltp' not in df.columns:
            logger.warning(f'  Skipping {slug} — empty or missing ltp')
            pd.DataFrame({'ts': pd.Series(dtype='datetime64[us, UTC]')}).to_parquet(out_path, engine='pyarrow', compression='snappy')
            processed += 1
            continue
        df_labeled = build_label_matrix(df, horizons=horizons, thresholds=thresholds)
        df_labeled.to_parquet(out_path, engine='pyarrow', compression='snappy')
        processed += 1
        if processed % 100 == 0:
            logger.info(f"  {processed}/{len(feat_files)} done")
    logger.info(f"Stage 4 done: {processed} labeled files")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("day", help="Day label, e.g. 25_Jun_26")
    parser.add_argument("--skip-1", action="store_true")
    parser.add_argument("--skip-2", action="store_true")
    parser.add_argument("--skip-3", action="store_true")
    parser.add_argument("--skip-4", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    if not args.skip_1:
        stage_1(args.day)
        gc.collect()
    else:
        logger.info("Skipping Stage 1")
    if not args.skip_2:
        stage_2(args.day)
        gc.collect()
    if not args.skip_3:
        stage_3(args.day)
        gc.collect()
    if not args.skip_4:
        stage_4(args.day)
        gc.collect()
    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed/60:.1f} min")
