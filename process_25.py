"""Focused pipeline for Jun 25 only — clean dom+tick, infer trades, features, labels."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from raw.data_cleaner import DataCleaner
from features.trade_inference import TradeInferenceEngine
from features.feature_factory import FeatureFactory
from labels import build_label_matrix
from utils.constants import PROCESSED_DIR, LABEL_HORIZONS, LABEL_THRESHOLDS
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger("process_25")

processed = Path(PROCESSED_DIR)
data_raw = Path("Data/raw")
date = "25_Jun_26"

# ---- Stage 1: Clean DOM + TICK ----
dom_files = sorted(data_raw.glob(f"dom_*_{date}.parquet"))
logger.info(f"DOM files: {len(dom_files)}")
for f in dom_files:
    slug = f.stem.replace("dom_", "")
    out = processed / f"cleaned_dom_{slug}.parquet"
    if out.exists():
        continue
    try:
        DataCleaner.clean_dom_file_duckdb(f, out, label=f.name)
    except Exception as e:
        logger.warning(f"DOM FAILED: {slug}: {e}")

tick_files = sorted(data_raw.glob(f"tick_*_{date}.parquet"))
logger.info(f"TICK files: {len(tick_files)}")
for f in tick_files:
    slug = f.stem.replace("tick_", "")
    out = processed / f"cleaned_ticks_{slug}.parquet"
    if out.exists():
        continue
    try:
        DataCleaner.clean_ticks_file_duckdb(f, out, label=f.name)
    except Exception as e:
        logger.warning(f"TICK FAILED: {slug}: {e}")

# ---- Stage 2: Infer trades ----
dom_files = sorted(processed.glob(f"cleaned_dom_*_{date}.parquet"))
logger.info(f"Inferring trades for {len(dom_files)} symbols")
for f in dom_files:
    slug = f.stem.replace("cleaned_dom_", "")
    out = processed / f"inferred_trades_{slug}.parquet"
    if out.exists():
        continue
    try:
        df = pd.read_parquet(f)
        df_trades = TradeInferenceEngine.infer_trades(df)
        df_trades.to_parquet(out, engine="pyarrow", compression="snappy")
    except Exception as e:
        logger.warning(f"TRADES FAILED: {slug}: {e}")

# ---- Stage 3: Features ----
dom_files = sorted(processed.glob(f"cleaned_dom_*_{date}.parquet"))
logger.info(f"Computing features for {len(dom_files)} symbols")
for f in dom_files:
    slug = f.stem.replace("cleaned_dom_", "")
    trades_file = processed / f"inferred_trades_{slug}.parquet"
    tick_file = processed / f"cleaned_ticks_{slug}.parquet"
    out = processed / f"features_1m_{slug}.parquet"
    if out.exists() or not trades_file.exists():
        continue
    try:
        df_dom = pd.read_parquet(f)
        df_trades = pd.read_parquet(trades_file)
        if tick_file.exists():
            df_tick = pd.read_parquet(tick_file)
            df_feat = FeatureFactory.build_feature_matrix(df_dom, df_tick=df_tick, df_trades=df_trades)
        else:
            df_feat = FeatureFactory.build_feature_matrix(df_dom, df_trades)
        df_feat.to_parquet(out, engine="pyarrow", compression="snappy")
    except Exception as e:
        logger.warning(f"FEATURES FAILED: {slug}: {e}")

# ---- Stage 4: Labels ----
feat_files = sorted(processed.glob(f"features_1m_*_{date}.parquet"))
logger.info(f"Generating labels for {len(feat_files)} symbols")
for f in feat_files:
    slug = f.stem.replace("features_1m_", "")
    out = processed / f"labeled_features_1m_{slug}.parquet"
    if out.exists():
        continue
    try:
        df_feat = pd.read_parquet(f)
        df_lab = build_label_matrix(df_feat, horizons=LABEL_HORIZONS, thresholds=LABEL_THRESHOLDS)
        df_lab.to_parquet(out, engine="pyarrow", compression="snappy")
    except Exception as e:
        logger.warning(f"LABELS FAILED: {slug}: {e}")

# ---- Cleanup: delete cleaned files for Jun 25 to free space ----
import glob
for f in glob.glob(f"02_processed/cleaned_*_{date}.parquet"):
    os.remove(f)
logger.info(f"Cleaned up intermediate files for {date}")

logger.info("All done.")
