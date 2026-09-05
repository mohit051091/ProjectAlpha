"""SCRIPT 40: GENERATE LABELS AND RUN FEATURE VS LABEL ANALYSIS

Usage:
    python scripts/40_generate_labels.py

Inputs:
    02_processed/features_1m_<slug>.parquet

Outputs:
    02_processed/labeled_features_1m_<slug>.parquet
    Summary printed to stdout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from labels import build_label_matrix, summarize_all_label_configs
from utils.constants import (
    PROCESSED_DIR,
    ACTIVE_FEATURES,
    EXPERIMENTAL_FEATURES,
    LABEL_HORIZONS,
    LABEL_THRESHOLDS,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def process_single_file(args_tuple):
    """Worker function to process a single feature file in parallel"""
    feature_file, slug, output_file, horizons, thresholds = args_tuple
    try:
        df_features = pd.read_parquet(feature_file)
        df_labels = build_label_matrix(df_features, horizons=horizons, thresholds=thresholds)
        df_labels.to_parquet(output_file, engine='pyarrow', compression='snappy')
        return True
    except Exception as e:
        return f"Error on {slug}: {str(e)}"


def print_label_summary(summaries):
    for summary in summaries:
        logger.info("\n--- Label config: %s @ %s%% ---" % (
            f"{summary['horizon_min']}m", int(summary['threshold'] * 100)
        ))
        logger.info(f"Total rows: {summary['total_rows']}")
        logger.info(f"Available labels: {summary['available_labels']}")
        logger.info(f"Unavailable labels: {summary['unavailable_labels']}")
        logger.info(f"Availability reason: {summary['availability_reason']}")
        logger.info(f"Counts: {summary['counts']}")
        logger.info(f"Percentages: {summary['percentages']}")


def feature_vs_label_analysis(df: pd.DataFrame, label_col: str, feature_cols: list):
    if label_col not in df.columns:
        logger.warning(f"Label column not found: {label_col}")
        return

    label_data = df[df[label_col].notna()].copy()
    if label_data.empty:
        logger.warning(f"No available labels for {label_col}.")
        return

    grouped = label_data.groupby(label_col)[feature_cols].mean()
    logger.info(f"\nFeature-vs-label means for {label_col}:")
    logger.info(grouped.to_string(float_format='%.6f'))


def main():
    import argparse
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    parser = argparse.ArgumentParser(description="Stage 4: Label Generation")
    parser.add_argument("--force", action="store_true", help="Force label regeneration.")
    args, unknown = parser.parse_known_args()

    processed = Path(PROCESSED_DIR)
    feature_files = sorted(processed.glob('features_1m_*.parquet'))

    if not feature_files:
        logger.error(f"No feature files found in {PROCESSED_DIR}. Run scripts/30_compute_features.py first.")
        return

    logger.info(f"Found {len(feature_files)} feature file(s) for label generation.")
    
    import pyarrow.parquet as pq

    existing_count = 0
    to_process = []
    for f in feature_files:
        slug = f.stem.replace('features_1m_', '')
        output_file = processed / f'labeled_features_1m_{slug}.parquet'
        if output_file.exists() and not args.force:
            try:
                schema = pq.read_schema(output_file)
                if "absorption_buyer_1m" in schema.names:
                    existing_count += 1
                    continue
            except Exception as e:
                pass
        to_process.append((f, slug, output_file))

    logger.info(f"  {existing_count} labeled files already exist. Processing remaining {len(to_process)} file(s).")

    if not to_process:
        logger.info("No files to process.")
        return

    # Package tasks
    tasks = [(f, slug, out_f, LABEL_HORIZONS, LABEL_THRESHOLDS) for f, slug, out_f in to_process]
    
    # Run in parallel
    num_workers = min(multiprocessing.cpu_count(), len(tasks))
    logger.info(f"Launching parallel label generation with {num_workers} workers...")
    
    success_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res is True:
                success_count += 1
            else:
                errors.append(res)
            
            if idx % 200 == 0 or idx == len(tasks):
                logger.info(f"[{idx}/{len(tasks)}] Labeled files complete...")

    if errors:
        logger.warning(f"Failed to process {len(errors)} files. Sample errors: {errors[:5]}")
        
    # Analyze and print summary for the last successful file processed
    last_file = to_process[-1][2]
    if last_file.exists():
        df_labels = pd.read_parquet(last_file)
        summaries = summarize_all_label_configs(df_labels, horizons=LABEL_HORIZONS, thresholds=LABEL_THRESHOLDS)
        print_label_summary(summaries)

        logger.info(f"\nActive features used in analysis (excluding experimental): {ACTIVE_FEATURES}")
        for horizon in LABEL_HORIZONS:
            for threshold in LABEL_THRESHOLDS:
                label_col = f'label_{horizon}m_{int(threshold*100)}pct'
                feature_vs_label_analysis(df_labels, label_col, ACTIVE_FEATURES)
                
    logger.info(f'\nLabel generation complete. Successfully processed {success_count} files.')

    logger.info('\nLabel generation and analysis complete.')


if __name__ == '__main__':
    main()
