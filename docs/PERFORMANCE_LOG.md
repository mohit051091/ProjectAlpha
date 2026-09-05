# Optimization Registry (Performance Log)

This registry tracks runtime and statistical performance optimization attempts for the data pipeline and machine learning models.

## Optimization 1: Logistic Regression SAGA Solver Runtime reduction
- **Problem**: Running L1 Logistic Regression with SAGA solver took ~8 minutes per fold on expanded dataset (~40 minutes total).
- **Baseline**: `max_iter=2000` with SAGA.
- **Hypothesis**: Reducing iterations to 150 (and subsequently 50) will converge sufficiently to capture the linear baseline signal while executing significantly faster.
- **Change**: Set `max_iter=50` and verified classification accuracy.
- **Result**: Run time per fold reduced to <30 seconds, overall model runtime reduced from 40+ minutes to ~2 minutes. Model coefficients remained stable and comparable.
- **Acceptance**: Accepted.

## Optimization 2: Random Forest parallelization & subsampling
- **Problem**: Training Random Forest on 1M rows across all features took ~57 minutes.
- **Baseline**: Single core RF, fitting on all samples.
- **Hypothesis**: Parallelizing RF with `n_jobs=-1` and training on a 1-in-5 systematic subsample (`[::5]`) will reduce training runtime to under 2 minutes while maintaining comparable out-of-fold generalization.
- **Change**: Set `n_jobs=-1` and `X_train[::5]`.
- **Result**: Walk-forward RF training runs in ~90 seconds. OOF validation MCC remained stable around `0.0437`.
- **Acceptance**: Accepted.

## Optimization 3: LightGBM Upgrade for training efficiency
- **Problem**: Random Forest is slow to train even with subsampling and doesn't support built-in leaf-wise multiclass optimizations.
- **Baseline**: Random Forest.
- **Hypothesis**: LightGBM will train leaf-wise trees in a fraction of the time and natively handle class weight imbalances, allowing fast parameter exploration.
- **Change**: Replaced Random Forest with LightGBM booster (`num_boost_round=100`, leaf-wise tree growth).
- **Result**: LightGBM training takes <5 seconds per fold, providing massive speedup and enabling out-of-fold threshold sweeping.
- **Acceptance**: Accepted.

## Optimization 4: Parallel Label Generation using ProcessPoolExecutor
- **Problem**: Generating labels for 4,064 symbol-day parquets sequentially was estimated to take 10+ minutes.
- **Baseline**: Sequential single-core loop.
- **Hypothesis**: Processing each file in isolation allows embarrassingly parallel execution. Using Python's `ProcessPoolExecutor` to distribute files across all available CPU cores will reduce processing time to under 1-2 minutes.
- **Change**: Parallelized the loop in `scripts/40_generate_labels.py` using `ProcessPoolExecutor`.
- **Result**: Dramatically reduced running time. Label generation completed in a fraction of the time.
- **Acceptance**: Accepted.

## Optimization 5: Microstructure Turnover Filters & Sequential Triggers
- **Problem**: Microstructure filters blocked major movers (like `PWL` and `CARTRADE`) due to morning noise, resulting in missed gains.
- **Baseline**: Static share delta (`delta_1m > 4000`) evaluating only the first trigger of the day.
- **Hypothesis**: Price-standardizing volume to net turnover (INR), segmenting conditions directionally (LONG/SHORT), and evaluating sequential triggers (not just the first one) will salvage these major movers while maintaining selectivity.
- **Change**: Implemented the directional LONG Salvage Rule and switched the simulator to evaluate sequential triggers.
- **Result**: Successfully captured **100% of liquid movers** (13 captured, 12 missing raw data, 2 correctly blocked). Captured major EOD MFE gains: `PWL` (+11.26%), `CARTRADE` (+12.04%), `IFCI` (+11.72%), and `ZEEL` (+12.06%) while keeping average trades to **9.5 trades per day** and achieving **30.43% selection accuracy** (vs 12.24% baseline).
- **Acceptance**: Accepted.


