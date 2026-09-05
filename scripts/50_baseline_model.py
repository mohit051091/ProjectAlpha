"""
Baseline Model Experiment — 50_baseline_model.py
================================================
Approved methodology per BASELINE_MODEL_DECISION.md.

Purpose:
  Detect whether engineered features contain directional signal for
  predicting 60-minute forward returns at 5% threshold.

Design:
  5-fold stock-level cross-validation
  16 features input (no pre-selection)
  Model A: Logistic Regression (L1) — linear floor
  Model B: Random Forest — non-linear baseline
  Evaluation: MCC per fold, mean ± std across folds

Usage:
  python scripts/50_baseline_model.py

Output:
  results/baseline_results.json       — per-fold and aggregate metrics
  results/baseline_fold_assignments.json — stock→fold mapping
  results/baseline_lr_coefficients.csv  — per-fold LR coefficients
  results/baseline_rf_feature_importance.csv — per-fold RF importance
  results/baseline_confusion_matrices.npz — per-fold confusion matrices
  logs/baseline_model.out.txt         — run log

Dependencies:
  pandas, numpy, scikit-learn, joblib, pathlib, json, time, sys
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json, time, sys, warnings, os
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────

TARGET_LABEL = "label_60m_1pct"

FEATURES = [
    "large_trade_ratio",
    "delta_1m",
    "volume_burst",
    "aggressor_ratio",
    "trade_count_burst",
    "imbalance_top5",
    "spread",
    "depth_drop_bid",
    "depth_drop_ask",
    "vwap_distance",
    "volatility_5m",
    "price_acceleration",
    "iceberg_score",
    "bid_replenishment_rate",
    "absorption_buyer_1m",
    "absorption_buyer_5m",
    "absorption_seller_1m",
    "absorption_seller_5m",
]

N_FOLDS = 5
RANDOM_STATE = 42
SHUFFLE_TEST_SEED = 99

# ─── Paths ──────────────────────────────────────────────────────────────────

PROC = Path("02_processed")
RESULTS = Path("results")
LOGS = Path("logs")
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

log_path = LOGS / "baseline_model.out.txt"
results_path = RESULTS / "baseline_results.json"
fold_path = RESULTS / "baseline_fold_assignments.json"
lr_coef_path = RESULTS / "baseline_lr_coefficients.csv"
rf_imp_path = RESULTS / "baseline_rf_feature_importance.csv"
cm_path = RESULTS / "baseline_confusion_matrices.npz"


# ─── Logging ────────────────────────────────────────────────────────────────

def log(msg, file=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if file:
        print(line, file=file, flush=True)


# ─── Step 1: Build BOTH slug list ───────────────────────────────────────────

def get_both_slugs(proc_dir: Path):
    """Return sorted list of slugs that have BOTH dom+tick data."""
    dom_slugs = {
        f.stem.replace("cleaned_dom_", "")
        for f in proc_dir.glob("cleaned_dom_*.parquet")
    }
    tick_slugs = {
        f.stem.replace("cleaned_ticks_", "")
        for f in proc_dir.glob("cleaned_ticks_*.parquet")
    }
    inf_slugs = {
        f.stem.replace("inferred_trades_", "")
        for f in proc_dir.glob("inferred_trades_*.parquet")
        if f.stem.replace("inferred_trades_", "").count("_") >= 2
    }
    both = sorted(dom_slugs & tick_slugs & inf_slugs)
    return both


# ─── Step 2: Load data ──────────────────────────────────────────────────────

def load_data(slugs: list, proc_dir: Path, columns: list, label: str):
    """
    Load labeled features for all slugs.
    Returns:
        X: np.ndarray (n_rows, n_features)
        y: np.ndarray (n_rows,) — string labels (LONG/SHORT/NO_TRADE)
        slug_idx: np.ndarray (n_rows,) — slug index per row for fold mapping
        slug_list: list — original slug order matching slug_idx values
    """
    all_X = []
    all_y = []
    all_slug_ids = []
    slug_list = []

    load_cols = columns + [label]

    for idx, slug in enumerate(slugs):
        fpath = proc_dir / f"labeled_features_1m_{slug}.parquet"
        if not fpath.exists():
            continue

        df = pd.read_parquet(fpath, columns=load_cols)

        # Drop rows where label or any feature is NaN
        df_clean = df.dropna(subset=[label] + columns)
        if len(df_clean) == 0:
            continue

        all_X.append(df_clean[columns].values)
        all_y.append(df_clean[label].values)
        all_slug_ids.append(np.full(len(df_clean), len(slug_list), dtype=int))
        slug_list.append(slug)

    if len(all_X) == 0:
        return np.empty((0, len(columns))), np.empty(0), np.empty(0, dtype=int), []

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    slug_idx = np.concatenate(all_slug_ids)

    return X, y, slug_idx, slug_list


# ─── Step 3: Assign folds ───────────────────────────────────────────────────

def assign_folds(slug_list: list, n_folds: int, seed: int):
    """
    Randomly assign each slug to a fold.
    Returns dict: slug -> fold_number (0-indexed)
    Also saves to JSON.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(slug_list))
    rng.shuffle(indices)

    fold_size = len(slug_list) // n_folds
    assignments = {}

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(slug_list)
        for pos in indices[start:end]:
            assignments[slug_list[pos]] = fold

    return assignments


def validate_folds(fold_assignments: dict, slug_list: list, n_folds: int):
    """
    Validate fold assignments before use.
    Checks:
      - All slugs are assigned to exactly one fold
      - Fold IDs are 0..n_folds-1
      - Every fold has at least one slug
    """
    assert isinstance(fold_assignments, dict), \
        f"fold_assignments must be dict, got {type(fold_assignments)}"
    assert len(fold_assignments) == len(slug_list), \
        f"Slug count mismatch: {len(fold_assignments)} assigned vs {len(slug_list)} loaded"
    assert set(fold_assignments.keys()) == set(slug_list), \
        "Assigned slugs do not match loaded slugs"

    folds_used = set(fold_assignments.values())
    expected = set(range(n_folds))
    assert folds_used == expected, \
        f"Fold IDs mismatch: expected {sorted(expected)}, got {sorted(folds_used)}"

    for f in range(n_folds):
        count = sum(1 for s in fold_assignments if fold_assignments[s] == f)
        assert count > 0, f"Fold {f} has 0 slugs assigned"

    return True


# ─── Step 4: Per-fold training ──────────────────────────────────────────────

def run_fold(X_train, y_train, X_val, y_val, fold_num, log_fh):
    """Train LR(L1) and RF on one fold. Return metrics dict."""

    # ── Model A: Logistic Regression (L1) ──
    scaler = RobustScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_val_lr = scaler.transform(X_val)

    lr = LogisticRegression(
        l1_ratio=1,
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=50,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    lr.fit(X_train_lr, y_train)
    lr_time = time.time() - t0

    pred_lr = lr.predict(X_val_lr)
    mcc_lr = matthews_corrcoef(y_val, pred_lr)

    lr_coef = {
        "class_" + str(cls): {FEATURES[i]: float(lr.coef_[c, i]) for i in range(len(FEATURES))}
        for c, cls in enumerate(lr.classes_)
    }

    # ── Shuffle-label test (LR) ──
    y_train_shuffled_lr = y_train.copy()
    rng_shuf = np.random.default_rng(SHUFFLE_TEST_SEED + fold_num)
    rng_shuf.shuffle(y_train_shuffled_lr)
    lr_shuf = LogisticRegression(
        l1_ratio=1,
        solver="saga",
        C=1.0,
        class_weight="balanced",
        max_iter=50,
        random_state=RANDOM_STATE,
    )
    lr_shuf.fit(X_train_lr, y_train_shuffled_lr)
    pred_lr_shuf = lr_shuf.predict(X_val_lr)
    mcc_lr_shuf = matthews_corrcoef(y_val, pred_lr_shuf)

    # ── Model B: Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    pred_rf = rf.predict(X_val)
    mcc_rf = matthews_corrcoef(y_val, pred_rf)

    rf_imp = {FEATURES[i]: float(rf.feature_importances_[i]) for i in range(len(FEATURES))}

    # ── Shuffle-label test (RF) ──
    y_train_shuffled_rf = y_train.copy()
    rng_shuf2 = np.random.default_rng(SHUFFLE_TEST_SEED + 100 + fold_num)
    rng_shuf2.shuffle(y_train_shuffled_rf)
    rf_shuf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_shuf.fit(X_train, y_train_shuffled_rf)
    pred_rf_shuf = rf_shuf.predict(X_val)
    mcc_rf_shuf = matthews_corrcoef(y_val, pred_rf_shuf)

    # ── Classification reports ──
    report_lr = classification_report(y_val, pred_lr, output_dict=True, zero_division=0)
    report_rf = classification_report(y_val, pred_rf, output_dict=True, zero_division=0)

    cm_rf = confusion_matrix(y_val, pred_rf, labels=["LONG", "SHORT", "NO_TRADE"])
    cm_lr = confusion_matrix(y_val, pred_lr, labels=["LONG", "SHORT", "NO_TRADE"])

    log(f"  Fold {fold_num+1}/{N_FOLDS} — LR MCC={mcc_lr:.4f} (shuf={mcc_lr_shuf:.4f}) "
        f"RF MCC={mcc_rf:.4f} (shuf={mcc_rf_shuf:.4f}) "
        f"LR={lr_time:.1f}s RF={rf_time:.1f}s | "
        f"train={len(X_train)} val={len(X_val)}", file=log_fh)

    return {
        "fold": fold_num,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "lr_time_s": round(lr_time, 2),
        "rf_time_s": round(rf_time, 2),
        "mcc_lr": round(float(mcc_lr), 6),
        "mcc_rf": round(float(mcc_rf), 6),
        "mcc_lr_shuffled": round(float(mcc_lr_shuf), 6),
        "mcc_rf_shuffled": round(float(mcc_rf_shuf), 6),
        "lr_coef": lr_coef,
        "rf_importance": rf_imp,
        "lr_report": {k: {sk: round(float(sv), 4) for sk, sv in v.items()}
                        if isinstance(v, dict) else round(float(v), 6)
                       for k, v in report_lr.items()},
        "rf_report": {k: {sk: round(float(sv), 4) for sk, sv in v.items()}
                        if isinstance(v, dict) else round(float(v), 6)
                       for k, v in report_rf.items()},
        "confusion_lr": cm_lr.tolist(),
        "confusion_rf": cm_rf.tolist(),
    }


# ─── Step 5: Main ───────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    with open(log_path, "w") as log_fh:
        log("=" * 60, file=log_fh)
        log("BASELINE MODEL EXPERIMENT", file=log_fh)
        log(f"Started: {datetime.now().isoformat()}", file=log_fh)
        log(f"N_FOLDS={N_FOLDS}, TARGET={TARGET_LABEL}, N_FEATURES={len(FEATURES)}", file=log_fh)
        log("=" * 60, file=log_fh)

        # ── Step 1: Build slug list ──
        log("Step 1: Building BOTH slug list...", file=log_fh)
        slugs = get_both_slugs(PROC)
        log(f"  Found {len(slugs)} BOTH slugs", file=log_fh)

        # ── Step 2: Load data ──
        log("Step 2: Loading data...", file=log_fh)
        X, y, slug_idx, slug_list = load_data(slugs, PROC, FEATURES, TARGET_LABEL)
        log(f"  Loaded {len(X)} rows, {X.shape[1]} features, {len(slug_list)} stocks with data", file=log_fh)
        log(f"  Label distribution: LONG={(y=='LONG').sum()}, "
            f"SHORT={(y=='SHORT').sum()}, "
            f"NO_TRADE={(y=='NO_TRADE').sum()}", file=log_fh)

        if len(X) == 0:
            log("  ERROR: No data loaded. Exiting.", file=log_fh)
            return

        # ── Step 3: Assign folds (load or create) ──
        log("Step 3: Assigning 5 folds...", file=log_fh)
        if fold_path.exists():
            with open(fold_path, "r") as f:
                fold_assignments = json.load(f)
            missing = set(slug_list) - set(fold_assignments.keys())
            extra = set(fold_assignments.keys()) - set(slug_list)
            if missing or extra:
                log(f"  Fold file stale: {len(missing)} missing, {len(extra)} extra. Rebuilding.", file=log_fh)
                fold_assignments = assign_folds(slug_list, N_FOLDS, RANDOM_STATE)
            else:
                log(f"  Loaded existing fold assignments from {fold_path}", file=log_fh)
        else:
            fold_assignments = assign_folds(slug_list, N_FOLDS, RANDOM_STATE)
            with open(fold_path, "w") as f:
                json.dump(fold_assignments, f, indent=2)
            log(f"  Fold assignments saved to {fold_path}", file=log_fh)

        validate_folds(fold_assignments, slug_list, N_FOLDS)
        log(f"  Fold validation passed.", file=log_fh)

        # Verify IFCI is in one fold
        ifci_slugs = [s for s in slug_list if "ifci" in s.lower()]
        if len(ifci_slugs) > 1:
            ifci_folds = set(fold_assignments[s] for s in ifci_slugs)
            log(f"  IFCI slugs: {ifci_slugs} -> folds {ifci_folds}", file=log_fh)
            if len(ifci_folds) > 1:
                log("  WARNING: IFCI slugs span multiple folds!", file=log_fh)

        # Log fold sizes
        for fold in range(N_FOLDS):
            fold_slugs = [s for s, f in fold_assignments.items() if f == fold]
            log(f"  Fold {fold+1}: {len(fold_slugs)} stocks", file=log_fh)

        # ── Step 4: Per-fold training ──
        log("Step 4: Running 5-fold CV...", file=log_fh)

        all_results = []
        lr_coef_rows = []
        rf_imp_rows = []
        all_cm_lr = []
        all_cm_rf = []

        for fold in range(N_FOLDS):
            log(f"  --- Fold {fold+1}/{N_FOLDS} ---", file=log_fh)

            # Get slugs for this fold
            val_slugs = {s for s, f in fold_assignments.items() if f == fold}
            train_mask = np.array([slug_list[slug_idx[i]] not in val_slugs for i in range(len(X))])
            val_mask = ~train_mask

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]

            result = run_fold(X_train, y_train, X_val, y_val, fold, log_fh)
            all_results.append(result)

            # Collect coefficients (one row per class per fold)
            for class_name, coef_dict in result["lr_coef"].items():
                row_lr = {"fold": fold, "class": class_name}
                row_lr.update(coef_dict)
                lr_coef_rows.append(row_lr)

            row_rf = {"fold": fold}
            row_rf.update(result["rf_importance"])
            rf_imp_rows.append(row_rf)

            all_cm_lr.append(result["confusion_lr"])
            all_cm_rf.append(result["confusion_rf"])

        # ── Step 5: Aggregate ──
        log("Step 5: Aggregating results...", file=log_fh)

        mcc_lr_vals = [r["mcc_lr"] for r in all_results]
        mcc_rf_vals = [r["mcc_rf"] for r in all_results]
        mcc_lr_shuf = [r["mcc_lr_shuffled"] for r in all_results]
        mcc_rf_shuf = [r["mcc_rf_shuffled"] for r in all_results]

        summary = {
            "experiment": "baseline_model_v1",
            "date": datetime.now().isoformat(),
            "target_label": TARGET_LABEL,
            "n_features": len(FEATURES),
            "features": FEATURES,
            "n_folds": N_FOLDS,
            "n_slugs_total": len(slugs),
            "n_slugs_with_data": len(slug_list),
            "n_rows_total": len(X),
            "model_a": "LogisticRegression(L1, saga, C=1.0)",
            "model_b": "RandomForest(n_estimators=200, max_depth=10, min_samples_leaf=20)",
            "results_per_fold": all_results,
            "aggregate": {
                "mcc_lr_mean": round(float(np.mean(mcc_lr_vals)), 6),
                "mcc_lr_std": round(float(np.std(mcc_lr_vals)), 6),
                "mcc_rf_mean": round(float(np.mean(mcc_rf_vals)), 6),
                "mcc_rf_std": round(float(np.std(mcc_rf_vals)), 6),
                "mcc_lr_shuffled_mean": round(float(np.mean(mcc_lr_shuf)), 6),
                "mcc_rf_shuffled_mean": round(float(np.mean(mcc_rf_shuf)), 6),
                "mcc_lr_values": mcc_lr_vals,
                "mcc_rf_values": mcc_rf_vals,
            },
            "runtime_s": round(time.time() - t_start, 2),
        }

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log(f"  Results saved to {results_path}", file=log_fh)

        # Save coefficients
        df_lr = pd.DataFrame(lr_coef_rows)
        df_lr.to_csv(lr_coef_path, index=False)
        log(f"  LR coefficients saved to {lr_coef_path}", file=log_fh)

        df_rf = pd.DataFrame(rf_imp_rows)
        df_rf.to_csv(rf_imp_path, index=False)
        log(f"  RF feature importances saved to {rf_imp_path}", file=log_fh)

        # Save confusion matrices
        np.savez(cm_path, lr=np.array(all_cm_lr), rf=np.array(all_cm_rf))
        log(f"  Confusion matrices saved to {cm_path}", file=log_fh)

        # ── Final report ──
        log("", file=log_fh)
        log("=" * 60, file=log_fh)
        log("BASELINE EXPERIMENT COMPLETE", file=log_fh)
        log("=" * 60, file=log_fh)
        log(f"  LR(L1) MCC: {summary['aggregate']['mcc_lr_mean']:.4f} ± {summary['aggregate']['mcc_lr_std']:.4f}  "
            f"(shuffled: {summary['aggregate']['mcc_lr_shuffled_mean']:.4f})", file=log_fh)
        log(f"  RF     MCC: {summary['aggregate']['mcc_rf_mean']:.4f} ± {summary['aggregate']['mcc_rf_std']:.4f}  "
            f"(shuffled: {summary['aggregate']['mcc_rf_shuffled_mean']:.4f})", file=log_fh)
        log(f"  Total runtime: {summary['runtime_s']:.1f}s", file=log_fh)
        log(f"  Rows: {len(X)} | Stocks: {len(slug_list)} | Features: {len(FEATURES)}", file=log_fh)
        log("=" * 60, file=log_fh)

    print(f"\nExperiment complete. Results: {results_path}")
    print(f"Log: {log_path}")
    print(f"Fold assignments: {fold_path}")


if __name__ == "__main__":
    main()
