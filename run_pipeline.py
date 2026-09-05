"""
MASTER ORCHESTRATOR: run_pipeline.py
====================================
Unified entry point to run the Order Flow & Pre-Move Detection data pipeline.

Quick reference:

  # First time (no model exists — auto-trains LGBM):
  python run_pipeline.py --use-v2 --rebuild --sweep

  # Add new trading days (keep existing model):
  python run_pipeline.py --use-v2 --skip-split --skip-model --rebuild --sweep

  # Retrain LGBM model + full rebuild + sweep:
  python run_pipeline.py --use-v2 --train --rebuild --sweep

  # Just generate sweep spreadsheet (OOF must already exist):
  python scripts/76_generate_v2_sweeps_v2.py

Stages:
  Stage -1: Auto-convert unified tick parquets to per-symbol files
  Stage 0:  Split day-level parquets into per-symbol raw files
  Stage 1:  Clean DOM and Tick data
  Stage 2:  Reconstruct trade flow histories
  Stage 3:  Build 1-minute feature matrix
  Stage 4:  Generate target return labels
  Stage 5:  Train baseline model
  --train:  Train final LightGBM model (80_train_final_models.py)
  --rebuild: Rebuild OOF predictions (99_rebuild_oof_all_v2.py)
  --sweep:  Generate performance sweep xlsx (76_generate_v2_sweeps_v2.py)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_stage(command: list, cwd: Path, env: dict) -> bool:
    stage_name = command[1] if len(command) > 1 else command[0]
    print(f"\n[RUNNING] {' '.join(command)}")
    t0 = time.time()
    result = subprocess.run(command, cwd=cwd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{stage_name}' failed with exit code: {result.returncode}")
        return False
    print(f"[SUCCESS] Stage finished in {elapsed:.1f}s")
    return True


MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
MONTH_NUM = {v: k for k, v in MONTH_ABBR.items()}
UNIFIED_PATTERN = re.compile(r"ticks_year=(\d+)_month=(\d+)_day=(\d+)_ticks\.parquet")


def detect_new_combined(data_dir: Path, proc_dir: Path) -> list:
    new_days = []
    for f in sorted(data_dir.glob("ticks_year=*_month=*_day=*_ticks.parquet")):
        m = UNIFIED_PATTERN.match(f.name)
        if not m:
            continue
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        slug_date = f"{day:02d}_{MONTH_ABBR[month]}_{year % 100:02d}"
        if list(proc_dir.glob(f"labeled_features_1m_*_{slug_date}.parquet")):
            continue
        new_days.append((year, month, day, slug_date, f))
    return new_days


def main():
    parser = argparse.ArgumentParser(description="Unified pipeline orchestrator.")
    parser.add_argument("--skip-split", action="store_true", help="Skip Stage 0 day-level splitter.")
    parser.add_argument("--skip-model", action="store_true", help="Skip Stage 5 baseline model training.")
    parser.add_argument("--force", action="store_true", help="Force rebuild (overwrite existing files).")
    parser.add_argument("--data-dir", default=None,
                        help="Path to data directory. Default: PROJECTALPHA_DATA env var, else Data/")
    parser.add_argument("--use-v2", action="store_true",
                        help="Use parallel _v2 scripts for Stages 1, 2, 3.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild OOF predictions. Auto-incremental if model unchanged; auto-trains if model missing.")
    parser.add_argument("--sweep", action="store_true",
                        help="After rebuild, generate performance sweep xlsx.")
    parser.add_argument("--train", action="store_true",
                        help="(Re)train final LightGBM model (80_train_final_models.py). Auto-triggered if --rebuild needs it.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()

    # Resolve data directory: env var > --data-dir > default Data/
    data_dir = args.data_dir or os.environ.get("PROJECTALPHA_DATA")
    if data_dir:
        data_dir = Path(data_dir).resolve()
    else:
        data_dir = project_root / "Data"

    print_header("ORDER FLOW & PRE-MOVE DETECTION SYSTEM — PIPELINE ORCHESTRATOR")
    print(f"Project directory : {project_root}")
    print(f"Data directory    : {data_dir}")
    print(f"Force overwrite   : {args.force}")
    print(f"Skip split stage  : {args.skip_split}")
    print(f"Skip model stage  : {args.skip_model}")
    print(f"Use v2 scripts    : {args.use_v2}")
    print(f"Rebuild OOF       : {args.rebuild}")
    print(f"Generate sweep    : {args.sweep}")
    print(f"Train LGBM model  : {args.train}")
    print("=" * 80)

    # Build custom env so subprocesses inherit PROJECTALPHA_DATA
    env = os.environ.copy()
    env["PROJECTALPHA_DATA"] = str(data_dir)

    proc_dir = project_root / "02_processed"

    # --- DYNAMIC WHITELIST UPDATE ---
    print_header("UPDATING SYMBOL WHITELIST FROM DHAN ONLINE")
    wl_result = subprocess.run(
        [sys.executable, "scripts/update_whitelist.py"], cwd=project_root, env=env
    )
    if wl_result.returncode != 0:
        print("\n[ERROR] Whitelist update failed. Exiting pipeline.")
        return 1

    # Pre-flight checks
    if not data_dir.exists():
        print(f"\n[ERROR] Data directory '{data_dir}' does not exist.")
        print("Set PROJECTALPHA_DATA env var or use --data-dir to point to your data folder.")
        return 1

    t_start = time.time()
    stages = []

    # --- STAGE -1: AUTO-DETECT NEW COMBINED PARQUETS ---
    new_days = detect_new_combined(data_dir, proc_dir)
    if new_days:
        print(f"\n[INFO] Found {len(new_days)} new combined tick file(s):")
        for _, _, _, slug, path in new_days:
            print(f"       {path.name} -> {slug}")
        cmd = [sys.executable, "scripts/06_convert_unified_ticks.py", "--all"]
        if args.force:
            cmd.append("--force")
        stages.append(("Stage -1 (Auto-Convert)", cmd))
    else:
        print("\n[INFO] No new combined tick files detected. All caught up.")

    # --- STAGE 0: SPLIT DAY-LEVEL PARQUETS ---
    if not args.skip_split:
        cmd = [sys.executable, "scripts/05_split_day_level.py"]
        if args.force:
            cmd.append("--force")
        stages.append(("Stage 0 (Day Splitter)", cmd))
    else:
        print("\n[INFO] Skipping Stage 0 (Day Splitter) as requested.")

    # --- STAGE 1: DATA CLEANER ---
    s1 = "scripts/10_prepare_data_v2.py" if args.use_v2 else "scripts/10_prepare_data.py"
    cmd_s1 = [sys.executable, s1]
    if args.force:
        cmd_s1.append("--force")
    stages.append(("Stage 1 (Data Cleaner)", cmd_s1))

    # --- STAGE 2: TRADE INFERENCE ---
    s2 = "scripts/20_infer_trades_v2.py" if args.use_v2 else "scripts/20_infer_trades.py"
    cmd_s2 = [sys.executable, s2]
    stages.append(("Stage 2 (Trade Inference)", cmd_s2))

    # --- STAGE 3: FEATURE FACTORY ---
    s3 = "scripts/30_compute_features_v2.py" if args.use_v2 else "scripts/30_compute_features.py"
    cmd_s3 = [sys.executable, s3]
    if args.force:
        cmd_s3.append("--force")
    stages.append(("Stage 3 (Feature Factory)", cmd_s3))

    # --- STAGE 4: LABEL GENERATION ---
    cmd_s4 = [sys.executable, "scripts/40_generate_labels.py"]
    if args.force:
        cmd_s4.append("--force")
    stages.append(("Stage 4 (Label Generation)", cmd_s4))

    # --- STAGE 5: BASELINE MODEL ---
    if not args.skip_model:
        cmd_s5 = [sys.executable, "scripts/50_baseline_model.py"]
        stages.append(("Stage 5 (Baseline Model)", cmd_s5))
    else:
        print("\n[INFO] Skipping Stage 5 (Baseline Model) as requested.")

    # Execute all scheduled stages
    execution_times = []
    for name, cmd in stages:
        print_header(f"RUNNING: {name}")
        t0 = time.time()
        success = run_stage(cmd, project_root, env)
        if not success:
            return 1
        execution_times.append((name, time.time() - t0))

    # --- POST-PIPELINE: TRAIN FINAL LGBM (if requested or needed by rebuild) ---
    model_path = project_root / "models" / "lgbm_model_60m_1pct_final.txt"
    needs_model = args.train or (args.rebuild and not model_path.exists())

    if needs_model:
        if args.train:
            print_header("POST-PIPELINE: TRAINING FINAL LIGHTGBM MODEL (--train flag)")
        else:
            print_header("POST-PIPELINE: NO MODEL FOUND — AUTO-TRAINING LIGHTGBM FOR --rebuild")
        cmd_t = [sys.executable, "scripts/80_train_final_models.py"]
        t0 = time.time()
        success = run_stage(cmd_t, project_root, env)
        if not success:
            return 1
        execution_times.append(("Train Final LGBM", time.time() - t0))

    # --- POST-PIPELINE: REBUILD OOF ---
    if args.rebuild:
        # Full rebuild if model was just trained (--train or auto), else incremental
        if needs_model:
            print_header("POST-PIPELINE: REBUILD OOF (FULL — after model training)")
            cmd_rb = [sys.executable, "scripts/99_rebuild_oof_all_v2.py"]
        else:
            print_header("POST-PIPELINE: REBUILD OOF (INCREMENTAL — model unchanged)")
            cmd_rb = [sys.executable, "scripts/99_rebuild_oof_all_v2.py", "--incremental"]
        t0 = time.time()
        success = run_stage(cmd_rb, project_root, env)
        if not success:
            return 1
        execution_times.append(("OOF Rebuild", time.time() - t0))

    # --- POST-PIPELINE: GENERATE SWEEP ---
    if args.sweep:
        print_header("POST-PIPELINE: GENERATE PERFORMANCE SWEEP")
        cmd_sw = [sys.executable, "scripts/76_generate_v2_sweeps_v2.py"]
        t0 = time.time()
        success = run_stage(cmd_sw, project_root, env)
        if not success:
            return 1
        execution_times.append(("Performance Sweep", time.time() - t0))

    # Final summary
    print_header("PIPELINE EXECUTION SUMMARY")
    print(f"Total pipeline elapsed time: {time.time() - t_start:.1f}s\n")
    print(f"{'Pipeline Stage':<30} | {'Elapsed Time (s)':<18}")
    print("-" * 52)
    for name, elapsed in execution_times:
        print(f"{name:<30} | {elapsed:>15.1f}s")
    print("=" * 80)
    print("[OK] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
