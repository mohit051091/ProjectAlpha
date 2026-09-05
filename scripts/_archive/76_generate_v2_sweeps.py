"""
Script 76: Generate threshold-pair sweep — Double ROC(3) state machine.
=======================================================================
Reads OOF predictions parquet, runs the full Double ROC(3) strategy
state machine for 8 threshold pairs, outputs single Excel with 8
Strategy_Sweep sheets + Summary, matching v2 schema.

Usage:
    python scripts/76_generate_v2_sweeps.py

Output:
    results/performance_sweeps.xlsx — 9 sheets (8 configs + Summary)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os

import numpy as np
import pandas as pd
from utils.logger import setup_logger

logger = setup_logger(__name__)

ROC_PERIOD = 3
ROC_THRESH = 1.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_LABEL = os.getenv("AR_TARGET_LABEL", "label_60m_1pct").strip()
OOF_PATH = Path(os.getenv("AR_OOF_PATH", "")).expanduser().resolve() if os.getenv("AR_OOF_PATH", "").strip() else (PROJECT_ROOT / "results" / f"oof_predictions_lgbm_{TARGET_LABEL}.parquet")
OUTPUT_PATH = Path(os.getenv("AR_SWEEP_OUTPUT", "")).expanduser().resolve() if os.getenv("AR_SWEEP_OUTPUT", "").strip() else (
    (PROJECT_ROOT / "results" / "performance_sweeps.xlsx")
    if TARGET_LABEL == "label_60m_1pct"
    else (PROJECT_ROOT / "results" / f"performance_sweeps_{TARGET_LABEL}.xlsx")
)

THRESHOLD_PAIRS = [
    (0.20, 0.30, "L20_S30"),
    (0.25, 0.30, "L25_S30"),
    (0.28, 0.30, "L28_S30"),
    (0.30, 0.30, "L30_S30"),
    (0.20, 0.25, "L20_S25"),
    (0.25, 0.25, "L25_S25"),
    (0.28, 0.25, "L28_S25"),
    (0.30, 0.25, "L30_S25"),
    (0.28, 0.28, "L28_S28"),
    (0.15, 0.15, "L15_S15"),
    (0.20, 0.20, "L20_S20"),
]


def compute_day_metrics(df_sorted):
    ltp = df_sorted["ltp"].values
    open_val = df_sorted["open_price"].iloc[0] if "open_price" in df_sorted.columns and not pd.isna(df_sorted["open_price"].iloc[0]) else ltp[0]
    return {
        "open": float(open_val),
        "close": float(ltp[-1]),
        "high": float(np.max(ltp)),
        "low": float(np.min(ltp)),
        "intraday_range_pct": float((np.max(ltp) - np.min(ltp)) / np.min(ltp) * 100),
    }


def run_direction_state_machine(df_sorted, theta, direction, day_met):
    n = len(df_sorted)
    slug = df_sorted["slug"].iloc[0]
    symbol = df_sorted["symbol"].iloc[0]
    date = df_sorted["date"].iloc[0]

    ts_str = df_sorted["ts_ist_str"].values
    ltp = df_sorted["ltp"].values
    roc = df_sorted["roc_3"].values
    delta = df_sorted["delta_1m"].values

    if direction == "LONG":
        prob = df_sorted["prob_long"].values
        roc_direction = 1.0
    else:
        prob = df_sorted["prob_short"].values
        roc_direction = -1.0

    max_prob_val = float(np.max(prob))
    max_prob_idx = int(np.argmax(prob))
    max_prob_time = ts_str[max_prob_idx]

    row_out = {
        "slug": slug,
        "symbol": symbol,
        "date": date,
        "direction": direction,
        "max_prob_long": float(np.max(df_sorted["prob_long"].values)),
        "max_prob_short": float(np.max(df_sorted["prob_short"].values)),
        "max_prob_time": max_prob_time,
        "theta_caught_time": None,
        "open": day_met["open"],
        "close": day_met["close"],
        "high": day_met["high"],
        "low": day_met["low"],
        "intraday_range_pct": day_met["intraday_range_pct"],
        "mark1_ts": None,
        "between_ts": None,
        "max_stage": "IDLE",
        "rejection_reason": "NO_ALERT",
        "entry_time_ist": None,
        "entry_price": None,
        "exit_price": None,
        "eod_pnl_pct": None,
        "mfe_pct": None,
        "mae_pct": None,
        "roc_at_entry": None,
        "daily_open_passed": False,
        "bypass_used": False,
        "delta_value_lakhs_at_theta": None,
        "delta_value_lakhs_at_max_prob": None,
    }

    state = "IDLE"
    theta_caught_idx = -1
    mark1_idx = -1
    between_idx = -1
    triggered_idx = -1
    roc_crossed = False

    for t in range(n):
        if state == "IDLE":
            if not np.isnan(prob[t]) and prob[t] >= theta:
                state = "ARMING"
                theta_caught_idx = t

        elif state == "ARMING":
            if not np.isnan(roc[t]):
                if (direction == "LONG" and roc[t] >= ROC_THRESH) or \
                   (direction == "SHORT" and roc[t] <= -ROC_THRESH):
                    state = "MARK1"
                    mark1_idx = t
                    roc_crossed = True

        elif state == "MARK1":
            if not np.isnan(roc[t]):
                below = (direction == "LONG" and roc[t] < ROC_THRESH) or \
                        (direction == "SHORT" and roc[t] > -ROC_THRESH)
                if below:
                    state = "BETWEEN"
                    between_idx = t

        elif state == "BETWEEN":
            if not np.isnan(roc[t]):
                if (direction == "LONG" and roc[t] >= ROC_THRESH) or \
                   (direction == "SHORT" and roc[t] <= -ROC_THRESH):
                    state = "TRIGGERED"
                    triggered_idx = t
                    break

    daily_open = day_met["open"]
    daily_open_passed = (ltp[t] > daily_open) if direction == "LONG" else (ltp[t] < daily_open)

    row_out["theta_caught_time"] = ts_str[theta_caught_idx] if theta_caught_idx >= 0 else None

    if state == "IDLE":
        row_out["max_stage"] = "IDLE"
        row_out["rejection_reason"] = "NO_ALERT"
        row_out["daily_open_passed"] = False

    elif state == "ARMING":
        row_out["max_stage"] = "ARMING"
        row_out["rejection_reason"] = "MARK1_NEVER"
        row_out["daily_open_passed"] = False

    elif state == "MARK1":
        row_out["max_stage"] = "MARK1"
        row_out["rejection_reason"] = "BETWEEN_NEVER"
        row_out["mark1_ts"] = ts_str[mark1_idx]
        row_out["daily_open_passed"] = False

    elif state == "BETWEEN":
        row_out["max_stage"] = "BETWEEN"
        row_out["rejection_reason"] = "MARK2_NEVER"
        row_out["mark1_ts"] = ts_str[mark1_idx]
        row_out["between_ts"] = ts_str[between_idx]
        row_out["daily_open_passed"] = False

    elif state == "TRIGGERED":
        entry_price = ltp[triggered_idx]
        exit_price = day_met["close"]
        exit_idx = n - 1

        if direction == "LONG":
            eod_pnl = (exit_price - entry_price) / entry_price
            future = ltp[triggered_idx:]
            mfe = (np.max(future) - entry_price) / entry_price
            mae = (np.min(future) - entry_price) / entry_price
        else:
            eod_pnl = (entry_price - exit_price) / entry_price
            future = ltp[triggered_idx:]
            mfe = (entry_price - np.min(future)) / entry_price
            mae = (entry_price - np.max(future)) / entry_price

        roc_at_entry = float(roc[triggered_idx]) if not np.isnan(roc[triggered_idx]) else None

        delta_val = delta[triggered_idx] * entry_price / 100000.0

        bypass_used = False
        rejection = "EXECUTED"
        if not daily_open_passed:
            if direction == "LONG":
                if prob[triggered_idx] >= 0.35:
                    bypass_used = True
                    daily_open_passed = True
                else:
                    rejection = "DAILY_OPEN_FAILED"
            else:
                rejection = "DAILY_OPEN_FAILED"

        row_out.update({
            "max_stage": "TRIGGERED",
            "rejection_reason": rejection,
            "entry_time_ist": ts_str[triggered_idx],
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "eod_pnl_pct": float(eod_pnl),
            "mfe_pct": float(mfe),
            "mae_pct": float(mae),
            "roc_at_entry": roc_at_entry,
            "daily_open_passed": daily_open_passed,
            "bypass_used": bypass_used,
            "mark1_ts": ts_str[mark1_idx],
            "between_ts": ts_str[between_idx],
            "delta_value_lakhs_at_theta": float(delta_val),
        })

    return row_out


def compute_summary(trades_df, label):
    executed = trades_df[trades_df["rejection_reason"] == "EXECUTED"]
    num = len(executed)
    if num == 0:
        return {
            "Pair": label, "Total Trades": 0, "Win Rate": "0.00%",
            "Mean Return": "0.00%", "Cumulative Return": "+0.00%",
            "Mean MFE": "0.00%", "Mean MAE": "0.00%",
            "MFE >=3%": "0.00%", "MFE >=5%": "0.00%",
            "LONG": 0, "SHORT": 0,
            "Total Rows": len(trades_df),
        }
    longs = executed[executed["direction"] == "LONG"]
    shorts = executed[executed["direction"] == "SHORT"]
    pnls = executed["eod_pnl_pct"].values
    mfes = executed["mfe_pct"].values
    wins = pnls[pnls > 0]
    return {
        "Pair": label,
        "Total Trades": num,
        "Win Rate": f"{len(wins) / len(pnls):.2%}",
        "Mean Return": f"{np.mean(pnls):.4%}",
        "Cumulative Return": f"{np.sum(pnls):+.2%}",
        "Mean MFE": f"{np.mean(mfes):.2%}",
        "Mean MAE": f"{np.mean(executed['mae_pct'].values):.2%}",
        "MFE >=3%": f"{np.mean(mfes >= 0.03):.2%}",
        "MFE >=5%": f"{np.mean(mfes >= 0.05):.2%}",
        "LONG": len(longs),
        "SHORT": len(shorts),
        "Total Rows": len(trades_df),
    }


def main():
    logger.info("=" * 70)
    logger.info(f"SCRIPT 76: DOUBLE ROC(3) STATE MACHINE — 8 THRESHOLD PAIRS | TARGET={TARGET_LABEL}")
    logger.info("=" * 70)

    if not OOF_PATH.exists():
        logger.error(f"OOF file not found: {OOF_PATH}")
        return

    df = pd.read_parquet(OOF_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    logger.info(f"Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols, {df['date'].nunique()} dates")

    # Always recompute ts_ist_str from ts (UTC) to ensure IST times are correct
    # regardless of what was stored in the parquet file
    if df["ts"].dt.tz is None:
        _ts_ist = df["ts"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    else:
        _ts_ist = df["ts"].dt.tz_convert("Asia/Kolkata")
    df["ts_ist_str"] = _ts_ist.dt.strftime("%H:%M:%S")
    logger.info("ts_ist_str recomputed from UTC -> Asia/Kolkata (sample: %s)", df["ts_ist_str"].iloc[0])

    # Build max prob delta lookup
    def _build_max_prob_delta_lookup(df_full):
        delta_lookup = {}
        for direction, prob_col in [("LONG", "prob_long"), ("SHORT", "prob_short")]:
            idxs = df_full.groupby(["symbol", "date"])[prob_col].idxmax()
            for (sym, dt), idx in idxs.items():
                row = df_full.loc[idx]
                delta_lookup[(sym, dt, direction)] = float(row["delta_1m"] * row["close_price"] / 100000.0)
        return delta_lookup

    delta_lookup = _build_max_prob_delta_lookup(df)
    logger.info(f"Max-prob delta lookup: {len(delta_lookup)} entries")

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary_rows = []

        for theta_long, theta_short, label in THRESHOLD_PAIRS:
            logger.info(f"\n--- {label} (LONG={theta_long}, SHORT={theta_short}) ---")
            all_rows = []

            for (symbol_name, date_str), grp in df.groupby(["symbol", "date"], sort=False):
                grp = grp.sort_values("ts").reset_index(drop=True)
                day_met = compute_day_metrics(grp)

                long_row = run_direction_state_machine(grp, theta_long, "LONG", day_met)
                short_row = run_direction_state_machine(grp, theta_short, "SHORT", day_met)

                for r in [long_row, short_row]:
                    r["delta_value_lakhs_at_max_prob"] = delta_lookup.get(
                        (r["symbol"], r["date"], r["direction"]), None
                    )

                all_rows.append(long_row)
                all_rows.append(short_row)

            df_out = pd.DataFrame(all_rows)

            STRATEGY_COLS = [
                "slug", "symbol", "date", "direction",
                "max_prob_long", "max_prob_short", "max_prob_time",
                "theta_caught_time",
                "open", "close", "high", "low", "intraday_range_pct",
                "mark1_ts", "between_ts", "max_stage", "rejection_reason",
                "entry_time_ist", "entry_price", "exit_price",
                "eod_pnl_pct", "mfe_pct", "mae_pct", "roc_at_entry",
                "daily_open_passed", "bypass_used",
                "delta_value_lakhs_at_theta", "delta_value_lakhs_at_max_prob",
            ]
            df_out = df_out[STRATEGY_COLS]

            sheet_name = label[:31]
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
            executed = df_out[df_out["rejection_reason"] == "EXECUTED"]
            logger.info(f"  Sheet '{sheet_name}': {len(df_out)} rows ({len(executed)} executed)")

            summary_rows.append(compute_summary(df_out, label))

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    logger.info(f"\n[SUCCESS] Output: {OUTPUT_PATH}")
    print("\n" + "=" * 90)
    print("THRESHOLD PAIR SWEEP — SUMMARY")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # Log rejection distribution for each pair
    for label in [lbl for _, _, lbl in THRESHOLD_PAIRS]:
        xls = pd.ExcelFile(OUTPUT_PATH)
        df_s = pd.read_excel(xls, sheet_name=label[:31])
        print(f"\n{label} — rejection breakdown:")
        print(df_s["rejection_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
