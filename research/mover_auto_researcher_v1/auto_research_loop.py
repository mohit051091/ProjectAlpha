import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ConfigMetrics:
    label: str
    target_label: str
    target_alias: str
    long_th: Optional[float]
    short_th: Optional[float]
    executed_trades: int
    unique_days: int
    trades_per_day: float
    win_rate: float
    mover_hit_rate: float
    cumulative_return: float
    mean_mfe: float
    universe_symbol_days: int
    actual_movers: int
    actual_non_movers: int
    predicted_movers: int
    true_positive_movers: int
    false_positive_non_movers: int
    missed_movers: int
    true_negative_non_movers: int
    mover_precision: float
    mover_recall: float
    non_mover_specificity: float
    false_positive_rate: float
    classification_accuracy: float
    mover_f1: float
    mean_daily_mover_precision: float
    mean_daily_mover_recall: float
    long_directional_precision: float
    short_directional_precision: float
    score: float = 0.0


def _parse_percentish(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_pair(label: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r"L(\d+)_S(\d+)", label.upper())
    if not m:
        return None, None
    return int(m.group(1)) / 100.0, int(m.group(2)) / 100.0


def _sheet_label(sheet_name: str) -> str:
    prefixes = ["Strategy_Sweep_", "Sweep_"]
    for p in prefixes:
        if sheet_name.startswith(p):
            return sheet_name[len(p):]
    return sheet_name


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
def _normalize_sweep_date_col(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%d_%b_%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s[missing], format="%Y-%m-%d", errors="coerce")
    return parsed

def _apply_date_window(df: pd.DataFrame, eval_start_date: Optional[str], eval_end_date: Optional[str]) -> pd.DataFrame:
    if df.empty or ("date" not in df.columns) or (not eval_start_date and not eval_end_date):
        return df
    out = df.copy()
    dt = _normalize_sweep_date_col(out["date"])
    mask = pd.Series(True, index=out.index)
    if eval_start_date:
        start = pd.to_datetime(eval_start_date, errors="coerce")
        if pd.notna(start):
            mask &= dt >= start
    if eval_end_date:
        end = pd.to_datetime(eval_end_date, errors="coerce")
        if pd.notna(end):
            mask &= dt <= end
    return out[mask].copy()

def compute_universe_mover_metrics(df: pd.DataFrame, mover_threshold_pct: float) -> Dict[str, float]:
    zero = {
        "universe_symbol_days": 0,
        "actual_movers": 0,
        "actual_non_movers": 0,
        "predicted_movers": 0,
        "true_positive_movers": 0,
        "false_positive_non_movers": 0,
        "missed_movers": 0,
        "true_negative_non_movers": 0,
        "mover_precision": 0.0,
        "mover_recall": 0.0,
        "non_mover_specificity": 0.0,
        "false_positive_rate": 0.0,
        "classification_accuracy": 0.0,
        "mover_f1": 0.0,
        "mean_daily_mover_precision": 0.0,
        "mean_daily_mover_recall": 0.0,
        "long_directional_precision": 0.0,
        "short_directional_precision": 0.0,
    }
    if df.empty:
        return zero

    work = df.copy()
    for c in ("open", "high", "low", "intraday_range_pct"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        else:
            work[c] = pd.NA
    if "direction" not in work.columns:
        work["direction"] = ""
    if "rejection_reason" not in work.columns:
        work["rejection_reason"] = ""

    base = work.drop_duplicates(subset=["date", "symbol"], keep="first").copy()
    if base.empty:
        return zero

    threshold = float(mover_threshold_pct)
    open_px = pd.to_numeric(base["open"], errors="coerce")
    safe_open = open_px.where(open_px != 0)
    up_move_pct = ((pd.to_numeric(base["high"], errors="coerce") - safe_open) / safe_open) * 100.0
    down_move_pct = ((pd.to_numeric(base["low"], errors="coerce") - safe_open) / safe_open) * 100.0
    intraday_range_pct = pd.to_numeric(base["intraday_range_pct"], errors="coerce")

    base["actual_up_mover"] = (up_move_pct >= threshold).fillna(False)
    base["actual_down_mover"] = (down_move_pct <= -threshold).fillna(False)
    fallback_range_mover = (intraday_range_pct >= threshold).fillna(False)
    base["actual_mover"] = (base["actual_up_mover"] | base["actual_down_mover"] | fallback_range_mover).fillna(False)

    executed = work[work["rejection_reason"] == "EXECUTED"][["date", "symbol", "direction"]].copy()
    if executed.empty:
        pred_flags = pd.DataFrame(columns=["date", "symbol", "predicted_mover", "predicted_long", "predicted_short"])
    else:
        pred_flags = (
            executed.groupby(["date", "symbol"])
            .agg(
                pred_count=("direction", "size"),
                predicted_long=("direction", lambda s: (s == "LONG").any()),
                predicted_short=("direction", lambda s: (s == "SHORT").any()),
            )
            .reset_index()
        )
        pred_flags["predicted_mover"] = pred_flags["pred_count"] > 0
        pred_flags = pred_flags[["date", "symbol", "predicted_mover", "predicted_long", "predicted_short"]]

    merged = base.merge(pred_flags, on=["date", "symbol"], how="left")
    for c in ("predicted_mover", "predicted_long", "predicted_short"):
        merged[c] = merged[c].fillna(False).astype(bool)

    universe_symbol_days = int(len(merged))
    actual_movers = int(merged["actual_mover"].sum())
    actual_non_movers = int(universe_symbol_days - actual_movers)
    predicted_movers = int(merged["predicted_mover"].sum())
    true_positive_movers = int((merged["predicted_mover"] & merged["actual_mover"]).sum())
    false_positive_non_movers = int((merged["predicted_mover"] & ~merged["actual_mover"]).sum())
    missed_movers = int((~merged["predicted_mover"] & merged["actual_mover"]).sum())
    true_negative_non_movers = int((~merged["predicted_mover"] & ~merged["actual_mover"]).sum())

    mover_precision = _safe_rate(true_positive_movers, predicted_movers)
    mover_recall = _safe_rate(true_positive_movers, actual_movers)
    non_mover_specificity = _safe_rate(true_negative_non_movers, actual_non_movers)
    false_positive_rate = _safe_rate(false_positive_non_movers, actual_non_movers)
    classification_accuracy = _safe_rate(true_positive_movers + true_negative_non_movers, universe_symbol_days)
    mover_f1 = _safe_rate(2 * mover_precision * mover_recall, mover_precision + mover_recall)

    daily_precision = []
    daily_recall = []
    for _, g in merged.groupby("date"):
        a_m = int(g["actual_mover"].sum())
        p_m = int(g["predicted_mover"].sum())
        tp = int((g["predicted_mover"] & g["actual_mover"]).sum())
        daily_precision.append(_safe_rate(tp, p_m))
        daily_recall.append(_safe_rate(tp, a_m))
    mean_daily_mover_precision = float(pd.Series(daily_precision, dtype=float).mean()) if daily_precision else 0.0
    mean_daily_mover_recall = float(pd.Series(daily_recall, dtype=float).mean()) if daily_recall else 0.0

    long_directional_precision = _safe_rate(
        int((merged["predicted_long"] & merged["actual_up_mover"]).sum()),
        int(merged["predicted_long"].sum()),
    )
    short_directional_precision = _safe_rate(
        int((merged["predicted_short"] & merged["actual_down_mover"]).sum()),
        int(merged["predicted_short"].sum()),
    )

    return {
        "universe_symbol_days": universe_symbol_days,
        "actual_movers": actual_movers,
        "actual_non_movers": actual_non_movers,
        "predicted_movers": predicted_movers,
        "true_positive_movers": true_positive_movers,
        "false_positive_non_movers": false_positive_non_movers,
        "missed_movers": missed_movers,
        "true_negative_non_movers": true_negative_non_movers,
        "mover_precision": mover_precision,
        "mover_recall": mover_recall,
        "non_mover_specificity": non_mover_specificity,
        "false_positive_rate": false_positive_rate,
        "classification_accuracy": classification_accuracy,
        "mover_f1": mover_f1,
        "mean_daily_mover_precision": mean_daily_mover_precision,
        "mean_daily_mover_recall": mean_daily_mover_recall,
        "long_directional_precision": long_directional_precision,
        "short_directional_precision": short_directional_precision,
    }


def compute_metrics_from_sheet(
    df: pd.DataFrame,
    label: str,
    target_label: str,
    target_alias: str,
    mover_threshold_pct: float,
    eval_start_date: Optional[str] = None,
    eval_end_date: Optional[str] = None,
) -> ConfigMetrics:
    scoped = _apply_date_window(df, eval_start_date, eval_end_date)
    universe = compute_universe_mover_metrics(scoped, mover_threshold_pct)
    if scoped.empty:
        long_th, short_th = _extract_pair(label)
        return ConfigMetrics(
            label=label,
            target_label=target_label,
            target_alias=target_alias,
            long_th=long_th,
            short_th=short_th,
            executed_trades=0,
            unique_days=0,
            trades_per_day=0.0,
            win_rate=0.0,
            mover_hit_rate=0.0,
            cumulative_return=0.0,
            mean_mfe=0.0,
            universe_symbol_days=universe["universe_symbol_days"],
            actual_movers=universe["actual_movers"],
            actual_non_movers=universe["actual_non_movers"],
            predicted_movers=universe["predicted_movers"],
            true_positive_movers=universe["true_positive_movers"],
            false_positive_non_movers=universe["false_positive_non_movers"],
            missed_movers=universe["missed_movers"],
            true_negative_non_movers=universe["true_negative_non_movers"],
            mover_precision=universe["mover_precision"],
            mover_recall=universe["mover_recall"],
            non_mover_specificity=universe["non_mover_specificity"],
            false_positive_rate=universe["false_positive_rate"],
            classification_accuracy=universe["classification_accuracy"],
            mover_f1=universe["mover_f1"],
            mean_daily_mover_precision=universe["mean_daily_mover_precision"],
            mean_daily_mover_recall=universe["mean_daily_mover_recall"],
            long_directional_precision=universe["long_directional_precision"],
            short_directional_precision=universe["short_directional_precision"],
        )

    work = scoped.copy()
    if "rejection_reason" in work.columns:
        work = work[work["rejection_reason"] == "EXECUTED"].copy()
    if work.empty:
        long_th, short_th = _extract_pair(label)
        return ConfigMetrics(
            label=label,
            target_label=target_label,
            target_alias=target_alias,
            long_th=long_th,
            short_th=short_th,
            executed_trades=0,
            unique_days=int(scoped["date"].nunique()) if "date" in scoped.columns else 0,
            trades_per_day=0.0,
            win_rate=0.0,
            mover_hit_rate=0.0,
            cumulative_return=0.0,
            mean_mfe=0.0,
            universe_symbol_days=universe["universe_symbol_days"],
            actual_movers=universe["actual_movers"],
            actual_non_movers=universe["actual_non_movers"],
            predicted_movers=universe["predicted_movers"],
            true_positive_movers=universe["true_positive_movers"],
            false_positive_non_movers=universe["false_positive_non_movers"],
            missed_movers=universe["missed_movers"],
            true_negative_non_movers=universe["true_negative_non_movers"],
            mover_precision=universe["mover_precision"],
            mover_recall=universe["mover_recall"],
            non_mover_specificity=universe["non_mover_specificity"],
            false_positive_rate=universe["false_positive_rate"],
            classification_accuracy=universe["classification_accuracy"],
            mover_f1=universe["mover_f1"],
            mean_daily_mover_precision=universe["mean_daily_mover_precision"],
            mean_daily_mover_recall=universe["mean_daily_mover_recall"],
            long_directional_precision=universe["long_directional_precision"],
            short_directional_precision=universe["short_directional_precision"],
        )

    for c in ("eod_pnl_pct", "mfe_pct"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        else:
            work[c] = 0.0

    executed = int(len(work))
    days = int(work["date"].nunique()) if "date" in work.columns else 0
    trades_per_day = _safe_rate(executed, days) if days else 0.0
    win_rate = float((work["eod_pnl_pct"] > 0).mean()) if executed else 0.0
    mover_threshold = mover_threshold_pct / 100.0
    mover_hit_rate = float((work["mfe_pct"] >= mover_threshold).mean()) if executed else 0.0
    cumulative_return = float(work["eod_pnl_pct"].sum()) if executed else 0.0
    mean_mfe = float(work["mfe_pct"].mean()) if executed else 0.0
    long_th, short_th = _extract_pair(label)

    return ConfigMetrics(
        label=label,
        target_label=target_label,
        target_alias=target_alias,
        long_th=long_th,
        short_th=short_th,
        executed_trades=executed,
        unique_days=days,
        trades_per_day=trades_per_day,
        win_rate=win_rate,
        mover_hit_rate=mover_hit_rate,
        cumulative_return=cumulative_return,
        mean_mfe=mean_mfe,
        universe_symbol_days=universe["universe_symbol_days"],
        actual_movers=universe["actual_movers"],
        actual_non_movers=universe["actual_non_movers"],
        predicted_movers=universe["predicted_movers"],
        true_positive_movers=universe["true_positive_movers"],
        false_positive_non_movers=universe["false_positive_non_movers"],
        missed_movers=universe["missed_movers"],
        true_negative_non_movers=universe["true_negative_non_movers"],
        mover_precision=universe["mover_precision"],
        mover_recall=universe["mover_recall"],
        non_mover_specificity=universe["non_mover_specificity"],
        false_positive_rate=universe["false_positive_rate"],
        classification_accuracy=universe["classification_accuracy"],
        mover_f1=universe["mover_f1"],
        mean_daily_mover_precision=universe["mean_daily_mover_precision"],
        mean_daily_mover_recall=universe["mean_daily_mover_recall"],
        long_directional_precision=universe["long_directional_precision"],
        short_directional_precision=universe["short_directional_precision"],
    )


def score_metrics(
    metrics: ConfigMetrics,
    min_trades_per_day: float,
    max_trades_per_day: float,
    score_weights: Optional[Dict[str, float]] = None,
    false_positive_rate_multiplier: float = 0.30,
    tpd_penalty_cap: float = 0.20,
) -> float:
    if score_weights is None:
        score_weights = {
            "mover_recall": 0.24,
            "mover_precision": 0.18,
            "non_mover_specificity": 0.12,
            "mover_hit_rate": 0.08,
            "win_rate": 0.08,
            "directional_precision": 0.05,
            "cum_return_component": 0.20,
            "mean_mfe_component": 0.05,
        }
    # Normalize cumulative return roughly into [0,1] with a soft clamp.
    cum_component = max(min((metrics.cumulative_return + 1.0) / 2.0, 1.0), 0.0)
    mean_mfe_component = min(max(metrics.mean_mfe / 0.10, 0.0), 1.0)
    directional_precision = (metrics.long_directional_precision + metrics.short_directional_precision) / 2.0
    base = (
        score_weights.get("mover_recall", 0.0) * metrics.mover_recall
        + score_weights.get("mover_precision", 0.0) * metrics.mover_precision
        + score_weights.get("non_mover_specificity", 0.0) * metrics.non_mover_specificity
        + score_weights.get("mover_hit_rate", 0.0) * metrics.mover_hit_rate
        + score_weights.get("win_rate", 0.0) * metrics.win_rate
        + score_weights.get("directional_precision", 0.0) * directional_precision
        + score_weights.get("cum_return_component", 0.0) * cum_component
        + score_weights.get("mean_mfe_component", 0.0) * mean_mfe_component
    )
    false_positive_penalty = min(metrics.false_positive_rate * false_positive_rate_multiplier, tpd_penalty_cap)
    tpd = metrics.trades_per_day
    if tpd == 0:
        tpd_penalty = tpd_penalty_cap
    elif tpd < min_trades_per_day:
        tpd_penalty = min((min_trades_per_day - tpd) * 0.01, tpd_penalty_cap)
    elif tpd > max_trades_per_day:
        tpd_penalty = min((tpd - max_trades_per_day) * 0.01, tpd_penalty_cap)
    else:
        tpd_penalty = 0.0
    return max(base - false_positive_penalty - tpd_penalty, 0.0)


def load_config(workspace_dir: Path) -> Dict:
    cfg_path = workspace_dir / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)
def _target_token(target_label: str) -> str:
    return target_label.replace("label_", "").replace("/", "_")

def resolve_target_candidates(cfg: Dict) -> List[Dict[str, str]]:
    objective = cfg.get("objective", {})
    raw = objective.get("target_label_candidates")
    default = [{"target_label": "label_60m_1pct", "alias": "balanced_60m_1pct"}]
    if not raw:
        single = str(objective.get("target_label", "label_60m_1pct")).strip()
        return [{"target_label": single, "alias": _target_token(single)}]
    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                target_label = item.strip()
                if target_label:
                    out.append({"target_label": target_label, "alias": _target_token(target_label)})
                continue
            if isinstance(item, dict):
                target_label = str(item.get("target_label", "")).strip()
                if not target_label:
                    continue
                alias = str(item.get("alias", _target_token(target_label))).strip() or _target_token(target_label)
                out.append({"target_label": target_label, "alias": alias})
    return out or default

def sweep_excel_path(project_root: Path, cfg_inputs: Dict, target_label: str) -> Path:
    template = cfg_inputs.get("sweep_excel_template", "results/performance_sweeps_{target_label}.xlsx")
    rendered = template.format(target_label=target_label)
    candidate = project_root / rendered
    if candidate.exists():
        return candidate
    legacy = project_root / cfg_inputs.get("sweep_excel_relative", "results/performance_sweeps.xlsx")
    if target_label == "label_60m_1pct" and legacy.exists():
        return legacy
    return candidate


def read_workbook_metrics(
    excel_path: Path,
    target_label: str,
    target_alias: str,
    mover_threshold_pct: float,
    eval_start_date: Optional[str] = None,
    eval_end_date: Optional[str] = None,
) -> List[ConfigMetrics]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Sweep workbook not found: {excel_path}")
    xls = pd.ExcelFile(excel_path)
    needed_cols = {
        "slug",
        "symbol",
        "date",
        "direction",
        "open",
        "high",
        "low",
        "intraday_range_pct",
        "rejection_reason",
        "eod_pnl_pct",
        "mfe_pct",
    }
    results: List[ConfigMetrics] = []
    for sheet in xls.sheet_names:
        if sheet.lower() == "summary":
            continue
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet,
            usecols=lambda c: c in needed_cols,
        )
        if "mfe_pct" not in df.columns:
            continue
        label = _sheet_label(sheet)
        results.append(
            compute_metrics_from_sheet(
                df,
                label,
                target_label,
                target_alias,
                mover_threshold_pct,
                eval_start_date=eval_start_date,
                eval_end_date=eval_end_date,
            )
        )
    return results


def propose_candidates(
    best: ConfigMetrics,
    step: float,
    long_min: float,
    long_max: float,
    short_min: float,
    short_max: float,
    max_candidates: int,
) -> List[Dict]:
    base_long = best.long_th if best.long_th is not None else 0.30
    base_short = best.short_th if best.short_th is not None else 0.25

    deltas = [
        (0.0, 0.0),
        (+step, 0.0),
        (-step, 0.0),
        (0.0, +step),
        (0.0, -step),
        (+step, -step),
        (-step, +step),
        (+2 * step, 0.0),
        (0.0, +2 * step),
    ]

    out = []
    seen = set()
    for dl, ds in deltas:
        l = round(base_long + dl, 2)
        s = round(base_short + ds, 2)
        l = min(max(l, long_min), long_max)
        s = min(max(s, short_min), short_max)
        key = (l, s)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "label": f"L{int(round(l * 100)):02d}_S{int(round(s * 100)):02d}",
                "long_th": l,
                "short_th": s,
                "hypothesis": (
                    f"Test LONG={l:.2f}, SHORT={s:.2f} for better mover capture while "
                    f"keeping trade frequency in target band."
                ),
            }
        )
        if len(out) >= max_candidates:
            break
    return out


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_plan_markdown(
    project_root: Path,
    excel_path: Path,
    best: ConfigMetrics,
    candidates: List[Dict],
    dry_run: bool,
) -> str:
    mode = "DRY-RUN (no commands executed)" if dry_run else "EXECUTE"
    lines = [
        "# Mover Auto-Research Plan Snapshot",
        f"- Generated: {datetime.utcnow().isoformat()}Z",
        f"- Mode: {mode}",
        f"- Project root: `{project_root}`",
        f"- Sweep source: `{excel_path}`",
        "",
        "## Current best config (mover objective)",
        f"- Target label family: `{best.target_label}` (`{best.target_alias}`)",
        f"- Label: `{best.label}`",
        f"- LONG_TH: `{best.long_th}`",
        f"- SHORT_TH: `{best.short_th}`",
        f"- Executed trades: `{best.executed_trades}`",
        f"- Trades/day: `{best.trades_per_day:.2f}`",
        f"- Win rate: `{best.win_rate:.2%}`",
        f"- Mover hit-rate (MFE>=4%): `{best.mover_hit_rate:.2%}`",
        f"- Universe symbol-days: `{best.universe_symbol_days}`",
        f"- Actual movers / non-movers: `{best.actual_movers}` / `{best.actual_non_movers}`",
        f"- Predicted movers: `{best.predicted_movers}`",
        f"- TP / FP / FN / TN: `{best.true_positive_movers}` / `{best.false_positive_non_movers}` / `{best.missed_movers}` / `{best.true_negative_non_movers}`",
        f"- Mover precision: `{best.mover_precision:.2%}`",
        f"- Mover recall: `{best.mover_recall:.2%}`",
        f"- Mover F1: `{best.mover_f1:.2%}`",
        f"- Classification accuracy: `{best.classification_accuracy:.2%}`",
        f"- Non-mover specificity: `{best.non_mover_specificity:.2%}`",
        f"- Non-mover false-positive rate: `{best.false_positive_rate:.2%}`",
        f"- Mean daily mover precision: `{best.mean_daily_mover_precision:.2%}`",
        f"- Mean daily mover recall: `{best.mean_daily_mover_recall:.2%}`",
        f"- Directional precision LONG/SHORT: `{best.long_directional_precision:.2%}` / `{best.short_directional_precision:.2%}`",
        f"- Cumulative return: `{best.cumulative_return:+.2%}`",
        f"- Score: `{best.score:.4f}`",
        "",
        "## Proposed next experiments",
    ]
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"{i}. `{c['label']}` (LONG={c['long_th']:.2f}, SHORT={c['short_th']:.2f}) — {c['hypothesis']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- This scaffold does not modify baseline code automatically.",
            "- Universe-level mover classification uses symbol-day highs/lows vs open: up-mover if `(high-open)/open >= mover_threshold_pct`, down-mover if `(low-open)/open <= -mover_threshold_pct`.",
            "- Predicted movers are symbol-days where at least one direction reached `EXECUTED` in the sweep sheet.",
            "- Threshold candidates are generated here; wiring these into sweep execution should be done in sandbox copy first.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_commands(commands: List[str], cwd: Path, run_dir: Path, env_overrides: Dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base_env = dict(**env_overrides)
    for i, command in enumerate(commands, start=1):
        log_path = run_dir / f"{i:02d}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {command}\n\n")
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                env={**dict(os.environ), **base_env},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed with exit code {proc.returncode}: {command}")


def main():
    parser = argparse.ArgumentParser(description="Isolated mover auto-research loop scaffold.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]), help="ProjectAlpha root path.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parent), help="Workspace path.")
    parser.add_argument("--dry-run", action="store_true", help="Generate plan/candidates only.")
    parser.add_argument("--execute", action="store_true", help="Execute configured commands.")
    parser.add_argument("--safe-copy-root", default="", help="Sandbox copy root for execution (recommended).")
    parser.add_argument("--max-exec", type=int, default=4, help="Max target label families to execute in one invocation.")
    args = parser.parse_args()

    workspace = Path(args.workspace_root).resolve()
    cfg = load_config(workspace)

    project_root = Path(args.project_root).resolve()
    if args.execute:
        if not args.safe_copy_root:
            raise ValueError("Execution requires --safe-copy-root to avoid touching current workspace.")
        project_root = Path(args.safe_copy_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"--safe-copy-root does not exist: {project_root}")

    objective = cfg["objective"]
    mover_threshold = float(objective.get("mover_threshold_pct", 4.0))
    min_tpd = float(objective.get("min_trades_per_day", 8))
    max_tpd = float(objective.get("max_trades_per_day", 25))
    eval_start_date = objective.get("eval_start_date")
    eval_end_date = objective.get("eval_end_date")
    score_weights = objective.get("score_weights")
    false_positive_rate_multiplier = float(objective.get("false_positive_rate_multiplier", 0.30))
    tpd_penalty_cap = float(objective.get("tpd_penalty_cap", 0.20))
    inputs_cfg = cfg.get("inputs", {})
    target_candidates = resolve_target_candidates(cfg)

    sweep_paths_by_target: Dict[str, Path] = {}

    if args.execute and not args.dry_run:
        commands = cfg["execution"]["commands"]
        if not commands:
            raise RuntimeError("No execution commands configured in config.json")
        run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        oof_source_rel = inputs_cfg.get("oof_source_relative", "results/oof_predictions_lgbm_label_60m_1pct.parquet")
        oof_source_path = project_root / oof_source_rel
        for idx, target in enumerate(target_candidates[: max(args.max_exec, 1)], start=1):
            target_label = target["target_label"]
            target_alias = target["alias"]
            sweep_path = sweep_excel_path(project_root, inputs_cfg, target_label)
            sweep_paths_by_target[target_label] = sweep_path
            run_dir = workspace / "runs" / f"{run_stamp}_{idx:02d}_{target_alias}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "target_label.json",
                {
                    "target_label": target_label,
                    "target_alias": target_alias,
                    "expected_sweep_path": str(sweep_path),
                },
            )
            run_commands(
                commands=commands,
                cwd=project_root,
                run_dir=run_dir,
                env_overrides={
                    "AR_TARGET_LABEL": target_label,
                    "AR_TARGET_ALIAS": target_alias,
                    "AR_SWEEP_OUTPUT": str(sweep_path),
                    "AR_OOF_SOURCE_PATH": str(oof_source_path),
                },
            )

    all_metrics: List[ConfigMetrics] = []
    evaluated_targets: List[Dict[str, str]] = []
    for target in target_candidates:
        target_label = target["target_label"]
        target_alias = target["alias"]
        excel_path = sweep_paths_by_target.get(target_label) or sweep_excel_path(project_root, inputs_cfg, target_label)
        sweep_paths_by_target[target_label] = excel_path
        if not excel_path.exists():
            continue
        target_metrics = read_workbook_metrics(
            excel_path,
            target_label=target_label,
            target_alias=target_alias,
            mover_threshold_pct=mover_threshold,
            eval_start_date=eval_start_date,
            eval_end_date=eval_end_date,
        )
        if not target_metrics:
            continue
        all_metrics.extend(target_metrics)
        evaluated_targets.append(
            {
                "target_label": target_label,
                "target_alias": target_alias,
                "sweep_workbook": str(excel_path),
            }
        )
    if not all_metrics:
        raise RuntimeError("No sweep sheets with mfe_pct found for configured target label families.")

    for m in all_metrics:
        m.score = score_metrics(
            m,
            min_tpd,
            max_tpd,
            score_weights=score_weights,
            false_positive_rate_multiplier=false_positive_rate_multiplier,
            tpd_penalty_cap=tpd_penalty_cap,
        )
    ranked = sorted(all_metrics, key=lambda x: x.score, reverse=True)
    best = ranked[0]

    p_cfg = cfg["proposal_engine"]
    candidates = propose_candidates(
        best=best,
        step=float(p_cfg.get("step_size", 0.02)),
        long_min=float(p_cfg.get("long_min", 0.2)),
        long_max=float(p_cfg.get("long_max", 0.4)),
        short_min=float(p_cfg.get("short_min", 0.2)),
        short_max=float(p_cfg.get("short_max", 0.35)),
        max_candidates=int(p_cfg.get("max_candidates", 6)),
    )

    proposals_payload = {
        "generated_at_utc": f"{datetime.utcnow().isoformat()}Z",
        "project_root": str(project_root),
        "objective": objective,
        "evaluated_targets": evaluated_targets,
        "current_best": asdict(best),
        "ranked_configs": [asdict(x) for x in ranked],
        "candidates": candidates,
    }
    write_json(workspace / "proposals" / "auto_generated_candidates.json", proposals_payload)

    best_excel_path = sweep_paths_by_target.get(best.target_label) or sweep_excel_path(project_root, inputs_cfg, best.target_label)

    md = build_plan_markdown(
        project_root=project_root,
        excel_path=best_excel_path,
        best=best,
        candidates=candidates,
        dry_run=(not args.execute) or args.dry_run,
    )
    (workspace / "reports").mkdir(parents=True, exist_ok=True)
    (workspace / "reports" / "latest_research_plan.md").write_text(md, encoding="utf-8")
    write_json(workspace / "reports" / "latest_result_snapshot.json", {"best": asdict(best)})
    write_json(
        workspace / "reports" / "live_recommended_settings.json",
        {
            "generated_at_utc": f"{datetime.utcnow().isoformat()}Z",
            "recommended_target_label": best.target_label,
            "recommended_target_alias": best.target_alias,
            "recommended_pair_label": best.label,
            "long_th": best.long_th,
            "short_th": best.short_th,
            "mover_threshold_pct": mover_threshold,
            "eval_start_date": eval_start_date,
            "eval_end_date": eval_end_date,
            "score": best.score,
            "mover_precision": best.mover_precision,
            "mover_recall": best.mover_recall,
            "mover_f1": best.mover_f1,
            "classification_accuracy": best.classification_accuracy,
            "non_mover_specificity": best.non_mover_specificity,
            "win_rate": best.win_rate,
            "cumulative_return": best.cumulative_return,
            "sweep_workbook": str(best_excel_path),
        },
    )

    print("Auto-research scaffold completed.")
    print(
        f"Best config: {best.target_alias}/{best.label} | "
        f"score={best.score:.4f} | mover_hit={best.mover_hit_rate:.2%}"
    )
    print(f"Candidates written: {workspace / 'proposals' / 'auto_generated_candidates.json'}")
    print(f"Plan written: {workspace / 'reports' / 'latest_research_plan.md'}")


if __name__ == "__main__":
    main()
