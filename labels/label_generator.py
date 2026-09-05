"""Label generation utilities for forward-return target creation."""

import pandas as pd
from typing import Dict, List, Optional
from utils.constants import LABEL_HORIZONS, LABEL_THRESHOLDS, Label


def compute_forward_return(
    df: pd.DataFrame,
    horizon_min: int,
    price_col: str = 'ltp',
) -> pd.Series:
    """Compute the forward return over a fixed horizon on aligned 1-minute bars."""
    if 'ts' in df.columns:
        df = df.sort_values('ts').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    future_price = df[price_col].shift(-horizon_min)
    return (future_price - df[price_col]) / df[price_col]


def assign_label(
    returns: pd.Series,
    threshold: float,
) -> pd.Series:
    """Assign LONG/SHORT/NO_TRADE labels from a forward return series."""
    labels = pd.Series(pd.NA, index=returns.index, dtype='object')
    labels = labels.where(returns.isna(), Label.NO_TRADE.value)
    labels = labels.mask(returns >= threshold, Label.LONG.value)
    labels = labels.mask(returns <= -threshold, Label.SHORT.value)
    return labels


def build_label_matrix(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
    price_col: str = 'ltp',
) -> pd.DataFrame:
    """Add forward returns and label columns for multiple horizons and thresholds."""
    horizons = horizons if horizons is not None else LABEL_HORIZONS
    thresholds = thresholds if thresholds is not None else LABEL_THRESHOLDS

    labeled = df.copy()
    if 'ts' in labeled.columns:
        labeled = labeled.sort_values('ts').reset_index(drop=True)

    for horizon in horizons:
        labeled[f'return_{horizon}m'] = compute_forward_return(labeled, horizon, price_col=price_col)
        for threshold in thresholds:
            label_col = f'label_{horizon}m_{int(threshold*100)}pct'
            labeled[label_col] = assign_label(labeled[f'return_{horizon}m'], threshold)
    return labeled


def summarize_label_counts(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> Dict[str, object]:
    """Summarize label counts and availability for one horizon/threshold pair."""
    label_col = f'label_{horizon}m_{int(threshold*100)}pct'
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    available = df[label_col].notna()
    counts = df[label_col].value_counts(dropna=True).to_dict()
    total = len(df)
    available_count = int(available.sum())
    unavailable_count = total - available_count
    percentages = {
        label: float(count) / available_count * 100 if available_count > 0 else 0.0
        for label, count in counts.items()
    }
    return {
        'horizon_min': horizon,
        'threshold': threshold,
        'label_column': label_col,
        'total_rows': total,
        'available_labels': available_count,
        'unavailable_labels': unavailable_count,
        'counts': counts,
        'percentages': percentages,
        'availability_reason': 'insufficient forward horizon data for the last rows' if unavailable_count > 0 else 'all rows available',
    }


def summarize_all_label_configs(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, object]]:
    horizons = horizons if horizons is not None else LABEL_HORIZONS
    thresholds = thresholds if thresholds is not None else LABEL_THRESHOLDS
    summaries = []
    for horizon in horizons:
        for threshold in thresholds:
            summaries.append(summarize_label_counts(df, horizon, threshold))
    return summaries
