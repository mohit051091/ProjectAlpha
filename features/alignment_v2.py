import numpy as np
import pandas as pd
from typing import Dict, Any
from utils.logger import setup_logger
from utils.constants import ALIGNMENT_CONFIG

logger = setup_logger(__name__)

class EventAlignmentEngine:
    """Align Tick and DOM events using event-time nearest-neighbor matching."""

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, ts_col: str = 'ts') -> pd.DataFrame:
        if ts_col not in df.columns:
            raise ValueError(f"Missing required timestamp column: {ts_col}")

        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
        else:
            df[ts_col] = df[ts_col].dt.tz_convert('UTC')

        return df.sort_values(ts_col).reset_index(drop=False).rename(columns={'index': 'orig_index'})

    @staticmethod
    def _asof_match(left: pd.DataFrame, right: pd.DataFrame, tolerance: pd.Timedelta) -> pd.DataFrame:
        right_df = right[[ 'ts', 'orig_index']].rename(columns={'ts': 'match_ts', 'orig_index': 'match_index'})
        result = pd.merge_asof(
            left,
            right_df,
            left_on='ts',
            right_on='match_ts',
            direction='nearest',
            tolerance=tolerance,
        )
        diff_seconds = (result['ts'] - result['match_ts']).dt.total_seconds().abs()
        result['alignment_distance_ms'] = diff_seconds * 1000.0
        return result

    @staticmethod
    def align_tick_dom(
        df_tick: pd.DataFrame,
        df_dom: pd.DataFrame,
        tolerance_s: float = 1.0,
        window: str = '1min',
    ) -> Dict[str, Any]:
        if len(df_tick) == 0 or len(df_dom) == 0:
            logger.warning("Either Tick or DOM data is empty. No alignment is possible.")
            empty_df = pd.DataFrame(columns=[
                'window_start', 'tick_count', 'dom_count', 'tick_matched_count',
                'dom_matched_count', 'unmatched_tick_count', 'unmatched_dom_count',
                'matched_tick_ratio', 'matched_dom_ratio', 'unmatched_ratio',
                'avg_alignment_distance_ms', 'max_alignment_distance_ms',
            ])
            return {
                'tick_matches': pd.DataFrame(),
                'dom_matches': pd.DataFrame(),
                'window_metrics': empty_df,
                'global_metrics': {
                    'total_tick_count': len(df_tick),
                    'total_dom_count': len(df_dom),
                    'matched_tick_count': 0,
                    'matched_dom_count': 0,
                    'unmatched_tick_count': len(df_tick),
                    'unmatched_dom_count': len(df_dom),
                    'matched_tick_ratio': 0.0,
                    'matched_dom_ratio': 0.0,
                    'unmatched_ratio': 1.0,
                    'avg_alignment_distance_ms': np.nan,
                    'max_alignment_distance_ms': np.nan,
                },
            }

        tick = EventAlignmentEngine._prepare_dataframe(df_tick, ts_col='ts')
        dom = EventAlignmentEngine._prepare_dataframe(df_dom, ts_col='ts')

        tolerance = pd.Timedelta(seconds=tolerance_s)

        tick_matches = EventAlignmentEngine._asof_match(tick, dom, tolerance)
        dom_matches = EventAlignmentEngine._asof_match(dom, tick, tolerance)

        total_tick_count = len(tick)
        total_dom_count = len(dom)
        matched_tick_count = int(tick_matches['match_index'].notna().sum())
        matched_dom_count = int(dom_matches['match_index'].notna().sum())

        unmatched_tick_count = total_tick_count - matched_tick_count
        unmatched_dom_count = total_dom_count - matched_dom_count

        avg_alignment_distance_ms = float(tick_matches['alignment_distance_ms'].dropna().mean()) if matched_tick_count > 0 else np.nan
        max_alignment_distance_ms = float(tick_matches['alignment_distance_ms'].dropna().max()) if matched_tick_count > 0 else np.nan

        matched_tick_ratio = matched_tick_count / total_tick_count if total_tick_count > 0 else 0.0
        matched_dom_ratio = matched_dom_count / total_dom_count if total_dom_count > 0 else 0.0
        total_events = total_tick_count + total_dom_count
        unmatched_ratio = (unmatched_tick_count + unmatched_dom_count) / total_events if total_events > 0 else 1.0

        tick_matches['window_start'] = tick_matches['ts'].dt.floor(window)
        dom_matches['window_start'] = dom_matches['ts'].dt.floor(window)

        tick_window = tick_matches.groupby('window_start').agg(
            tick_count=('orig_index', 'count'),
            tick_matched_count=('match_index', lambda x: x.notna().sum()),
            avg_alignment_distance_ms=('alignment_distance_ms', 'mean'),
            max_alignment_distance_ms=('alignment_distance_ms', 'max'),
        )

        dom_window = dom_matches.groupby('window_start').agg(
            dom_count=('orig_index', 'count'),
            dom_matched_count=('match_index', lambda x: x.notna().sum()),
            dom_avg_alignment_distance_ms=('alignment_distance_ms', 'mean'),
            dom_max_alignment_distance_ms=('alignment_distance_ms', 'max'),
        )

        window_metrics = pd.merge(
            tick_window,
            dom_window,
            how='outer',
            left_index=True,
            right_index=True,
        ).reset_index().rename(columns={'index': 'window_start'})

        window_metrics['tick_count'] = window_metrics['tick_count'].fillna(0).astype('int64')
        window_metrics['dom_count'] = window_metrics['dom_count'].fillna(0).astype('int64')
        window_metrics['tick_matched_count'] = window_metrics['tick_matched_count'].fillna(0).astype('int64')
        window_metrics['dom_matched_count'] = window_metrics['dom_matched_count'].fillna(0).astype('int64')

        window_metrics['unmatched_tick_count'] = window_metrics['tick_count'] - window_metrics['tick_matched_count']
        window_metrics['unmatched_dom_count'] = window_metrics['dom_count'] - window_metrics['dom_matched_count']

        window_metrics['matched_tick_ratio'] = np.where(
            window_metrics['tick_count'] > 0,
            window_metrics['tick_matched_count'] / window_metrics['tick_count'],
            0.0,
        )
        window_metrics['matched_dom_ratio'] = np.where(
            window_metrics['dom_count'] > 0,
            window_metrics['dom_matched_count'] / window_metrics['dom_count'],
            0.0,
        )
        window_metrics['unmatched_ratio'] = np.where(
            (window_metrics['tick_count'] + window_metrics['dom_count']) > 0,
            (window_metrics['unmatched_tick_count'] + window_metrics['unmatched_dom_count']) / (window_metrics['tick_count'] + window_metrics['dom_count']),
            1.0,
        )

        window_metrics['avg_alignment_distance_ms'] = window_metrics[['avg_alignment_distance_ms', 'dom_avg_alignment_distance_ms']].mean(axis=1, skipna=True)
        window_metrics['max_alignment_distance_ms'] = window_metrics[['max_alignment_distance_ms', 'dom_max_alignment_distance_ms']].max(axis=1)

        window_metrics = window_metrics.drop(columns=['dom_avg_alignment_distance_ms', 'dom_max_alignment_distance_ms'], errors='ignore')

        global_metrics = {
            'total_tick_count': total_tick_count,
            'total_dom_count': total_dom_count,
            'matched_tick_count': matched_tick_count,
            'matched_dom_count': matched_dom_count,
            'unmatched_tick_count': unmatched_tick_count,
            'unmatched_dom_count': unmatched_dom_count,
            'matched_tick_ratio': matched_tick_ratio,
            'matched_dom_ratio': matched_dom_ratio,
            'unmatched_ratio': unmatched_ratio,
            'avg_alignment_distance_ms': avg_alignment_distance_ms,
            'max_alignment_distance_ms': max_alignment_distance_ms,
        }

        return {
            'tick_matches': tick_matches,
            'dom_matches': dom_matches,
            'window_metrics': window_metrics,
            'global_metrics': global_metrics,
        }

class WindowValidationEngine:
    """Compute window-level quality scores and validity flags."""

    @staticmethod
    def compute_scores(
        df_windows: pd.DataFrame,
        tolerance_ms: float = ALIGNMENT_CONFIG['tolerance_ms'],
        aqs_weights: Dict[str, float] = ALIGNMENT_CONFIG['aqs_weights'],
        aqs_threshold: float = ALIGNMENT_CONFIG['aqs_threshold'],
        dcs_threshold: float = ALIGNMENT_CONFIG['dcs_threshold'],
        wvs_threshold: float = ALIGNMENT_CONFIG['wvs_threshold'],
        min_tick_count: int = ALIGNMENT_CONFIG['min_tick_count'],
        min_dom_count: int = ALIGNMENT_CONFIG['min_dom_count'],
    ) -> pd.DataFrame:
        df = df_windows.copy()
        if df.empty:
            return df

        df['normalized_avg_distance'] = np.where(
            df['avg_alignment_distance_ms'].notna(),
            np.minimum(df['avg_alignment_distance_ms'] / tolerance_ms, 1.0),
            1.0,
        )

        df['aqs'] = 1.0 - (
            aqs_weights['unmatched_ratio'] * df['unmatched_ratio'] +
            aqs_weights['avg_distance'] * df['normalized_avg_distance']
        )
        df['aqs'] = df['aqs'].clip(lower=0.0, upper=1.0)

        df['dcs'] = np.minimum(df['matched_tick_ratio'], df['matched_dom_ratio'])
        df['wvs'] = np.minimum(df['aqs'], df['dcs'])

        df['window_valid'] = (
            (df['tick_count'] >= min_tick_count) &
            (df['dom_count'] >= min_dom_count) &
            (df['aqs'] >= aqs_threshold) &
            (df['dcs'] >= dcs_threshold) &
            (df['wvs'] >= wvs_threshold)
        )

        cond_map = {
            'insufficient_tick_events': (df['tick_count'] < min_tick_count).values,
            'insufficient_dom_events': (df['dom_count'] < min_dom_count).values,
            'low_alignment_quality': (df['aqs'] < aqs_threshold).values,
            'low_data_completeness': (df['dcs'] < dcs_threshold).values,
            'low_window_validity': (df['wvs'] < wvs_threshold).values,
        }
        result = np.full(len(df), '', dtype=object)
        for reason, mask in cond_map.items():
            accum = np.where(result != '', result + ';' + reason, reason)
            result = np.where(mask, accum, result)
        df['failure_reason'] = result
        return df
