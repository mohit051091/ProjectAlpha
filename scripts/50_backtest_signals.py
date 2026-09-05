"""Backtest rule-based signals and persist signal outputs for analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils.constants import PROCESSED_DIR, LABEL_HORIZONS

FEATURES = ['trade_count_burst', 'volume_burst', 'spread', 'delta_5m', 'delta_1m']
RULE_FEATURES = ['trade_count_burst', 'volume_burst', 'spread']


def build_rules(df: pd.DataFrame, horizon: int):
    return_col = f'return_{horizon}m'
    label_col = f'label_{horizon}m_3pct'

    q10 = df[FEATURES].quantile(0.10)
    q20 = df[FEATURES].quantile(0.20)
    q80 = df[FEATURES].quantile(0.80)
    q90 = df[FEATURES].quantile(0.90)

    rules = []

    # single-feature decile rules using sign direction from correlation
    for feature in FEATURES:
        corr = df[[feature, return_col]].corr().iloc[0, 1]
        if pd.isna(corr):
            continue
        if corr >= 0:
            rules.append((feature, 'top_decile_long', df[feature] >= q90[feature], q90[feature], None))
            rules.append((feature, 'bottom_decile_short', df[feature] <= q10[feature], None, q10[feature]))
        else:
            rules.append((feature, 'top_decile_short', df[feature] >= q90[feature], q90[feature], None))
            rules.append((feature, 'bottom_decile_long', df[feature] <= q10[feature], None, q10[feature]))

    # percentile threshold rules for selected strong features
    for feature in RULE_FEATURES:
        rules.extend([
            (feature, 'thresh_>=90pct', df[feature] >= q90[feature], q90[feature], None),
            (feature, 'thresh_>=80pct', df[feature] >= q80[feature], q80[feature], None),
            (feature, 'thresh_<=10pct', df[feature] <= q10[feature], None, q10[feature]),
            (feature, 'thresh_<=20pct', df[feature] <= q20[feature], None, q20[feature]),
        ])

    # combination rules using top-decile direction for each feature
    combos = [
        (['delta_1m', 'trade_count_burst'], 'delta_1m+trade_count_burst'),
        (['delta_5m', 'volume_burst'], 'delta_5m+volume_burst'),
        (['trade_count_burst', 'volume_burst'], 'trade_count_burst+volume_burst'),
        (['trade_count_burst', 'spread'], 'trade_count_burst+spread'),
    ]
    for comb, name in combos:
        masks = []
        for feature in comb:
            corr = df[[feature, return_col]].corr().iloc[0, 1]
            q90v = q90[feature]
            q10v = q10[feature]
            if corr >= 0:
                masks.append(df[feature] >= q90v)
            else:
                masks.append(df[feature] <= q10v)
        combined = masks[0]
        for mask in masks[1:]:
            combined &= mask
        rules.append((name, 'top_decile_and', combined, None, None))

    return rules


def backtest_signals(slug: str, save_parquet: bool = True, save_csv: bool = False):
    processed = Path(PROCESSED_DIR)
    input_file = processed / f'labeled_features_1m_{slug}.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f'Input feature file not found: {input_file}')

    df = pd.read_parquet(input_file)
    symbol = slug.split('_')[0].upper()
    signals = []

    for horizon in LABEL_HORIZONS:
        return_col = f'return_{horizon}m'
        label_col = f'label_{horizon}m_3pct'
        if return_col not in df.columns:
            continue
        rules = build_rules(df, horizon)

        for feature, rule_name, mask, threshold_high, threshold_low in rules:
            if mask.sum() == 0:
                continue
            df_signal = df.loc[mask, ['ts', 'ltp', label_col] + FEATURES].copy()
            df_signal = df_signal.rename(columns={'ltp': 'entry_price', label_col: 'label'})
            df_signal['stock'] = symbol
            df_signal['slug'] = slug
            df_signal['rule_name'] = rule_name
            df_signal['feature'] = feature
            df_signal['horizon'] = horizon
            df_signal['forward_return'] = df.loc[mask, return_col].values
            df_signal['threshold_high'] = threshold_high
            df_signal['threshold_low'] = threshold_low
            signals.append(df_signal)

    if not signals:
        raise ValueError('No signals generated for slug %s' % slug)

    df_output = pd.concat(signals, ignore_index=True)
    output_base = processed / f'backtest_signals_{slug}'
    if save_parquet:
        out_parquet = output_base.with_suffix('.parquet')
        df_output.to_parquet(out_parquet, engine='pyarrow', compression='snappy')
        print(f'Saved backtest signals to: {out_parquet}')
    if save_csv:
        out_csv = output_base.with_suffix('.csv')
        df_output.to_csv(out_csv, index=False)
        print(f'Saved backtest signals to: {out_csv}')

    return df_output


def main():
    slug = 'ifci_3_Jun_26'
    backtest_signals(slug, save_parquet=True, save_csv=False)


if __name__ == '__main__':
    main()
