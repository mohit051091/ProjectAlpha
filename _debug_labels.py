"""Debug which feature file 101 fails label generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from labels import build_label_matrix
from utils.constants import LABEL_HORIZONS, LABEL_THRESHOLDS

files = sorted(Path('02_processed').glob('features_1m_*_25_Jun_26.parquet'))
print(f'Total files: {len(files)}')

for i in range(95, 105):
    slug = files[i].stem.replace('features_1m_', '')
    df = pd.read_parquet(files[i])
    has_ltp = 'ltp' in df.columns
    print(f'{i}: {slug} rows={len(df)} has_ltp={has_ltp}')
    if not has_ltp:
        print(f'  FAIL: no ltp')
    else:
        try:
            labeled = build_label_matrix(df, horizons=LABEL_HORIZONS, thresholds=LABEL_THRESHOLDS)
            print(f'  OK')
        except Exception as e:
            print(f'  FAIL: {e}')
