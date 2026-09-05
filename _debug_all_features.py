"""Test label generation on ALL 500 feature files to identify the failing one."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from labels import build_label_matrix
from utils.constants import LABEL_HORIZONS, LABEL_THRESHOLDS

files = sorted(Path('02_processed').glob('features_1m_*_25_Jun_26.parquet'))

for i, fp in enumerate(files):
    slug = fp.stem.replace('features_1m_', '')
    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        print(f'{i}: {slug} READ FAIL: {e}')
        continue
    
    if 'ltp' not in df.columns:
        print(f'{i}: {slug} NO LTP COL! cols={list(df.columns)[:8]}')
        continue
    
    if len(df) == 0:
        print(f'{i}: {slug} EMPTY DF')
        continue
    
    try:
        labeled = build_label_matrix(df, horizons=LABEL_HORIZONS, thresholds=LABEL_THRESHOLDS)
    except Exception as e:
        print(f'{i}: {slug} FAIL: {e}')
        print(f'  rows={len(df)}, ltp_in_cols={"ltp" in df.columns}')
        break
    
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/500 OK')

print('Done testing')
