"""Debug the first feature file that fails label generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from labels import build_label_matrix
from utils.constants import LABEL_HORIZONS, LABEL_THRESHOLDS

files = sorted(Path('02_processed').glob('features_1m_*_25_Jun_26.parquet'))
print(f'Total files: {len(files)}')

slug0 = files[0].stem.replace('features_1m_', '')
print(f'File 0: {slug0}')
df = pd.read_parquet(files[0])
print(f'rows={len(df)}, ltp={"ltp" in df.columns}')
if 'ltp' in df.columns:
    print(f'ltp first 3: {df["ltp"].head(3).tolist()}')
    try:
        labeled = build_label_matrix(df, horizons=LABEL_HORIZONS, thresholds=LABEL_THRESHOLDS)
        print('OK')
    except Exception as e:
        print(f'FAIL: {e}')
else:
    print('NO LTP COLUMN!')
    print(list(df.columns))
