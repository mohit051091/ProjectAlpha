"""Check MSUMI trigger after roc_3 fix."""
import pandas as pd

xls = pd.ExcelFile('results/performance_sweeps_v2.xlsx')
df = pd.read_excel(xls, 'L30_S25')
print('L30_S25: All rows')
exe = (df.rejection_reason == 'EXECUTED').sum()
print(f'Total: {len(df)}, Executed: {exe}')

d25 = df[df['date'] == '25_Jun_26']
print(f'\nJun 25 rows: {len(d25)}')
print(d25['rejection_reason'].value_counts().to_string())

ms = df[df['slug'].str.contains('MSUMI', na=False)]
print(f'\nMSUMI any date:')
for _, r in ms.iterrows():
    print(f'  {r.date} {r.direction} stage={r.max_stage} reason={r.rejection_reason} entry={r.entry_time_ist} pnl={r.eod_pnl_pct}')

ms25 = d25[d25['slug'].str.contains('MSUMI', na=False)]
print(f'\nMSUMI Jun 25:')
for _, r in ms25.iterrows():
    print(f'  {r.direction} stage={r.max_stage} reason={r.rejection_reason} entry={r.entry_time_ist} pnl={r.eod_pnl_pct}')
