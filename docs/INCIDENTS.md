# Incident & RCA Registry

This registry tracks bugs, outages, and issues discovered in the data, pipeline, or model code.

## Incident 1: Cross-Symbol LTP Data Contamination in Stage 0
- **Discovery Date**: 2026-06-08
- **Symptoms**: Feature files (`features_1m_*.parquet`) showed extremely large prices/LTP ranges (P99 Daily Range = 35,901%).
- **Root Cause**: In `scripts/05_split_day_level.py`, `group_by("symbol")` aggregated symbols in hash-table order, but the row slicing logic iterated using alphabetical sorting order. This mismatch caused data from different symbols to be written to wrong target files.
- **Fix**: Aligned the sorting of the hash table and slicing logic. Regenerated the split files.
- **Preventive Actions**: Added validation checks in Stage 1 data cleaner to assert that the symbol column matches the file's slug name.

## Incident 2: Timezone Mismatch in Nearest-Neighbor Alignment
- **Discovery Date**: 2026-06-12
- **Symptoms**: Merging DOM and Tick tables using `pandas.merge_asof` failed with timezone errors or yielded empty datasets.
- **Root Cause**: Datetime values in raw parquet streams were naive, and different engines (Pandas vs. DuckDB) localized them differently depending on system local time (e.g., IST on local laptop).
- **Fix**: Standardized all timestamp columns to UTC at the start of feature engineering, and configured DuckDB database connections explicitly to UTC.
- **Preventive Actions**: Added timezone verification tests in `tests/test_alignment.py`.
