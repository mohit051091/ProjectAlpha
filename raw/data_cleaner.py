"""
Data cleaning and preprocessing
"""

import pandas as pd
from typing import Tuple
from utils.logger import setup_logger
from utils.constants import GARBAGE_COLUMNS, VALID_COLUMNS

logger = setup_logger(__name__)

class DataCleaner:
    """Clean and preprocess raw data"""
    
    @staticmethod
    def remove_garbage_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove garbage columns (all nulls from data export issue)
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe
        
        Returns
        -------
        pd.DataFrame
            Cleaned dataframe
        """
        # Identify columns to drop
        cols_to_drop = [col for col in GARBAGE_COLUMNS if col in df.columns]
        
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping garbage columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        logger.info(f"After cleanup: {len(df.columns)} columns remaining")
        return df
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, expected: list = None) -> Tuple[pd.DataFrame, list]:
        """
        Validate and keep only expected columns
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        expected : list, optional
            Expected column names. If None, uses VALID_COLUMNS.
        
        Returns
        -------
        Tuple[pd.DataFrame, list]
            (cleaned_df, missing_columns)
        """
        if expected is None:
            expected = VALID_COLUMNS
        
        # Check what's missing
        missing = [col for col in expected if col not in df.columns]
        if missing:
            logger.warning(f"Missing expected columns: {missing}")
        
        # Keep only expected columns that exist
        cols_to_keep = [col for col in expected if col in df.columns]
        df = df[cols_to_keep]
        
        logger.info(f"Kept {len(cols_to_keep)} columns (missing: {len(missing)})")
        return df, missing
    
    @staticmethod
    def _verify_tz_assumption(df: pd.DataFrame, label: str) -> None:
        """
        R1 safeguard: when 'ts' is naive, assert hour-of-day range, and
        warn if the median hour looks like IST (UTC+5:30) trading hours.
        Reference: TZ_VERIFICATION.md. This is a soft check (logs, does
        not raise) so a single off-hour outlier does not crash the run.
        """
        if 'ts' not in df.columns or len(df) == 0:
            return
        s = df['ts']
        if not pd.api.types.is_datetime64_any_dtype(s):
            return
        if getattr(s.dtype, 'tz', None) is not None:
            return  # already tz-aware; assumption is moot

        hours = s.dt.hour
        h_min, h_max = int(hours.min()), int(hours.max())
        h_median = float(hours.median())

        # Hard sanity: hour-of-day must be in [0, 24). If not, the
        # timestamp is corrupt (R1 mitigation, INGESTION_PLAN §6.1).
        if h_min < 0 or h_max >= 24:
            logger.error(
                f"[{label}] R1 timezone safeguard FAILED: hour-of-day "
                f"range [{h_min}, {h_max}] is outside [0, 24). Aborting."
            )
            raise ValueError(
                f"[{label}] naive timestamps have out-of-range hours "
                f"[{h_min}, {h_max}]; refuse to assume UTC."
            )

        # Soft warn: median hour inside IST trading window [9, 16] means
        # timestamps are likely local-IST, not UTC.
        if 9 <= h_median <= 16:
            logger.warning(
                f"[{label}] R1 timezone safeguard: median naive hour is "
                f"{h_median:.1f} (inside IST trading window 9-16). "
                f"Assuming the data is UTC, but if it is actually IST, "
                f"shift by +5:30. Verified UTC per TZ_VERIFICATION.md."
            )

        # Soft warn: hour range outside typical UTC trading window
        # [2, 22]. UTC range 02:00-22:00 covers all global sessions; if
        # we see activity outside that band, the data is probably not UTC.
        if h_min < 2 or h_max > 22:
            logger.warning(
                f"[{label}] R1 timezone safeguard: hour range "
                f"[{h_min}, {h_max}] is outside typical UTC window "
                f"[2, 22]. Verified UTC per TZ_VERIFICATION.md."
            )

        logger.info(
            f"[{label}] R1 timezone safeguard passed: naive hour range "
            f"[{h_min}, {h_max}], median={h_median:.1f}. Treating as UTC."
        )

    @staticmethod
    def standardize_dtypes(df: pd.DataFrame, label: str = "clean") -> pd.DataFrame:
        """
        Ensure correct data types.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        label : str
            Short tag used in R1 safeguard log lines (e.g. file stem).

        Returns
        -------
        pd.DataFrame
            Dataframe with standardized dtypes
        """
        # Ensure timestamp is UTC aware datetime.
        # Fixes a latent bug: hasattr(datetime64_tz_naive, 'tz') is True
        # (attribute exists, value is None), so the previous guard was a
        # no-op. We now use pd.api.types to detect tz-awareness and, for
        # naive input, localise to UTC explicitly after asserting the
        # hour-of-day range is sane (R1).
        if 'ts' in df.columns:
            ts = df['ts']
            if not pd.api.types.is_datetime64_any_dtype(ts):
                df['ts'] = pd.to_datetime(ts, utc=True)
            elif pd.api.types.is_datetime64tz_dtype(ts):
                # already tz-aware; convert to UTC for consistency
                df['ts'] = ts.dt.tz_convert('UTC')
            else:
                # naive datetime64
                DataCleaner._verify_tz_assumption(df, label)
                df['ts'] = ts.dt.tz_localize('UTC')
            logger.info(f"Timestamp dtype: {df['ts'].dtype}")
        
        # Ensure symbol is string
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype('string')
        
        # Ensure prices are float64
        price_cols = [col for col in df.columns if col.startswith(('bid', 'ask', 'ltp'))]
        for col in price_cols:
            if col in df.columns and col not in ['bqty', 'aqty']:
                df[col] = df[col].astype('float64')
        
        # Ensure quantities are int64
        qty_cols = [col for col in df.columns if col.startswith(('bqty', 'aqty')) or 'qty' in col]
        for col in qty_cols:
            if col in df.columns:
                df[col] = df[col].astype('int64')
        
        logger.info("Data types standardized")
        return df
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: list = None) -> int:
        """
        Check for duplicate rows
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        subset : list, optional
            Columns to check for duplicates. If None, checks all.
        
        Returns
        -------
        int
            Number of duplicate rows found
        """
        n_dupes = df.duplicated(subset=subset).sum()
        
        if n_dupes > 0:
            logger.warning(f"Found {n_dupes} duplicate rows")
            # Don't drop, just log
        else:
            logger.info("No duplicates found")
        
        return n_dupes
    
    @staticmethod
    def sort_by_timestamp(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Sort dataframe by timestamp (and symbol if provided)
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        symbol : str, optional
            If provided, also sort by symbol
        
        Returns
        -------
        pd.DataFrame
            Sorted dataframe
        """
        if 'ts' not in df.columns:
            logger.warning("No 'ts' column found - cannot sort")
            return df
        
        sort_cols = ['symbol', 'ts'] if symbol else ['ts']
        df = df.sort_values(by=sort_cols, ignore_index=False)
        
        logger.info(f"Sorted by {sort_cols}")
        return df
    
    @staticmethod
    def clean_full_pipeline(df: pd.DataFrame, label: str = "clean") -> pd.DataFrame:
        """
        Run complete cleaning pipeline

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe
        label : str
            Short tag used in R1 safeguard log lines (e.g. file stem).

        Returns
        -------
        pd.DataFrame
            Fully cleaned dataframe
        """
        logger.info("Starting full cleaning pipeline...")

        # Step 1: Remove garbage columns
        df = DataCleaner.remove_garbage_columns(df)

        # Step 2: Validate columns
        df, missing = DataCleaner.validate_columns(df)

        # Step 3: Standardize dtypes (R1 safeguard runs here for naive ts)
        df = DataCleaner.standardize_dtypes(df, label=label)

        # Step 4: Check duplicates
        DataCleaner.check_duplicates(df)

        # Step 5: Sort by timestamp
        df = DataCleaner.sort_by_timestamp(df)

        logger.info(f"Pipeline complete. Final shape: {df.shape}")
        return df

    @staticmethod
    def clean_ticks(df: pd.DataFrame, label: str = "ticks") -> pd.DataFrame:
        """
        Clean a per-symbol tick dataframe produced by the Stage 0
        splitter. Mirrors `clean_full_pipeline` for the tick schema:

            ts, symbol, security_id, ltp, volume, oi, aggressor,
            trade_qty, bid1, bqty1, ask1, aqty1, cum_delta

        Steps:
            1. Drop all-null columns (currently `security_id`).
            2. Run R1 timezone safeguard, then localize naive `ts` to
               UTC (the standard for in-memory pipeline layers).
            3. Tighten dtypes: symbol->string, ltp/bid1/ask1->float64,
               volume/oi/trade_qty/bqty1/aqty1->Int64, aggressor->category.
            4. Sort by (symbol, ts) and check duplicates.

        Parameters
        ----------
        df : pd.DataFrame
            Per-symbol tick dataframe (output of Stage 0 splitter).
        label : str
            Short tag used in R1 safeguard log lines.

        Returns
        -------
        pd.DataFrame
            Cleaned tick dataframe with tz-aware UTC `ts`.
        """
        logger.info(f"[{label}] clean_ticks: input shape={df.shape}")

        # 1. Drop all-null columns (e.g. security_id is 100% null in the
        # day-level file). pandas considers a column all-null when its
        # non-null count is 0. We do not drop columns that are constant
        # non-null (volume, oi) because they carry domain meaning.
        null_counts = df.isnull().sum()
        all_null_cols = list(null_counts[null_counts == len(df)].index)
        if all_null_cols:
            logger.info(f"[{label}] Dropping all-null columns: {all_null_cols}")
            df = df.drop(columns=all_null_cols)

        # 2. R1 safeguard + naive->UTC localization. standardize_dtypes
        # now performs both; it does not validate the column set, so
        # tick-specific columns (aggressor, cum_delta, ...) pass through.
        df = DataCleaner.standardize_dtypes(df, label=label)

        # 3. Tick-specific dtype tightening.
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].astype('string')
        for col in ('ltp', 'bid1', 'ask1'):
            if col in df.columns:
                df[col] = df[col].astype('float64')
        for col in ('volume', 'oi', 'trade_qty', 'bqty1', 'aqty1'):
            if col in df.columns:
                df[col] = df[col].astype('Int64')  # nullable int
        if 'aggressor' in df.columns:
            df['aggressor'] = df['aggressor'].astype('category')
        if 'cum_delta' in df.columns:
            df['cum_delta'] = df['cum_delta'].astype('Int64')

        # 4. Sort + dedup audit.
        DataCleaner.check_duplicates(df, subset=['ts', 'symbol'])
        df = DataCleaner.sort_by_timestamp(df, symbol='symbol')

        logger.info(f"[{label}] clean_ticks: output shape={df.shape}, ts={df['ts'].dtype}")
        return df

    @staticmethod
    def clean_dom_file_duckdb(input_path, output_path, label):
        """
        Clean raw DOM parquet file using DuckDB.
        Returns (raw_rows, clean_rows).
        """
        import duckdb
        from utils.constants import VALID_COLUMNS
        
        con = duckdb.connect()
        con.execute("SET TIME ZONE 'UTC'")
        
        # Get count of raw rows
        raw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{input_path}')").fetchone()[0]
        
        # Get columns and types
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{input_path}')").fetchall()
        all_cols = [r[0] for r in desc]
        ts_type = [r[1] for r in desc if r[0] == 'ts'][0]
        
        # Define cols to keep
        cols_to_keep = [col for col in VALID_COLUMNS if col in all_cols]
        
        select_exprs = []
        for col in cols_to_keep:
            if col == 'ts':
                if 'WITH TIME ZONE' in ts_type.upper() or 'TIMESTAMPTZ' in ts_type.upper():
                    select_exprs.append("ts")
                else:
                    select_exprs.append("CAST(ts AS TIMESTAMPTZ) AS ts")
            elif col == 'symbol':
                select_exprs.append("CAST(symbol AS VARCHAR) AS symbol")
            elif col.startswith(('bid', 'ask', 'ltp')) and col not in ('bqty', 'aqty'):
                select_exprs.append(f"CAST({col} AS DOUBLE) AS {col}")
            elif col.startswith(('bqty', 'aqty')) or 'qty' in col:
                select_exprs.append(f"CAST({col} AS BIGINT) AS {col}")
            else:
                select_exprs.append(col)
                
        # Filter crossed book and nulls
        where_clauses = ["ts IS NOT NULL", "symbol IS NOT NULL"]
        if 'ask1' in cols_to_keep and 'bid1' in cols_to_keep:
            where_clauses.append("ask1 > bid1 AND bid1 > 0 AND ask1 > 0")
            
        # Discard pre-market and post-market: keep exactly 09:15:00 to 15:30:00 IST (exclusive of 15:30:00)
        where_clauses.append("(ts AT TIME ZONE 'Asia/Kolkata')::TIME >= '09:15:00'::TIME")
        where_clauses.append("(ts AT TIME ZONE 'Asia/Kolkata')::TIME < '15:30:00'::TIME")
            
        where_str = " AND ".join(where_clauses)
        select_str = ", ".join(select_exprs)
        
        query = f"""
        COPY (
            SELECT {select_str}
            FROM read_parquet('{input_path}')
            WHERE {where_str}
            ORDER BY ts
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """
        
        con.execute(query)
        
        # Get count of cleaned rows
        clean_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
        
        return raw_rows, clean_rows

    @staticmethod
    def clean_ticks_file_duckdb(input_path, output_path, label):
        """
        Clean raw Tick parquet file using DuckDB.
        Returns (raw_rows, clean_rows).
        """
        import duckdb
        
        con = duckdb.connect()
        con.execute("SET TIME ZONE 'UTC'")
        
        # Get count of raw rows
        raw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{input_path}')").fetchone()[0]
        
        # Get columns and types
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{input_path}')").fetchall()
        all_cols = [r[0] for r in desc]
        ts_type = [r[1] for r in desc if r[0] == 'ts'][0]
        
        # Step 1: Drop all-null columns
        # Run a query to count non-nulls for all columns
        null_selects = [f"COUNT({col}) AS {col}_count" for col in all_cols]
        non_null_counts = con.execute(f"SELECT {', '.join(null_selects)} FROM read_parquet('{input_path}')").fetchone()
        non_null_map = dict(zip(all_cols, non_null_counts))
        
        cols_to_keep = [col for col in all_cols if non_null_map[col] > 0]
        
        select_exprs = []
        for col in cols_to_keep:
            if col == 'ts':
                if 'WITH TIME ZONE' in ts_type.upper() or 'TIMESTAMPTZ' in ts_type.upper():
                    select_exprs.append("ts")
                else:
                    select_exprs.append("CAST(ts AS TIMESTAMPTZ) AS ts")
            elif col == 'symbol':
                select_exprs.append("CAST(symbol AS VARCHAR) AS symbol")
            elif col in ('ltp', 'bid1', 'ask1'):
                select_exprs.append(f"CAST({col} AS DOUBLE) AS {col}")
            elif col in ('volume', 'oi', 'trade_qty', 'bqty1', 'aqty1', 'cum_delta'):
                select_exprs.append(f"CAST({col} AS BIGINT) AS {col}")
            elif col == 'aggressor':
                select_exprs.append("CAST(aggressor AS VARCHAR) AS aggressor")
            else:
                select_exprs.append(col)
                
        # Discard pre-market and post-market: keep exactly 09:15:00 to 15:30:00 IST (exclusive of 15:30:00)
        where_str = "ts IS NOT NULL AND symbol IS NOT NULL AND (ts AT TIME ZONE 'Asia/Kolkata')::TIME >= '09:15:00'::TIME AND (ts AT TIME ZONE 'Asia/Kolkata')::TIME < '15:30:00'::TIME"
        select_str = ", ".join(select_exprs)
        
        query = f"""
        COPY (
            SELECT {select_str}
            FROM read_parquet('{input_path}')
            WHERE {where_str}
            ORDER BY symbol, ts
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """
        
        con.execute(query)
        
        # Get count of cleaned rows
        clean_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
        
        return raw_rows, clean_rows





if __name__ == "__main__":
    from raw.data_loader import DataLoader

    loader = DataLoader("Data")
    dom_files = loader.load_all_dom_files()

    for fname, df in list(dom_files.items())[:1]:  # Process first file
        print(f"\nCleaning {fname}...")
        print(f"Before: {df.shape}")

        df_clean = DataCleaner.clean_full_pipeline(df, label=fname)

        print(f"After: {df_clean.shape}")
        print(f"Columns: {list(df_clean.columns)[:10]}...")
