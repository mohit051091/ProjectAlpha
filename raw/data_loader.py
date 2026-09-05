"""
Raw data loading and basic validation
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from utils.logger import setup_logger
from utils.constants import GARBAGE_COLUMNS, VALID_COLUMNS, DATA_DIR

logger = setup_logger(__name__)

class DataLoader:
    """Load parquet files and basic validation"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = DATA_DIR
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        logger.info(f"DataLoader initialized with path: {data_path}")
    
    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load a parquet file
        
        Parameters
        ----------
        filename : str
            Name of parquet file (e.g., "dom_ifci_3_Jun_26.parquet")
        
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_parquet(filepath, engine='pyarrow')
            logger.info(f"Loaded {filename}: {len(df):,} rows × {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def load_all_dom_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all DOM (order book) files. Walks both `Data/dom_*.parquet`
        (pre-existing per-symbol files) and `Data/raw/dom_*.parquet`
        (Stage 0 splitter outputs, post-2026-05-29 ingest).

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of {filename: dataframe}
        """
        search_dirs = [self.data_path]
        raw_dir = self.data_path / "raw"
        if raw_dir.exists():
            search_dirs.append(raw_dir)

        result = {}
        seen = set()
        for d in search_dirs:
            for f in sorted(d.glob("dom_*.parquet")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                try:
                    df = pd.read_parquet(f, engine='pyarrow')
                    result[f.name] = df
                    logger.info(f"Loaded {f.name}: {len(df):,} rows")
                except Exception as e:
                    logger.warning(f"Could not load {f.name}: {e}")

        if not result:
            raise FileNotFoundError("No DOM files loaded. Check data path.")

        logger.info(f"Loaded {len(result)} DOM files")
        return result

    def load_all_tick_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all tick (trade) files. Walks both `Data/tick_*.parquet`
        (pre-existing per-symbol files) and `Data/raw/tick_*.parquet`
        (Stage 0 splitter outputs, post-2026-05-29 ingest). Returns an
        empty dict (not an error) if no tick files exist -- the original
        2-day corpus has no cleaned tick files.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of {filename: dataframe}
        """
        search_dirs = [self.data_path]
        raw_dir = self.data_path / "raw"
        if raw_dir.exists():
            search_dirs.append(raw_dir)

        result = {}
        seen = set()
        for d in search_dirs:
            for f in sorted(d.glob("tick_*.parquet")):
                if f.name in seen:
                    continue
                seen.add(f.name)
                try:
                    df = pd.read_parquet(f, engine='pyarrow')
                    result[f.name] = df
                    logger.info(f"Loaded {f.name}: {len(df):,} rows")
                except Exception as e:
                    logger.warning(f"Could not load {f.name}: {e}")

        if not result:
            logger.warning("No tick files loaded (expected if no Stage 0 split has run).")

        return result
    
    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str] = None) -> bool:
        """
        Validate dataframe schema
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        expected_columns : List[str], optional
            Expected column names. If None, checks for valid DOM columns.
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if expected_columns is None:
            expected_columns = VALID_COLUMNS
        
        # Check columns exist
        missing = set(expected_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        # Check dtypes for critical columns
        critical_checks = {
            'ts': 'datetime64[ns, UTC]',
            'symbol': 'object',
            'ltp': 'float64',
        }
        
        for col, expected_dtype in critical_checks.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not actual_dtype.startswith(expected_dtype.split('[')[0]):
                    logger.warning(f"Column {col}: expected {expected_dtype}, got {actual_dtype}")
        
        logger.info("Schema validation passed")
        return True
    
    def check_nulls(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for null values
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check
        
        Returns
        -------
        Dict[str, int]
            {column: null_count}
        """
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        
        if len(nulls) > 0:
            logger.warning(f"Found nulls:\n{nulls}")
        else:
            logger.info("No null values found")
        
        return nulls.to_dict()


if __name__ == "__main__":
    loader = DataLoader("Data")
    dom_files = loader.load_all_dom_files()
    for fname, df in dom_files.items():
        print(f"\n{fname}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:10]}...")  # First 10
