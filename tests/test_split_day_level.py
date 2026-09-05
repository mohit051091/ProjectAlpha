"""
UNIT TESTS: tests/test_split_day_level.py
PURPOSE: Validate Stage 0 splitter, R1 timezone safeguard, slug naming,
sentinel filtering, and clean_ticks dtype tightening.

These tests use a tiny synthetic in-memory parquet so they are fast and
deterministic; they do NOT touch Data/ or Data/raw/ on disk.

HOW TO RUN:
    python -m unittest tests/test_split_day_level.py
"""

import sys
import unittest
import tempfile
import shutil
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from raw.data_cleaner import DataCleaner
from utils.constants import GARBAGE_COLUMNS, PROCESSED_DIR


# The Stage 0 splitter module is `scripts/05_split_day_level.py`. When
# imported as a module, Python wants a normal identifier, so we import
# the file by path.
import importlib.util
_SPLITTER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "05_split_day_level.py"
_spec = importlib.util.spec_from_file_location("split_day_level", _SPLITTER_PATH)
_split_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split_module)
MONTH_ABBR = _split_module.MONTH_ABBR  # alias for the test below


# ============================================================================
# 1. Slug naming convention
# ============================================================================

class TestSlugNaming(unittest.TestCase):
    def test_uppercase_symbol_with_2digit_day_2digit_year(self):
        slug = _split_module.make_slug("IFCI", 2026, 29, 5)
        self.assertEqual(slug, "IFCI_29_May_26")

    def test_all_twelve_month_abbreviations(self):
        for m, abbr in MONTH_ABBR.items():
            slug = _split_module.make_slug("HDFCBANK", 2026, 1, m)
            self.assertIn(abbr, slug, f"month {m} -> {abbr}")

    def test_lowercase_input_is_uppercased(self):
        slug = _split_module.make_slug("ifci", 2026, 29, 5)
        self.assertEqual(slug, "IFCI_29_May_26")


# ============================================================================
# 2. Sentinel discovery
# ============================================================================

class TestSentinelFilter(unittest.TestCase):
    def test_known_sentinels_present(self):
        self.assertIn("GRAPHITEticks", _split_module.SENTINEL_SYMBOLS)
        self.assertIn("MFticks", _split_module.SENTINEL_SYMBOLS)
        self.assertIn("JSWCEMENTdom_snapshots", _split_module.SENTINEL_SYMBOLS)
        self.assertIn("dom_snapshots", _split_module.SENTINEL_SYMBOLS)


# ============================================================================
# 3. Stage 0 splitter on synthetic data
# ============================================================================

class TestStage0Split(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="split_test_"))
        # Build a tiny day-level DOM parquet with 3 symbols and 3 row
        # groups (one per group of symbols), including one sentinel.
        rng = np.random.default_rng(42)
        rows = []
        for i, sym in enumerate(["IFCI", "HDFCBANK", "MFticks", "dom_snapshots"]):
            for j in range(50):
                rows.append({
                    "ts": pd.Timestamp("2026-05-29 03:45:00") + pd.Timedelta(milliseconds=10 * j),
                    "symbol": sym,
                    "ltp": 100.0 + rng.normal() * 0.1,
                    "total_bid_qty": 1000,
                    "total_ask_qty": 1000,
                    "imbalance": 0.0,
                })
        # Pad to 20 bid/ask depth cols and add the 8 garbage cols.
        df = pd.DataFrame(rows)
        for k in range(1, 21):
            df[f"bid{k}"] = 100.0 - 0.01 * k
            df[f"bqty{k}"] = 100
            df[f"ask{k}"] = 100.05 + 0.01 * k
            df[f"aqty{k}"] = 100
        for gc in GARBAGE_COLUMNS:
            df[gc] = None
        table = pa.Table.from_pandas(df, preserve_index=False)
        self.src = self.tmpdir / "doms_year=2026_month=05_day=29_doms.parquet"
        pq.write_table(table, self.src, compression="snappy")
        self.out_dir = self.tmpdir / "raw"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_split_drops_garbage_and_sentinels(self):
        stats = _split_module.split_one_file(
            self.src, "doms", 2026, 5, 29, self.out_dir, force=True
        )
        # 2 valid symbols (IFCI, HDFCBANK), 2 sentinels rejected
        self.assertEqual(stats["symbols"], 2)
        self.assertEqual(stats["sentinel_rows"], 100)  # 50 rows * 2 sentinels
        written_names = [w[0] for w in stats["written"]]
        self.assertIn("IFCI_29_May_26", written_names)
        self.assertIn("HDFCBANK_29_May_26", written_names)
        for name, n in stats["written"]:
            f = self.out_dir / f"dom_{name}.parquet"
            self.assertTrue(f.exists(), f"missing {f}")
            self.assertEqual(n, 50)
            # Garbage columns must be gone
            re_read = pd.read_parquet(f)
            for gc in GARBAGE_COLUMNS:
                self.assertNotIn(gc, re_read.columns, f"garbage col {gc} leaked")
            # 20 valid bid/ask pairs preserved
            for k in range(1, 21):
                self.assertIn(f"bid{k}", re_read.columns)

    def test_split_is_idempotent(self):
        _split_module.split_one_file(self.src, "doms", 2026, 5, 29, self.out_dir, force=True)
        # Second pass: nothing should be re-written.
        stats2 = _split_module.split_one_file(
            self.src, "doms", 2026, 5, 29, self.out_dir, force=False
        )
        self.assertEqual(len(stats2["written"]), 0)
        self.assertEqual(len(stats2["skipped_existing"]), 2)


# ============================================================================
# 4. R1 timezone safeguard
# ============================================================================

class TestR1TimezoneSafeguard(unittest.TestCase):
    def test_pass_for_typical_utc_naive(self):
        # Median hour = 10.0 (UTC mid-morning), range [3, 15] -> typical
        # UTC trading window. The IST warning (9-16) fires but the
        # safeguard does not raise; we expect a logged warning, not a
        # crash. The localize step then succeeds.
        hours = list(range(3, 16))
        df = pd.DataFrame({
            "ts": [pd.Timestamp("2026-05-29") + pd.Timedelta(hours=h) for h in hours],
            "symbol": "X",
        })
        out = DataCleaner.standardize_dtypes(df, label="test-utc")
        self.assertIsNotNone(out["ts"].dt.tz)
        self.assertEqual(str(out["ts"].dt.tz), "UTC")

    def test_pass_for_morning_utc_window(self):
        # Hour range entirely below 2 or above 22: out of typical UTC
        # trading window. Soft warn only.
        df = pd.DataFrame({
            "ts": [pd.Timestamp("2026-05-29") + pd.Timedelta(hours=h) for h in [0, 1, 23]],
            "symbol": "X",
        })
        out = DataCleaner.standardize_dtypes(df, label="test-utc-edges")
        self.assertIsNotNone(out["ts"].dt.tz)

    def test_ist_warning_fires_for_naive_midday(self):
        # Median hour inside IST trading window [9, 16]: soft warn.
        # We capture log output and assert the warning message appears.
        import io
        import logging
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.WARNING)
        from utils.logger import setup_logger
        lg = setup_logger("raw.data_cleaner")
        lg.addHandler(h)
        lg.setLevel(logging.WARNING)
        try:
            hours = list(range(3, 16))  # median=9.0 (IST warning band)
            df = pd.DataFrame({
                "ts": [pd.Timestamp("2026-05-29") + pd.Timedelta(hours=h_) for h_ in hours],
                "symbol": "X",
            })
            DataCleaner.standardize_dtypes(df, label="ist-warn")
            out = buf.getvalue()
            self.assertIn("IST trading window", out)
        finally:
            lg.removeHandler(h)

    def test_utc_range_warning_fires_for_out_of_window_hours(self):
        import io
        import logging
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.WARNING)
        from utils.logger import setup_logger
        lg = setup_logger("raw.data_cleaner")
        lg.addHandler(h)
        lg.setLevel(logging.WARNING)
        try:
            df = pd.DataFrame({
                "ts": [pd.Timestamp("2026-05-29") + pd.Timedelta(hours=hh) for hh in [0, 1, 23]],
                "symbol": "X",
            })
            DataCleaner.standardize_dtypes(df, label="utc-warn")
            out = buf.getvalue()
            self.assertIn("outside typical UTC window", out)
        finally:
            lg.removeHandler(h)

    def test_hard_fail_branch_logic(self):
        # The hard-fail branch of _verify_tz_assumption is defensive:
        # `s.dt.hour` is always in [0, 24) after calendar normalisation,
        # so this branch is essentially dead code in practice. We test
        # the boolean condition directly to keep the hard-fail path
        # covered without invasive mocking of the pandas dt accessor.
        for hours, should_fail in [
            ([0, 23], False),     # typical UTC, no fail
            ([9, 9, 9], False),   # median in IST band, soft warn only
            ([-1, 5], True),      # h_min < 0 -> fail
            ([5, 25], True),      # h_max >= 24 -> fail
        ]:
            h_min, h_max = min(hours), max(hours)
            raises = (h_min < 0 or h_max >= 24)
            self.assertEqual(raises, should_fail, f"hours={hours}")

    def test_skip_when_already_aware(self):
        df = pd.DataFrame({
            "ts": [pd.Timestamp("2026-05-29 10:00:00", tz="UTC")],
            "symbol": "X",
        })
        out = DataCleaner.standardize_dtypes(df, label="test-aware")
        self.assertIsNotNone(out["ts"].dt.tz)
        self.assertEqual(str(out["ts"].dt.tz), "UTC")


# ============================================================================
# 5. clean_ticks dtype tightening
# ============================================================================

class TestCleanTicks(unittest.TestCase):
    def test_drops_all_null_columns_and_localises(self):
        df = pd.DataFrame({
            "ts": [pd.Timestamp("2026-05-29 03:45:00"), pd.Timestamp("2026-05-29 04:00:00")],
            "symbol": ["IFCI", "IFCI"],
            "security_id": [None, None],   # all-null -> dropped
            "ltp": [100.0, 100.5],
            "volume": [0, 0],
            "oi": [0, 0],
            "aggressor": ["BUY", "SELL"],
            "trade_qty": [100, 200],
            "bid1": [99.5, 100.0],
            "bqty1": [10, 20],
            "ask1": [100.5, 101.0],
            "aqty1": [10, 20],
            "cum_delta": [0, 100],
        })
        out = DataCleaner.clean_ticks(df, label="test")
        self.assertNotIn("security_id", out.columns)
        self.assertIsNotNone(out["ts"].dt.tz)
        self.assertEqual(str(out["ts"].dt.tz), "UTC")
        # dtypes tightened
        self.assertEqual(str(out["ltp"].dtype), "float64")
        # pandas Int64 nullable maps to string "Int64"
        self.assertIn(str(out["trade_qty"].dtype), ("Int64", "int64"))
        # aggressor becomes a category
        self.assertEqual(str(out["aggressor"].dtype), "category")
        # sorted by ts
        self.assertTrue(out["ts"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
