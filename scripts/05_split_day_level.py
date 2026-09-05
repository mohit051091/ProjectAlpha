"""
SCRIPT 05: SPLIT DAY-LEVEL PARQUETS INTO PER-SYMBOL FILES (STAGE 0, ONE-TIME)

Streams the hive-style day-level parquets in `Data/`
(`doms_year=YYYY_month=MM_day=DD_doms.parquet` and
`ticks_year=YYYY_month=MM_day=DD_ticks.parquet`) row-group by row-group,
uses `pyarrow.compute` for sentinel filtering and per-symbol splitting,
and writes one parquet per symbol per day to `Data/raw/` via
`pyarrow.parquet.ParquetWriter` (streaming append).

Algorithm (per row group):
    1. `pf.read_row_group(rg)` -> pyarrow.Table
    2. `tbl.drop(GARBAGE_COLUMNS)` (DOMs only)
    3. `tbl.filter(~pc.is_in(tbl['symbol'], SENTINELS))` -> sentinels out
    4. `tbl.group_by('symbol').aggregate([('ts', 'count')])` -> per-symbol counts
    5. `pc.sort_indices(tbl, sort_keys=[('symbol', 'ascending')])` -> sort order
    6. `tbl.take(indices)` -> sorted view (zero-copy)
    7. `sorted_tbl.slice(off, n)` -> per-symbol sub-table
    8. `ParquetWriter.write_table(sub)` -> append to per-symbol output

Naming convention (per INGESTION_PLAN.md §5.7):
    Data/raw/dom_<SYM>_<DD>_<Mon>_<YY>.parquet
    Data/raw/tick_<SYM>_<DD>_<Mon>_<YY>.parquet
    e.g.  Data/raw/dom_IFCI_29_May_26.parquet

Properties:
    - Read-only on `Data/`. The day-level source parquets are never
      modified; they remain the canonical rollback target.
    - Writes only to `Data/raw/` (created if missing).
    - Idempotent: per-row-group write of the same symbol is atomic on
      tmp+rename at close. Re-running with a prior successful run is a
      no-op (overwrite).
    - Drops the 4 known sentinels: GRAPHITEticks, MFticks,
      JSWCEMENTdom_snapshots, dom_snapshots.
    - Drops `GARBAGE_COLUMNS` on the DOM side.
    - Does NOT localize timestamps. Stage 0 preserves the source
      dtype exactly. Localization to UTC happens in `DataCleaner`
      (Stage 1), which is the single, audited timezone boundary.

Usage:
    python scripts/05_split_day_level.py
"""

import argparse
import re
import sys
import time
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.logger import setup_logger
from utils.constants import DATA_DIR, GARBAGE_COLUMNS

logger = setup_logger(__name__)

DAY_LEVEL_PATTERN = re.compile(
    r"^(doms|ticks|dom_snapshots)_year=(?P<year>\d{4})_month=(?P<month>\d{2})_day=(?P<day>\d{2})_(doms|ticks|dom_snapshots)\.parquet$"
)

SENTINEL_SYMBOLS = {
    "GRAPHITEticks",
    "MFticks",
    "JSWCEMENTdom_snapshots",
    "dom_snapshots",
}

MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def make_slug(symbol: str, year: int, day: int, month: int) -> str:
    """Build a slug of the form `<SYM>_<DD>_<Mon>_<YY>` (uppercase symbol)."""
    return f"{str(symbol).upper()}_{int(day):02d}_{MONTH_ABBR[int(month)]}_{int(year) % 100:02d}"


def discover_day_level_files(data_dir: Path) -> list:
    """Return [(path, kind, year, month, day)] for every day-level parquet."""
    found = []
    for f in sorted(data_dir.glob("*.parquet")):
        m = DAY_LEVEL_PATTERN.match(f.name)
        if not m:
            continue
        kind = "doms" if "dom" in m.group(1) else "ticks"
        found.append(
            (
                f,
                kind,
                int(m.group("year")),
                int(m.group("month")),
                int(m.group("day")),
            )
        )
    return found


class _SymbolSink:
    """
    Lazily opens a `ParquetWriter` per symbol and appends sub-tables
    to it. At `close_all()`, every temp file is renamed to its final
    name atomically, guaranteeing no half-written outputs in
    `Data/raw/`.
    """

    def __init__(self, out_dir: Path, prefix: str, year: int, month: int, day: int, force: bool = False):
        self.out_dir = out_dir
        self.prefix = prefix
        self.year, self.month, self.day = year, month, day
        self.force = force
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writers: dict = {}        # sym -> ParquetWriter
        self.tmp_paths: dict = {}      # sym -> tmp Path
        self.final_paths: dict = {}    # sym -> final Path
        self.row_counts: dict = {}     # sym -> int
        self.skipped_symbols: set = set()

    def _open(self, sym: str, schema: pa.Schema) -> None:
        slug = make_slug(sym, self.year, self.day, self.month)
        final = self.out_dir / f"{self.prefix}_{slug}.parquet"
        tmp = self.out_dir / f".{self.prefix}_{slug}.parquet.tmp"
        writer = pq.ParquetWriter(str(tmp), schema, compression="snappy")
        self.writers[sym] = writer
        self.tmp_paths[sym] = tmp
        self.final_paths[sym] = final
        self.row_counts[sym] = 0

    def append(self, sym: str, sub: pa.Table) -> None:
        if sym in self.skipped_symbols:
            return
        if sym not in self.writers:
            slug = make_slug(sym, self.year, self.day, self.month)
            final = self.out_dir / f"{self.prefix}_{slug}.parquet"
            if not self.force and final.exists():
                self.skipped_symbols.add(sym)
                return
            self._open(sym, sub.schema)
        self.writers[sym].write_table(sub)
        self.row_counts[sym] += sub.num_rows

    def close_all(self) -> None:
        import os
        for sym, writer in self.writers.items():
            writer.close()
            tmp = self.tmp_paths[sym]
            final = self.final_paths[sym]
            os.replace(str(tmp), str(final))


def split_one_file(
    src: Path,
    kind: str,
    year: int,
    month: int,
    day: int,
    out_dir: Path,
    force: bool = False,
) -> dict:
    """
    Stream one day-level parquet, group by `symbol`, write per-symbol files.

    Returns a stats dict with row counts and skip / write / sentinel
    tallies for the audit log.
    """
    stats = {
        "src": src.name,
        "kind": kind,
        "row_groups": 0,
        "src_rows": 0,
        "written": [],
        "skipped_existing": [],
        "sentinel_rows": 0,
        "symbols": 0,
    }

    pf = pq.ParquetFile(src)
    stats["row_groups"] = pf.num_row_groups
    logger.info(
        f"Streaming {src.name} | row_groups={pf.num_row_groups} | "
        f"num_rows={pf.metadata.num_rows:,} | schema_cols={pf.schema_arrow.names[:5]}..."
    )

    prefix = "dom" if kind == "doms" else "tick"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vectorised sentinel filter built once.
    sentinel_array = pa.array(list(SENTINEL_SYMBOLS))

    # Per-row-group timing
    sink = _SymbolSink(out_dir, prefix, year, month, day, force=force)
    t_loop0 = time.time()
    chunk_size = 1 if kind == "doms" else 20
    num_rgs = pf.num_row_groups
    num_chunks = (num_rgs + chunk_size - 1) // chunk_size
    t_rg_times = []

    for rg_idx in range(pf.num_row_groups):
        t_rg0 = time.time()
        tbl = pf.read_row_group(rg_idx)
        stats["src_rows"] += tbl.num_rows

        # Drop garbage columns (DOMs only).
        if kind == "doms":
            drop_names = [c for c in GARBAGE_COLUMNS if c in tbl.column_names]
            if drop_names:
                tbl = tbl.drop(drop_names)

        # Filter sentinels using `arrow.compute.is_in` (one pass).
        if "symbol" in tbl.column_names and tbl.num_rows > 0:
            mask_in = pc.is_in(tbl["symbol"], value_set=sentinel_array)
            n_removed = int(pc.sum(pc.cast(mask_in, pa.int64())).as_py() or 0)
            stats["sentinel_rows"] += n_removed
            mask_keep = pc.invert(mask_in)
            tbl = tbl.filter(mask_keep)

        if tbl.num_rows == 0 or "symbol" not in tbl.column_names:
            continue

        # Sort by symbol once, then slice into per-symbol sub-tables.
        # group_by aggregate gives us per-symbol row counts in one hash
        # pass; sort_indices + take is the cheapest path to a sorted view.
        agg = tbl.group_by("symbol").aggregate([("ts", "count")]).sort_by([("symbol", "ascending")])
        syms = agg["symbol"].to_pylist()
        counts = agg["ts_count"].to_pylist()

        # Sort the source table by symbol so we can slice contiguously.
        sort_idx = pc.sort_indices(tbl, sort_keys=[("symbol", "ascending")])
        sorted_tbl = tbl.take(sort_idx)

        # Slice offsets in the sorted view.
        off = 0
        for sym, n in zip(syms, counts):
            if not sym or pd_is_nan(sym) or sym == "":
                off += n
                continue
            sub = sorted_tbl.slice(off, n)
            sink.append(sym, sub)
            off += n

        t_rg = time.time() - t_rg0
        t_rg_times.append(t_rg)
        elapsed = time.time() - t_loop0
        avg_rg = elapsed / (rg_idx + 1)
        eta = avg_rg * (pf.num_row_groups - rg_idx - 1)
        logger.info(
            f"  rg={rg_idx + 1:3d}/{pf.num_row_groups} | rows={tbl.num_rows:>9,} | "
            f"rg_time={t_rg:5.1f}s | elapsed={elapsed:6.1f}s | "
            f"eta={eta:6.1f}s | symbols_seen={len(sink.writers)}"
        )

    # Close all writers; atomic rename of .tmp -> final paths.
    sink.close_all()
    stats["symbols"] = len(sink.row_counts) + len(sink.skipped_symbols)
    for sym, n in sink.row_counts.items():
        slug = sink.final_paths[sym].stem.replace(f"{prefix}_", "")
        stats["written"].append((slug, n))
    for sym in sorted(sink.skipped_symbols):
        slug = make_slug(sym, year, day, month)
        stats["skipped_existing"].append(slug)

    return stats


def pd_is_nan(x) -> bool:
    """Cheap is-NaN check that doesn't require importing pandas."""
    return x is None


def main():
    parser = argparse.ArgumentParser(description="Stage 0: split day-level parquets per symbol.")
    parser.add_argument("--force", action="store_true", help="Re-write even when output exists.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with day-level parquets.")
    parser.add_argument("--out-dir", default=str(Path(DATA_DIR) / "raw"),
                        help="Directory to write per-symbol parquets.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STEP 05: SPLIT DAY-LEVEL PARQUETS -> PER-SYMBOL (Stage 0, one-time)")
    logger.info("=" * 80)
    logger.info(f"  data_dir = {data_dir.resolve()}")
    logger.info(f"  out_dir  = {out_dir.resolve()}")
    logger.info(f"  force    = {args.force}")
    logger.info(f"  sentinels excluded: {sorted(SENTINEL_SYMBOLS)}")
    logger.info(f"  garbage columns dropped from DOMs: {GARBAGE_COLUMNS}")

    files = discover_day_level_files(data_dir)
    if not files:
        logger.error(f"No day-level parquets found in {data_dir}. Nothing to do.")
        return 1

    logger.info(f"Discovered {len(files)} day-level parquet file(s).")

    t0 = time.time()
    summary = []
    for src, kind, year, month, day in files:
        day_slug = f"_{day:02d}_{MONTH_ABBR[month]}_{year%100:02d}.parquet"
        prefix = "dom" if kind == "doms" else "tick"
        
        # Skip if labeled features for this day already exist in 02_processed/ (saves massive processing time)
        processed_dir = out_dir.parent.parent / "02_processed"
        existing_processed = list(processed_dir.glob(f"labeled_features_1m_*{day_slug}"))
        if not args.force and len(existing_processed) > 0:
            logger.info(f"Skipping {src.name} (date {year:04d}-{month:02d}-{day:02d}) - {len(existing_processed)} processed files already exist in {processed_dir.name}.")
            continue

        # Fallback: Skip if split files for this day already exist to save streaming time
        existing_files = list(out_dir.glob(f"{prefix}_*{day_slug}"))
        if not args.force and len(existing_files) > 0:
            logger.info(f"Skipping {src.name} (date {year:04d}-{month:02d}-{day:02d}) - {len(existing_files)} split files already exist.")
            continue

        logger.info("")
        logger.info("-" * 80)
        logger.info(f"Processing: {src.name}  (kind={kind}, date={year:04d}-{month:02d}-{day:02d})")
        logger.info("-" * 80)
        try:
            stats = split_one_file(src, kind, year, month, day, out_dir, force=args.force)
        except Exception as e:
            logger.error(f"FAILED to split {src.name}: {e}")
            continue
        summary.append(stats)
        logger.info(
            f"Summary for {src.name}: "
            f"row_groups={stats['row_groups']} src_rows={stats['src_rows']:,} "
            f"symbols={stats['symbols']} written={len(stats['written'])} "
            f"skipped_existing={len(stats['skipped_existing'])} "
            f"sentinel_rows={stats['sentinel_rows']:,}"
        )

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 80)
    logger.info("STAGE 0 COMPLETE [OK]")
    logger.info("=" * 80)
    logger.info(f"  files processed        : {len(summary)}")
    logger.info(f"  symbols written        : {sum(len(s['written']) for s in summary)}")
    logger.info(f"  symbols skipped (idemp): {sum(len(s['skipped_existing']) for s in summary)}")
    logger.info(f"  total src rows         : {sum(s['src_rows'] for s in summary):,}")
    logger.info(f"  sentinel rows rejected : {sum(s['sentinel_rows'] for s in summary):,}")
    logger.info(f"  elapsed (s)            : {elapsed:.1f}")
    logger.info(f"  outputs in             : {out_dir.resolve()}")
    logger.info("Next step: run scripts/10_prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
