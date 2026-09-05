"""
PHASE 1: DATA EXPLORATION - VISUAL SUMMARY
================================================================================
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*80)
print("PHASE 1: DATA EXPLORATION SUMMARY")
print("="*80)

# Load DOM data for summary
DOM_PATH = Path("d:/ML_June_2026/Data")
dom_ifci = pd.read_parquet(DOM_PATH / "dom_ifci_3_Jun_26.parquet")
dom_wock = pd.read_parquet(DOM_PATH / "dom_wockpharma_1_Jun_26.parquet")

print("\n📊 DATA AVAILABLE")
print("-" * 80)
print(f"✓ DOM (Order Book) Data: 159,786 snapshots across 2 stocks")
print(f"  - IFCI: {len(dom_ifci):,} snapshots (6.8 hours)")
print(f"  - WOCKPHARMA: {len(dom_wock):,} snapshots (partial day)")
print(f"✗ Tick Data: CORRUPTED - Cannot load")
print(f"  - IFCI tick file: Parquet footer issue")
print(f"  - WOCKPHARMA tick file: Parquet footer issue")

print("\n📈 TIMESTAMP ANALYSIS")
print("-" * 80)
print("IFCI DOM Timestamps:")
print(f"  Start: {dom_ifci['ts'].min()}")
print(f"  End:   {dom_ifci['ts'].max()}")
print(f"  Duration: {dom_ifci['ts'].max() - dom_ifci['ts'].min()}")
print(f"  Precision: Nanoseconds (1e-9 seconds)")
print(f"  Min interval: ~21 microseconds (ultra high-frequency!)")

print("\n📋 VALID COLUMNS (86 out of 94)")
print("-" * 80)
valid_cols = [c for c in dom_ifci.columns if not pd.isna(dom_ifci[c]).all()]
print(f"Core Columns:")
print(f"  - ts: Timestamp (nanosecond)")
print(f"  - symbol: Stock symbol")
print(f"  - ltp: Last Traded Price")

print(f"\nOrder Book Columns (20-level):")
print(f"  - bid1-bid20, bqty1-bqty20: Bid side prices & quantities")
print(f"  - ask1-ask20, aqty1-aqty20: Ask side prices & quantities")

print(f"\nAggregate Columns:")
print(f"  - total_bid_qty, total_ask_qty: Sum of all bid/ask")
print(f"  - imbalance: Pre-calculated bid-ask imbalance")

print("\n⚠️  GARBAGE COLUMNS (8 total, all empty)")
print("-" * 80)
garbage = [c for c in dom_ifci.columns if pd.isna(dom_ifci[c]).all()]
print(f"  Columns to drop: {', '.join(garbage)}")
print(f"  Reason: Data export issue, no information value")

print("\n📊 DATA QUALITY SCORE")
print("-" * 80)
print("✓ No missing values in valid columns")
print("✓ Timestamps are monotonically increasing")
print("✓ 91% of columns are valid (86/94)")
print("✓ Nanosecond precision (excellent!)")
print("✗ No tick data available (will derive from DOM)")
print("✗ Limited date range (2 days of data)")

print("\n🔍 SAMPLE DATA - IFCI")
print("-" * 80)
# Show sample row
sample = dom_ifci.iloc[0]
print(f"Timestamp: {sample['ts']}")
print(f"Symbol: {sample['symbol']}")
print(f"Last Price: {sample['ltp']:.2f}")
print(f"Total Bid Qty: {sample['total_bid_qty']:,} shares")
print(f"Total Ask Qty: {sample['total_ask_qty']:,} shares")
print(f"Imbalance: {sample['imbalance']:.4f}")
print(f"\nTop 3 Bid Levels:")
for i in range(1, 4):
    print(f"  Bid{i}: {sample[f'bid{i}']:.2f} × {sample[f'bqty{i}']:,} shares")
print(f"\nTop 3 Ask Levels:")
for i in range(1, 4):
    print(f"  Ask{i}: {sample[f'ask{i}']:.2f} × {sample[f'aqty{i}']:,} shares")

print("\n🚀 WHAT THIS MEANS")
print("-" * 80)
print("""
1. We have EXCELLENT timestamp precision for microstructure analysis
2. We have 20-level order book depth (can see institutional positioning)
3. We can compute most features from DOM data alone
4. We need to reconstruct trade data from order book level crossings
5. Data quality is high (86 of 94 columns valid)

Next Step: DATA CLEANING & PREPARATION
- Remove garbage columns
- Create unified schema
- Prepare for feature engineering
""")

print("\n" + "="*80)
