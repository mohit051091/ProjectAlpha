"""
Constants, enums, and shared definitions
"""

from enum import Enum
from typing import Dict, List

# ============================================================================
# ENUMS
# ============================================================================

class Label(Enum):
    """Price movement labels"""
    LONG = "LONG"           # Expected move > +5%
    SHORT = "SHORT"         # Expected move < -5%
    NO_TRADE = "NO_TRADE"   # Expected move between -5% and +5%

class TradeDirection(Enum):
    """Trade direction (aggressor side)"""
    BUY = "BUY"             # Aggressive buy (hit ask)
    SELL = "SELL"           # Aggressive sell (hit bid)
    UNKNOWN = "UNKNOWN"     # Cannot determine

class TradeSignificance(Enum):
    """Trade size classification"""
    SMALL = "SMALL"         # < median size
    NORMAL = "NORMAL"       # median to 5x median
    LARGE = "LARGE"         # > 5x median

# ============================================================================
# COLUMN GROUPS
# ============================================================================

GARBAGE_COLUMNS = [
    'symbolbid17', 'aqk9', 'asbqty15', 'bid1d15', 
    '19', 'im', 'askqty', 'aqtybid1'
]

CORE_COLUMNS = ['ts', 'symbol', 'ltp']

AGGREGATE_COLUMNS = ['total_bid_qty', 'total_ask_qty', 'imbalance']

DOM_PRICE_COLUMNS = [
    'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 
    'bid6', 'bid7', 'bid8', 'bid9', 'bid10',
    'bid11', 'bid12', 'bid13', 'bid14', 'bid15',
    'bid16', 'bid17', 'bid18', 'bid19', 'bid20',
    'ask1', 'ask2', 'ask3', 'ask4', 'ask5',
    'ask6', 'ask7', 'ask8', 'ask9', 'ask10',
    'ask11', 'ask12', 'ask13', 'ask14', 'ask15',
    'ask16', 'ask17', 'ask18', 'ask19', 'ask20',
]

DOM_QTY_COLUMNS = [
    'bqty1', 'bqty2', 'bqty3', 'bqty4', 'bqty5',
    'bqty6', 'bqty7', 'bqty8', 'bqty9', 'bqty10',
    'bqty11', 'bqty12', 'bqty13', 'bqty14', 'bqty15',
    'bqty16', 'bqty17', 'bqty18', 'bqty19', 'bqty20',
    'aqty1', 'aqty2', 'aqty3', 'aqty4', 'aqty5',
    'aqty6', 'aqty7', 'aqty8', 'aqty9', 'aqty10',
    'aqty11', 'aqty12', 'aqty13', 'aqty14', 'aqty15',
    'aqty16', 'aqty17', 'aqty18', 'aqty19', 'aqty20',
]

VALID_COLUMNS = CORE_COLUMNS + AGGREGATE_COLUMNS + DOM_PRICE_COLUMNS + DOM_QTY_COLUMNS

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

class FeatureGroup(Enum):
    """Feature groups matching the brief"""
    TRADE_FLOW = "Trade Flow"
    MARKET_DEPTH = "Market Depth"
    PRICE_DERIVED = "Price Derived"
    MICROSTRUCTURE = "Microstructure"

FEATURE_REGISTRY: Dict[str, Dict] = {
    # Group A: Trade Flow (6 features)
    'delta_1m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Buy volume - Sell volume (1 min)',
        'source': 'Tick',
    },
    'delta_5m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Buy volume - Sell volume (5 min)',
        'source': 'Tick',
    },
    'volume_burst': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Current volume / rolling avg (20 periods)',
        'source': 'Tick',
    },
    'aggressor_ratio': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Aggressive buy trades / total trades',
        'source': 'Tick',
    },
    'trade_count_burst': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Trade count / rolling avg',
        'source': 'Tick',
    },
    'large_trade_ratio': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Trades > 10x median / total',
        'source': 'Tick',
    },
    
    # Group B: Market Depth (6 features)
    'imbalance_top5': {
        'group': FeatureGroup.MARKET_DEPTH,
        'description': '(BidQty - AskQty) / (BidQty + AskQty) - top 5 levels',
        'source': 'DOM',
    },
    'spread': {
        'group': FeatureGroup.MARKET_DEPTH,
        'description': 'ask1 - bid1 (best bid-ask gap)',
        'source': 'DOM',
    },
    'depth_drop_bid': {
        'group': FeatureGroup.MARKET_DEPTH,
        'description': 'Change in total bid qty',
        'source': 'DOM',
    },
    'depth_drop_ask': {
        'group': FeatureGroup.MARKET_DEPTH,
        'description': 'Change in total ask qty',
        'source': 'DOM',
    },
    
    # Group C: Price Derived (3 features)
    'vwap_distance': {
        'group': FeatureGroup.PRICE_DERIVED,
        'description': '(Price - VWAP) / VWAP',
        'source': 'Tick',
    },
    'volatility_5m': {
        'group': FeatureGroup.PRICE_DERIVED,
        'description': 'Std dev of prices (5 min window)',
        'source': 'Tick',
    },
    'price_acceleration': {
        'group': FeatureGroup.PRICE_DERIVED,
        'description': '2nd derivative of price',
        'source': 'Tick',
    },
    
    # Group D: Microstructure (3 features)
    'iceberg_score': {
        'group': FeatureGroup.MICROSTRUCTURE,
        'description': 'Executed qty / displayed qty - hidden order detection',
        'source': 'Tick+DOM',
    },
    'order_cancel_rate': {
        'group': FeatureGroup.MICROSTRUCTURE,
        'description': 'Orders cancelled / placed (spoofing signal)',
        'source': 'Tick',
    },
    'bid_replenishment_rate': {
        'group': FeatureGroup.MICROSTRUCTURE,
        'description': 'How fast bid qty refills (absorption signal)',
        'source': 'DOM',
    },
    'absorption_buyer_1m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Passive buyer absorption of aggressive selling (1 min)',
        'source': 'Tick+DOM',
    },
    'absorption_buyer_5m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Passive buyer absorption of aggressive selling (5 min)',
        'source': 'Tick+DOM',
    },
    'absorption_seller_1m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Passive seller absorption of aggressive buying (1 min)',
        'source': 'Tick+DOM',
    },
    'absorption_seller_5m': {
        'group': FeatureGroup.TRADE_FLOW,
        'description': 'Passive seller absorption of aggressive buying (5 min)',
        'source': 'Tick+DOM',
    },
}

# Get features by group
FEATURES_BY_GROUP = {}
for fname, finfo in FEATURE_REGISTRY.items():
    group = finfo['group']
    if group not in FEATURES_BY_GROUP:
        FEATURES_BY_GROUP[group] = []
    FEATURES_BY_GROUP[group].append(fname)

# ============================================================================
# NUMERIC RANGES FOR VALIDATION
# ============================================================================

FEATURE_RANGES = {
    'delta_1m': (-1_000_000, 1_000_000),
    'delta_5m': (-5_000_000, 5_000_000),
    'volume_burst': (0.01, 100),
    'aggressor_ratio': (0, 1),
    'trade_count_burst': (0.01, 100),
    'large_trade_ratio': (0, 1),
    'imbalance_top5': (-1, 1),
    'spread': (0.01, 1000),
    'depth_drop_bid': (-1_000_000, 1_000_000),
    'depth_drop_ask': (-1_000_000, 1_000_000),
    'vwap_distance': (-0.5, 0.5),
    'volatility_5m': (0, 50),
    'price_acceleration': (-10, 10),
    'iceberg_score': (0, 100),
    'order_cancel_rate': (0, 1),
    'bid_replenishment_rate': (0, 1),
    'absorption_buyer_1m': (0, 1.0),
    'absorption_buyer_5m': (0, 1.0),
    'absorption_seller_1m': (0, 1.0),
    'absorption_seller_5m': (0, 1.0),
}

EXPERIMENTAL_FEATURES = [
    'iceberg_score',
    'order_cancel_rate',
]

ACTIVE_FEATURES = [
    fname for fname in FEATURE_REGISTRY.keys()
    if fname not in EXPERIMENTAL_FEATURES
]

LABEL_HORIZONS = [30, 60, 120]
LABEL_THRESHOLDS = [0.01, 0.03, 0.05, 0.07]

# ============================================================================
# ALIGNMENT CONFIGURATION
# ============================================================================

ALIGNMENT_CONFIG = {
    'tolerance_ms': 1000,
    'aqs_weights': {
        'unmatched_ratio': 0.6,
        'avg_distance': 0.4,
    },
    'aqs_threshold': 0.7,
    'dcs_threshold': 0.6,
    'wvs_threshold': 0.6,
    'min_tick_count': 1,
    'min_dom_count': 1,
}

# ============================================================================
# PATHS (portable — can be overridden by PROJECTALPHA_DATA env var)
# ============================================================================

import os as _os
from pathlib import Path as _Path

def _resolve_data_dir() -> str:
    env = _os.environ.get("PROJECTALPHA_DATA")
    if env:
        return str(_Path(env).resolve())
    return "Data"

DATA_DIR = _resolve_data_dir()
PROCESSED_DIR = "02_processed"
FEATURES_DIR = "03_features"
MODELS_DIR = "06_models"
CONFIG_DIR = "config"
SCRIPTS_DIR = "scripts"

# ============================================================================
# THRESHOLDS
# ============================================================================

MOVE_THRESHOLD_DEFAULT = 0.05  # 5% move
MOVE_THRESHOLD_LONG = 0.05     # > 5% = LONG
MOVE_THRESHOLD_SHORT = -0.05   # < -5% = SHORT

LOOK_AHEAD_HOURS = 2           # 2-hour forward prediction window
LOOK_AHEAD_MINUTES = 120       # Same as above

AGGREGATION_WINDOW = "1min"    # 1-minute bars

# ============================================================================
# PRINT INFO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONSTANTS, ENUMS, AND DEFINITIONS")
    print("="*80)
    print(f"\nValid DOM columns: {len(VALID_COLUMNS)}")
    print(f"Garbage columns to drop: {len(GARBAGE_COLUMNS)}")
    print(f"Total features defined: {len(FEATURE_REGISTRY)}")
    print(f"Features by group:")
    for group, features in FEATURES_BY_GROUP.items():
        print(f"  {group.value}: {len(features)} features")
        for f in features:
            print(f"    - {f}")
    print("\n" + "="*80)
