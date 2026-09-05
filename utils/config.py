"""
Configuration management
Load and manage config from YAML files
"""

import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration container"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key: str, default=None) -> Any:
        """Get config value by key (supports nested keys with dots)"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access"""
        return self.get(key)
    
    def __repr__(self):
        return str(self._config)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_default_config() -> Config:
    """Get default configuration (built-in)"""
    default = {
        'data': {
            'input_path': './Data',
            'output_path': './02_processed',
            'parquet_engine': 'pyarrow',
            'compression': 'snappy',
        },
        'features': {
            'aggregation_window': '1min',
            'lookback_windows': [1, 5, 20],
            'dom_levels': 20,
        },
        'labels': {
            'forward_window': '2h',
            'move_threshold': 0.05,
            'long_threshold': 0.05,
            'short_threshold': -0.05,
        },
        'model': {
            'type': 'lightgbm',
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 7,
                'num_leaves': 31,
                'min_child_samples': 20,
            },
        },
        'validation': {
            'method': 'walk_forward',
            'train_size': 0.7,
            'val_size': 0.1,
            'test_size': 0.2,
        },
        'performance': {
            'workers': 4,
            'batch_size': 10000,
            'verbose': True,
        },
    }
    return Config(default)
