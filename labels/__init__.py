"""Label generation package."""

from .label_generator import (
    assign_label,
    build_label_matrix,
    compute_forward_return,
    summarize_all_label_configs,
    summarize_label_counts,
)

__all__ = [
    'assign_label',
    'build_label_matrix',
    'compute_forward_return',
    'summarize_all_label_configs',
    'summarize_label_counts',
]
