"""
Convenience imports for the CVD library.

This package now splits utilities into dedicated modules:
- feature_engineering: standardization, bucketing, domain helpers
- eda: quick and deep analysis helpers
- visualization: plotting helpers

Existing code that imported from cvd.library.data_utils will continue to work.
These modules simply re-export the relevant functions from data_utils for a
cleaner API surface without code duplication.
"""
from . import feature_engineering, eda, visualization

__all__ = [
    'feature_engineering',
    'eda',
    'visualization',
]
