"""
Updated helpers.py to use the new configuration system.

This module now imports from config and provides backward-compatible
wrappers for existing code.
"""

import pandas as pd
from config import colors, analysis


RISK_PALETTE = colors.RISK_PALETTE
GENDER_PALETTE = colors.GENDER_PALETTE


def get_risk_cat(val):
    """Categorize a risk value into 5-band classification."""
    return analysis.get_risk_category(val, use_5band=True)


def get_risk_cat_4band(val):
    """Categorize a risk value into 4-band classification."""
    return analysis.get_risk_category(val, use_5band=False)


def get_risk_color(category: str, use_hex: bool = True) -> str:
    """Get color for a risk category."""
    return colors.get_risk_color(category, use_hex=use_hex)


def get_gender_color(gender: str) -> str:
    """Get color for a gender category."""
    return colors.get_gender_color(gender)


def validate_risk_value(value: float) -> bool:
    """Check if a risk value is valid."""
    return analysis.validate_risk_value(value)


def convert_risk_horizon(risk_10y: float, to_5year: bool = True) -> float:
    """Convert between 10-year and 5-year risk."""
    if to_5year:
        return analysis.convert_10y_to_5y_risk(risk_10y)
    return risk_10y


def add_risk_categories(df: pd.DataFrame, 
                        risk_col: str = 'risk_nonlab',
                        output_col: str = 'risk_cat',
                        use_5band: bool = True) -> pd.DataFrame:
    """Add risk category column to dataframe."""
    df = df.copy()
    bins = analysis.RISK_BINS if use_5band else analysis.RISK_BINS_4BAND
    labels = analysis.RISK_LABELS if use_5band else analysis.RISK_LABELS_4BAND
    
    df[output_col] = pd.cut(
        df[risk_col],
        bins=bins,
        labels=labels,
        right=False
    )
    return df


def add_age_bands(df: pd.DataFrame,
                  age_col: str = 'age',
                  output_col: str = 'age_band') -> pd.DataFrame:
    """Add age band column to dataframe."""
    df = df.copy()
    df[output_col] = pd.cut(
        df[age_col],
        bins=analysis.AGE_BINS,
        labels=analysis.AGE_LABELS,
        right=False
    )
    return df


def filter_cohort(df: pd.DataFrame,
                  age_col: str = 'age',
                  risk_col: str = 'risk_nonlab') -> pd.DataFrame:
    """Filter dataframe to valid CVD cohort (age 40-74, non-missing risk)."""
    df = df.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    df[risk_col] = pd.to_numeric(df[risk_col], errors='coerce')
    
    mask = (
        (df[age_col] >= analysis.AGE_MIN) & 
        (df[age_col] <= analysis.AGE_MAX) & 
        (df[risk_col].notna())
    )
    
    return df[mask]


__all__ = [
    'RISK_PALETTE',
    'GENDER_PALETTE',
    'get_risk_cat',
    'get_risk_cat_4band',
    
    'get_risk_color',
    'get_gender_color',
    
    'validate_risk_value',
    
    'convert_risk_horizon',
    
    'add_risk_categories',
    'add_age_bands',
    'filter_cohort',
]
