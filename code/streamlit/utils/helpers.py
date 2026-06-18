"""
Updated helpers.py to use the new configuration system.

This module now imports from config and provides backward-compatible
wrappers for existing code.
"""

import pandas as pd
from config import colors, analysis

# ==================== Legacy Exports (Backward Compatibility) ====================

# Export palettes for backward compatibility with existing code
RISK_PALETTE = colors.RISK_PALETTE  # Alias to RISK_COLORS_HEX
GENDER_PALETTE = colors.GENDER_PALETTE  # Alias to GENDER_COLORS

# ==================== Helper Functions ====================

def get_risk_cat(val):
    """
    Categorize a risk value into 5-band classification.
    
    This is a legacy wrapper that delegates to the config system.
    New code should use: analysis.get_risk_category(val, use_5band=True)
    
    Args:
        val: Risk percentage value (0-100)
        
    Returns:
        Risk category label
        
    Examples:
        >>> get_risk_cat(3.5)
        '<5%'
        >>> get_risk_cat(15.2)
        '10% to <20%'
    """
    return analysis.get_risk_category(val, use_5band=True)


def get_risk_cat_4band(val):
    """
    Categorize a risk value into 4-band classification.
    
    This is a legacy wrapper that delegates to the config system.
    New code should use: analysis.get_risk_category(val, use_5band=False)
    
    Args:
        val: Risk percentage value (0-100)
        
    Returns:
        Risk category label
        
    Examples:
        >>> get_risk_cat_4band(8.5)
        '5% to <10%'
        >>> get_risk_cat_4band(25.0)
        '≥20%'
    """
    return analysis.get_risk_category(val, use_5band=False)


# ==================== Additional Helper Functions ====================

def get_risk_color(category: str, use_hex: bool = True) -> str:
    """
    Get color for a risk category.
    
    Args:
        category: Risk category label
        use_hex: If True, return hex code; else return color name
        
    Returns:
        Color code or name
    """
    return colors.get_risk_color(category, use_hex=use_hex)


def get_gender_color(gender: str) -> str:
    """
    Get color for a gender category.
    
    Args:
        gender: Gender label (Male, Female, etc.)
        
    Returns:
        Color hex code
    """
    return colors.get_gender_color(gender)


def validate_risk_value(value: float) -> bool:
    """
    Check if a risk value is valid.
    
    Args:
        value: Risk percentage value
        
    Returns:
        True if valid (0-100 or NaN)
    """
    return analysis.validate_risk_value(value)


def convert_risk_horizon(risk_10y: float, to_5year: bool = True) -> float:
    """
    Convert between 10-year and 5-year risk.
    
    Args:
        risk_10y: 10-year risk percentage
        to_5year: If True, convert to 5-year; else return as-is
        
    Returns:
        Converted risk percentage
    """
    if to_5year:
        return analysis.convert_10y_to_5y_risk(risk_10y)
    return risk_10y


# ==================== DataFrame Helpers ====================

def add_risk_categories(df: pd.DataFrame, 
                        risk_col: str = 'risk_nonlab',
                        output_col: str = 'risk_cat',
                        use_5band: bool = True) -> pd.DataFrame:
    """
    Add risk category column to dataframe.
    
    Args:
        df: Input dataframe
        risk_col: Column containing risk values
        output_col: Name for output category column
        use_5band: Use 5-band (True) or 4-band (False) classification
        
    Returns:
        DataFrame with added risk category column
    """
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
    """
    Add age band column to dataframe.
    
    Args:
        df: Input dataframe
        age_col: Column containing age values
        output_col: Name for output age band column
        
    Returns:
        DataFrame with added age band column
    """
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
    """
    Filter dataframe to valid CVD cohort (age 40-74, non-missing risk).
    
    Args:
        df: Input dataframe
        age_col: Column containing age values
        risk_col: Column containing risk values
        
    Returns:
        Filtered dataframe
    """
    df = df.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    df[risk_col] = pd.to_numeric(df[risk_col], errors='coerce')
    
    mask = (
        (df[age_col] >= analysis.AGE_MIN) & 
        (df[age_col] <= analysis.AGE_MAX) & 
        (df[risk_col].notna())
    )
    
    return df[mask]


# ==================== Module Info ====================

__all__ = [
    # Legacy exports
    'RISK_PALETTE',
    'GENDER_PALETTE',
    'get_risk_cat',
    'get_risk_cat_4band',
    
    # Color helpers
    'get_risk_color',
    'get_gender_color',
    
    # Validation
    'validate_risk_value',
    
    # Conversion
    'convert_risk_horizon',
    
    # DataFrame helpers
    'add_risk_categories',
    'add_age_bands',
    'filter_cohort',
]
