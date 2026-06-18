"""
Configuration Management System for CVD Risk Tracker Streamlit Application

This package provides centralized configuration management for:
- Data file paths and versions
- Analysis parameters and thresholds
- Visualization settings and color schemes
- Statistical analysis defaults
- UI/UX settings

Usage:
    from config import paths, analysis, colors, ui
    
    # Access configuration values
    data_path = paths.get_analyzed_data_path('cvd_paired.csv')
    risk_bins = analysis.RISK_BINS
    color_palette = colors.RISK_COLORS
"""

from .paths import PathConfig
from .analysis import AnalysisConfig
from .colors import ColorConfig
from .ui import UIConfig

# Initialize configuration instances
paths = PathConfig()
analysis = AnalysisConfig()
colors = ColorConfig()
ui = UIConfig()

__all__ = [
    'paths',
    'analysis', 
    'colors',
    'ui',
    'PathConfig',
    'AnalysisConfig',
    'ColorConfig',
    'UIConfig'
]

__version__ = '1.0.0'
