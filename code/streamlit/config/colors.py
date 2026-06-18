"""
Color Configuration Module

Centralizes all color schemes, palettes, and styling constants for visualizations.
"""

from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class ColorConfig:
    """
    Centralized color configuration for all visualizations.
    
    Provides consistent color schemes across:
    - Risk categories
    - Gender/demographics
    - Urban/rural classifications
    - Chart elements
    """
    
    
    RISK_COLORS: Dict[str, str] = field(default_factory=lambda: {
        '<5%': 'green',
        '5% to <10%': 'gold',
        '10% to <20%': 'orange',
        '20% to <30%': 'red',
        '≥30%': 'darkred'
    })
    
    RISK_COLORS_HEX: Dict[str, str] = field(default_factory=lambda: {
        '<5%': '#2a9d8f',
        '5% to <10%': '#e9c46a',
        '10% to <20%': '#f4a261',
        '20% to <30%': '#e76f51',
        '≥30%': '#d62828',
        '≥20%': '#e76f51'
    })
    
    RISK_COLORS_4BAND: Dict[str, str] = field(default_factory=lambda: {
        '<5%': '#2a9d8f',
        '5% to <10%': '#e9c46a',
        '10% to <20%': '#f4a261',
        '≥20%': '#e76f51'
    })
    
    
    GENDER_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'Male': '#457b9d',
        'Female': '#e63946',
        'Men': '#457b9d',
        'Women': '#e63946',
        'Other': '#6c757d'
    })
    
    LOCATION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'Urban': '#1d3557',
        'Rural': '#52b788',
        'Semi-urban': '#f77f00',
        'Semi-Urban': '#f77f00',
        'Unknown': '#adb5bd'
    })
    
    
    PRIMARY_COLOR: str = '#006a4e'
    SECONDARY_COLOR: str = '#f42a41'
    
    BACKGROUND_COLOR: str = '#fcfcfc'
    GRID_COLOR: str = '#e0e0e0'
    
    TEXT_PRIMARY: str = '#212529'
    TEXT_SECONDARY: str = '#6c757d'
    TEXT_LIGHT: str = '#adb5bd'
    
    BORDER_COLOR: str = '#dee2e6'
    BORDER_HOVER: str = '#ced4da'
    
    SUCCESS_COLOR: str = '#28a745'
    WARNING_COLOR: str = '#ffc107'
    DANGER_COLOR: str = '#dc3545'
    INFO_COLOR: str = '#17a2b8'
    
    
    SEQUENTIAL_BLUE: List[str] = field(default_factory=lambda: [
        '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'
    ])
    
    SEQUENTIAL_RED: List[str] = field(default_factory=lambda: [
        '#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'
    ])
    
    SEQUENTIAL_GREEN: List[str] = field(default_factory=lambda: [
        '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'
    ])
    
    DIVERGING_PALETTE: List[str] = field(default_factory=lambda: [
        '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'
    ])
    
    
    PLOTLY_TEMPLATE: str = 'plotly_white'
    
    DEFAULT_MARKER_SIZE: int = 8
    DEFAULT_LINE_WIDTH: int = 2
    
    MARKER_OPACITY: float = 0.8
    FILL_OPACITY: float = 0.3
    
    
    def get_risk_color(self, risk_category: str, use_hex: bool = False) -> str:
        """Get color for a risk category."""
        palette = self.RISK_COLORS_HEX if use_hex else self.RISK_COLORS
        return palette.get(risk_category, '#cccccc' if use_hex else 'gray')
    
    def get_gender_color(self, gender: str) -> str:
        """Get color for a gender category."""
        return self.GENDER_COLORS.get(gender, self.GENDER_COLORS.get('Other'))
    
    def get_location_color(self, location: str) -> str:
        """Get color for a location type."""
        return self.LOCATION_COLORS.get(location, self.LOCATION_COLORS.get('Unknown'))
    
    def get_color_scale(self, palette_type: str = 'blue') -> List[str]:
        """Get a color scale for continuous data."""
        palettes = {
            'blue': self.SEQUENTIAL_BLUE,
            'red': self.SEQUENTIAL_RED,
            'green': self.SEQUENTIAL_GREEN,
            'diverging': self.DIVERGING_PALETTE
        }
        return palettes.get(palette_type, self.SEQUENTIAL_BLUE)
    
    def create_custom_palette(self, categories: List[str], palette_name: str = 'risk') -> Dict[str, str]:
        """Create a custom color palette for specific categories."""
        base_palettes = {
            'risk': self.RISK_COLORS_HEX,
            'gender': self.GENDER_COLORS,
            'location': self.LOCATION_COLORS
        }
        
        base = base_palettes.get(palette_name, self.RISK_COLORS_HEX)
        
        result = {}
        for cat in categories:
            result[cat] = base.get(cat, '#cccccc')
        
        return result
    
    
    def get_plotly_layout_defaults(self) -> dict:
        """Get default layout settings for Plotly figures."""
        return {
            'template': self.PLOTLY_TEMPLATE,
            'font': {
                'family': 'Inter, sans-serif',
                'size': 12,
                'color': self.TEXT_PRIMARY
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
            'hovermode': 'closest',
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1
            }
        }
    
    def get_plotly_marker_defaults(self) -> dict:
        """Get default marker settings for Plotly scatter plots."""
        return {
            'size': self.DEFAULT_MARKER_SIZE,
            'opacity': self.MARKER_OPACITY,
            'line': {
                'width': 1,
                'color': 'white'
            }
        }
    
    
    @property
    def RISK_PALETTE(self) -> Dict[str, str]:
        """Legacy property for backward compatibility."""
        return self.RISK_COLORS_HEX
    
    @property
    def GENDER_PALETTE(self) -> Dict[str, str]:
        """Legacy property for backward compatibility."""
        return self.GENDER_COLORS
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ColorConfig(risk_colors={len(self.RISK_COLORS)}, "
            f"primary={self.PRIMARY_COLOR}, "
            f"theme={self.PLOTLY_TEMPLATE})"
        )
