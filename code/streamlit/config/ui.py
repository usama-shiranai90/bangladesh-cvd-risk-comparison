"""
UI Configuration Module

Centralizes all UI/UX settings, page configurations, and Streamlit-specific settings.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class UIConfig:
    """
    Configuration for UI elements and Streamlit page settings.
    
    Contains settings for:
    - Page configuration and layout
    - Navigation and menu items
    - Form defaults and controls
    - Text and messaging
    """
    
    
    PAGE_TITLE: str = "Bangladesh CVD Risk Tracker"
    PAGE_ICON: str = "❤️"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    
    MAIN_MENU_ITEMS: List[str] = field(default_factory=lambda: [
        "Overview",
        "RQ0: Baseline Burden",
        "RQ1: Safety & Discordance",
        "RQ1.1: Safety & Discordance",
       
      
        "Site Heterogeneity",
        "Risk Calculator",
        "Data Browser"
    ])
    
    MENU_ICONS: Dict[str, str] = field(default_factory=lambda: {
        "Overview": "📊",
        "RQ0: Baseline Burden": "📋",
        "RQ1: Safety & Discordance": "⚖️",
        "RQ1.1: Safety & Discordance": "⚖️",
     

        "Site Heterogeneity": "🗺️",
        "Risk Calculator": "🧮",
        "Data Browser": "🔍"
    })
    
    
    CUSTOM_CSS: str = """
    <style>
        .main { 
            background-color: #fcfcfc; 
        }
        
        h1, h2, h3 { 
            color: #006a4e; 
            font-family: 'Inter', sans-serif; 
        }
        
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        div.stButton > button {
            background-color: #006a4e;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            border: none;
            transition: all 0.2s;
        }
        
        div.stButton > button:hover {
            background-color: #005a42;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        [data-testid="stSidebar"] {
            border-right: 1px solid #ddd;
            background-color: #f8f9fa;
        }
        
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            background-color: white;
            border-radius: 8px 8px 0 0;
            border: 1px solid #e0e0e0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #006a4e;
            color: white;
        }
        
        /* Alert/Info box styling */
        .stAlert {
            border-radius: 8px;
        }
    </style>
    """
    
    
    NUMBER_INPUT_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        'min_site_n': {
            'min_value': 10,
            'max_value': 500,
            'default': 50,
            'step': 10,
            'label': "Minimum Patients per Site"
        },
        'confidence_level': {
            'min_value': 0.90,
            'max_value': 0.99,
            'default': 0.95,
            'step': 0.01,
            'label': "Confidence Level"
        }
    })
    
    RADIO_OPTIONS: Dict[str, List[str]] = field(default_factory=lambda: {
        'risk_threshold': [
            "High Risk (≥10%)",
            "Very High Risk (≥20%)"
        ],
        'risk_horizon': [
            "10-Year (Standard)",
            "5-Year (Derived)"
        ],
        'metric_display': [
            "Crude Prevalence",
            "Age-Standardized",
            "Bayesian Smoothed"
        ],
        'model_type': [
            "Cluster-Robust Logistic Regression (Binary Outcome)",
            "Quantile Regression (Continuous Risk Score)"
        ]
    })
    
    
    ERROR_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        'no_data': "No data loaded. Please select a dataset from the sidebar.",
        'insufficient_data': "Insufficient data for this analysis. Minimum {min_n} records required.",
        'missing_column': "Required column '{column}' not found in dataset.",
        'invalid_age_range': "No data in valid age range (40-74 years).",
        'model_failed': "Model fitting failed. Please check data quality and try again."
    })
    
    WARNING_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        'low_sample': "⚠️ Warning: Sample size is below recommended minimum (n < {min_n}).",
        'high_missing': "⚠️ Warning: High missing data rate ({rate:.1%}) detected.",
        'unstable_model': "⚠️ Warning: Modeling events for ≥20% risk may be unstable due to low event counts. ≥10% is recommended.",
        'no_variation': "⚠️ Not enough variation in {variable} for analysis."
    })
    
    INFO_MESSAGES: Dict[str, str] = field(default_factory=lambda: {
        'data_loaded': "✅ Successfully loaded {n:,} records.",
        'analysis_complete': "✅ Analysis completed successfully.",
        'filtering_applied': "📊 Filters applied: {description}"
    })
    
    
    RQ0_TABS: List[str] = field(default_factory=lambda: [
        "📋 Baseline Table 1",
        "📊 Distributions",
        "📋 Table 2: Risk Distribution",
        "📋 Table 3: Risk by Gender",
        "📈 Prevalence Tables",
        "🏥 Site Heterogeneity & Meta-Analysis",
        "🧮 Advanced Modeling",
        "📊 Paired Data Analysis (Lab vs Non-Lab)"
    ])
    
    
    CHART_HEIGHT_SMALL: int = 400
    CHART_HEIGHT_MEDIUM: int = 600
    CHART_HEIGHT_LARGE: int = 800
    
    MIN_CHART_HEIGHT: int = 400
    PIXELS_PER_ROW: int = 25
    
    
    def get_page_config(self) -> Dict[str, str]:
        """Get Streamlit page configuration dictionary."""
        return {
            'page_title': self.PAGE_TITLE,
            'page_icon': self.PAGE_ICON,
            'layout': self.LAYOUT,
            'initial_sidebar_state': self.INITIAL_SIDEBAR_STATE
        }
    
    def get_error_message(self, key: str, **kwargs) -> str:
        """Get formatted error message."""
        msg = self.ERROR_MESSAGES.get(key, "An error occurred.")
        return msg.format(**kwargs) if kwargs else msg
    
    def get_warning_message(self, key: str, **kwargs) -> str:
        """Get formatted warning message."""
        msg = self.WARNING_MESSAGES.get(key, "Warning!")
        return msg.format(**kwargs) if kwargs else msg
    
    def get_info_message(self, key: str, **kwargs) -> str:
        """Get formatted info message."""
        msg = self.INFO_MESSAGES.get(key, "Information.")
        return msg.format(**kwargs) if kwargs else msg
    
    def calculate_dynamic_height(self, num_items: int) -> int:
        """Calculate dynamic chart height based on number of items."""
        calculated = self.MIN_CHART_HEIGHT + (num_items * self.PIXELS_PER_ROW)
        return max(self.MIN_CHART_HEIGHT, calculated)
    
    def get_menu_item_with_icon(self, item: str) -> str:
        """Get menu item with its icon."""
        icon = self.MENU_ICONS.get(item, "")
        return f"{icon} {item}" if icon else item
    
    
    @staticmethod
    def format_number(value: float, decimals: int = 1) -> str:
        """Format number with thousands separator."""
        if isinstance(value, (int, float)):
            return f"{value:,.{decimals}f}"
        return str(value)
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format value as percentage."""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def format_confidence_interval(lower: float, upper: float, decimals: int = 1) -> str:
        """Format confidence interval."""
        return f"95% CI: [{lower:.{decimals}f}, {upper:.{decimals}f}]"
    
    
    SESSION_STATE_KEYS: Dict[str, str] = field(default_factory=lambda: {
        'path_main': 'path_main',
        'path_lab': 'path_lab',
        'path_paired': 'path_paired',
        'path_sites': 'path_sites',
        'selected_page': 'selected_page',
        'data_version': 'data_version',
        'cache_timestamp': 'cache_timestamp'
    })
    
    def init_session_state(self, st_session_state) -> None:
        """Initialize session state with default values."""
        for key in self.SESSION_STATE_KEYS.values():
            if key not in st_session_state:
                st_session_state[key] = None
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UIConfig(title='{self.PAGE_TITLE}', "
            f"layout='{self.LAYOUT}', "
            f"menu_items={len(self.MAIN_MENU_ITEMS)})"
        )
