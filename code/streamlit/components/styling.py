"""
Updated styling.py to use the configuration system.

Applies custom CSS from ui config for consistent branding.
"""

import streamlit as st
from config import ui, colors


def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app.
    
    Uses centralized CSS from config.ui.CUSTOM_CSS for consistency.
    """
    st.markdown(ui.CUSTOM_CSS, unsafe_allow_html=True)


def get_brand_colors():
    """
    Get brand colors for custom components.
    
    Returns:
        dict: Dictionary with primary and secondary colors
    """
    return {
        'primary': colors.PRIMARY_COLOR,
        'secondary': colors.SECONDARY_COLOR,
        'background': colors.BACKGROUND_COLOR,
        'text': colors.TEXT_PRIMARY
    }


def apply_custom_metric_style(label: str, value: str, delta: str = None):
    """
    Create a custom styled metric with brand colors.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta/change value
        
    Returns:
        HTML string for custom metric
    """
    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size: 0.9rem; color: {colors.TEXT_SECONDARY};">{delta}</div>'
    
    return f"""
    <div style="
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {colors.BORDER_COLOR};
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="font-size: 0.85rem; color: {colors.TEXT_SECONDARY}; margin-bottom: 5px;">
            {label}
        </div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {colors.PRIMARY_COLOR};">
            {value}
        </div>
        {delta_html}
    </div>
    """
