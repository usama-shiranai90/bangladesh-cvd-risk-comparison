"""
Analysis Configuration Module

Centralizes all analysis parameters, statistical thresholds, and domain-specific settings.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """
    Configuration for CVD risk analysis parameters and thresholds.
    
    Contains all constants used for:
    - Risk categorization and binning
    - Age stratification
    - Statistical analysis defaults
    - Cohort definitions
    """
    
    
    RISK_BINS: List[float] = field(default_factory=lambda: [-np.inf, 5, 10, 20, 30, np.inf])
    RISK_LABELS: List[str] = field(default_factory=lambda: ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%'])
    
    RISK_BINS_4BAND: List[float] = field(default_factory=lambda: [-np.inf, 5, 10, 20, np.inf])
    RISK_LABELS_4BAND: List[str] = field(default_factory=lambda: ['<5%', '5% to <10%', '10% to <20%', '≥20%'])
    
    RISK_THRESHOLD_HIGH: float = 10.0
    RISK_THRESHOLD_VERY_HIGH: float = 20.0
    RISK_THRESHOLD_LOW: float = 5.0
    
    
    AGE_BINS: List[int] = field(default_factory=lambda: [40, 45, 50, 55, 60, 65, 70, 75])
    AGE_LABELS: List[str] = field(default_factory=lambda: ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"])
    
    AGE_MIN: int = 40
    AGE_MAX: int = 74
    
    
    CONFIDENCE_LEVEL: float = 0.95
    ALPHA: float = 0.05
    Z_SCORE_95CI: float = 1.96
    
    MIN_SITES_FOR_META: int = 3
    DEFAULT_MIN_SITE_N: int = 50
    
    CHI_SQUARE_ALPHA: float = 0.05
    
    
    RISK_HORIZON_10Y: str = "10-Year (Standard)"
    RISK_HORIZON_5Y: str = "5-Year (Derived)"
    RISK_HORIZON_OPTIONS: List[str] = field(default_factory=lambda: ["10-Year (Standard)", "5-Year (Derived)"])
    
    RISK_5Y_EXPONENT: float = 0.5
    
    
    MIN_SAMPLE_SIZE_STRATIFIED: int = 5
    MIN_SAMPLE_SIZE_REGRESSION: int = 30
    
    MAX_MISSING_RATE: float = 0.30
    
    
    STANDARD_COLUMNS: Dict[str, str] = field(default_factory=lambda: {
        'age': 'age',
        'gender': 'gender',
        'site_id': 'site_id',
        'site_name': 'site_name',
        'site_title': 'site_title',
        'risk_nonlab': 'risk_nonlab',
        'risk_lab': 'risk_lab',
        'location_type': 'location_type',
        'urban_rural': 'urban_rural',
        'age_band': 'age_band',
    })
    
    
    def get_risk_category(self, risk_value: float, use_5band: bool = True) -> str:
        """Categorize a risk value into its appropriate band."""
        if np.isnan(risk_value):
            return "Unknown"
        
        bins = self.RISK_BINS if use_5band else self.RISK_BINS_4BAND
        labels = self.RISK_LABELS if use_5band else self.RISK_LABELS_4BAND
        
        for i, threshold in enumerate(bins[1:], start=0):
            if risk_value < threshold:
                return labels[i] if i < len(labels) else labels[-1]
        
        return labels[-1]
    
    def convert_10y_to_5y_risk(self, risk_10y: float) -> float:
        """Convert 10-year risk to 5-year risk using constant hazard assumption."""
        if np.isnan(risk_10y):
            return np.nan
        
        p10 = np.clip(risk_10y / 100.0, 0, 1)
        p5 = 1 - np.power((1 - p10), self.RISK_5Y_EXPONENT)
        return p5 * 100.0
    
    def get_age_band(self, age: int) -> str:
        """Get age band label for a given age."""
        if age < self.AGE_MIN or age >= self.AGE_MAX:
            return "Out of Range"
        
        for i, threshold in enumerate(self.AGE_BINS[1:], start=0):
            if age < threshold:
                return self.AGE_LABELS[i] if i < len(self.AGE_LABELS) else self.AGE_LABELS[-1]
        
        return self.AGE_LABELS[-1]
    
    def is_high_risk(self, risk_value: float, threshold: str = "10%") -> bool:
        """Check if a risk value qualifies as high risk."""
        target = self.RISK_THRESHOLD_VERY_HIGH if "20" in threshold else self.RISK_THRESHOLD_HIGH
        return risk_value >= target if not np.isnan(risk_value) else False
    
    def get_outcome_label(self, threshold: str, horizon: str) -> str:
        """Generate a standardized outcome label."""
        return f"{threshold} ({horizon})"
    
    
    def validate_risk_value(self, value: float) -> bool:
        """Check if a risk value is valid (0-100 or NaN)."""
        return np.isnan(value) or (0 <= value <= 100)
    
    def validate_age(self, age: int) -> bool:
        """Check if age is within valid cohort range."""
        return self.AGE_MIN <= age <= self.AGE_MAX
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AnalysisConfig(risk_bands={len(self.RISK_LABELS)}, "
            f"age_range={self.AGE_MIN}-{self.AGE_MAX}, "
            f"risk_thresholds=[{self.RISK_THRESHOLD_HIGH}, {self.RISK_THRESHOLD_VERY_HIGH}])"
        )
