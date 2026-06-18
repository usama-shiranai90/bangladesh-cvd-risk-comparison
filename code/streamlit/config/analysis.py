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
    
    # ==================== Risk Categories ====================
    
    # 5-Band Risk Classification (Standard)
    RISK_BINS: List[float] = field(default_factory=lambda: [-np.inf, 5, 10, 20, 30, np.inf])
    RISK_LABELS: List[str] = field(default_factory=lambda: ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%'])
    
    # 4-Band Risk Classification (Alternative)
    RISK_BINS_4BAND: List[float] = field(default_factory=lambda: [-np.inf, 5, 10, 20, np.inf])
    RISK_LABELS_4BAND: List[str] = field(default_factory=lambda: ['<5%', '5% to <10%', '10% to <20%', '≥20%'])
    
    # Risk Thresholds
    RISK_THRESHOLD_HIGH: float = 10.0  # High risk threshold (≥10%)
    RISK_THRESHOLD_VERY_HIGH: float = 20.0  # Very high risk threshold (≥20%)
    RISK_THRESHOLD_LOW: float = 5.0  # Low risk threshold (<5%)
    
    # ==================== Age Stratification ====================
    
    # Standard age bands for CVD analysis (WHO guideline: 40-74 years)
    AGE_BINS: List[int] = field(default_factory=lambda: [40, 45, 50, 55, 60, 65, 70, 75])
    AGE_LABELS: List[str] = field(default_factory=lambda: ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"])
    
    # Cohort age limits
    AGE_MIN: int = 40
    AGE_MAX: int = 74
    
    # ==================== Statistical Parameters ====================
    
    # Confidence Interval
    CONFIDENCE_LEVEL: float = 0.95
    ALPHA: float = 0.05  # Significance level
    Z_SCORE_95CI: float = 1.96  # Z-score for 95% CI
    
    # Meta-Analysis
    MIN_SITES_FOR_META: int = 3  # Minimum number of sites for meta-analysis
    DEFAULT_MIN_SITE_N: int = 50  # Default minimum patients per site
    
    # Statistical Tests
    CHI_SQUARE_ALPHA: float = 0.05
    
    # ==================== Risk Horizon ====================
    
    # Default risk time horizons
    RISK_HORIZON_10Y: str = "10-Year (Standard)"
    RISK_HORIZON_5Y: str = "5-Year (Derived)"
    RISK_HORIZON_OPTIONS: List[str] = field(default_factory=lambda: ["10-Year (Standard)", "5-Year (Derived)"])
    
    # 5-year risk conversion factor (constant hazard assumption)
    RISK_5Y_EXPONENT: float = 0.5  # p5 = 1 - (1 - p10)^0.5
    
    # ==================== Data Quality ====================
    
    # Minimum sample sizes
    MIN_SAMPLE_SIZE_STRATIFIED: int = 5  # Minimum for stratified analysis
    MIN_SAMPLE_SIZE_REGRESSION: int = 30  # Minimum for regression models
    
    # Missing data thresholds
    MAX_MISSING_RATE: float = 0.30  # Maximum acceptable missing rate (30%)
    
    # ==================== Column Name Mappings ====================
    
    # Standard column names expected in datasets
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
    
    # ==================== Helper Methods ====================
    
    def get_risk_category(self, risk_value: float, use_5band: bool = True) -> str:
        """
        Categorize a risk value into its appropriate band.
        
        Args:
            risk_value: Risk percentage (0-100)
            use_5band: If True, use 5-band classification, else 4-band
            
        Returns:
            Risk category label
        """
        if np.isnan(risk_value):
            return "Unknown"
        
        bins = self.RISK_BINS if use_5band else self.RISK_BINS_4BAND
        labels = self.RISK_LABELS if use_5band else self.RISK_LABELS_4BAND
        
        for i, threshold in enumerate(bins[1:], start=0):
            if risk_value < threshold:
                return labels[i] if i < len(labels) else labels[-1]
        
        return labels[-1]
    
    def convert_10y_to_5y_risk(self, risk_10y: float) -> float:
        """
        Convert 10-year risk to 5-year risk using constant hazard assumption.
        
        Formula: p5 = 1 - (1 - p10)^0.5
        
        Args:
            risk_10y: 10-year risk (0-100)
            
        Returns:
            5-year risk (0-100)
        """
        if np.isnan(risk_10y):
            return np.nan
        
        p10 = np.clip(risk_10y / 100.0, 0, 1)
        p5 = 1 - np.power((1 - p10), self.RISK_5Y_EXPONENT)
        return p5 * 100.0
    
    def get_age_band(self, age: int) -> str:
        """
        Get age band label for a given age.
        
        Args:
            age: Age in years
            
        Returns:
            Age band label or "Out of Range" if outside cohort definition
        """
        if age < self.AGE_MIN or age >= self.AGE_MAX:
            return "Out of Range"
        
        for i, threshold in enumerate(self.AGE_BINS[1:], start=0):
            if age < threshold:
                return self.AGE_LABELS[i] if i < len(self.AGE_LABELS) else self.AGE_LABELS[-1]
        
        return self.AGE_LABELS[-1]
    
    def is_high_risk(self, risk_value: float, threshold: str = "10%") -> bool:
        """
        Check if a risk value qualifies as high risk.
        
        Args:
            risk_value: Risk percentage
            threshold: "10%" or "20%"
            
        Returns:
            True if risk meets or exceeds threshold
        """
        target = self.RISK_THRESHOLD_VERY_HIGH if "20" in threshold else self.RISK_THRESHOLD_HIGH
        return risk_value >= target if not np.isnan(risk_value) else False
    
    def get_outcome_label(self, threshold: str, horizon: str) -> str:
        """
        Generate a standardized outcome label.
        
        Args:
            threshold: Risk threshold (e.g., "≥10%", "≥20%")
            horizon: Risk horizon (e.g., "10-Year", "5-Year")
            
        Returns:
            Formatted outcome label
        """
        return f"{threshold} ({horizon})"
    
    # ==================== Validation ====================
    
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
