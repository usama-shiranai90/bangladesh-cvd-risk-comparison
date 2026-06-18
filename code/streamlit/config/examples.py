"""
Example usage of the configuration management system.

This file demonstrates how to use the new config system across different scenarios.
Run this file to see examples and validate the configuration setup.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import paths, analysis, colors, ui


def example_paths():
    """Demonstrate path configuration usage."""
    print("=" * 60)
    print("PATH CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Basic path access
    print(f"\n📁 Resource Directory: {paths.resource_dir}")
    print(f"📁 Analyzed Data Directory: {paths.analyzed_dir}")
    print(f"📁 Sites File: {paths.sites_file}")
    
    # Get specific dataset paths
    print(f"\n📊 Paired Data: {paths.get_analyzed_data_path('cvd_paired.csv')}")
    print(f"📊 WHO NonLab: {paths.get_dataset_path('who_nonlab')}")
    
    # Validate paths
    print("\n🔍 Path Validation:")
    validation = paths.validate_paths()
    for path_name, exists in validation.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {path_name}: {exists}")
    
    # Get all datasets
    print(f"\n📚 Available Datasets: {list(paths.dataset_mapping.keys())}")
    
    # Legacy compatibility
    base_res, base_an = paths.get_cvd_paths()
    print(f"\n🔄 Legacy Format: base_res={base_res}, base_an={base_an}")


def example_analysis():
    """Demonstrate analysis configuration usage."""
    print("\n" + "=" * 60)
    print("ANALYSIS CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Risk categorization
    test_risks = [3.5, 7.2, 15.8, 25.3, 32.1]
    print("\n🎯 Risk Categorization (5-band):")
    for risk in test_risks:
        category = analysis.get_risk_category(risk, use_5band=True)
        print(f"  {risk:5.1f}% → {category}")
    
    # Risk conversion
    print("\n🔄 10-Year to 5-Year Risk Conversion:")
    for risk_10y in [10.0, 20.0, 30.0]:
        risk_5y = analysis.convert_10y_to_5y_risk(risk_10y)
        print(f"  {risk_10y:.1f}% (10-year) → {risk_5y:.1f}% (5-year)")
    
    # Age banding
    print("\n👥 Age Banding:")
    test_ages = [42, 48, 53, 67, 72]
    for age in test_ages:
        band = analysis.get_age_band(age)
        print(f"  Age {age} → {band}")
    
    # High risk check
    print("\n⚠️ High Risk Classification:")
    for risk in [8.5, 12.3, 22.7]:
        is_high_10 = analysis.is_high_risk(risk, "10%")
        is_high_20 = analysis.is_high_risk(risk, "20%")
        print(f"  {risk:.1f}% - High (≥10%): {is_high_10}, Very High (≥20%): {is_high_20}")
    
    # Constants
    print(f"\n📊 Risk Bins: {analysis.RISK_BINS}")
    print(f"📊 Risk Labels: {analysis.RISK_LABELS}")
    print(f"📊 Age Range: {analysis.AGE_MIN}-{analysis.AGE_MAX}")
    print(f"📊 Confidence Level: {analysis.CONFIDENCE_LEVEL}")


def example_colors():
    """Demonstrate color configuration usage."""
    print("\n" + "=" * 60)
    print("COLOR CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Risk colors
    print("\n🎨 Risk Category Colors:")
    for label in analysis.RISK_LABELS:
        color_name = colors.get_risk_color(label, use_hex=False)
        color_hex = colors.get_risk_color(label, use_hex=True)
        print(f"  {label:15} → {color_name:10} ({color_hex})")
    
    # Gender colors
    print("\n👥 Gender Colors:")
    for gender in ['Male', 'Female', 'Other']:
        color = colors.get_gender_color(gender)
        print(f"  {gender:10} → {color}")
    
    # Location colors
    print("\n🗺️ Location Colors:")
    for location in ['Urban', 'Rural', 'Semi-urban']:
        color = colors.get_location_color(location)
        print(f"  {location:12} → {color}")
    
    # Brand colors
    print(f"\n🏷️ Primary Color: {colors.PRIMARY_COLOR}")
    print(f"🏷️ Secondary Color: {colors.SECONDARY_COLOR}")
    
    # Color scales
    print(f"\n🌈 Blue Scale ({len(colors.SEQUENTIAL_BLUE)} colors):")
    print(f"   {', '.join(colors.SEQUENTIAL_BLUE[:3])} ... {colors.SEQUENTIAL_BLUE[-1]}")


def example_ui():
    """Demonstrate UI configuration usage."""
    print("\n" + "=" * 60)
    print("UI CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Page config
    page_config = ui.get_page_config()
    print("\n📄 Page Configuration:")
    for key, value in page_config.items():
        print(f"  {key}: {value}")
    
    # Messages
    print("\n💬 Message Examples:")
    print(f"  Error: {ui.get_error_message('no_data')}")
    print(f"  Warning: {ui.get_warning_message('low_sample', min_n=50)}")
    print(f"  Info: {ui.get_info_message('data_loaded', n=10000)}")
    
    # Formatting
    print("\n🔢 Formatting Examples:")
    print(f"  Number: {ui.format_number(1234567.89, decimals=2)}")
    print(f"  Percentage: {ui.format_percentage(45.678)}")
    print(f"  CI: {ui.format_confidence_interval(10.2, 15.8)}")
    
    # Dynamic height
    print("\n📏 Dynamic Chart Heights:")
    for n_items in [10, 30, 50]:
        height = ui.calculate_dynamic_height(n_items)
        print(f"  {n_items} items → {height}px")
    
    # Menu items
    print(f"\n📋 Menu Items ({len(ui.MAIN_MENU_ITEMS)}):")
    for item in ui.MAIN_MENU_ITEMS[:5]:
        icon_item = ui.get_menu_item_with_icon(item)
        print(f"  {icon_item}")
    print(f"  ... and {len(ui.MAIN_MENU_ITEMS) - 5} more")


def example_integration():
    """Demonstrate integrated usage across modules."""
    print("\n" + "=" * 60)
    print("INTEGRATED USAGE EXAMPLE")
    print("=" * 60)
    
    # Scenario: Prepare analysis settings
    print("\n📊 Scenario: Setting up RQ0 Analysis")
    
    # 1. Get data path
    data_path = paths.get_analyzed_data_path('cvd_who_nonlab_domain.csv')
    print(f"\n1️⃣ Data Path: {data_path}")
    
    # 2. Define risk categorization
    print(f"\n2️⃣ Risk Categories:")
    print(f"   Bins: {analysis.RISK_BINS}")
    print(f"   Labels: {analysis.RISK_LABELS}")
    
    # 3. Get color scheme
    print(f"\n3️⃣ Color Scheme:")
    for label in analysis.RISK_LABELS:
        color = colors.get_risk_color(label, use_hex=True)
        print(f"   {label:15} → {color}")
    
    # 4. UI settings
    print(f"\n4️⃣ UI Settings:")
    print(f"   Page Title: {ui.PAGE_TITLE}")
    print(f"   Chart Height (30 sites): {ui.calculate_dynamic_height(30)}px")
    
    # 5. Analysis parameters
    print(f"\n5️⃣ Analysis Parameters:")
    print(f"   Age Range: {analysis.AGE_MIN}-{analysis.AGE_MAX}")
    print(f"   High Risk Threshold: ≥{analysis.RISK_THRESHOLD_HIGH}%")
    print(f"   Confidence Level: {analysis.CONFIDENCE_LEVEL}")


def validate_configuration():
    """Validate all configuration modules."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Check paths
    print("\n✓ PathConfig initialized")
    print(f"  Version: {paths.data_version}")
    print(f"  Streamlit Dir: {paths.streamlit_dir}")
    
    # Check analysis
    print("\n✓ AnalysisConfig initialized")
    print(f"  Risk Bands: {len(analysis.RISK_LABELS)}")
    print(f"  Age Bands: {len(analysis.AGE_LABELS)}")
    
    # Check colors
    print("\n✓ ColorConfig initialized")
    print(f"  Risk Colors: {len(colors.RISK_COLORS)}")
    print(f"  Gender Colors: {len(colors.GENDER_COLORS)}")
    
    # Check ui
    print("\n✓ UIConfig initialized")
    print(f"  Menu Items: {len(ui.MAIN_MENU_ITEMS)}")
    print(f"  Error Messages: {len(ui.ERROR_MESSAGES)}")
    
    print("\n" + "=" * 60)
    print("✅ ALL CONFIGURATIONS VALIDATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "🔧" * 30)
    print("CONFIGURATION MANAGEMENT SYSTEM - EXAMPLES")
    print("🔧" * 30)
    
    # Run all examples
    example_paths()
    example_analysis()
    example_colors()
    example_ui()
    example_integration()
    validate_configuration()
    
    print("\n" + "✅" * 30)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("✅" * 30 + "\n")
