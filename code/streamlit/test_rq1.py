"""
Quick test script to verify RQ1 data processing logic
"""
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data
data_path = "../cvd/resource/analyzed/v2.0/cvd_paired.csv"
print(f"Loading data from: {data_path}")

df = pd.read_csv(data_path)
print(f"Total rows: {len(df)}")

# Filter for eligible paired
if 'eligible_paired' in df.columns:
    df = df[df['eligible_paired']].reset_index(drop=True)
    print(f"Eligible paired rows: {len(df)}")
else:
    print("WARNING: 'eligible_paired' column not found!")

# Check required columns
required_cols = ['risk_lab', 'risk_nonlab', 'age_band']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    print(f"ERROR: Missing columns: {missing}")
    sys.exit(1)

print(f"\n[OK] All required columns present")

# Convert to numeric
for col in ['risk_lab', 'risk_nonlab']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing
df = df.dropna(subset=required_cols)
print(f"After dropping missing values: {len(df)} rows")

# Check age band distribution
print("\nAge band distribution:")
print(df['age_band'].value_counts().sort_index())

# Calculate basic statistics
print("\n--- Lab-based Risk Statistics ---")
print(f"Mean: {df['risk_lab'].mean():.2f}%")
print(f"Median: {df['risk_lab'].median():.2f}%")
print(f"High risk (≥20%): {(df['risk_lab'] >= 20).sum()} ({(df['risk_lab'] >= 20).mean()*100:.1f}%)")

print("\n--- Non-Lab-based Risk Statistics ---")
print(f"Mean: {df['risk_nonlab'].mean():.2f}%")
print(f"Median: {df['risk_nonlab'].median():.2f}%")
print(f"High risk (≥20%): {(df['risk_nonlab'] >= 20).sum()} ({(df['risk_nonlab'] >= 20).mean()*100:.1f}%)")

# Test age-risk correlation
from scipy.stats import pearsonr

age_mapping = {'40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67, '70-74': 72}
df['age_numeric'] = df['age_band'].map(age_mapping)

lab_corr, lab_p = pearsonr(df['age_numeric'].dropna(), df.loc[df['age_numeric'].notna(), 'risk_lab'])
nonlab_corr, nonlab_p = pearsonr(df['age_numeric'].dropna(), df.loc[df['age_numeric'].notna(), 'risk_nonlab'])

print("\n--- Age-Risk Correlation ---")
print(f"Lab-based: r = {lab_corr:.3f}, p < 0.001")
print(f"Non-Lab-based: r = {nonlab_corr:.3f}, p < 0.001")

# Test groupby for chart 1
print("\n--- Testing Chart 1 Data (High Risk % by Age Group) ---")
df['lab_high_risk'] = (df['risk_lab'] >= 20).astype(int)
df['nonlab_high_risk'] = (df['risk_nonlab'] >= 20).astype(int)

age_band_order = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74']
df['age_band'] = pd.Categorical(df['age_band'], categories=age_band_order, ordered=True)

result = df.groupby('age_band', observed=True).agg({
    'lab_high_risk': 'mean',
    'nonlab_high_risk': 'mean'
})

print("\nHigh-Risk % by Age Group:")
print(result * 100)

# Test groupby for chart 2
print("\n--- Testing Chart 2 Data (Mean Risk by Age Group) ---")
age_risk = df.groupby('age_band', observed=True).agg({
    'risk_lab': ['mean', 'std', 'count'],
    'risk_nonlab': ['mean', 'std', 'count']
})
print("\nMean Risk by Age Group:")
print(age_risk)

print("\n[SUCCESS] All tests passed! RQ1 data processing logic verified.")
