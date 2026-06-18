"""
Exploratory Data Analysis (EDA) helpers for the CVD project.

This module re-exports EDA utilities from data_utils to keep a clear separation
of concerns while preserving backward compatibility.
"""
import pandas as pd

import numpy as np
from IPython.display import display


def quick_analysis(df, name="DataFrame", show_examples=False, top_n=5):
    """Print a quick summary of the given DataFrame."""
    if df is None or df.empty:
        print(f"\n📊 Quick Analysis — {name}")
        print("⚠️ DataFrame is empty or None.\n")
        return

    print(f"\n📊 Quick Analysis — {name}")
    print("=" * (20 + len(name)))
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().sum() / len(df) * 100).round(2),
        "unique": df.nunique(dropna=True)
    })

    if show_examples:
        summary["examples"] = [
            ", ".join(map(str, df[col].dropna().unique()[:top_n]))
            if df[col].notna().any()
            else ""
            for col in df.columns
        ]

    try:
        display(summary.sort_index())
    except ImportError:
        print(summary.sort_index())

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print("\n🔹 Memory usage: {:.2f} MB".format(mem_mb))
    print("🔹 Duplicated rows:", df.duplicated().sum())
    print("🔹 Columns with any missing values:", int(df.isnull().any().sum()))


def deep_analysis(df: pd.DataFrame, name: str = "DataFrame", show_examples: bool = True, top_n: int = 5):
    """Perform an in-depth quick analysis of a pandas DataFrame."""

    if df is None or df.empty:
        print(f"\n📊 Deep Analysis — {name}")
        print("⚠️ DataFrame is empty or None.\n")
        return

    print(f"\n📊 Deep Analysis — {name}")
    print("=" * (22 + len(name)))
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().sum() / len(df) * 100).round(2),
        "unique": df.nunique(dropna=True),
    })

    if show_examples:
        summary["examples"] = [
            ", ".join(map(str, df[col].dropna().unique()[:top_n]))
            if df[col].notna().any() else ""
            for col in df.columns
        ]

    num_summary = df.describe(include=[np.number]).T
    num_summary = num_summary[["mean", "std", "min", "max"]].round(2)
    summary = summary.join(num_summary, how="left")

    try:
        from IPython.display import display as _display
        _display(summary.sort_index())
    except ImportError:
        print(summary.sort_index())

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print("\n🧾 Memory usage: {:.2f} MB".format(mem_mb))
    print("🔁 Duplicated rows:", df.duplicated().sum())
    print("🧩 Columns with missing values:", int(df.isnull().any().sum()))

    print("\n🔸 Top columns by missing percentage:")
    print(summary["null_%"].sort_values(ascending=False).head(10))

    print("\n🔹 Example value distributions (categorical cols):")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols[:5]:
        print(f"\n▶ {col}:")
        print(df[col].value_counts(dropna=False).head(top_n))

    print("\n✅ Deep analysis complete.\n")


"""
analysis.py — Exploratory Data Analysis

This module provides functions for:
- Quick data profiling and summaries
- Deep statistical analysis
- Missing value analysis
- Data quality reporting

Author: OneEyeOwl
"""

from typing import Optional


def missing_value_report(
        df: pd.DataFrame,
        threshold: float = 0.0,
        verbose: bool = True
) -> pd.DataFrame:
    """Missing value report."""
    missing_stats = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "total_count": len(df),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2)
    })

    missing_stats = missing_stats[missing_stats["missing_pct"] > threshold]
    missing_stats = missing_stats.sort_values("missing_pct", ascending=False)

    if verbose:
        print("\n🔍 Missing Value Report")
        print("=" * 50)

        if missing_stats.empty:
            print(f"✅ No columns with missing values above {threshold}%")
        else:
            print(f"\nColumns with missing values > {threshold}%:\n")
            print(missing_stats.to_string())

            total_missing = df.isnull().sum().sum()
            total_cells = df.size
            overall_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

            print(f"\n📊 Overall Statistics:")
            print(f"  Total missing values: {total_missing:,}")
            print(f"  Total cells: {total_cells:,}")
            print(f"  Overall missing %: {overall_pct:.2f}%")

    return missing_stats


def completeness_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness matrix."""
    return df.notna().astype(int)


def numeric_summary(
        df: pd.DataFrame,
        percentiles: Optional[list] = None
) -> pd.DataFrame:
    """Numeric summary."""
    if percentiles is None:
        percentiles = [.25, .5, .75]

    summary = df.describe(percentiles=percentiles, include=[np.number]).T

    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    summary["zeros"] = (df == 0).sum()
    summary["negative"] = (df < 0).sum()

    col_order = ["count", "missing", "missing_pct", "mean", "std", "min"] + \
                [f"{int(p * 100)}%" for p in percentiles] + \
                ["max", "zeros", "negative"]

    available_cols = [c for c in col_order if c in summary.columns]

    return summary[available_cols]


def categorical_summary(
        df: pd.DataFrame,
        top_n: int = 10
) -> dict:
    """Categorical summary."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    summaries = {}

    for col in cat_cols:
        summaries[col] = {
            "unique_count": df[col].nunique(),
            "missing_count": df[col].isnull().sum(),
            "missing_pct": (df[col].isnull().sum() / len(df) * 100),
            "top_values": df[col].value_counts().head(top_n).to_dict()
        }

    return summaries


def data_quality_score(df: pd.DataFrame) -> dict:
    """Data quality score."""
    total_cells = df.size
    non_null_cells = df.notna().sum().sum()

    completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
    uniqueness = (len(df.drop_duplicates()) / len(df) * 100) if len(df) > 0 else 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    type_consistency = len(numeric_cols) / len(df.columns) * 100 if len(df.columns) > 0 else 0

    return {
        "completeness_pct": round(completeness, 2),
        "uniqueness_pct": round(uniqueness, 2),
        "type_consistency_pct": round(type_consistency, 2),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": total_cells - non_null_cells,
        "duplicate_rows": len(df) - len(df.drop_duplicates())
    }


def print_quality_report(df: pd.DataFrame, name: str = "DataFrame"):
    """Print quality report."""
    print(f"\n📋 Data Quality Report — {name}")
    print("=" * (25 + len(name)))

    metrics = data_quality_score(df)

    print(f"\n📊 Overall Metrics:")
    print(f"  Rows: {metrics['total_rows']:,}")
    print(f"  Columns: {metrics['total_columns']}")
    print(f"  Completeness: {metrics['completeness_pct']:.2f}%")
    print(f"  Uniqueness: {metrics['uniqueness_pct']:.2f}%")
    print(f"  Missing values: {metrics['missing_values']:,}")
    print(f"  Duplicate rows: {metrics['duplicate_rows']:,}")

    print(f"\n🔍 Column-Level Issues:")

    high_missing = df.columns[df.isnull().sum() / len(df) > 0.5]
    if len(high_missing) > 0:
        print(f"  ⚠️  High missing rate (>50%): {len(high_missing)} columns")
        for col in high_missing:
            pct = (df[col].isnull().sum() / len(df) * 100)
            print(f"     - {col}: {pct:.1f}%")
    else:
        print(f"  ✅ No columns with >50% missing values")

    print(f"\n✅ Report complete.\n")


def correlation_summary(
        df: pd.DataFrame,
        method: str = "pearson",
        threshold: float = 0.7,
        verbose: bool = True
) -> pd.DataFrame:
    """Correlation summary."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        if verbose:
            print("⚠️  Not enough numeric columns for correlation analysis")
        return pd.DataFrame()

    corr_matrix = numeric_df.corr(method=method)

    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) >= threshold:
                high_corr.append({
                    "variable_1": col1,
                    "variable_2": col2,
                    "correlation": round(corr_val, 3)
                })

    high_corr_df = pd.DataFrame(high_corr).sort_values("correlation",
                                                       key=abs,
                                                       ascending=False)

    if verbose:
        print(f"\n🔗 Correlation Analysis ({method.title()})")
        print("=" * 50)
        print(f"Threshold: |r| >= {threshold}")

        if high_corr_df.empty:
            print(f"\n✅ No correlations above threshold")
        else:
            print(f"\n⚠️  High correlations found:\n")
            print(high_corr_df.to_string(index=False))

    return high_corr_df


__all__ = [
    'quick_analysis',
    'deep_analysis',
    'UNIVARIATE_METRICS',
    'MULTIVARIATE_METRICS',
    'LAB_FEATURES',
    'NON_LAB_FEATURES',
    'DIABETES_FEATURES',
    'cohort_summary',
]

LAB_FEATURES = ["age", "gender", "smoker", "sbp", "cholesterol_mmolL", "has_diabetes"]
NON_LAB_FEATURES = ["age", "gender", "smoker", "sbp", "bmi"]
DIABETES_FEATURES = ["has_diabetes", "bs", "bsType", "bg_mgdl", "bg_pbs_equiv_mgdl"]

UNIVARIATE_METRICS = [
    "count", "mean", "median", "mode", "std", "var",
    "min", "max", "range", "q1", "q3", "iqr",
    "skewness", "kurtosis"
]

MULTIVARIATE_METRICS = [
    "pearson", "spearman", "kendall",
    "partial_correlation", "autocorrelation", "cross_correlation",
    "point_biserial", "cramers_v"
]


def cohort_summary(df: pd.DataFrame, name: str) -> dict:
    """Cohort summary."""
    out = {"name": name, "n": int(len(df))}
    print(f"\n✅ {name}")
    print(f"   Records: {out['n']:,}")

    if "age" in df.columns and df["age"].notna().any():
        out["age_min"] = float(df["age"].min())
        out["age_max"] = float(df["age"].max())
        print(f"   Age range: {out['age_min']:.0f} - {out['age_max']:.0f} years")

    if "gender_key" in df.columns:
        out["sex_counts"] = df["gender_key"].value_counts(dropna=False).to_dict()
        print(f"   Sex: {out['sex_counts']}")

    if "site_id" in df.columns:
        out["top_sites"] = df["site_id"].value_counts(dropna=False).head(10).to_dict()
        print(f"   Sites: {out['top_sites']}")

    return out
