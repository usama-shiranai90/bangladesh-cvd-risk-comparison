"""
Data visualization helpers for the CVD project.

This module re-exports visualization-related utilities from data_utils to
provide a dedicated API surface while maintaining backward compatibility.
"""
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np
import math
import seaborn as sns

try:
    plt.style.use(['science', 'nature', 'no-latex'])
except Exception:
    pass  # Fallback to default style
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})


# sns.set_style("whitegrid")


# Implementations moved here from data_utils to avoid circular imports

def annotate_bars(ax, total, fmt="{:.1f}%"):
    for p in ax.patches:
        h = p.get_height()
        if not h:
            continue
        pct = (h / total) * 100
        ax.annotate(f"{h}\n({fmt.format(pct)})",
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10,
                    xytext=(0, 5), textcoords="offset points")


def check_Outliers(dataframe, plt, sns):
    # 1. Identify Numeric Columns (exclude IDs)
    drop_cols = {"pid", "cid", "patient_id", "checkup_id", "barcode_id"}

    # Select numeric types
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out IDs and completely empty columns
    cols_to_check = [c for c in numeric_cols if c not in drop_cols and dataframe[c].notna().any()]

    if not cols_to_check:
        print("No suitable numeric columns found for outlier analysis.")
        return

    # Work with a slice for calculation
    numeric_df = dataframe[cols_to_check].copy()

    # 2. Calculate IQR Statistics
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 3. Identify Outliers
    # Returns a boolean DataFrame of the same shape
    outlier_mask = (numeric_df < lower_bound) | (numeric_df > upper_bound)

    # Flag rows in original dataframe (Create 'Outlier' column: True if ANY column has an outlier)
    dataframe["Outlier"] = outlier_mask.any(axis=1)

    # 4. Summary Table
    summary = pd.DataFrame({
        "non_null": numeric_df.count(),
        "outliers": outlier_mask.sum(),
        "outlier_rate": (outlier_mask.sum() / numeric_df.count()).replace([np.inf], np.nan)
    }).sort_values("outliers", ascending=False)

    print("--- Outlier Summary ---")
    from IPython.display import display as _display
    _display(summary)

    # 5. Visualization

    # Plot A: Overview Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_df)
    plt.title("Overview: Distribution of Numeric Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Plot B: Detailed Stripplots (2 columns per row)
    n = len(cols_to_check)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_check):
        ax = axes[i]

        # Data for this specific column
        s = numeric_df[col].dropna()
        m = outlier_mask.loc[s.index, col]  # Mask for valid values only

        # Horizontal Boxplot + Stripplot
        sns.boxplot(x=s.values, orient="h", ax=ax, color="lightgray")
        sns.stripplot(
            x=s.values,
            orient="h",
            hue=m.values,
            palette={False: "#4c72b0", True: "#c44e52"},  # Blue=Normal, Red=Outlier
            alpha=0.6,
            size=3,
            ax=ax,
            legend=False,
            jitter=True
        )

        count = m.sum()
        total = len(s)
        ax.set_title(f"{col}\n(Outliers: {int(count)} / {total} -> {count / total:.1%})")

    # Turn off unused axes
    for ax in axes[len(cols_to_check):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


__all__ = [
    'annotate_bars',
    'check_Outliers',
    'plot_cohort_counts',
    'plot_missingness_by_site',
    'plot_who_bin_occupancy',
    'plot_distributions',
    # Age impact & trend utilities
    'calculate_age_prevalence',
    'run_trend_tests',
    'plot_age_impact',
]


def plot_data_quality_audit(df, filename="Fig_Suppl_Data_Quality_Audit"):
    df = df.copy()

    # 1. Define the Flags to Check
    flags = {
        'BMI Calculation Mismatch': 'flag_bmi_discrepancy',
        'WHR Calculation Mismatch': 'flag_whr_discrepancy',
        'Implausible Blood Glucose': 'bg_flag_implausible',
        'Missing Height/Weight': ['height', 'weight'],
        'Missing Blood Pressure': ['sbp', 'dbp']
    }

    stats = []

    # 2. Calculate counts and percentages
    total_n = len(df)

    for label, col in flags.items():
        if isinstance(col, str) and col in df.columns:
            n_flagged = df[col].sum()
            stats.append({
                'Issue': label,
                'Count': n_flagged,
                'Percent': (n_flagged / total_n) * 100
            })

        elif isinstance(col, list):
            mask = df[col].isna().any(axis=1)
            n_flagged = mask.sum()
            stats.append({
                'Issue': label,
                'Count': n_flagged,
                'Percent': (n_flagged / total_n) * 100
            })

    stats_df = pd.DataFrame(stats).sort_values('Percent', ascending=True)

    # 3. Plotting with Seaborn
    sns.set(style="whitegrid")  # optional

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Sequential Reds palette, ordered by Percent
    palette = sns.color_palette("Reds", n_colors=len(stats_df))  # [web:10]

    # Horizontal barplot: x = Percent, y = Issue
    sns.barplot(
        data=stats_df,
        x="Percent",
        y="Issue",
        palette=palette,
        edgecolor="black",
        ax=ax,
        orient="h"
    )  # [web:3][web:6]

    ax.set_xlabel('Percentage of Records Flagged (%)', fontsize=12, fontweight='bold')
    ax.set_title('Data Quality Audit: Prevalence of Data Irregularities',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, stats_df['Percent'].max() * 1.2)

    # Add labels to bar ends
    for i, (p, n) in enumerate(zip(ax.patches, stats_df['Count'])):
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(
            width + 0.1,
            y,
            f"{width:.1f}%  (n={n:,})",
            va='center',
            fontsize=10,
            color='#333333'
        )  # [web:8]

    # Footnote
    plt.figtext(
        0.5, -0.05,
        f"Total Cohort Size: N={total_n:,}. 'Mismatch' indicates calculated metric differed from raw record.",
        ha="center",
        fontsize=9,
        style='italic'
    )

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Generated {filename}.png")
    plt.show()


def plot_cohort_counts(df: pd.DataFrame):
    def _sum(col: str) -> int:
        if col not in df.columns:
            return 0
        s = df[col]
        if s.dtype == bool or str(s.dtype) == "boolean":
            return int(s.fillna(False).sum())
        return int(pd.to_numeric(s, errors="coerce").fillna(0).sum())

    counts = {
        "eligible_nonlab": _sum("eligible_nonlab"),
        "eligible_lab": _sum("eligible_lab"),
        "eligible_paired": _sum("eligible_paired"),
        "who_domain_ok_nonlab": _sum("who_domain_ok_nonlab"),
        "who_domain_ok_lab": _sum("who_domain_ok_lab"),
    }

    fig, ax = plt.subplots()
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title("Cohort Sizes (Eligibility + WHO Domain)")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_missingness_by_site(
        df: pd.DataFrame,
        cols: list[str],
        site_col: str = "site_id",
        min_records_per_site: int = 30,
        top_n_sites: int = 30,
):
    if site_col not in df.columns:
        raise KeyError(f"'{site_col}' not found in df columns")

    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("None of the requested columns exist in df")

    site_counts = df[site_col].value_counts(dropna=False)
    keep_sites = site_counts[site_counts >= min_records_per_site].head(top_n_sites).index
    d = df[df[site_col].isin(keep_sites)].copy()

    miss = d.groupby(site_col)[cols].apply(lambda g: g.isna().mean())
    miss = miss.loc[keep_sites]
    n_by_site = d.groupby(site_col).size().reindex(keep_sites)

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cols)), max(6, 0.35 * len(miss.index))))
    im = ax.imshow(miss.values, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_title(f"Missingness by Site (Top {len(miss.index)} sites, n≥{min_records_per_site})")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(miss.index)))
    ax.set_yticklabels([f"{sid} (n={int(n_by_site.loc[sid])})" for sid in miss.index])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Missing fraction")

    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_who_bin_occupancy(df: pd.DataFrame, which: str = "nonlab"):
    flag = "who_domain_ok_nonlab" if which == "nonlab" else "who_domain_ok_lab"
    d = df[df.get(flag, False)].copy()
    if d.empty:
        print(f"No records in WHO domain for {which}.")
        return None, None

    pivot = pd.crosstab(d["age_band"], d["sbp_band"])

    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(f"WHO Domain OK ({which}): Age band × SBP band")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_distributions(df: pd.DataFrame, cols=None, bins=40):
    if cols is None:
        cols = ["age", "sbp", "bmi", "cholesterol_mmolL"]
    for col in cols:
        if col in df.columns and df[col].notna().any():
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=bins)
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            fig.tight_layout()
            plt.show()


# ==========================================
# Age impact & trend visualizations (moved from notebooks)
# ==========================================

def calculate_age_prevalence(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Calculates percent of high risk (>=10% and >=20%) per age band.
    Returns a DataFrame with columns: n, n_10, n_20, pct_10, pct_20, se_10, se_20
    - df: DataFrame containing at least 'age_band' and the given score column.
    - score_col: Column with WHO/ASCVD risk percentage values.
    """
    if 'age_band' not in df.columns:
        raise KeyError("'age_band' column is required in df")
    d = df.copy()
    score = pd.to_numeric(d.get(score_col), errors="coerce")
    d['high_risk_10'] = score >= 10
    d['high_risk_20'] = score >= 20

    stats_df = d.groupby('age_band', observed=True).agg(
        n=('age_band', 'count'),
        n_10=('high_risk_10', 'sum'),
        n_20=('high_risk_20', 'sum'),
    )

    # Percentages
    stats_df['pct_10'] = (stats_df['n_10'] / stats_df['n']) * 100
    stats_df['pct_20'] = (stats_df['n_20'] / stats_df['n']) * 100

    # Standard error for binomial proportions (expressed in percentage points)
    stats_df['se_10'] = np.sqrt((stats_df['pct_10'] / 100 * (1 - stats_df['pct_10'] / 100)) / stats_df['n']) * 100
    stats_df['se_20'] = np.sqrt((stats_df['pct_20'] / 100 * (1 - stats_df['pct_20'] / 100)) / stats_df['n']) * 100

    return stats_df


def run_trend_tests(df: pd.DataFrame, score_col: str, cohort_name: str):
    """
    Prints simple trend diagnostics for risk vs age.
    - Linear regression on continuous age vs risk score.
    - Logistic trend by ordinal age band for >=10% risk.
    Dependencies (imported lazily): scipy, statsmodels.
    """
    print(f"\n=== Statistical Trend Tests: {cohort_name} ===")

    # A) Continuous trend (Linear Regression)
    try:
        from scipy import stats as _scistats  # lazy import
    except Exception:
        _scistats = None

    df_lr = df[[c for c in ['age', score_col] if c in df.columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if _scistats is not None and {'age', score_col}.issubset(df_lr.columns):
        slope, intercept, r_value, p_value, std_err = _scistats.linregress(
            df_lr['age'],
            pd.to_numeric(df_lr[score_col], errors='coerce')
        )
        print("1. Continuous Trend (Linear Reg):")
        print(f"   Slope: {slope:.3f} (risk score points per year of age)")
        print(f"   P-value: {p_value:.2e} {'***' if p_value < 0.001 else ''}")
        print(f"   R-squared: {r_value ** 2:.3f}")
    else:
        print("1. Continuous Trend: skipped (missing scipy.stats or required columns)")

    # B) Categorical trend (Logistic Regression)
    try:
        import statsmodels.api as sm  # lazy import
    except Exception:
        sm = None

    if sm is not None and 'age_band' in df.columns:
        df_reg = df.copy()
        df_reg['outcome_10'] = (pd.to_numeric(df_reg[score_col], errors='coerce') >= 10).astype('Int64')
        # Ordinal mapping from categorical order if present
        if hasattr(df_reg['age_band'], 'cat'):
            categories = df_reg['age_band'].cat.categories
            cat_map = {cat: i for i, cat in enumerate(categories)}
            df_reg['age_ordinal'] = df_reg['age_band'].map(cat_map)
        else:
            # Fallback: sort unique labels naturally
            uniq = pd.unique(df_reg['age_band'].astype(str))
            try:
                # Try numeric sort if possible
                uniq_sorted = sorted(uniq, key=lambda x: float(x.split('-')[0]))
            except Exception:
                uniq_sorted = sorted(uniq)
            df_reg['age_ordinal'] = df_reg['age_band'].astype(str).map({v: i for i, v in enumerate(uniq_sorted)})

        X = sm.add_constant(df_reg['age_ordinal'])
        y = df_reg['outcome_10']
        mask = X['age_ordinal'].notna() & y.notna()
        try:
            model = sm.Logit(y[mask].astype(int), X[mask]).fit(disp=0)
            or_val = float(np.exp(model.params.get('age_ordinal', np.nan)))
            p_val = float(model.pvalues.get('age_ordinal', np.nan))
            print("2. Categorical Trend (Logistic Reg on Ordinal Bands):")
            print(f"   Odds Ratio (per band step): {or_val:.2f}")
            print(f"   P-value: {p_val:.2e} {'***' if p_val < 0.001 else ''}")
        except Exception as e:
            print("2. Categorical Trend: model failed:", e)
    else:
        print("2. Categorical Trend: skipped (missing statsmodels or 'age_band')")

    print("-" * 40)


def plot_age_impact(
        df_main: pd.DataFrame,
        df_valid: pd.DataFrame,
        score_col_main: str,
        score_col_valid: str,
        save_path: str | None = "Fig_Age_Impact_Analysis.png",
):
    """
    Plot prevalence of >=10% and >=20% risk across age bands for the main cohort,
    with the validation cohort (paired subset) overlaid as markers.
    Returns (fig, ax, stats_main, stats_valid).
    """
    stats_main = calculate_age_prevalence(df_main, score_col_main)
    stats_valid = calculate_age_prevalence(df_valid, score_col_valid)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(stats_main.index))

    # Main cohort lines
    ax.plot(x, stats_main['pct_10'], marker='o', color='#d62828', linewidth=2, label='Main (≥10% Risk)')
    ax.fill_between(x,
                    stats_main['pct_10'] - 1.96 * stats_main['se_10'],
                    stats_main['pct_10'] + 1.96 * stats_main['se_10'],
                    color='#d62828', alpha=0.1)
    ax.plot(x, stats_main['pct_20'], marker='s', color='#f4a261', linewidth=2, linestyle='--',
            label='Main (≥20% Risk)')

    # Validation cohort overlay
    ax.scatter(x, stats_valid['pct_10'], color='#2a9d8f', marker='x', s=80, zorder=5,
               label='Validation (≥10% Risk)')

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(stats_main.index, fontsize=11, fontweight='bold')
    ax.set_xlabel("Age Band (Years)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Prevalence of High Risk (%)", fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(100, float(np.nanmax(stats_main[['pct_10', 'pct_20']].values)) * 1.15))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_title("Age-Based Escalation of Cardiovascular Risk\n(Main vs Validation)", fontsize=14, pad=15,
                 fontweight='bold')

    ax.legend(loc='upper left', frameon=True, fontsize=10)

    # Add N counts
    for i, n in enumerate(stats_main['n']):
        try:
            ax.text(i, 5, f"n={int(n)}", ha='center', fontsize=8, color='gray')
        except Exception:
            pass

    plt.tight_layout()
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Generated Chart: {save_path}")
        except Exception as e:
            print("Warning: failed to save figure:", e)
    return fig, ax, stats_main, stats_valid
