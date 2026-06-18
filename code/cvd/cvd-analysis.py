"""
Comprehensive CVD Risk Stratification Analysis
WHO 2019 Laboratory vs Non-Laboratory Risk Charts in Bangladesh

Research Questions:
RQ1: Distribution of WHO 2019 non-laboratory 10-year CVD risk
RQ2: Agreement between laboratory and non-laboratory risk categories
RQ3: Missed high-risk individuals analysis
RQ4: Machine learning augmentation for high-risk detection
RQ5: Decision utility under risk-threshold treatment strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report, brier_score_loss, recall_score, precision_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CVDRiskAnalysis:
    """Complete CVD Risk Analysis Pipeline"""

    def __init__(self, df_nonlab, df_lab, df_paired, df_who_nonlab, df_who_lab, df_sites):
        """Initialize with all required dataframes"""
        self.df_nonlab = df_nonlab
        self.df_lab = df_lab
        self.df_paired = df_paired
        self.df_who_nonlab = df_who_nonlab
        self.df_who_lab = df_who_lab
        self.df_sites = df_sites

        self.df_who_nonlab = self._merge_sites(df_who_nonlab)
        self.df_who_lab = self._merge_sites(df_who_lab)
        self.df_paired_analysis = self._merge_sites(df_paired)

        self.results = {}
        self.figures = {}
        self.tables = {}

    def _merge_sites(self, df):
        """Merge with site information"""
        return df.merge(self.df_sites[['site_id', 'location_type', 'division_name',
                                       'district_name', 'sedentary']],
                        on='site_id', how='left')


    def rq1_baseline_distribution(self):
        """RQ1: Distribution of WHO 2019 non-laboratory 10-year CVD risk"""
        print("\n" + "=" * 80)
        print("RQ1: BASELINE CVD RISK BURDEN & HETEROGENEITY")
        print("=" * 80)

        df = self.df_who_nonlab.copy()

        print("\n1. OVERALL RISK DISTRIBUTION (n={:,})".format(len(df)))
        print("-" * 60)
        risk_stats = df['risk_nonlab'].describe()
        print(risk_stats)

        print("\nRisk Categories:")
        risk_cat_dist = df['risk_nonlab_cat'].value_counts(dropna=False).sort_index()
        risk_cat_pct = (risk_cat_dist / len(df) * 100).round(2)
        for cat, count in risk_cat_dist.items():
            print(f"  {cat}: {count:,} ({risk_cat_pct[cat]}%)")

        high_risk = (df['risk_nonlab'] >= 20).sum()
        print(f"\nHigh-Risk (≥20%): {high_risk:,} ({high_risk / len(df) * 100:.2f}%)")

        results = {
            'overall': risk_stats.to_dict(),
            'risk_categories': risk_cat_dist.to_dict(),
            'high_risk_n': high_risk,
            'high_risk_pct': high_risk / len(df) * 100
        }

        print("\n2. BY SEX")
        print("-" * 60)
        sex_stats = df.groupby('gender')['risk_nonlab'].agg([
            'count', 'mean', 'std', 'median',
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).round(2)
        print(sex_stats)

        sex_highrisk = df.groupby('gender').apply(
            lambda x: (x['risk_nonlab'] >= 20).sum() / len(x) * 100
        ).round(2)
        print("\nHigh-Risk Prevalence by Sex:")
        print(sex_highrisk)

        male_risk = df[df['gender'] == 'M']['risk_nonlab'].dropna()
        female_risk = df[df['gender'] == 'F']['risk_nonlab'].dropna()
        u_stat, p_val = stats.mannwhitneyu(male_risk, female_risk)
        print(f"\nMann-Whitney U test: U={u_stat:.0f}, p={p_val:.4f}")

        results['by_sex'] = sex_stats.to_dict()
        results['sex_highrisk'] = sex_highrisk.to_dict()
        results['sex_test'] = {'u_statistic': u_stat, 'p_value': p_val}

        print("\n3. BY AGE BAND")
        print("-" * 60)
        age_stats = df.groupby('age_band')['risk_nonlab'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(2)
        print(age_stats)

        age_highrisk = df.groupby('age_band').apply(
            lambda x: (x['risk_nonlab'] >= 20).sum() / len(x) * 100
        ).round(2)
        print("\nHigh-Risk Prevalence by Age:")
        print(age_highrisk)

        age_groups = [df[df['age_band'] == age]['risk_nonlab'].dropna()
                      for age in df['age_band'].unique() if pd.notna(age)]
        h_stat, p_val = kruskal(*age_groups)
        print(f"\nKruskal-Wallis test: H={h_stat:.2f}, p={p_val:.4f}")

        results['by_age'] = age_stats.to_dict()
        results['age_highrisk'] = age_highrisk.to_dict()
        results['age_test'] = {'h_statistic': h_stat, 'p_value': p_val}

        print("\n4. BY LOCATION TYPE (URBAN/RURAL)")
        print("-" * 60)
        loc_stats = df.groupby('location_type')['risk_nonlab'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(2)
        print(loc_stats)

        loc_highrisk = df.groupby('location_type').apply(
            lambda x: (x['risk_nonlab'] >= 20).sum() / len(x) * 100
        ).round(2)
        print("\nHigh-Risk Prevalence by Location:")
        print(loc_highrisk)

        if len(df['location_type'].dropna().unique()) == 2:
            urban = df[df['location_type'] == 'Urban']['risk_nonlab'].dropna()
            rural = df[df['location_type'] == 'Rural']['risk_nonlab'].dropna()
            u_stat, p_val = stats.mannwhitneyu(urban, rural)
            print(f"\nMann-Whitney U test: U={u_stat:.0f}, p={p_val:.4f}")
            results['location_test'] = {'u_statistic': u_stat, 'p_value': p_val}

        results['by_location'] = loc_stats.to_dict()
        results['location_highrisk'] = loc_highrisk.to_dict()

        print("\n5. BY SITE (Top 10)")
        print("-" * 60)
        site_stats = df.groupby('site_id')['risk_nonlab'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(2).sort_values('count', ascending=False).head(10)
        print(site_stats)

        site_groups = [df[df['site_id'] == site]['risk_nonlab'].dropna()
                       for site in df['site_id'].unique() if len(df[df['site_id'] == site]) > 10]
        if len(site_groups) > 2:
            h_stat, p_val = kruskal(*site_groups)
            print(f"\nKruskal-Wallis test across sites: H={h_stat:.2f}, p={p_val:.4f}")
            results['site_test'] = {'h_statistic': h_stat, 'p_value': p_val}

        results['by_site'] = site_stats.to_dict()

        print("\n6. AGE × SEX INTERACTION")
        print("-" * 60)
        age_sex = df.groupby(['age_band', 'gender'])['risk_nonlab'].agg([
            'count', 'mean', 'median'
        ]).round(2)
        print(age_sex)

        results['age_sex_interaction'] = age_sex.to_dict()

        self.results['rq1'] = results

        self._rq1_visualizations(df)

        return results

    def _rq1_visualizations(self, df):
        """Create RQ1 visualizations"""
        fig = plt.figure(figsize=(20, 12))

        ax1 = plt.subplot(3, 3, 1)
        df['risk_nonlab'].hist(bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(20, color='red', linestyle='--', linewidth=2, label='High-Risk Threshold')
        plt.xlabel('10-Year CVD Risk (%)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Overall Risk Distribution', fontsize=12, fontweight='bold')
        plt.legend()

        ax2 = plt.subplot(3, 3, 2)
        risk_cat_order = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
        cat_data = df['risk_nonlab_cat'].value_counts().reindex(risk_cat_order, fill_value=0)
        cat_data.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.xlabel('Risk Category', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.title('Risk Category Distribution', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        ax3 = plt.subplot(3, 3, 3)
        df.boxplot(column='risk_nonlab', by='gender', ax=ax3)
        plt.axhline(20, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Sex', fontsize=10)
        plt.ylabel('10-Year CVD Risk (%)', fontsize=10)
        plt.title('Risk Distribution by Sex', fontsize=12, fontweight='bold')
        plt.suptitle('')

        ax4 = plt.subplot(3, 3, 4)
        age_order = sorted([age for age in df['age_band'].unique() if pd.notna(age)])
        df.boxplot(column='risk_nonlab', by='age_band', ax=ax4)
        plt.axhline(20, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Age Band', fontsize=10)
        plt.ylabel('10-Year CVD Risk (%)', fontsize=10)
        plt.title('Risk Distribution by Age', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.suptitle('')

        ax5 = plt.subplot(3, 3, 5)
        df.boxplot(column='risk_nonlab', by='location_type', ax=ax5)
        plt.axhline(20, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Location Type', fontsize=10)
        plt.ylabel('10-Year CVD Risk (%)', fontsize=10)
        plt.title('Risk Distribution by Location', fontsize=12, fontweight='bold')
        plt.suptitle('')

        ax6 = plt.subplot(3, 3, 6)
        pivot_data = df.pivot_table(values='risk_nonlab', index='age_band',
                                    columns='gender', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Mean Risk (%)'}, ax=ax6)
        plt.title('Mean Risk: Age × Sex', fontsize=12, fontweight='bold')

        ax7 = plt.subplot(3, 3, 7)
        highrisk_age = df.groupby('age_band').apply(
            lambda x: (x['risk_nonlab'] >= 20).sum() / len(x) * 100
        ).sort_index()
        highrisk_age.plot(kind='bar', color='darkred', edgecolor='black', ax=ax7)
        plt.xlabel('Age Band', fontsize=10)
        plt.ylabel('High-Risk Prevalence (%)', fontsize=10)
        plt.title('High-Risk (≥20%) Prevalence by Age', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        ax8 = plt.subplot(3, 3, 8)
        site_mean = df.groupby('site_id')['risk_nonlab'].mean().sort_values(ascending=False).head(10)
        site_mean.plot(kind='barh', color='teal', edgecolor='black', ax=ax8)
        plt.xlabel('Mean Risk (%)', fontsize=10)
        plt.ylabel('Site ID', fontsize=10)
        plt.title('Mean Risk by Site (Top 10)', fontsize=12, fontweight='bold')

        ax9 = plt.subplot(3, 3, 9)
        age_subset = df[df['age_band'].isin(['40-44', '50-54', '60-64', '70-74'])]
        sns.violinplot(data=age_subset, x='age_band', y='risk_nonlab',
                       hue='gender', split=True, ax=ax9)
        plt.axhline(20, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Age Band', fontsize=10)
        plt.ylabel('10-Year CVD Risk (%)', fontsize=10)
        plt.title('Risk Distribution: Age × Sex', fontsize=12, fontweight='bold')
        plt.legend(title='Sex')

        plt.tight_layout()
        plt.savefig('resource/images/rq1_baseline_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.figures['rq1'] = 'resource/images/rq1_baseline_distribution.png'
        print("\n✓ RQ1 visualizations saved")


    def rq2_concordance_analysis(self):
        """RQ2: Agreement between laboratory and non-laboratory risk categories"""
        print("\n" + "=" * 80)
        print("RQ2: CONCORDANCE & SYSTEMATIC BIAS ANALYSIS")
        print("=" * 80)

        df = self.df_paired_analysis[self.df_paired_analysis["who_domain_ok_lab"]].copy()
        print(f"\n1. PAIRED SAMPLE (raw n={len(df)})")
        print("-" * 60)

        df["risk_lab"] = pd.to_numeric(df.get("risk_lab"), errors="coerce")
        df["risk_nonlab"] = pd.to_numeric(df.get("risk_nonlab"), errors="coerce")

        df_cont = df.dropna(subset=["risk_lab", "risk_nonlab"]).copy()
        print(f"Usable for continuous agreement: n={len(df_cont)}")

        df_cont["risk_diff"] = df_cont["risk_nonlab"] - df_cont["risk_lab"]
        df_cont["abs_diff"] = df_cont["risk_diff"].abs()

        print("\nContinuous Risk Agreement:")
        print(f"  Mean difference (non-lab − lab): {df_cont['risk_diff'].mean():.2f}%")
        print(f"  Median difference: {df_cont['risk_diff'].median():.2f}%")
        print(f"  SD of differences: {df_cont['risk_diff'].std():.2f}%")
        print(f"  Mean absolute difference: {df_cont['abs_diff'].mean():.2f}%")

        corr = df_cont[["risk_nonlab", "risk_lab"]].corr().iloc[0, 1]
        print(f"\nPearson correlation: r = {corr:.3f}")

        t_stat, p_val = stats.ttest_rel(df_cont["risk_nonlab"], df_cont["risk_lab"])
        print(f"Paired t-test: t = {t_stat:.2f}, p = {p_val:.4f}")

        try:
            w_stat, w_pval = stats.wilcoxon(df_cont["risk_nonlab"], df_cont["risk_lab"])
        except Exception:
            w_stat, w_pval = np.nan, np.nan
        print(
            f"Wilcoxon signed-rank: W = {w_stat if pd.notna(w_stat) else 'NA'}, p = {w_pval if pd.notna(w_pval) else 'NA'}")

        print("\n2. CATEGORICAL AGREEMENT")
        print("-" * 60)

        cat_order = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]

        def _clean_cat(s: pd.Series) -> pd.Series:
            """Clean cat."""
            s = s.astype("string").str.strip()

            replacements = {
                "<5": "<5%",
                "<5 %": "<5%",
                "< 5%": "<5%",
                ">=30%": "≥30%",
                "≥30": "≥30%",
                ">= 30%": "≥30%",
                "30%+": "≥30%",
                "30% or more": "≥30%",
                "5-10%": "5% to <10%",
                "5% to 10%": "5% to <10%",
                "10-20%": "10% to <20%",
                "10% to 20%": "10% to <20%",
                "20-30%": "20% to <30%",
                "20% to 30%": "20% to <30%",
            }
            s = s.replace(replacements)
            return s

        if "risk_lab_cat" not in df.columns:
            df["risk_lab_cat"] = pd.cut(
                df["risk_lab"],
                bins=[0, 5, 10, 20, 30, np.inf],
                labels=cat_order,
                right=False,
                include_lowest=True,
            ).astype("string")

        if "risk_nonlab_cat" not in df.columns:
            df["risk_nonlab_cat"] = pd.cut(
                df["risk_nonlab"],
                bins=[0, 5, 10, 20, 30, np.inf],
                labels=cat_order,
                right=False,
                include_lowest=True,
            ).astype("string")

        df["risk_lab_cat"] = _clean_cat(df["risk_lab_cat"])
        df["risk_nonlab_cat"] = _clean_cat(df["risk_nonlab_cat"])

        df_cat = df[df["risk_lab_cat"].isin(cat_order) & df["risk_nonlab_cat"].isin(cat_order)].copy()
        print(f"Usable for categorical agreement: n={len(df_cat)}")

        confusion = pd.crosstab(
            df_cat["risk_lab_cat"],
            df_cat["risk_nonlab_cat"],
            rownames=["Lab"],
            colnames=["Non-Lab"],
            dropna=False,
        ).reindex(index=cat_order, columns=cat_order, fill_value=0)

        print("\nConfusion Matrix (counts):")
        print(confusion)

        total_cat = len(df_cat)
        exact_agreement = (df_cat["risk_nonlab_cat"] == df_cat["risk_lab_cat"]).sum()
        agreement_pct = (exact_agreement / total_cat * 100) if total_cat else np.nan
        print(
            f"\nExact agreement: {exact_agreement}/{total_cat} ({agreement_pct:.2f}%)" if total_cat else "\nExact agreement: NA (no valid categorical pairs)")

        lab_codes = pd.Categorical(df_cat["risk_lab_cat"], categories=cat_order, ordered=True).codes
        nonlab_codes = pd.Categorical(df_cat["risk_nonlab_cat"], categories=cat_order, ordered=True).codes
        within_one = (np.abs(lab_codes - nonlab_codes) <= 1).sum() if total_cat else 0
        within_one_pct = (within_one / total_cat * 100) if total_cat else np.nan
        print(
            f"Within 1 category: {within_one}/{total_cat} ({within_one_pct:.2f}%)" if total_cat else "Within 1 category: NA")

        from sklearn.metrics import cohen_kappa_score

        if total_cat:
            kappa = cohen_kappa_score(df_cat["risk_lab_cat"], df_cat["risk_nonlab_cat"], labels=cat_order)
            kappa_weighted = cohen_kappa_score(
                df_cat["risk_lab_cat"],
                df_cat["risk_nonlab_cat"],
                labels=cat_order,
                weights="linear",
            )
            print(f"\nCohen's kappa: κ = {kappa:.3f}")
            print(f"Weighted kappa: κw = {kappa_weighted:.3f}")
        else:
            kappa, kappa_weighted = np.nan, np.nan
            print("\nCohen's kappa: NA (no valid categorical pairs)")
            print("Weighted kappa: NA (no valid categorical pairs)")

        print("\n3. HIGH-RISK (≥20%) AGREEMENT")
        print("-" * 60)

        df_hr = df_cont.copy()
        df_hr["highrisk_lab"] = df_hr["risk_lab"] >= 20
        df_hr["highrisk_nonlab"] = df_hr["risk_nonlab"] >= 20

        highrisk_confusion = pd.crosstab(
            df_hr["highrisk_lab"],
            df_hr["highrisk_nonlab"],
            rownames=["Lab ≥20%"],
            colnames=["Non-Lab ≥20%"],
            dropna=False,
        )
        print("\n2×2 Contingency Table:")
        print(highrisk_confusion)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        if len(df_hr):
            accuracy = accuracy_score(df_hr["highrisk_lab"], df_hr["highrisk_nonlab"])
            sensitivity = recall_score(df_hr["highrisk_lab"], df_hr["highrisk_nonlab"])
            specificity = recall_score(~df_hr["highrisk_lab"], ~df_hr["highrisk_nonlab"])
            ppv = precision_score(df_hr["highrisk_lab"], df_hr["highrisk_nonlab"], zero_division=0)
            npv = precision_score(~df_hr["highrisk_lab"], ~df_hr["highrisk_nonlab"], zero_division=0)
            f1 = f1_score(df_hr["highrisk_lab"], df_hr["highrisk_nonlab"], zero_division=0)
        else:
            accuracy = sensitivity = specificity = ppv = npv = f1 = np.nan

        print(f"\nDiagnostic Performance (Non-Lab vs Lab):")
        print(f"  Accuracy: {accuracy:.3f}" if pd.notna(accuracy) else "  Accuracy: NA")
        print(f"  Sensitivity (Recall): {sensitivity:.3f}" if pd.notna(sensitivity) else "  Sensitivity: NA")
        print(f"  Specificity: {specificity:.3f}" if pd.notna(specificity) else "  Specificity: NA")
        print(f"  PPV (Precision): {ppv:.3f}" if pd.notna(ppv) else "  PPV: NA")
        print(f"  NPV: {npv:.3f}" if pd.notna(npv) else "  NPV: NA")
        print(f"  F1-Score: {f1:.3f}" if pd.notna(f1) else "  F1-Score: NA")

        print("\n4. SYSTEMATIC BIAS ANALYSIS")
        print("-" * 60)

        total_cont = len(df_cont)
        underestimates = (df_cont["risk_diff"] < 0).sum()
        overestimates = (df_cont["risk_diff"] > 0).sum()
        exact_match = (df_cont["risk_diff"] == 0).sum()

        print(
            f"Non-lab underestimates: {underestimates} ({underestimates / total_cont * 100:.2f}%)" if total_cont else "Non-lab underestimates: NA")
        print(
            f"Non-lab overestimates: {overestimates} ({overestimates / total_cont * 100:.2f}%)" if total_cont else "Non-lab overestimates: NA")
        print(
            f"Exact match: {exact_match} ({exact_match / total_cont * 100:.2f}%)" if total_cont else "Exact match: NA")

        print("\nBias by Lab Risk Level:")
        risk_bins = [0, 5, 10, 20, 30, 100]
        risk_labels = ["<5%", "5-10%", "10-20%", "20-30%", "≥30%"]
        df_cont["lab_risk_bin"] = pd.cut(df_cont["risk_lab"], bins=risk_bins, labels=risk_labels, include_lowest=True)

        bias_by_risk = df_cont.groupby("lab_risk_bin")["risk_diff"].agg(["count", "mean", "std"])
        print(bias_by_risk)

        corr_risk_bias = df_cont[["risk_lab", "risk_diff"]].corr().iloc[0, 1] if total_cont else np.nan
        print(f"\nCorrelation (Lab risk vs Bias): r = {corr_risk_bias:.3f}" if pd.notna(
            corr_risk_bias) else "\nCorrelation (Lab risk vs Bias): NA")

        results = {
            "n_paired": int(total_cont),
            "n_paired_raw": int(len(df)),
            "n_continuous": int(len(df_cont)),
            "n_categorical": int(len(df_cat)),
            "n_highrisk_eval": int(len(df_hr)),
            "continuous_agreement": {
                "mean_diff": float(df_cont["risk_diff"].mean()) if len(df_cont) else np.nan,
                "median_diff": float(df_cont["risk_diff"].median()) if len(df_cont) else np.nan,
                "sd_diff": float(df_cont["risk_diff"].std()) if len(df_cont) else np.nan,
                "mean_abs_diff": float(df_cont["abs_diff"].mean()) if len(df_cont) else np.nan,
                "correlation": float(corr) if pd.notna(corr) else np.nan,
                "paired_ttest": {"t": float(t_stat) if pd.notna(t_stat) else np.nan,
                                 "p": float(p_val) if pd.notna(p_val) else np.nan},
                "wilcoxon": {"w": float(w_stat) if pd.notna(w_stat) else np.nan,
                             "p": float(w_pval) if pd.notna(w_pval) else np.nan},
            },
            "categorical_agreement": {
                'exact_agreement': float(agreement_pct),
                'within_one_category': float(within_one_pct),

                "exact_agreement_pct": float(agreement_pct) if pd.notna(agreement_pct) else np.nan,
                "within_one_category_pct": float(within_one_pct) if pd.notna(within_one_pct) else np.nan,
                "cohen_kappa": float(kappa) if pd.notna(kappa) else np.nan,
                "weighted_kappa": float(kappa_weighted) if pd.notna(kappa_weighted) else np.nan,
                "confusion_matrix": confusion.to_dict(),
            },
            "highrisk_agreement": {
                "accuracy": float(accuracy) if pd.notna(accuracy) else np.nan,
                "sensitivity": float(sensitivity) if pd.notna(sensitivity) else np.nan,
                "specificity": float(specificity) if pd.notna(specificity) else np.nan,
                "ppv": float(ppv) if pd.notna(ppv) else np.nan,
                "npv": float(npv) if pd.notna(npv) else np.nan,
                "f1_score": float(f1) if pd.notna(f1) else np.nan,
                "contingency_table": highrisk_confusion.to_dict(),
            },
            "systematic_bias": {
                "underestimates_pct": float(underestimates / total_cont * 100) if total_cont else np.nan,
                "overestimates_pct": float(overestimates / total_cont * 100) if total_cont else np.nan,
                "bias_by_risk": bias_by_risk.to_dict(),
                "risk_bias_correlation": float(corr_risk_bias) if pd.notna(corr_risk_bias) else np.nan,
            },
        }

        self.results["rq2"] = results

        df_viz = df_cont.copy()
        df_viz["risk_lab_cat"] = _clean_cat(df_viz["risk_lab_cat"])
        df_viz["risk_nonlab_cat"] = _clean_cat(df_viz["risk_nonlab_cat"])
        df_viz["highrisk_lab"] = df_viz["risk_lab"] >= 20
        df_viz["highrisk_nonlab"] = df_viz["risk_nonlab"] >= 20
        df_viz["lab_risk_bin"] = pd.cut(df_viz["risk_lab"], bins=risk_bins, labels=risk_labels, include_lowest=True)

        self._rq2_visualizations(df_viz)

        return results

    def _rq2_visualizations(self, df):
        """Create RQ2 visualizations (robust to missing/partial categories)"""
        fig = plt.figure(figsize=(20, 16))

        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(df["risk_lab"], df["risk_nonlab"], alpha=0.5, s=20)
        plt.plot([0, 40], [0, 40], "r--", linewidth=2, label="Identity line")
        plt.axhline(20, color="gray", linestyle=":", alpha=0.7)
        plt.axvline(20, color="gray", linestyle=":", alpha=0.7)
        plt.xlabel("Laboratory Risk (%)", fontsize=10)
        plt.ylabel("Non-Laboratory Risk (%)", fontsize=10)
        plt.title("Lab vs Non-Lab Risk Scores", fontsize=12, fontweight="bold")
        plt.legend()
        plt.xlim(0, 40)
        plt.ylim(0, 40)

        ax2 = plt.subplot(3, 3, 2)
        mean_risk = (df["risk_lab"] + df["risk_nonlab"]) / 2
        diff = df["risk_nonlab"] - df["risk_lab"]
        plt.scatter(mean_risk, diff, alpha=0.5, s=20)
        plt.axhline(0, color="black", linestyle="-", linewidth=2)
        plt.axhline(diff.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {diff.mean():.2f}%")
        plt.axhline(diff.mean() + 1.96 * diff.std(), color="red", linestyle=":", linewidth=2, label="±1.96 SD")
        plt.axhline(diff.mean() - 1.96 * diff.std(), color="red", linestyle=":", linewidth=2)
        plt.xlabel("Mean Risk (Lab + Non-Lab) / 2 (%)", fontsize=10)
        plt.ylabel("Difference (Non-Lab − Lab) (%)", fontsize=10)
        plt.title("Bland-Altman Plot", fontsize=12, fontweight="bold")
        plt.legend()

        ax3 = plt.subplot(3, 3, 3)
        diff.hist(bins=50, edgecolor="black", alpha=0.7)
        plt.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        plt.axvline(diff.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {diff.mean():.2f}%")
        plt.xlabel("Risk Difference (Non-Lab − Lab) (%)", fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
        plt.title("Distribution of Differences", fontsize=12, fontweight="bold")
        plt.legend()

        ax4 = plt.subplot(3, 3, 4)
        cat_order = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
        confusion = pd.crosstab(df["risk_lab_cat"], df["risk_nonlab_cat"], dropna=False).reindex(
            index=cat_order, columns=cat_order, fill_value=0
        )
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"}, ax=ax4)
        plt.xlabel("Non-Lab Category", fontsize=10)
        plt.ylabel("Lab Category", fontsize=10)
        plt.title("Categorical Agreement Matrix", fontsize=12, fontweight="bold")

        ax5 = plt.subplot(3, 3, 5)
        df_box = df.dropna(subset=["lab_risk_bin"]).copy()
        if len(df_box):
            df_box.assign(risk_diff=diff.loc[df_box.index]).boxplot(column="risk_diff", by="lab_risk_bin", ax=ax5)
            plt.axhline(0, color="red", linestyle="--", linewidth=2)
            plt.xlabel("Laboratory Risk Level", fontsize=10)
            plt.ylabel("Bias (Non-Lab − Lab) (%)", fontsize=10)
            plt.title("Bias by Lab Risk Level", fontsize=12, fontweight="bold")
            plt.suptitle("")
            plt.xticks(rotation=45, ha="right")
        else:
            ax5.text(0.5, 0.5, "No data for lab_risk_bin", ha="center", va="center")
            ax5.set_axis_off()

        ax6 = plt.subplot(3, 3, 6)
        highrisk_confusion = pd.crosstab(df["highrisk_lab"], df["highrisk_nonlab"], dropna=False)
        sns.heatmap(highrisk_confusion, annot=True, fmt="d", cmap="RdYlGn_r", cbar_kws={"label": "Count"}, ax=ax6)
        plt.xlabel("Non-Lab ≥20%", fontsize=10)
        plt.ylabel("Lab ≥20%", fontsize=10)
        plt.title("High-Risk Agreement (≥20%)", fontsize=12, fontweight="bold")

        ax7 = plt.subplot(3, 3, 7)
        df_plot = df.copy()
        df_plot["risk_diff"] = df_plot["risk_nonlab"] - df_plot["risk_lab"]

        under_data = df_plot[df_plot["risk_diff"] < -2]
        exact_data = df_plot[df_plot["risk_diff"].abs() <= 2]
        over_data = df_plot[df_plot["risk_diff"] > 2]

        if len(under_data):
            plt.scatter(under_data["risk_lab"], under_data["risk_nonlab"], alpha=0.6, s=30, label="Underestimated")
        if len(exact_data):
            plt.scatter(exact_data["risk_lab"], exact_data["risk_nonlab"], alpha=0.3, s=20, label="±2% agreement")
        if len(over_data):
            plt.scatter(over_data["risk_lab"], over_data["risk_nonlab"], alpha=0.6, s=30, label="Overestimated")

        plt.plot([0, 40], [0, 40], "k--", linewidth=2, alpha=0.5)
        plt.axhline(20, color="gray", linestyle=":", alpha=0.7)
        plt.axvline(20, color="gray", linestyle=":", alpha=0.7)
        plt.xlabel("Laboratory Risk (%)", fontsize=10)
        plt.ylabel("Non-Laboratory Risk (%)", fontsize=10)
        plt.title("Agreement Patterns", fontsize=12, fontweight="bold")
        plt.legend()
        plt.xlim(0, 40)
        plt.ylim(0, 40)

        ax8 = plt.subplot(3, 3, 8)
        lab_sorted = np.sort(df["risk_lab"].dropna().values)
        nonlab_sorted = np.sort(df["risk_nonlab"].dropna().values)
        if len(lab_sorted):
            plt.plot(lab_sorted, np.arange(len(lab_sorted)) / len(lab_sorted), label="Laboratory", linewidth=2)
        if len(nonlab_sorted):
            plt.plot(nonlab_sorted, np.arange(len(nonlab_sorted)) / len(nonlab_sorted), label="Non-Laboratory",
                     linewidth=2)
        plt.axvline(20, color="red", linestyle="--", linewidth=2, alpha=0.7)
        plt.xlabel("CVD Risk (%)", fontsize=10)
        plt.ylabel("Cumulative Probability", fontsize=10)
        plt.title("Cumulative Distribution Functions", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)

        ax9 = plt.subplot(3, 3, 9)
        if "age_band" in df.columns and "gender" in df.columns:
            pivot_bias = df_plot.pivot_table(values="risk_diff", index="age_band", columns="gender", aggfunc="mean")
            pivot_bias.plot(kind="bar", ax=ax9)
            plt.axhline(0, color="black", linestyle="--", linewidth=2)
            plt.xlabel("Age Band", fontsize=10)
            plt.ylabel("Mean Bias (Non-Lab − Lab) (%)", fontsize=10)
            plt.title("Bias by Age and Sex", fontsize=12, fontweight="bold")
            plt.legend(title="Sex")
            plt.xticks(rotation=45, ha="right")
        else:
            ax9.text(0.5, 0.5, "Missing age_band/gender columns", ha="center", va="center")
            ax9.set_axis_off()

        plt.tight_layout()
        plt.savefig("resource/images/rq2_concordance_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        self.figures["rq2"] = "resource/images/rq2_concordance_analysis.png"
        print("\n✓ RQ2 visualizations saved")


    def rq3_missed_highrisk_analysis(self):
        """RQ3: Analysis of lab-defined high-risk individuals missed by non-lab chart"""
        print("\n" + "=" * 80)
        print("RQ3: MISSED HIGH-RISK INDIVIDUALS ANALYSIS")
        print("=" * 80)

        df = self.df_paired_analysis[self.df_paired_analysis['who_domain_ok_lab']].copy()

        df['highrisk_lab'] = df['risk_lab'] >= 20
        df['highrisk_nonlab'] = df['risk_nonlab'] >= 20
        df['missed'] = df['highrisk_lab'] & ~df['highrisk_nonlab']

        n_highrisk_lab = df['highrisk_lab'].sum()
        n_missed = df['missed'].sum()
        missed_rate = n_missed / n_highrisk_lab * 100 if n_highrisk_lab > 0 else 0

        print(f"\n1. MISSED HIGH-RISK BURDEN")
        print("-" * 60)
        print(f"Lab-defined high-risk (≥20%): {n_highrisk_lab}")
        print(f"Missed by non-lab chart: {n_missed}")
        print(f"Missed rate: {missed_rate:.2f}%")
        print(f"Sensitivity of non-lab chart: {100 - missed_rate:.2f}%")

        print("\n2. CHARACTERISTICS OF MISSED INDIVIDUALS")
        print("-" * 60)

        missed_df = df[df['missed']].copy()
        not_missed_hr = df[df['highrisk_lab'] & ~df['missed']].copy()

        print("\nAge:")
        print(f"  Missed: {missed_df['age'].mean():.1f} ± {missed_df['age'].std():.1f}")
        print(f"  Not missed: {not_missed_hr['age'].mean():.1f} ± {not_missed_hr['age'].std():.1f}")

        print("\nSex distribution:")
        print("  Missed:")
        print(missed_df['gender'].value_counts())
        print("  Not missed:")
        print(not_missed_hr['gender'].value_counts())

        print("\nSystolic BP:")
        print(f"  Missed: {missed_df['sbp'].mean():.1f} ± {missed_df['sbp'].std():.1f}")
        print(f"  Not missed: {not_missed_hr['sbp'].mean():.1f} ± {not_missed_hr['sbp'].std():.1f}")

        print("\nBMI:")
        print(f"  Missed: {missed_df['bmi'].mean():.1f} ± {missed_df['bmi'].std():.1f}")
        print(f"  Not missed: {not_missed_hr['bmi'].mean():.1f} ± {not_missed_hr['bmi'].std():.1f}")

        print("\nSmoking status:")
        print("  Missed:")
        print(missed_df['smoker_who'].value_counts())
        print("  Not missed:")
        print(not_missed_hr['smoker_who'].value_counts())

        print("\nDiabetes status:")
        print("  Missed:")
        print(missed_df['has_diabetes'].value_counts())
        print("  Not missed:")
        print(not_missed_hr['has_diabetes'].value_counts())

        print("\n3. PREDICTORS OF BEING MISSED (Logistic Regression)")
        print("-" * 60)

        highrisk_subset = df[df['highrisk_lab']].copy()

        highrisk_subset['sex_male'] = (highrisk_subset['gender'] == 'M').astype(int)
        highrisk_subset['smoker'] = (highrisk_subset['smoker_who'] == 'Smoker').astype(int)
        highrisk_subset['diabetes'] = highrisk_subset['has_diabetes'].astype(int)
        highrisk_subset['urban'] = (highrisk_subset['location_type'] == 'Urban').astype(int)

        features = ['age', 'sex_male', 'sbp', 'bmi', 'smoker', 'diabetes']
        X = highrisk_subset[features].copy()
        y = highrisk_subset['missed'].astype(int)

        X = X.fillna(X.mean())

        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X, y)

        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': lr_model.coef_[0],
            'Odds Ratio': np.exp(lr_model.coef_[0])
        }).sort_values('Coefficient', ascending=False)

        print("\nLogistic Regression Coefficients:")
        print(coef_df.to_string(index=False))

        print("\n4. STATISTICAL COMPARISONS (Missed vs Not Missed)")
        print("-" * 60)

        t_stat, p_val = stats.ttest_ind(missed_df['age'].dropna(),
                                        not_missed_hr['age'].dropna())
        print(f"\nAge: t = {t_stat:.2f}, p = {p_val:.4f}")

        t_stat, p_val = stats.ttest_ind(missed_df['sbp'].dropna(),
                                        not_missed_hr['sbp'].dropna())
        print(f"SBP: t = {t_stat:.2f}, p = {p_val:.4f}")

        t_stat, p_val = stats.ttest_ind(missed_df['bmi'].dropna(),
                                        not_missed_hr['bmi'].dropna())
        print(f"BMI: t = {t_stat:.2f}, p = {p_val:.4f}")

        cont_table = pd.crosstab(highrisk_subset['missed'], highrisk_subset['sex_male'])
        chi2, p_val, dof, expected = chi2_contingency(cont_table)
        print(f"Sex: χ² = {chi2:.2f}, p = {p_val:.4f}")

        cont_table = pd.crosstab(highrisk_subset['missed'], highrisk_subset['smoker'])
        chi2, p_val, dof, expected = chi2_contingency(cont_table)
        print(f"Smoking: χ² = {chi2:.2f}, p = {p_val:.4f}")

        cont_table = pd.crosstab(highrisk_subset['missed'], highrisk_subset['diabetes'])
        chi2, p_val, dof, expected = chi2_contingency(cont_table)
        print(f"Diabetes: χ² = {chi2:.2f}, p = {p_val:.4f}")

        results = {
            'n_highrisk_lab': int(n_highrisk_lab),
            'n_missed': int(n_missed),
            'missed_rate': missed_rate,
            'sensitivity_nonlab': 100 - missed_rate,
            'characteristics_comparison': {
                'age_missed': missed_df['age'].mean(),
                'age_not_missed': not_missed_hr['age'].mean(),
                'sbp_missed': missed_df['sbp'].mean(),
                'sbp_not_missed': not_missed_hr['sbp'].mean(),
                'bmi_missed': missed_df['bmi'].mean(),
                'bmi_not_missed': not_missed_hr['bmi'].mean()
            },
            'predictors': coef_df.to_dict('records')
        }

        self.results['rq3'] = results

        self._rq3_visualizations(df, missed_df, not_missed_hr, coef_df)

        return results

    def _rq3_visualizations(self, df, missed_df, not_missed_hr, coef_df):
        """Create RQ3 visualizations"""
        fig = plt.figure(figsize=(20, 12))

        ax1 = plt.subplot(3, 3, 1)
        n_highrisk = df['highrisk_lab'].sum()
        n_missed = df['missed'].sum()
        n_detected = n_highrisk - n_missed

        labels = ['Detected\nby Non-Lab', 'Missed\nby Non-Lab']
        sizes = [n_detected, n_missed]
        colors = ['#4CAF50', '#F44336']
        explode = (0, 0.1)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        plt.title(f'High-Risk Detection Rate\n(n={n_highrisk} lab-defined ≥20%)',
                  fontsize=12, fontweight='bold')

        ax2 = plt.subplot(3, 3, 2)
        plt.hist([missed_df['age'].dropna(), not_missed_hr['age'].dropna()],
                 bins=15, label=['Missed', 'Detected'], alpha=0.7, edgecolor='black')
        plt.xlabel('Age (years)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Age Distribution: Missed vs Detected', fontsize=12, fontweight='bold')
        plt.legend()

        ax3 = plt.subplot(3, 3, 3)
        data_to_plot = [missed_df['sbp'].dropna(), not_missed_hr['sbp'].dropna()]
        bp = plt.boxplot(data_to_plot, labels=['Missed', 'Detected'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#F44336')
        bp['boxes'][1].set_facecolor('#4CAF50')
        plt.ylabel('Systolic BP (mmHg)', fontsize=10)
        plt.title('SBP: Missed vs Detected', fontsize=12, fontweight='bold')

        ax4 = plt.subplot(3, 3, 4)
        data_to_plot = [missed_df['bmi'].dropna(), not_missed_hr['bmi'].dropna()]
        bp = plt.boxplot(data_to_plot, labels=['Missed', 'Detected'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#F44336')
        bp['boxes'][1].set_facecolor('#4CAF50')
        plt.ylabel('BMI (kg/m²)', fontsize=10)
        plt.title('BMI: Missed vs Detected', fontsize=12, fontweight='bold')

        ax5 = plt.subplot(3, 3, 5)
        sex_missed = missed_df['gender'].value_counts()
        sex_detected = not_missed_hr['gender'].value_counts()
        x = np.arange(len(sex_missed))
        width = 0.35
        plt.bar(x - width / 2, sex_missed.values, width, label='Missed', color='#F44336',
                edgecolor='black')
        plt.bar(x + width / 2, sex_detected.values, width, label='Detected', color='#4CAF50',
                edgecolor='black')
        plt.xlabel('Sex', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.title('Sex Distribution', fontsize=12, fontweight='bold')
        plt.xticks(x, sex_missed.index)
        plt.legend()

        ax6 = plt.subplot(3, 3, 6)
        smoke_missed = missed_df['smoker_who'].value_counts()
        smoke_detected = not_missed_hr['smoker_who'].value_counts()
        all_categories = list(set(smoke_missed.index) | set(smoke_detected.index))
        smoke_missed_vals = [smoke_missed.get(cat, 0) for cat in all_categories]
        smoke_detected_vals = [smoke_detected.get(cat, 0) for cat in all_categories]

        x = np.arange(len(all_categories))
        width = 0.35
        plt.bar(x - width / 2, smoke_missed_vals, width, label='Missed', color='#F44336',
                edgecolor='black')
        plt.bar(x + width / 2, smoke_detected_vals, width, label='Detected', color='#4CAF50',
                edgecolor='black')
        plt.xlabel('Smoking Status', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.title('Smoking Status Distribution', fontsize=12, fontweight='bold')
        plt.xticks(x, all_categories, rotation=45, ha='right')
        plt.legend()

        ax7 = plt.subplot(3, 3, 7)
        diab_missed = missed_df['has_diabetes'].value_counts()
        diab_detected = not_missed_hr['has_diabetes'].value_counts()
        all_categories = list(set(diab_missed.index) | set(diab_detected.index))
        diab_missed_vals = [diab_missed.get(cat, 0) for cat in all_categories]
        diab_detected_vals = [diab_detected.get(cat, 0) for cat in all_categories]

        x = np.arange(len(all_categories))
        width = 0.35
        plt.bar(x - width / 2, diab_missed_vals, width, label='Missed', color='#F44336',
                edgecolor='black')
        plt.bar(x + width / 2, diab_detected_vals, width, label='Detected', color='#4CAF50',
                edgecolor='black')
        plt.xlabel('Diabetes Status', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.title('Diabetes Status Distribution', fontsize=12, fontweight='bold')
        plt.xticks(x, all_categories)
        plt.legend()

        ax8 = plt.subplot(3, 3, 8)
        coef_df_sorted = coef_df.sort_values('Odds Ratio')
        colors_or = ['red' if x < 1 else 'green' for x in coef_df_sorted['Odds Ratio']]
        plt.barh(coef_df_sorted['Feature'], coef_df_sorted['Odds Ratio'],
                 color=colors_or, edgecolor='black')
        plt.axvline(1, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Odds Ratio', fontsize=10)
        plt.title('Predictors of Being Missed\n(Odds Ratios from Logistic Regression)',
                  fontsize=12, fontweight='bold')
        plt.xscale('log')

        ax9 = plt.subplot(3, 3, 9)
        plt.scatter(missed_df['risk_lab'], missed_df['risk_nonlab'],
                    c='red', alpha=0.6, s=50, label='Missed', edgecolor='black')
        plt.scatter(not_missed_hr['risk_lab'], not_missed_hr['risk_nonlab'],
                    c='green', alpha=0.4, s=30, label='Detected')
        plt.plot([0, 50], [0, 50], 'k--', linewidth=2, alpha=0.5)
        plt.axhline(20, color='gray', linestyle=':', linewidth=2)
        plt.axvline(20, color='gray', linestyle=':', linewidth=2)
        plt.xlabel('Laboratory Risk (%)', fontsize=10)
        plt.ylabel('Non-Laboratory Risk (%)', fontsize=10)
        plt.title('Risk Scores: Missed vs Detected', fontsize=12, fontweight='bold')
        plt.legend()

        plt.tight_layout()
        plt.savefig('resource/images/rq3_missed_highrisk.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.figures['rq3'] = 'resource/images/rq3_missed_highrisk.png'
        print("\n✓ RQ3 visualizations saved")


    def rq4_ml_augmentation(self):
        """RQ4: Machine learning model for improved high-risk detection"""
        print("\n" + "=" * 80)
        print("RQ4: MACHINE LEARNING AUGMENTATION FOR HIGH-RISK DETECTION")
        print("=" * 80)

        df = self.df_paired_analysis[self.df_paired_analysis['who_domain_ok_lab']].copy()

        df['highrisk_lab'] = (df['risk_lab'] >= 20).astype(int)

        print("\n1. FEATURE PREPARATION")
        print("-" * 60)

        df['sex_male'] = (df['gender'] == 'M').astype(int)
        df['smoker'] = (df['smoker_who'] == 'Smoker').astype(int)
        df['has_diabetes'] = df['has_diabetes'].astype(int)

        df['highrisk_lab'] = (df['risk_lab'] >= 20).astype(int)


        feature_cols = ['age', 'sex_male', 'sbp', 'bmi', 'smoker', 'has_diabetes']

        df_ml = df[feature_cols + ['highrisk_lab', 'site_id', 'risk_nonlab']].copy()
        df_ml = df_ml.dropna(subset=feature_cols)

        X = df_ml[feature_cols].values
        y = df_ml['highrisk_lab'].values
        sites = df_ml['site_id'].values

        print(f"Samples: {len(df_ml)}")
        print(f"Features: {feature_cols}")
        print(f"Positive class (≥20%): {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")

        baseline_pred = (df_ml['risk_nonlab'] >= 20).astype(int).values

        from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

        baseline_auc = roc_auc_score(y, baseline_pred)
        baseline_sens = recall_score(y, baseline_pred)
        baseline_spec = recall_score(1 - y, 1 - baseline_pred)
        baseline_ppv = precision_score(y, baseline_pred) if baseline_pred.sum() > 0 else 0

        print(f"\nBaseline (WHO Non-Lab Chart):")
        print(f"  AUC: {baseline_auc:.3f}")
        print(f"  Sensitivity: {baseline_sens:.3f}")
        print(f"  Specificity: {baseline_spec:.3f}")
        print(f"  PPV: {baseline_ppv:.3f}")

        print("\n2. MODEL TRAINING & CROSS-VALIDATION")
        print("-" * 60)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5,
                                                    random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                            random_state=42)
        }

        results = {}
        best_model_name = None
        best_auc = 0

        for name, model in models.items():
            print(f"\n{name}:")

            y_pred_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            auc = roc_auc_score(y, y_pred_proba)
            ap = average_precision_score(y, y_pred_proba)
            sens = recall_score(y, y_pred)
            spec = recall_score(1 - y, 1 - y_pred)
            ppv = precision_score(y, y_pred) if y_pred.sum() > 0 else 0
            f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0

            print(f"  AUC: {auc:.3f} (+{auc - baseline_auc:.3f})")
            print(f"  AUPRC: {ap:.3f}")
            print(f"  Sensitivity: {sens:.3f} (+{sens - baseline_sens:.3f})")
            print(f"  Specificity: {spec:.3f}")
            print(f"  PPV: {ppv:.3f}")
            print(f"  F1-Score: {f1:.3f}")

            results[name] = {
                'auc': auc,
                'auprc': ap,
                'sensitivity': sens,
                'specificity': spec,
                'ppv': ppv,
                'f1': f1,
                'predictions': y_pred_proba
            }

            if auc > best_auc:
                best_auc = auc
                best_model_name = name

        print(f"\n✓ Best model: {best_model_name} (AUC = {best_auc:.3f})")

        best_model = models[best_model_name]
        best_model.fit(X, y)

        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print("\nFeature Importance:")
            print(importance_df.to_string(index=False))
        elif hasattr(best_model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': best_model.coef_[0],
                'Abs Coefficient': np.abs(best_model.coef_[0])
            }).sort_values('Abs Coefficient', ascending=False)
            print("\nFeature Coefficients:")
            print(coef_df.to_string(index=False))
            importance_df = coef_df[['Feature', 'Abs Coefficient']].copy()
            importance_df.columns = ['Feature', 'Importance']

        print("\n3. MODEL CALIBRATION")
        print("-" * 60)

        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
        calibrated_model.fit(X, y)

        y_pred_calib = cross_val_predict(calibrated_model, X, y, cv=skf, method='predict_proba')[:, 1]

        brier_uncalib = brier_score_loss(y, results[best_model_name]['predictions'])
        brier_calib = brier_score_loss(y, y_pred_calib)

        print(f"Brier score (uncalibrated): {brier_uncalib:.4f}")
        print(f"Brier score (calibrated): {brier_calib:.4f}")
        print(f"Improvement: {brier_uncalib - brier_calib:.4f}")

        print("\n4. SITE GENERALIZABILITY (Leave-One-Site-Out)")
        print("-" * 60)

        site_aucs = []
        unique_sites = np.unique(sites)

        for site in unique_sites:
            test_mask = (sites == site)
            if test_mask.sum() < 10:
                continue

            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if y_test.sum() == 0 or y_test.sum() == len(y_test):
                continue

            model_site = models[best_model_name]
            model_site.fit(X_train, y_train)
            y_pred_site = model_site.predict_proba(X_test)[:, 1]

            auc_site = roc_auc_score(y_test, y_pred_site)
            site_aucs.append(auc_site)

        mean_site_auc = np.mean(site_aucs)
        std_site_auc = np.std(site_aucs)

        print(f"Mean AUC across sites: {mean_site_auc:.3f} ± {std_site_auc:.3f}")
        print(f"Min AUC: {np.min(site_aucs):.3f}")
        print(f"Max AUC: {np.max(site_aucs):.3f}")

        rq4_results = {
            'baseline': {
                'auc': baseline_auc,
                'sensitivity': baseline_sens,
                'specificity': baseline_spec,
                'ppv': baseline_ppv
            },
            'models': {name: {k: v for k, v in res.items() if k != 'predictions'}
                       for name, res in results.items()},
            'best_model': best_model_name,
            'feature_importance': importance_df.to_dict('records'),
            'calibration': {
                'brier_uncalibrated': brier_uncalib,
                'brier_calibrated': brier_calib
            },
            'site_generalizability': {
                'mean_auc': mean_site_auc,
                'std_auc': std_site_auc,
                'site_aucs': site_aucs
            }
        }

        self.results['rq4'] = rq4_results

        self._rq4_visualizations(X, y, results, best_model_name, y_pred_calib,
                                 importance_df, feature_cols, baseline_pred)

        return rq4_results, best_model

    def _rq4_visualizations(self, X, y, results, best_model_name, y_pred_calib,
                            importance_df, feature_cols, baseline_pred):
        """Create RQ4 visualizations"""
        fig = plt.figure(figsize=(20, 12))

        ax1 = plt.subplot(2, 3, 1)

        from sklearn.metrics import roc_curve
        fpr_base, tpr_base, _ = roc_curve(y, baseline_pred)
        baseline_auc = roc_auc_score(y, baseline_pred)
        plt.plot(fpr_base, tpr_base, label=f'WHO Non-Lab (AUC={baseline_auc:.3f})',
                 linewidth=2, linestyle='--')

        colors = ['blue', 'green', 'orange']
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y, res['predictions'])
            plt.plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})',
                     linewidth=2, color=color)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title('ROC Curves', fontsize=12, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        ax2 = plt.subplot(2, 3, 2)

        from sklearn.metrics import precision_recall_curve

        prec_base, rec_base, _ = precision_recall_curve(y, baseline_pred)
        plt.plot(rec_base, prec_base, label='WHO Non-Lab', linewidth=2, linestyle='--')

        for (name, res), color in zip(results.items(), colors):
            prec, rec, _ = precision_recall_curve(y, res['predictions'])
            plt.plot(rec, prec, label=f'{name} (AP={res["auprc"]:.3f})',
                     linewidth=2, color=color)

        plt.xlabel('Recall (Sensitivity)', fontsize=10)
        plt.ylabel('Precision (PPV)', fontsize=10)
        plt.title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        ax3 = plt.subplot(2, 3, 3)
        importance_sorted = importance_df.sort_values('Importance', ascending=True)
        plt.barh(importance_sorted['Feature'], importance_sorted['Importance'],
                 color='steelblue', edgecolor='black')
        plt.xlabel('Importance', fontsize=10)
        plt.title(f'Feature Importance ({best_model_name})', fontsize=12, fontweight='bold')

        ax4 = plt.subplot(2, 3, 4)

        prob_true, prob_pred = calibration_curve(y, y_pred_calib, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='ML Model')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        plt.xlabel('Predicted Probability', fontsize=10)
        plt.ylabel('Observed Frequency', fontsize=10)
        plt.title('Calibration Curve', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)

        ax5 = plt.subplot(2, 3, 5)

        metrics_names = ['Sensitivity', 'Specificity', 'PPV', 'F1-Score']
        x = np.arange(len(metrics_names))
        width = 0.15

        baseline_metrics = [
            recall_score(y, baseline_pred),
            recall_score(1 - y, 1 - baseline_pred),
            precision_score(y, baseline_pred) if baseline_pred.sum() > 0 else 0,
            2 * precision_score(y, baseline_pred) * recall_score(y, baseline_pred) /
            (precision_score(y, baseline_pred) + recall_score(y, baseline_pred))
            if (precision_score(y, baseline_pred) + recall_score(y, baseline_pred)) > 0 else 0
        ]

        plt.bar(x - width, baseline_metrics, width, label='WHO Non-Lab',
                color='gray', edgecolor='black')

        best_results = results[best_model_name]
        ml_metrics = [
            best_results['sensitivity'],
            best_results['specificity'],
            best_results['ppv'],
            best_results['f1']
        ]

        plt.bar(x, ml_metrics, width, label=best_model_name,
                color='steelblue', edgecolor='black')

        plt.xlabel('Metric', fontsize=10)
        plt.ylabel('Score', fontsize=10)
        plt.title('Performance Comparison', fontsize=12, fontweight='bold')
        plt.xticks(x, metrics_names, rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1.1)

        for i, (b1, b2) in enumerate(zip(baseline_metrics, ml_metrics)):
            plt.text(i - width, b1 + 0.02, f'{b1:.2f}', ha='center', fontsize=8)
            plt.text(i, b2 + 0.02, f'{b2:.2f}', ha='center', fontsize=8)

        ax6 = plt.subplot(2, 3, 6)

        best_pred = (results[best_model_name]['predictions'] >= 0.5).astype(int)
        cm_ml = confusion_matrix(y, best_pred)
        cm_base = confusion_matrix(y, baseline_pred)

        fig_cm, (ax_base, ax_ml) = plt.subplots(1, 2, figsize=(12, 5))

        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax_base,
                    xticklabels=['<20%', '≥20%'], yticklabels=['<20%', '≥20%'])
        ax_base.set_xlabel('Predicted', fontsize=10)
        ax_base.set_ylabel('Actual (Lab)', fontsize=10)
        ax_base.set_title('WHO Non-Lab Chart', fontsize=12, fontweight='bold')

        sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Greens', ax=ax_ml,
                    xticklabels=['<20%', '≥20%'], yticklabels=['<20%', '≥20%'])
        ax_ml.set_xlabel('Predicted', fontsize=10)
        ax_ml.set_ylabel('Actual (Lab)', fontsize=10)
        ax_ml.set_title(f'{best_model_name}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('resource/images/rq4_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close(fig_cm)

        plt.tight_layout()
        plt.savefig('resource/images/rq4_ml_augmentation.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.figures['rq4'] = 'resource/images/rq4_ml_augmentation.png'
        self.figures['rq4_cm'] = 'resource/images/rq4_confusion_matrices.png'
        print("\n✓ RQ4 visualizations saved")


    def rq5_decision_utility(self, ml_model):
        """RQ5: Decision utility under risk-threshold treatment strategies"""
        print("\n" + "=" * 80)
        print("RQ5: DECISION UTILITY & TREATMENT STRATEGIES")
        print("=" * 80)

        df = self.df_paired_analysis[self.df_paired_analysis['who_domain_ok_lab']].copy()

        df['highrisk_lab'] = (df['risk_lab'] >= 20).astype(int)

        df['sex_male'] = (df['gender'] == 'M').astype(int)
        df['smoker'] = (df['smoker_who'] == 'Smoker').astype(int)
        df['has_diabetes'] = df['has_diabetes'].astype(int)

        feature_cols = ['age', 'sex_male', 'sbp', 'bmi', 'smoker', 'has_diabetes']

        needed_cols = feature_cols + ['highrisk_lab', 'risk_nonlab', 'risk_lab']
        df_ml = df.dropna(subset=feature_cols + ['risk_nonlab', 'risk_lab']).copy()
        df_ml = df_ml[needed_cols].copy()

        X = df_ml[feature_cols].values

        df_ml['ml_risk_score'] = ml_model.predict_proba(X)[:, 1] * 100.0

        thresholds = [10, 20]
        results = {}

        n_total = len(df_ml)

        for threshold in thresholds:
            print(f"\n{'=' * 60}")
            print(f"THRESHOLD: ≥{threshold}% RISK")
            print('=' * 60)

            df_ml[f'hr_lab_{threshold}'] = (df_ml['risk_lab'] >= threshold).astype(int)
            df_ml[f'hr_nonlab_{threshold}'] = (df_ml['risk_nonlab'] >= threshold).astype(int)
            df_ml[f'hr_ml_{threshold}'] = (df_ml['ml_risk_score'] >= threshold).astype(int)

            n_eligible_lab = int(df_ml[f'hr_lab_{threshold}'].sum())
            n_eligible_nonlab = int(df_ml[f'hr_nonlab_{threshold}'].sum())
            n_eligible_ml = int(df_ml[f'hr_ml_{threshold}'].sum())

            print(f"\n1. TREATMENT ELIGIBILITY (n={n_total})")
            print("-" * 60)
            print(f"Lab (gold standard): {n_eligible_lab} ({n_eligible_lab / n_total * 100:.1f}%)")
            print(f"Non-lab chart: {n_eligible_nonlab} ({n_eligible_nonlab / n_total * 100:.1f}%)")
            print(f"ML-augmented: {n_eligible_ml} ({n_eligible_ml / n_total * 100:.1f}%)")

            n_missed_nonlab = int(((df_ml[f'hr_lab_{threshold}'] == 1) & (df_ml[f'hr_nonlab_{threshold}'] == 0)).sum())
            n_missed_ml = int(((df_ml[f'hr_lab_{threshold}'] == 1) & (df_ml[f'hr_ml_{threshold}'] == 0)).sum())

            reduction_missed = n_missed_nonlab - n_missed_ml
            pct_reduction = (reduction_missed / n_missed_nonlab * 100) if n_missed_nonlab > 0 else 0.0

            print(f"\n2. MISSED HIGH-RISK INDIVIDUALS")
            print("-" * 60)
            denom = n_eligible_lab if n_eligible_lab > 0 else 1
            print(f"Non-lab chart: {n_missed_nonlab} ({n_missed_nonlab / denom * 100:.1f}% of lab-defined)")
            print(f"ML-augmented: {n_missed_ml} ({n_missed_ml / denom * 100:.1f}% of lab-defined)")
            print(f"Reduction: {reduction_missed} ({pct_reduction:.1f}% reduction)")

            n_overtreat_nonlab = int(
                ((df_ml[f'hr_lab_{threshold}'] == 0) & (df_ml[f'hr_nonlab_{threshold}'] == 1)).sum())
            n_overtreat_ml = int(((df_ml[f'hr_lab_{threshold}'] == 0) & (df_ml[f'hr_ml_{threshold}'] == 1)).sum())

            print(f"\n3. OVERTREATMENT (False Positives)")
            print("-" * 60)
            denom_nonlab = n_eligible_nonlab if n_eligible_nonlab > 0 else 1
            denom_ml = n_eligible_ml if n_eligible_ml > 0 else 1
            print(f"Non-lab chart: {n_overtreat_nonlab} ({n_overtreat_nonlab / denom_nonlab * 100:.1f}% of identified)")
            print(f"ML-augmented: {n_overtreat_ml} ({n_overtreat_ml / denom_ml * 100:.1f}% of identified)")

            print(f"\n4. TWO-STAGE WORKFLOW SIMULATION")
            print("-" * 60)
            print("Stage 1: ML screening → Stage 2: Lab testing for ML-positive")

            lab_tests_universal = n_total
            lab_tests_twostage = n_eligible_ml
            lab_reduction = lab_tests_universal - lab_tests_twostage
            lab_reduction_pct = lab_reduction / lab_tests_universal * 100.0

            df_ml_positive = df_ml[df_ml[f'hr_ml_{threshold}'] == 1]
            n_confirmed_highrisk = int(df_ml_positive[f'hr_lab_{threshold}'].sum())

            print(f"Lab tests needed (universal screening): {lab_tests_universal}")
            print(f"Lab tests needed (two-stage): {lab_tests_twostage}")
            print(f"Lab test reduction: {lab_reduction} ({lab_reduction_pct:.1f}%)")
            print(f"\nConfirmed high-risk from two-stage: {n_confirmed_highrisk}")
            print(f"Detection rate: {n_confirmed_highrisk / denom * 100:.1f}% of all lab-defined")

            print(f"\n5. NET BENEFIT ANALYSIS")
            print("-" * 60)

            pt = threshold / 100.0
            w = pt / (1 - pt)

            tp_nonlab = int(((df_ml[f'hr_lab_{threshold}'] == 1) & (df_ml[f'hr_nonlab_{threshold}'] == 1)).sum())
            fp_nonlab = n_overtreat_nonlab
            nb_nonlab = (tp_nonlab / n_total) - (fp_nonlab / n_total) * w

            tp_ml = int(((df_ml[f'hr_lab_{threshold}'] == 1) & (df_ml[f'hr_ml_{threshold}'] == 1)).sum())
            fp_ml = n_overtreat_ml
            nb_ml = (tp_ml / n_total) - (fp_ml / n_total) * w

            nb_all = (n_eligible_lab / n_total) - ((n_total - n_eligible_lab) / n_total) * w
            nb_none = 0.0

            print(f"Net Benefit (treat all): {nb_all:.4f}")
            print(f"Net Benefit (non-lab chart): {nb_nonlab:.4f}")
            print(f"Net Benefit (ML-augmented): {nb_ml:.4f}")
            print(f"Net Benefit (treat none): {nb_none:.4f}")
            print(f"\nIncremental NB (ML vs Non-Lab): {nb_ml - nb_nonlab:.4f}")

            results[f'threshold_{threshold}'] = {
                'treatment_eligibility': {'lab': n_eligible_lab, 'nonlab': n_eligible_nonlab, 'ml': n_eligible_ml},
                'missed_highrisk': {'nonlab': n_missed_nonlab, 'ml': n_missed_ml,
                                    'reduction': reduction_missed, 'pct_reduction': pct_reduction},
                'overtreatment': {'nonlab': n_overtreat_nonlab, 'ml': n_overtreat_ml},
                'twostage_workflow': {
                    'lab_tests_universal': lab_tests_universal,
                    'lab_tests_twostage': lab_tests_twostage,
                    'lab_reduction': lab_reduction,
                    'lab_reduction_pct': lab_reduction_pct,
                    'confirmed_highrisk': n_confirmed_highrisk
                },
                'net_benefit': {
                    'treat_all': nb_all,
                    'nonlab': nb_nonlab,
                    'ml': nb_ml,
                    'treat_none': nb_none,
                    'incremental_ml_vs_nonlab': nb_ml - nb_nonlab
                }
            }

        self.results['rq5'] = results
        self._rq5_visualizations(df_ml, results, thresholds)
        return results

    def _rq5_visualizations(self, df_ml, results, thresholds):
        """Create RQ5 visualizations (clean + correct)"""
        fig = plt.figure(figsize=(20, 10))

        n_total = len(df_ml)

        for idx, threshold in enumerate(thresholds):
            base_col = idx * 3

            ax1 = plt.subplot(2, 3, base_col + 1)
            methods = ['Lab\n(Gold)', 'Non-Lab\nChart', 'ML\nAugmented']
            eligible = [
                results[f'threshold_{threshold}']['treatment_eligibility']['lab'],
                results[f'threshold_{threshold}']['treatment_eligibility']['nonlab'],
                results[f'threshold_{threshold}']['treatment_eligibility']['ml'],
            ]
            bars = plt.bar(methods, eligible, edgecolor='black', linewidth=1)
            plt.ylabel('Number Eligible', fontsize=10)
            plt.title(f'Treatment Eligibility (≥{threshold}%)', fontsize=12, fontweight='bold')
            for bar, val in zip(bars, eligible):
                pct = (val / n_total * 100) if n_total else 0
                plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                         f"{int(val)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

            ax2 = plt.subplot(2, 3, base_col + 2)
            methods2 = ['Non-Lab\nChart', 'ML\nAugmented']
            missed = [
                results[f'threshold_{threshold}']['missed_highrisk']['nonlab'],
                results[f'threshold_{threshold}']['missed_highrisk']['ml'],
            ]
            bars2 = plt.bar(methods2, missed, edgecolor='black', linewidth=1)
            plt.ylabel('Missed High-Risk', fontsize=10)
            plt.title(f'Missed High-Risk (≥{threshold}%)', fontsize=12, fontweight='bold')
            denom = results[f'threshold_{threshold}']['treatment_eligibility']['lab'] or 1
            for bar, val in zip(bars2, missed):
                pct = val / denom * 100
                plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                         f"{int(val)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

            ax3 = plt.subplot(2, 3, base_col + 3)
            strategies = ['Universal\nLab', 'Two-Stage\n(ML+Lab)']
            lab_tests = [
                results[f'threshold_{threshold}']['twostage_workflow']['lab_tests_universal'],
                results[f'threshold_{threshold}']['twostage_workflow']['lab_tests_twostage'],
            ]
            bars3 = plt.bar(strategies, lab_tests, edgecolor='black', linewidth=1)
            plt.ylabel('Lab Tests Required', fontsize=10)
            plt.title(f'Lab Testing Volume (≥{threshold}%)', fontsize=12, fontweight='bold')
            for bar, val in zip(bars3, lab_tests):
                plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                         f"{int(val)}", ha="center", va="bottom", fontsize=9)

            red_pct = results[f'threshold_{threshold}']['twostage_workflow']['lab_reduction_pct']
            plt.text(0.5, max(lab_tests) * 0.95, f"Reduction: {red_pct:.1f}%",
                     ha='center', va='top', transform=ax3.transAxes, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('resource/images/rq5_decision_utility.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig2 = plt.figure(figsize=(16, 6))

        threshold_range = np.linspace(0.05, 0.50, 50)
        n_total = len(df_ml)

        for idx, threshold in enumerate(thresholds):
            ax = plt.subplot(1, 2, idx + 1)

            hr_lab = df_ml[f'hr_lab_{threshold}'].values
            hr_nonlab = df_ml[f'hr_nonlab_{threshold}'].values
            hr_ml = df_ml[f'hr_ml_{threshold}'].values

            nb_all_range, nb_nonlab_range, nb_ml_range = [], [], []

            n_eligible_lab = int(hr_lab.sum())

            for pt in threshold_range:
                w = pt / (1 - pt)

                tp_nonlab = int(((hr_lab == 1) & (hr_nonlab == 1)).sum())
                fp_nonlab = int(((hr_lab == 0) & (hr_nonlab == 1)).sum())
                nb_nonlab = (tp_nonlab / n_total) - (fp_nonlab / n_total) * w

                tp_ml = int(((hr_lab == 1) & (hr_ml == 1)).sum())
                fp_ml = int(((hr_lab == 0) & (hr_ml == 1)).sum())
                nb_m = (tp_ml / n_total) - (fp_ml / n_total) * w

                nb_all = (n_eligible_lab / n_total) - ((n_total - n_eligible_lab) / n_total) * w

                nb_nonlab_range.append(nb_nonlab)
                nb_ml_range.append(nb_m)
                nb_all_range.append(nb_all)

            plt.plot(threshold_range, nb_all_range, label='Treat All', linewidth=2, linestyle=':')
            plt.plot(threshold_range, nb_nonlab_range, label='Non-Lab Chart', linewidth=2)
            plt.plot(threshold_range, nb_ml_range, label='ML-Augmented', linewidth=2)
            plt.axhline(0, linewidth=1, linestyle='--', alpha=0.7, label='Treat None')

            plt.axvline(threshold / 100.0, linestyle='--', linewidth=2, alpha=0.7)

            plt.xlabel('Risk Threshold (Probability)', fontsize=11)
            plt.ylabel('Net Benefit', fontsize=11)
            plt.title(f'Decision Curve (≥{threshold}% rule)', fontsize=12, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.xlim(0.05, 0.50)
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('resource/images/rq5_net_benefit_curves.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        self.figures['rq5'] = 'resource/images/rq5_decision_utility.png'
        self.figures['rq5_nb'] = 'resource/images/rq5_net_benefit_curves.png'
        print("\n✓ RQ5 visualizations saved")

    def generate_summary_report(self):
        """Generate comprehensive summary report (robust + UTF-8 safe on Windows)"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("=" * 80)

        def get(d, path, default="NA"):
            """Safe nested dict getter: get(d, ['a','b','c'])"""
            cur = d
            for k in path:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        def fmt(x, nd=2):
            """Format numbers safely"""
            try:
                if x == "NA" or x is None:
                    return "NA"
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        report = []
        report.append("=" * 80)
        report.append("CARDIOVASCULAR RISK STRATIFICATION ANALYSIS")
        report.append("WHO 2019 Lab vs Non-Lab Charts in Bangladeshi Primary Care")
        report.append("=" * 80)
        report.append("")

        if 'rq1' in self.results:
            rq1 = self.results['rq1']
            report.append("RQ1: BASELINE RISK BURDEN & HETEROGENEITY")
            report.append("-" * 80)
            n = get(rq1, ['overall', 'count'], "NA")
            report.append(f"Sample size: {n:,.0f} adults (40-74 years)" if n != "NA" else "Sample size: NA")

            report.append(
                f"Mean risk: {fmt(get(rq1, ['overall', 'mean']), 2)}% (SD: {fmt(get(rq1, ['overall', 'std']), 2)}%)")
            report.append(f"High-risk prevalence (≥20%): {fmt(get(rq1, ['high_risk_pct']), 2)}%")
            report.append(f"Sex difference (p={fmt(get(rq1, ['sex_test', 'p_value']), 4)})")
            report.append(f"Age heterogeneity (p={fmt(get(rq1, ['age_test', 'p_value']), 4)})")
            report.append("")

        if 'rq2' in self.results:
            rq2 = self.results['rq2']
            report.append("RQ2: CONCORDANCE & SYSTEMATIC BIAS")
            report.append("-" * 80)

            n_paired = rq2.get('n_paired', rq2.get('n_paired_continuous', rq2.get('n', "NA")))
            report.append(
                f"Paired sample: {n_paired:,}" if isinstance(n_paired, (int, float)) else f"Paired sample: {n_paired}")

            report.append(f"Mean bias (non-lab − lab): {fmt(get(rq2, ['continuous_agreement', 'mean_diff']), 2)}%")
            report.append(f"Correlation: r = {fmt(get(rq2, ['continuous_agreement', 'correlation']), 3)}")

            exact_agree = get(rq2, ['categorical_agreement', 'exact_agreement'], "NA")
            if exact_agree == "NA":
                exact_agree = get(rq2, ['categorical_agreement', 'exact_agreement_pct'], "NA")
            report.append(f"Exact categorical agreement: {fmt(exact_agree, 2)}%")

            report.append(f"Cohen's kappa: κ = {fmt(get(rq2, ['categorical_agreement', 'cohen_kappa']), 3)}")
            report.append(f"High-risk (≥20%) sensitivity: {fmt(get(rq2, ['highrisk_agreement', 'sensitivity']), 3)}")
            report.append(f"High-risk (≥20%) specificity: {fmt(get(rq2, ['highrisk_agreement', 'specificity']), 3)}")
            report.append("")

        if 'rq3' in self.results:
            rq3 = self.results['rq3']
            report.append("RQ3: MISSED HIGH-RISK INDIVIDUALS")
            report.append("-" * 80)
            report.append(f"Lab-defined high-risk: {get(rq3, ['n_highrisk_lab'], 'NA')}")
            report.append(
                f"Missed by non-lab chart: {get(rq3, ['n_missed'], 'NA')} ({fmt(get(rq3, ['missed_rate']), 2)}%)")
            report.append(f"Sensitivity of non-lab chart: {fmt(get(rq3, ['sensitivity_nonlab']), 2)}%")
            report.append("")
            report.append("Key predictors of being missed:")
            preds = rq3.get('predictors', [])
            for pred in preds[:3]:
                feat = pred.get('Feature', 'NA')
                orv = pred.get('Odds Ratio', 'NA')
                report.append(f"  {feat}: OR = {fmt(orv, 2)}")
            report.append("")

        if 'rq4' in self.results:
            rq4 = self.results['rq4']
            report.append("RQ4: MACHINE LEARNING AUGMENTATION")
            report.append("-" * 80)
            best = rq4.get('best_model', 'NA')
            report.append(f"Best model: {best}")

            best_auc = get(rq4, ['models', best, 'auc'], "NA")
            base_auc = get(rq4, ['baseline', 'auc'], "NA")
            if best_auc != "NA" and base_auc != "NA":
                try:
                    delta_auc = float(best_auc) - float(base_auc)
                    report.append(f"AUC: {fmt(best_auc, 3)} (+{fmt(delta_auc, 3)} vs baseline)")
                except Exception:
                    report.append(f"AUC: {fmt(best_auc, 3)} (baseline: {fmt(base_auc, 3)})")
            else:
                report.append(f"AUC: {fmt(best_auc, 3)}")

            best_sens = get(rq4, ['models', best, 'sensitivity'], "NA")
            base_sens = get(rq4, ['baseline', 'sensitivity'], "NA")
            if best_sens != "NA" and base_sens != "NA":
                try:
                    delta_s = float(best_sens) - float(base_sens)
                    report.append(f"Sensitivity: {fmt(best_sens, 3)} (+{fmt(delta_s, 3)})")
                except Exception:
                    report.append(f"Sensitivity: {fmt(best_sens, 3)}")
            else:
                report.append(f"Sensitivity: {fmt(best_sens, 3)}")

            report.append(f"Calibration (Brier): {fmt(get(rq4, ['calibration', 'brier_calibrated']), 4)}")
            report.append(f"Site generalizability: AUC = {fmt(get(rq4, ['site_generalizability', 'mean_auc']), 3)} "
                          f"± {fmt(get(rq4, ['site_generalizability', 'std_auc']), 3)}")
            report.append("")

        if 'rq5' in self.results:
            rq5 = self.results['rq5']
            report.append("RQ5: DECISION UTILITY & TREATMENT STRATEGIES")
            report.append("-" * 80)

            for threshold in [20, 10]:
                key = f"threshold_{threshold}"
                if key in rq5:
                    th = rq5[key]
                    report.append(f"\nAt ≥{threshold}% risk threshold:")
                    report.append(
                        f"  Missed high-risk reduction: {fmt(get(th, ['missed_highrisk', 'pct_reduction']), 1)}%")
                    report.append(
                        f"  Lab test reduction (two-stage): {fmt(get(th, ['twostage_workflow', 'lab_reduction_pct']), 1)}%")
                    report.append(
                        f"  Incremental net benefit: {fmt(get(th, ['net_benefit', 'incremental_ml_vs_nonlab']), 4)}")
            report.append("")

        report.append("=" * 80)
        report.append("ANALYSIS COMPLETE")
        report.append("=" * 80)

        report_text = "\n".join(report)
        with open('resource/images/analysis_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print("Done")
        return report_text


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("CARDIOVASCULAR RISK STRATIFICATION ANALYSIS")
    print("Complete Analysis Pipeline for 5 Research Questions")
    print("=" * 80)

    print("\nLoading data...")
    try:
        df_nonlab = pd.read_csv('resource/analyzed/v2.0/cvd_nonlab.csv')
        df_lab = pd.read_csv('resource/analyzed/v2.0/cvd_lab.csv')
        df_paired = pd.read_csv('resource/analyzed/v2.0/cvd_paired.csv')
        df_who_nonlab = pd.read_csv('resource/analyzed/v2.0/cvd_who_nonlab_domain.csv')
        df_who_lab = pd.read_csv('resource/analyzed/v2.0/cvd_who_lab_domain.csv')
        df_sites = pd.read_csv('resource/reports/service_site_with_geographical_points.csv')

        print("✓ Data loaded successfully")

        analysis = CVDRiskAnalysis(df_nonlab, df_lab, df_paired, df_who_nonlab, df_who_lab, df_sites)

        print("\n" + "=" * 80)
        print("EXECUTING COMPREHENSIVE ANALYSIS")
        print("=" * 80)

        results_rq1 = analysis.rq1_baseline_distribution()

        results_rq2 = analysis.rq2_concordance_analysis()

        results_rq3 = analysis.rq3_missed_highrisk_analysis()

        results_rq4, ml_model = analysis.rq4_ml_augmentation()

        results_rq5 = analysis.rq5_decision_utility(ml_model)

        summary_report = analysis.generate_summary_report()

        import json
        def convert_types(obj):
            """Convert types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj


        results_to_save = convert_types(analysis.results)

        with open('resource/images/all_results.txt', 'w', encoding='utf-8') as f:
            f.write(str(results_to_save))

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  - rq1_baseline_distribution.png")
        print("  - rq2_concordance_analysis.png")
        print("  - rq3_missed_highrisk.png")
        print("  - rq4_ml_augmentation.png")
        print("  - rq4_confusion_matrices.png")
        print("  - rq5_decision_utility.png")
        print("  - rq5_net_benefit_curves.png")
        print("  - analysis_summary_report.txt")
        print("  - overall_output.json")

        return analysis

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find data files.")
        print(f"Please upload the following CSV files:")
        print("  - df_nonlab.csv")
        print("  - df_lab.csv")
        print("  - df_paired.csv")
        print("  - df_who_nonlab_domain.csv")
        print("  - df_who_lab_domain.csv")
        print("  - df_sites.csv")
        print(f"\nError details: {e}")
        return None


if __name__ == "__main__":
    analysis = main()
