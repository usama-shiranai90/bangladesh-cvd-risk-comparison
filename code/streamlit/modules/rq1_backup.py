import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import io
from utils.helpers import get_risk_cat, RISK_PALETTE

def render_backup_rq1(df_paired, datasets=None):
    """Render backup RQ1."""
    st.title("RQ1: WHO Non-Lab vs Lab Agreement & Safety")
    
    st.markdown("""
    **Research Question:** In paired WHO-domain data, how well does WHO non-lab agree with WHO lab (reference), 
    what is the direction/magnitude of bias (non-lab − lab), and which profiles predict missed lab-defined high-risk (≥20%) 
    accounting for site clustering?
    
    - **RQ1a:** Agreement & Classification (Kappa, Sensitivity/Specificity)
    - **RQ1b:** Bias Magnitude (Direction, Distribution by Risk Level)
    - **RQ1c:** Who Gets Missed (Safety - Site-Adjusted Regression with Location Interaction)
    """)

    if df_paired is None:
        st.error("No paired data loaded. Please ensure 'cvd_paired.csv' is selected in the sidebar.")
        return

    df = df_paired.copy()
    
    cols = ['risk_lab', 'risk_nonlab', 'age']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df.dropna(subset=['risk_lab', 'risk_nonlab'])
    
    risk_order = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
    df['risk_lab_cat'] = df['risk_lab'].apply(get_risk_cat).astype(pd.CategoricalDtype(risk_order, ordered=True))
    df['risk_nonlab_cat'] = df['risk_nonlab'].apply(get_risk_cat).astype(pd.CategoricalDtype(risk_order, ordered=True))
    
    for t in [10, 20]:
        df[f'lab_ge_{t}'] = (df['risk_lab'] >= t).astype(int)
        df[f'nonlab_ge_{t}'] = (df['risk_nonlab'] >= t).astype(int)
        df[f'missed_highrisk_{t}'] = ((df['risk_lab'] >= t) & (df['risk_nonlab'] < t)).astype(int)
    
    df['risk_diff'] = df['risk_nonlab'] - df['risk_lab']
    
    st.markdown(f"**Paired Observations:** {len(df):,}")

    tab1a, tab1b, tab1c, tab_stats, tab_summary = st.tabs([
        "📊 RQ1a: Agreement & Classification",
        "⚖️ RQ1b: Bias Magnitude & Direction",
        "⚠️ RQ1c: Who Gets Missed (Safety)",
        "📈 Statistical Deep Dive",
        "📝 Publication Summary"
    ])

    with tab1a:
        st.subheader("RQ1a: Agreement & Classification Performance")
        
        st.markdown("### 1️⃣ Categorical Agreement (5-Band Risk)")
        
        cm = pd.crosstab(df['risk_lab_cat'], df['risk_nonlab_cat'], margins=True, margins_name='Total')
        
        cat_map = {c: i for i, c in enumerate(risk_order)}
        y_true = df['risk_lab_cat'].map(cat_map).dropna()
        y_pred = df['risk_nonlab_cat'].map(cat_map).dropna()
        
        common_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_idx]
        y_pred = y_pred.loc[common_idx]
        
        kappa_weighted = cohen_kappa_score(y_true, y_pred, weights='linear')
        kappa_unweighted = cohen_kappa_score(y_true, y_pred, weights=None)
        
        col_k1, col_k2 = st.columns(2)
        col_k1.metric("Weighted Kappa (Linear)", f"{kappa_weighted:.3f}", 
                     help="Accounts for degree of disagreement")
        col_k2.metric("Unweighted Kappa", f"{kappa_unweighted:.3f}",
                     help="Exact agreement only")
        
        if kappa_weighted >= 0.81:
            st.success("✅ **Excellent Agreement** (κ ≥ 0.81)")
        elif kappa_weighted >= 0.61:
            st.info("✔️ **Substantial Agreement** (κ = 0.61-0.80)")
        elif kappa_weighted >= 0.41:
            st.warning("⚠️ **Moderate Agreement** (κ = 0.41-0.60)")
        else:
            st.error("🚨 **Poor Agreement** (κ < 0.41)")
        
        fig_hm = px.imshow(cm.iloc[:-1, :-1], text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Non-Lab Risk", y="Lab Risk (Reference)", color="Count"),
                           title="Confusion Matrix: Non-Lab vs Lab Risk")
        st.plotly_chart(fig_hm, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'confusion_matrix'}})
        
        st.caption("**Rows:** Lab Risk (Gold Standard) | **Columns:** Non-Lab Risk (Test) | **Diagonal:** Perfect Agreement")
        
        st.markdown("---")
        st.markdown("### 2️⃣ Binary Classification Performance")
        st.caption("Diagnostic accuracy at clinical thresholds for pharmacotherapy eligibility")
        
        classification_results = []
        for thresh in [20, 10]:
            lab_col = f'lab_ge_{thresh}'
            test_col = f'nonlab_ge_{thresh}'
            
            cm_binary = confusion_matrix(df[lab_col], df[test_col])
            if cm_binary.shape == (2, 2):
                tn, fp, fn, tp = cm_binary.ravel()
            else:
                tp = ((df[lab_col]==1) & (df[test_col]==1)).sum()
                fp = ((df[lab_col]==0) & (df[test_col]==1)).sum()
                fn = ((df[lab_col]==1) & (df[test_col]==0)).sum()
                tn = ((df[lab_col]==0) & (df[test_col]==0)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            missed_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            
            classification_results.append({
                'Threshold': f'≥{thresh}%',
                'TP': tp,
                'FN': fn,
                'TN': tn,
                'FP': fp,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'Missed Rate': missed_rate
            })
        
        col_20, col_10 = st.columns(2)
        
        for col, result in zip([col_20, col_10], classification_results):
            with col:
                st.markdown(f"#### {result['Threshold']} (Primary)" if '20' in result['Threshold'] else f"#### {result['Threshold']} (Sensitivity)")
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Missed Rate'],
                    'Value': [
                        f"{result['Sensitivity']:.1%}",
                        f"{result['Specificity']:.1%}",
                        f"{result['PPV']:.1%}",
                        f"{result['NPV']:.1%}",
                        f"{result['Missed Rate']:.1%}"
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                st.caption(f"""
                **Contingency:**  
                - TP (Correct High-Risk): {result['TP']}  
                - **FN (Missed Cases):** {result['FN']} 🚨  
                - TN (Correct Low-Risk): {result['TN']}  
                - FP (False Alarms): {result['FP']}
                """)
        
        primary_result = classification_results[0]
        if primary_result['Missed Rate'] > 0.20:
            st.error(f"🚨 **Safety Alert:** {primary_result['Missed Rate']:.1%} of lab-defined high-risk patients (≥20%) are missed by non-lab charts!")
        elif primary_result['Missed Rate'] > 0.10:
            st.warning(f"⚠️ **Safety Concern:** {primary_result['Missed Rate']:.1%} missed high-risk rate")
        else:
            st.success(f"✅ **Acceptable Safety:** {primary_result['Missed Rate']:.1%} missed high-risk rate")

        st.markdown("---")
        st.markdown("### 3️⃣ Agreement Tables by Gender and Age")
        st.caption("Cross-tabulation of Non-Laboratory vs Laboratory risk categories stratified by demographics")
        
        RISK_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
        RISK_DISPLAY = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
        
        def create_agreement_table(subset_df, group_label):
            """Create agreement cross-tabulation with Kappa"""
            if len(subset_df) == 0:
                return None, None
            
            ct = pd.crosstab(
                subset_df['risk_nonlab_cat'], 
                subset_df['risk_lab_cat'],
                dropna=False
            ).reindex(index=RISK_LABELS, columns=RISK_LABELS, fill_value=0)
            
            cat_map = {c: i for i, c in enumerate(RISK_LABELS)}
            y_true = subset_df['risk_lab_cat'].map(cat_map).dropna()
            y_pred = subset_df['risk_nonlab_cat'].map(cat_map).dropna()
            
            common_idx = y_true.index.intersection(y_pred.index)
            if len(common_idx) > 0:
                kappa = cohen_kappa_score(y_true.loc[common_idx], y_pred.loc[common_idx], weights='linear')
            else:
                kappa = np.nan
            
            ct['Total'] = ct.sum(axis=1)
            ct.loc['Total'] = ct.sum(axis=0)
            
            return ct, kappa
        
        st.markdown("#### Agreement by Gender")
        
        gender_tabs = st.tabs(["👨 Males", "👩 Females"])
        
        with gender_tabs[0]:
            male_df = df[df['gender'].isin(['M', 'Male', 'men'])]
            
            if len(male_df) > 0:
                if 'age_band' not in male_df.columns:
                    male_df = male_df.copy()
                    bins = [40, 45, 50, 55, 60, 65, 70, 75]
                    labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                    male_df['age_band'] = pd.cut(male_df['age'], bins=bins, labels=labels, right=False)
                
                ct_male, kappa_male = create_agreement_table(male_df, "Males")
                
                if ct_male is not None:
                    st.markdown(f"##### Overall Males (n={len(male_df):,}, κ={kappa_male:.3f})")
                    
                    ct_male_display = ct_male.copy()
                    ct_male_display.columns = RISK_DISPLAY + ['Total']
                    ct_male_display.index = RISK_DISPLAY + ['Total'] if 'Total' in ct_male.index else RISK_DISPLAY
                    
                    st.dataframe(ct_male_display, use_container_width=True)
                    
                    st.markdown("##### Males by Age Group")
                    
                    age_results_male = []
                    for age_band in male_df['age_band'].dropna().unique():
                        age_subset = male_df[male_df['age_band'] == age_band]
                        if len(age_subset) > 10:
                            _, k = create_agreement_table(age_subset, f"Males {age_band}")
                            age_results_male.append({
                                'Age Band': age_band,
                                'n': len(age_subset),
                                'Weighted Kappa': f"{k:.3f}" if not np.isnan(k) else "N/A"
                            })
                    
                    if age_results_male:
                        age_df_male = pd.DataFrame(age_results_male)
                        st.dataframe(age_df_male, use_container_width=True, hide_index=True)
            else:
                st.info("No male data available.")
        
        with gender_tabs[1]:
            female_df = df[df['gender'].isin(['F', 'Female', 'women'])]
            
            if len(female_df) > 0:
                if 'age_band' not in female_df.columns:
                    female_df = female_df.copy()
                    bins = [40, 45, 50, 55, 60, 65, 70, 75]
                    labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                    female_df['age_band'] = pd.cut(female_df['age'], bins=bins, labels=labels, right=False)
                
                ct_female, kappa_female = create_agreement_table(female_df, "Females")
                
                if ct_female is not None:
                    st.markdown(f"##### Overall Females (n={len(female_df):,}, κ={kappa_female:.3f})")
                    
                    ct_female_display = ct_female.copy()
                    ct_female_display.columns = RISK_DISPLAY + ['Total']
                    ct_female_display.index = RISK_DISPLAY + ['Total'] if 'Total' in ct_female.index else RISK_DISPLAY
                    
                    st.dataframe(ct_female_display, use_container_width=True)
                    
                    st.markdown("##### Females by Age Group")
                    
                    age_results_female = []
                    for age_band in female_df['age_band'].dropna().unique():
                        age_subset = female_df[female_df['age_band'] == age_band]
                        if len(age_subset) > 10:
                            _, k = create_agreement_table(age_subset, f"Females {age_band}")
                            age_results_female.append({
                                'Age Band': age_band,
                                'n': len(age_subset),
                                'Weighted Kappa': f"{k:.3f}" if not np.isnan(k) else "N/A"
                            })
                    
                    if age_results_female:
                        age_df_female = pd.DataFrame(age_results_female)
                        st.dataframe(age_df_female, use_container_width=True, hide_index=True)
            else:
                st.info("No female data available.")

        st.markdown("---")
        st.markdown("### 4️⃣ Clinical Performance of the WHO Non-Laboratory Model")
        st.caption("Diagnostic performance metrics with 95% confidence intervals")
        
        def wilson_ci(successes, total, z=1.96):
            """Calculate Wilson score confidence interval"""
            if total == 0:
                return 0, 0
            p = successes / total
            denom = 1 + z**2 / total
            center = (p + z**2 / (2 * total)) / denom
            margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denom
            return max(0, center - margin), min(1, center + margin)
        
        def calculate_mcc(tp, tn, fp, fn):
            """Calculate Matthews Correlation Coefficient"""
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if denominator == 0:
                return 0
            return numerator / denominator
        
        cutoff_scenarios = [
            ("20% for Lab / 20% for Non-Lab", 20, 20),
            ("20% for Lab / 10% for Non-Lab", 20, 10)
        ]
        
        performance_data = []
        
        for scenario_name, lab_thresh, nonlab_thresh in cutoff_scenarios:
            lab_positive = (df['risk_lab'] >= lab_thresh).astype(int)
            nonlab_positive = (df['risk_nonlab'] >= nonlab_thresh).astype(int)
            
            tp = ((lab_positive == 1) & (nonlab_positive == 1)).sum()
            fn = ((lab_positive == 1) & (nonlab_positive == 0)).sum()
            fp = ((lab_positive == 0) & (nonlab_positive == 1)).sum()
            tn = ((lab_positive == 0) & (nonlab_positive == 0)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            mcc = calculate_mcc(tp, tn, fp, fn)
            
            sens_low, sens_high = wilson_ci(tp, tp + fn)
            spec_low, spec_high = wilson_ci(tn, tn + fp)
            ppv_low, ppv_high = wilson_ci(tp, tp + fp)
            npv_low, npv_high = wilson_ci(tn, tn + fn)
            
            performance_data.append({
                'Cut-off Point': scenario_name,
                'Sensitivity % (95% CI)': f"{sensitivity*100:.1f} ({sens_low*100:.1f}–{sens_high*100:.1f})",
                'Specificity % (95% CI)': f"{specificity*100:.1f} ({spec_low*100:.1f}–{spec_high*100:.1f})",
                'PPV % (95% CI)': f"{ppv*100:.1f} ({ppv_low*100:.1f}–{ppv_high*100:.1f})",
                'NPV % (95% CI)': f"{npv*100:.1f} ({npv_low*100:.1f}–{npv_high*100:.1f})",
                'MCC': f"{mcc:.3f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **Interpretation:**
        - **Sensitivity:** Proportion of lab-defined high-risk correctly identified by non-lab model
        - **Specificity:** Proportion of lab-defined low-risk correctly identified by non-lab model  
        - **PPV:** Probability that a non-lab positive is truly lab-positive
        - **NPV:** Probability that a non-lab negative is truly lab-negative
        - **MCC:** Matthews Correlation Coefficient (-1 to +1, 0 = random, +1 = perfect)
        
        *Using different cut-points for Non-Lab (10%) increases sensitivity at the cost of specificity.*
        """)
        
        st.markdown("**Key Insight:**")
        if len(performance_data) >= 2:
            sens_20_20 = float(performance_data[0]['Sensitivity % (95% CI)'].split()[0])
            sens_20_10 = float(performance_data[1]['Sensitivity % (95% CI)'].split()[0])
            sens_gain = sens_20_10 - sens_20_20
            
            spec_20_20 = float(performance_data[0]['Specificity % (95% CI)'].split()[0])
            spec_20_10 = float(performance_data[1]['Specificity % (95% CI)'].split()[0])
            spec_loss = spec_20_20 - spec_20_10
            
            st.write(f"Using a 10% non-lab threshold to identify lab ≥20% patients:")
            st.write(f"- **Sensitivity gain:** +{sens_gain:.1f} percentage points")
            st.write(f"- **Specificity trade-off:** -{spec_loss:.1f} percentage points")


    with tab_stats:
        st.subheader("Statistical Deep Dive & Visualization")

        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        score_non_lab = df['risk_nonlab']
        score_lab = df['risk_lab']

        stat_n, p_n = stats.shapiro(score_non_lab - score_lab)

        if p_n > 0.05:
            test_res = stats.ttest_rel(score_non_lab, score_lab)
            test_name = "Paired t-test"
        else:
            test_res = stats.wilcoxon(score_non_lab, score_lab)
            test_name = "Wilcoxon signed-rank test"

        corr_val, corr_p = stats.spearmanr(score_non_lab, score_lab)

        
        RISK_ORDER = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']

        df['risk_nonlab_cat'] = pd.Categorical(df['risk_nonlab_cat'], categories=RISK_ORDER, ordered=True)
        df['risk_lab_cat']    = pd.Categorical(df['risk_lab_cat'],    categories=RISK_ORDER, ordered=True)

        ct = pd.crosstab(
            df['risk_nonlab_cat'],
            df['risk_lab_cat'],
            dropna=False
        ).reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)

        st.dataframe(ct)

        exact_match_pct = (np.diag(ct).sum() / ct.values.sum()) * 100

        fig = plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        sns.scatterplot(x=score_non_lab, y=score_lab, alpha=0.5)
        plt.plot([0, 30], [0, 30], 'r--', label='Identity (x=y)')
        plt.title('Score Comparison: Non-Lab vs Lab')
        plt.xlabel('Risk Score (Non-Lab)')
        plt.ylabel('Risk Score (Lab)')
        plt.legend()

        plt.subplot(2, 2, 2)
        mean_score = (score_non_lab + score_lab) / 2
        diff_score = score_lab - score_non_lab
        md = diff_score.mean()
        sd = diff_score.std()
        plt.scatter(mean_score, diff_score, alpha=0.5)
        plt.axhline(md, color='red', linestyle='-', label=f'Mean Diff: {md:.2f}')
        plt.axhline(md + 1.96*sd, color='red', linestyle='--', label='+1.96 SD')
        plt.axhline(md - 1.96*sd, color='red', linestyle='--', label='-1.96 SD')
        plt.title('Bland-Altman Plot (Bias Check)')
        plt.xlabel('Mean of Scores')
        plt.ylabel('Difference (Lab - Non-Lab)')
        plt.legend()

        plt.subplot(2, 2, 3)
        sns.kdeplot(score_non_lab, label='Non-Lab', fill=True)
        sns.kdeplot(score_lab, label='Lab', fill=True)
        plt.title('Distribution of Scores')
        plt.xlabel('Risk Score')
        plt.legend()

        plt.subplot(2, 2, 4)
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Agreement Heatmap (Match: {exact_match_pct:.1f}%)')
        plt.xlabel('Risk Range (Lab)')
        plt.ylabel('Risk Range (Non-Lab)')

        plt.tight_layout()
        st.pyplot(fig)
        
        img = io.BytesIO()
        fig.savefig(img, format='svg', bbox_inches='tight')
        img.seek(0)
        st.download_button(
            label="📥 Download Figure as SVG",
            data=img,
            file_name="rq1_statistical_deep_dive.svg",
            mime="image/svg+xml"
        )

        st.markdown(f"### Statistical Summary")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write(f"**Test Performed:** {test_name}")
            st.write(f"**P-value:** {test_res.pvalue:.4e}")
            st.write(f"**Spearman Correlation:** {corr_val:.3f}")

        with col_res2:
            st.write(f"**Mean Score (Non-Lab):** {score_non_lab.mean():.2f}")
            st.write(f"**Mean Score (Lab):** {score_lab.mean():.2f}")
            st.write(f"**Exact Range Agreement:** {exact_match_pct:.2f}%")

        range_mismatch = (df['risk_nonlab_cat'] != df['risk_lab_cat']).sum()
        total_records = len(df)
        cat_error_pct = (range_mismatch / total_records) * 100

        mape = np.mean(np.abs((df['risk_lab'] - df['risk_nonlab']) / df['risk_lab'])) * 100

        under_estimate = (df['risk_nonlab'] < df['risk_lab']).sum() / total_records * 100
        over_estimate = (df['risk_nonlab'] > df['risk_lab']).sum() / total_records * 100

        st.info(f"**Error Analysis Report**\n\n"
                f"- Total Disagreement in Risk Range: {cat_error_pct:.2f}%\n"
                f"- Average Numerical Score Error (MAPE): {mape:.2f}%\n"
                f"- Non-Lab Under-estimation Bias: {under_estimate:.2f}%\n"
                f"- Non-Lab Over-estimation Bias: {over_estimate:.2f}%")

        st.markdown("---")
        st.markdown("### 📊 Stratified Bland-Altman Plots")
        st.caption("Bland-Altman analysis showing agreement between laboratory-based and non-laboratory-based CVD risk predictions, stratified by clinical subgroups")
        
        def create_bland_altman_subplot(ax, data, title, color='#3498db'):
            """Create a single Bland-Altman plot on the given axes"""
            if len(data) < 5:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(data)})', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(title, fontweight='bold', fontsize=10)
                return
            
            mean_scores = (data['risk_nonlab'] + data['risk_lab']) / 2
            diff_scores = data['risk_lab'] - data['risk_nonlab']
            
            mean_diff = diff_scores.mean()
            std_diff = diff_scores.std()
            loa_upper = mean_diff + 1.96 * std_diff
            loa_lower = mean_diff - 1.96 * std_diff
            
            ax.scatter(mean_scores, diff_scores, alpha=0.4, s=15, c=color, edgecolors='none')
            
            ax.axhline(mean_diff, color='#e74c3c', linestyle='-', linewidth=1.5, 
                       label=f'Mean: {mean_diff:.2f}')
            
            ax.axhline(loa_upper, color='#e74c3c', linestyle='--', linewidth=1, 
                       label=f'+1.96 SD: {loa_upper:.2f}')
            ax.axhline(loa_lower, color='#e74c3c', linestyle='--', linewidth=1, 
                       label=f'-1.96 SD: {loa_lower:.2f}')
            
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            
            ax.fill_between(ax.get_xlim(), loa_lower, loa_upper, alpha=0.1, color='#e74c3c')
            
            ax.set_xlabel('Mean of Lab & Non-Lab (%)', fontsize=9)
            ax.set_ylabel('Difference (Lab - Non-Lab)', fontsize=9)
            ax.set_title(f'{title}\n(n={len(data):,}, Mean Diff={mean_diff:.2f})', fontweight='bold', fontsize=10)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        st.markdown("#### 1️⃣ By Gender")
        
        if 'gender' in df.columns:
            male_df = df[df['gender'].isin(['M', 'Male', 'men'])]
            female_df = df[df['gender'].isin(['F', 'Female', 'women'])]
            
            fig_gender_ba, axes_gender = plt.subplots(1, 2, figsize=(12, 5))
            
            create_bland_altman_subplot(axes_gender[0], male_df, 'Males', color='#3498db')
            create_bland_altman_subplot(axes_gender[1], female_df, 'Females', color='#e74c3c')
            
            fig_gender_ba.suptitle('Bland-Altman Plots Stratified by Gender', fontweight='bold', fontsize=12, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_gender_ba)
            
            img_gender = io.BytesIO()
            fig_gender_ba.savefig(img_gender, format='svg', bbox_inches='tight')
            img_gender.seek(0)
            st.download_button(
                label="📥 Download Gender-Stratified BA Plot (SVG)",
                data=img_gender,
                file_name="bland_altman_by_gender.svg",
                mime="image/svg+xml",
                key="ba_gender_svg"
            )
            plt.close(fig_gender_ba)
            
            gender_stats = []
            for name, subset in [('Male', male_df), ('Female', female_df)]:
                if len(subset) > 0:
                    diff = subset['risk_lab'] - subset['risk_nonlab']
                    gender_stats.append({
                        'Gender': name,
                        'n': len(subset),
                        'Mean Diff': f"{diff.mean():.2f}",
                        'SD': f"{diff.std():.2f}",
                        'LoA Lower': f"{diff.mean() - 1.96*diff.std():.2f}",
                        'LoA Upper': f"{diff.mean() + 1.96*diff.std():.2f}"
                    })
            
            if gender_stats:
                st.dataframe(pd.DataFrame(gender_stats), use_container_width=True, hide_index=True)
        else:
            st.info("Gender data not available for stratification.")
        
        st.markdown("#### 2️⃣ By Diabetes Status")
        
        diabetes_col = None
        for col in ['has_diabetes', 'diabetes', 'diab_group']:
            if col in df.columns:
                diabetes_col = col
                break
        
        if diabetes_col:
            df_diab = df.copy()
            
            if diabetes_col == 'diab_group':
                diab_df = df_diab[df_diab['diab_group'] == 'with_diabetes']
                non_diab_df = df_diab[df_diab['diab_group'] == 'no_diabetes']
            else:
                diab_df = df_diab[df_diab[diabetes_col].isin([True, 'Yes', 1, '1'])]
                non_diab_df = df_diab[df_diab[diabetes_col].isin([False, 'No', 0, '0'])]
            
            fig_diab_ba, axes_diab = plt.subplots(1, 2, figsize=(12, 5))
            
            create_bland_altman_subplot(axes_diab[0], non_diab_df, 'Without Diabetes', color='#27ae60')
            create_bland_altman_subplot(axes_diab[1], diab_df, 'With Diabetes', color='#8e44ad')
            
            fig_diab_ba.suptitle('Bland-Altman Plots Stratified by Diabetes Status', fontweight='bold', fontsize=12, y=1.02)
            plt.tight_layout()
            st.pyplot(fig_diab_ba)
            
            img_diab = io.BytesIO()
            fig_diab_ba.savefig(img_diab, format='svg', bbox_inches='tight')
            img_diab.seek(0)
            st.download_button(
                label="📥 Download Diabetes-Stratified BA Plot (SVG)",
                data=img_diab,
                file_name="bland_altman_by_diabetes.svg",
                mime="image/svg+xml",
                key="ba_diabetes_svg"
            )
            plt.close(fig_diab_ba)
            
            diab_stats = []
            for name, subset in [('Without Diabetes', non_diab_df), ('With Diabetes', diab_df)]:
                if len(subset) > 0:
                    diff = subset['risk_lab'] - subset['risk_nonlab']
                    diab_stats.append({
                        'Diabetes Status': name,
                        'n': len(subset),
                        'Mean Diff': f"{diff.mean():.2f}",
                        'SD': f"{diff.std():.2f}",
                        'LoA Lower': f"{diff.mean() - 1.96*diff.std():.2f}",
                        'LoA Upper': f"{diff.mean() + 1.96*diff.std():.2f}"
                    })
            
            if diab_stats:
                st.dataframe(pd.DataFrame(diab_stats), use_container_width=True, hide_index=True)
        else:
            st.info("Diabetes status data not available for stratification.")
        
        st.markdown("#### 3️⃣ By Systolic Blood Pressure Category")
        
        if 'sbp' in df.columns:
            df_sbp = df.dropna(subset=['sbp']).copy()
            
            def categorize_sbp(sbp):
                """Categorize sbp."""
                if sbp < 120:
                    return 'Normal (<120)'
                elif sbp < 130:
                    return 'Elevated (120-129)'
                elif sbp < 140:
                    return 'HTN Stage 1 (130-139)'
                else:
                    return 'HTN Stage 2 (≥140)'
            
            df_sbp['sbp_category'] = df_sbp['sbp'].apply(categorize_sbp)
            
            sbp_categories = ['Normal (<120)', 'Elevated (120-129)', 'HTN Stage 1 (130-139)', 'HTN Stage 2 (≥140)']
            sbp_colors = ['#27ae60', '#f39c12', '#e67e22', '#c0392b']
            
            fig_sbp_ba, axes_sbp = plt.subplots(2, 2, figsize=(12, 10))
            axes_sbp = axes_sbp.flatten()
            
            for i, (cat, color) in enumerate(zip(sbp_categories, sbp_colors)):
                subset = df_sbp[df_sbp['sbp_category'] == cat]
                create_bland_altman_subplot(axes_sbp[i], subset, cat, color=color)
            
            fig_sbp_ba.suptitle('Bland-Altman Plots Stratified by Systolic Blood Pressure', fontweight='bold', fontsize=12, y=1.01)
            plt.tight_layout()
            st.pyplot(fig_sbp_ba)
            
            img_sbp = io.BytesIO()
            fig_sbp_ba.savefig(img_sbp, format='svg', bbox_inches='tight')
            img_sbp.seek(0)
            st.download_button(
                label="📥 Download SBP-Stratified BA Plot (SVG)",
                data=img_sbp,
                file_name="bland_altman_by_sbp.svg",
                mime="image/svg+xml",
                key="ba_sbp_svg"
            )
            plt.close(fig_sbp_ba)
            
            sbp_stats = []
            for cat in sbp_categories:
                subset = df_sbp[df_sbp['sbp_category'] == cat]
                if len(subset) > 0:
                    diff = subset['risk_lab'] - subset['risk_nonlab']
                    sbp_stats.append({
                        'SBP Category': cat,
                        'n': len(subset),
                        'Mean Diff': f"{diff.mean():.2f}",
                        'SD': f"{diff.std():.2f}",
                        'LoA Lower': f"{diff.mean() - 1.96*diff.std():.2f}",
                        'LoA Upper': f"{diff.mean() + 1.96*diff.std():.2f}"
                    })
            
            if sbp_stats:
                st.dataframe(pd.DataFrame(sbp_stats), use_container_width=True, hide_index=True)
            
            if len(sbp_stats) >= 2:
                diffs = [float(s['Mean Diff']) for s in sbp_stats]
                if max(diffs) - min(diffs) > 2:
                    st.warning(f"""
                    ⚠️ **Clinical Insight:** The mean difference between Lab and Non-Lab risk estimates 
                    varies substantially across SBP categories (range: {min(diffs):.2f} to {max(diffs):.2f}). 
                    This suggests differential calibration of the non-laboratory model across blood pressure strata.
                    """)
                else:
                    st.success(f"""
                    ✅ **Clinical Insight:** The mean difference between Lab and Non-Lab risk estimates 
                    is relatively consistent across SBP categories, suggesting stable calibration.
                    """)
        else:
            st.info("Systolic blood pressure (sbp) data not available for stratification.")
        
        st.markdown("---")
        st.markdown("### 📝 Interpretation Guide")
        st.info("""
        **Bland-Altman Plot Interpretation:**
        
        - **Mean Difference (red solid line):** Represents systematic bias. Values > 0 indicate Lab estimates are higher than Non-Lab.
        - **Limits of Agreement (red dashed lines):** 95% of differences fall within these bounds (Mean ± 1.96×SD).
        - **Narrow LoA:** Indicates good agreement between methods.
        - **Wide LoA:** Indicates poor agreement with large individual-level variability.
        - **Proportional Bias:** If scatter shows a funnel pattern (wider at higher means), bias depends on risk level.
        - **Stratification Value:** Differences in patterns across subgroups reveal where non-laboratory charts may need recalibration.
        """)


    with tab1b:
        st.subheader("RQ1b: Bias Magnitude & Direction")
        st.caption("**Bias Definition:** risk_nonlab − risk_lab (negative = underestimation)")
        
        st.markdown("### 1️⃣ Overall Bias Distribution")
        
        mean_bias = df['risk_diff'].mean()
        median_bias = df['risk_diff'].median()
        std_bias = df['risk_diff'].std()
        underestimation_rate = (df['risk_diff'] < 0).mean()
        
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Mean Bias", f"{mean_bias:.2f} pp", 
                     delta=f"{'Under' if mean_bias < 0 else 'Over'}estimation",
                     delta_color="inverse" if mean_bias < 0 else "normal")
        col_b2.metric("Median Bias", f"{median_bias:.2f} pp")
        col_b3.metric("SD of Bias", f"{std_bias:.2f} pp")
        col_b4.metric("Underestimation Rate", f"{underestimation_rate:.1%}",
                     help="% of cases where non-lab < lab")
        
        fig_hist = px.histogram(df, x='risk_diff', nbins=50, 
                                title="Distribution of Risk Difference (Non-Lab − Lab)",
                                color_discrete_sequence=['#457b9d'])
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                          annotation_text="Zero Bias", annotation_position="top")
        fig_hist.add_vline(x=mean_bias, line_dash="dot", line_color="black", 
                          annotation_text=f"Mean: {mean_bias:.2f}", annotation_position="bottom right")
        fig_hist.update_xaxes(title="Risk Difference (percentage points)")
        fig_hist.update_yaxes(title="Count")
        st.plotly_chart(fig_hist, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'bias_histogram'}})
        
        st.markdown("---")
        st.markdown("### 2️⃣ Bias Stratified by True Risk Level")
        st.caption("Does bias magnitude/direction vary by lab-based risk band?")
        
        bias_by_band = df.groupby('risk_lab_cat')['risk_diff'].agg([
            ('N', 'count'),
            ('Mean Bias', 'mean'),
            ('Median Bias', 'median'),
            ('SD', 'std'),
            ('% Underestimated', lambda x: (x < 0).mean() * 100)
        ]).reset_index()
        
        st.dataframe(
            bias_by_band.style.format({
                'Mean Bias': '{:.2f}',
                'Median Bias': '{:.2f}',
                'SD': '{:.2f}',
                '% Underestimated': '{:.1f}%'
            }).background_gradient(cmap='RdYlGn_r', subset=['Mean Bias'], vmin=-10, vmax=10),
            use_container_width=True
        )
        
        fig_box = px.box(df, x='risk_lab_cat', y='risk_diff', color='risk_lab_cat',
                         color_discrete_map=RISK_PALETTE,
                         title="Bias Distribution by WHO Lab Risk Band",
                         labels={'risk_lab_cat': 'Lab Risk Band', 'risk_diff': 'Bias (Non-Lab − Lab, pp)'})
        fig_box.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Zero Bias", annotation_position="left")
        st.plotly_chart(fig_box, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'bias_boxplot'}})
        
        high_risk_bias = bias_by_band[bias_by_band['risk_lab_cat'].isin(['20% to <30%', '≥30%'])]
        if not high_risk_bias.empty:
            mean_high_bias = high_risk_bias['Mean Bias'].mean()
            if mean_high_bias < -3:
                st.error(f"🚨 **Systematic Underestimation in High-Risk Bands:** Mean bias = {mean_high_bias:.2f} pp. This is a clinical safety concern.")
            elif mean_high_bias < -1:
                st.warning(f"⚠️ **Mild Underestimation in High-Risk Bands:** Mean bias = {mean_high_bias:.2f} pp")
            elif abs(mean_high_bias) <= 1:
                st.success(f"✅ **Well-Calibrated in High-Risk Bands:** Mean bias = {mean_high_bias:.2f} pp")
            else:
                st.info(f"ℹ️ **Overestimation in High-Risk Bands:** Mean bias = {mean_high_bias:.2f} pp (conservative)")

    with tab1c:
        st.subheader("RQ1c: Predictors of Missed High-Risk Cases")
        
        st.markdown("""
        **Clinical Question:** Which patient profiles are systematically missed by WHO non-lab charts?
        
        **Outcome:** `missed_highrisk_20 = 1` if Lab Risk ≥20% AND Non-Lab Risk <20%  
        **Model:** Site-Clustered Logistic Regression with Location Interaction
        """)
        
        safe_thresh = st.radio("Analysis Threshold", [20, 10], 
                              format_func=lambda x: f"≥{x}% (Primary)" if x==20 else f"≥{x}% (Sensitivity)",
                              horizontal=True,
                              key="safety_thresh")
        
        missed_col = f'missed_highrisk_{safe_thresh}'
        n_missed = df[missed_col].sum()
        n_total_highrisk = df[f'lab_ge_{safe_thresh}'].sum()
        missed_rate = n_missed / n_total_highrisk if n_total_highrisk > 0 else 0
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Missed Cases", n_missed)
        col_s2.metric("Total Lab High-Risk", n_total_highrisk)
        col_s3.metric("Missed Rate", f"{missed_rate:.1%}")
        
        if n_missed < 10:
            st.warning(f"⚠️ **Low Event Count:** Only {n_missed} missed cases. Consider using ≥10% threshold for stable regression.")
        
        st.markdown("---")
        st.markdown("### Regression Analysis")
        
        base_predictors = ["age", "gender"]
        
        location_var = None
        for var in ['urban_rural', 'location_type', 'site_type']:
            if var in df.columns and df[var].notna().sum() > 0:
                location_var = var
                break
        
        clinical_candidates = ['bmi', 'sbp', 'dbp', 'diabetes', 'smoker', 'waist']
        available_clinical = [f for f in clinical_candidates if f in df.columns and df[f].notna().sum() > 0]
        
        st.markdown("**Configure Predictors:**")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            include_clinical = st.multiselect(
                "Clinical Factors",
                options=available_clinical,
                default=available_clinical[:3] if len(available_clinical) >= 3 else available_clinical,
                help="Select additional clinical predictors"
            )
        
        with col_c2:
            include_interaction = False
            if location_var:
                include_interaction = st.checkbox(
                    f"Include Location Interaction ({location_var} × Age)",
                    value=True,
                    help="Tests if location modifies the effect of age on being missed"
                )
        
        predictors = base_predictors + include_clinical
        if location_var and location_var not in predictors:
            predictors.append(location_var)
        
        if include_interaction and location_var:
            formula = f"{missed_col} ~ {' + '.join(predictors)} + age:{location_var}"
        else:
            formula = f"{missed_col} ~ {' + '.join(predictors)}"
        
        st.code(f"Model: {formula}", language="r")
        
        if st.button("🚀 Run Site-Adjusted Regression", type="primary"):
            try:
                req_cols = [missed_col] + predictors
                if 'site_id' in df.columns:
                    req_cols.append('site_id')
                
                m_df = df[req_cols].dropna().copy()
                
                if location_var and location_var in m_df.columns:
                    m_df[location_var] = m_df[location_var].astype(str)
                if 'gender' in m_df.columns:
                    m_df['gender'] = m_df['gender'].astype(str)
                
                if len(m_df) < 50:
                    st.error(f"Insufficient data after dropping missing values (n={len(m_df)})")
                    st.stop()
                
                model = smf.glm(formula=formula, data=m_df, family=sm.families.Binomial())
                
                if 'site_id' in m_df.columns and m_df['site_id'].nunique() > 1:
                    res = model.fit(cov_type='cluster', cov_kwds={'groups': m_df['site_id']})
                    se_type = "Cluster-Robust (by site_id)"
                else:
                    res = model.fit()
                    se_type = "Standard"
                
                st.success(f"✅ Model Converged (N={len(m_df)}, SE: {se_type})")
                
                st.markdown("### Regression Results")
                
                params = res.params
                conf = res.conf_int()
                conf.columns = ['CI_Lower', 'CI_Upper']
                
                results_df = pd.DataFrame({
                    'Coefficient': params,
                    'OR': np.exp(params),
                    'OR 95% CI Lower': np.exp(conf['CI_Lower']),
                    'OR 95% CI Upper': np.exp(conf['CI_Upper']),
                    'P-value': res.pvalues,
                    'Significant': res.pvalues < 0.05
                })
                
                st.dataframe(
                    results_df.style.format({
                        'Coefficient': '{:.3f}',
                        'OR': '{:.3f}',
                        'OR 95% CI Lower': '{:.3f}',
                        'OR 95% CI Upper': '{:.3f}',
                        'P-value': '{:.4f}'
                    }).apply(lambda x: ['background-color: #ffe6e6' if v < 0.05 else '' 
                                       for v in x], subset=['P-value']),
                    use_container_width=True
                )
                
                st.caption("""
                **Interpretation:**  
                - OR > 1: Higher odds of being MISSED by non-lab charts  
                - OR < 1: Lower odds of being missed (protective factor)  
                - P < 0.05: Statistically significant (highlighted in pink)
                """)
                
                st.markdown("---")
                st.markdown("### Key Findings")
                
                sig_predictors = results_df[results_df['Significant']].sort_values('P-value')
                
                if len(sig_predictors) > 0:
                    st.markdown("**Significant Risk Factors for Being Missed:**")
                    for idx, row in sig_predictors.iterrows():
                        direction = "increases" if row['OR'] > 1 else "decreases"
                        magnitude = abs(row['OR'] - 1) * 100
                        st.write(f"- **{idx}:** OR = {row['OR']:.2f} (95% CI: {row['OR 95% CI Lower']:.2f}-{row['OR 95% CI Upper']:.2f}, p = {row['P-value']:.4f})")
                        if ':' in idx:
                            st.caption(f"  → Interaction effect detected")
                        elif row['OR'] > 1:
                            st.caption(f"  → Each unit increase {direction} odds of being missed by {magnitude:.0f}%")
                        else:
                            st.caption(f"  → Protective factor: {direction} odds of being missed")
                else:
                    st.info("ℹ️ No predictors reached statistical significance at p < 0.05")
                
                with st.expander("📊 Full Model Summary"):
                    st.text(res.summary())
                
            except Exception as e:
                st.error(f"❌ Model fitting failed: {e}")
                st.caption("Try removing some predictors or checking for multicollinearity")

    with tab_summary:
        st.subheader("📝 Publication-Ready Summary")
        st.caption("Copy-paste text for your Results section")
        
        summary_text = f"""
## RQ1: WHO Non-Lab vs Lab Agreement & Safety (N={len(df):,} paired observations)

### RQ1a: Agreement & Classification

**Categorical Agreement:**  
Weighted kappa between WHO non-laboratory and laboratory risk categories was {kappa_weighted:.3f} 
(unweighted κ = {kappa_unweighted:.3f}), indicating {"excellent" if kappa_weighted >= 0.81 else "substantial" if kappa_weighted >= 0.61 else "moderate" if kappa_weighted >= 0.41 else "poor"} agreement.

**Binary Classification at ≥20% Threshold (Primary):**  
The non-laboratory algorithm demonstrated sensitivity of {classification_results[0]['Sensitivity']:.1%} and 
specificity of {classification_results[0]['Specificity']:.1%} for detecting lab-defined high-risk (≥20%). 
Of {classification_results[0]['TP'] + classification_results[0]['FN']} patients with lab risk ≥20%, 
{classification_results[0]['FN']} ({classification_results[0]['Missed Rate']:.1%}) were missed (false negatives). 
PPV was {classification_results[0]['PPV']:.1%} and NPV was {classification_results[0]['NPV']:.1%}.

**Sensitivity Analysis at ≥10% Threshold:**  
At the ≥10% threshold, sensitivity was {classification_results[1]['Sensitivity']:.1%}, specificity {classification_results[1]['Specificity']:.1%}, 
and missed rate {classification_results[1]['Missed Rate']:.1%} ({classification_results[1]['FN']}/{classification_results[1]['TP'] + classification_results[1]['FN']} lab high-risk patients).

### RQ1b: Bias Magnitude & Direction

**Overall Bias:**  
Mean bias (non-lab − lab) was {mean_bias:.2f} percentage points (pp) (median: {median_bias:.2f} pp, SD: {std_bias:.2f} pp). 
{"Negative values indicate systematic underestimation by non-lab charts." if mean_bias < 0 else "Positive values indicate systematic overestimation by non-lab charts."}
Overall, {underestimation_rate:.1%} of patients had non-lab risk lower than lab risk.

**Bias Stratification by Risk Level:**  
{bias_by_band.to_string(index=False)}


### RQ1c: Predictors of Missed High-Risk Cases

**Outcome Definition:**  
Missed high-risk was defined as lab risk ≥{safe_thresh}% AND non-lab risk <{safe_thresh}% (n={n_missed}/{n_total_highrisk}, {missed_rate:.1%}).

**Statistical Model:**  
Site-clustered logistic regression was employed to account for within-site correlation. {f"Interaction between location type and age was {'' if include_interaction else 'not '}included." if location_var else "No location variable was available."}

**[RESULTS WILL DEPEND ON REGRESSION OUTPUT - Run analysis to see significant predictors]**

---

## Methods Section Text

**Statistical Analysis:**  
Agreement between WHO laboratory and non-laboratory cardiovascular risk scores was assessed using weighted kappa 
(linear weights) for categorical risk bands and diagnostic test metrics (sensitivity, specificity, PPV, NPV) at 
clinically relevant thresholds (≥20% for primary analysis, ≥10% for sensitivity analysis). Bias was quantified as 
the mean difference (non-lab − lab) with distributions examined overall and stratified by laboratory risk band. 

To identify patient characteristics associated with being missed by non-laboratory charts (false negatives), we fit 
logistic regression models with cluster-robust standard errors to account for within-site correlation. The outcome 
was defined as lab risk ≥20% AND non-lab risk <20%. Predictors included age, sex, {", ".join(include_clinical) if include_clinical else "and clinical factors (BMI, blood pressure, diabetes, smoking)"}. 
{"Location-by-age interaction was tested to assess modification of risk by geographic setting." if include_interaction and location_var else ""}

All analyses were conducted in Python using statsmodels (v0.14+) and scikit-learn (v1.3+). Statistical significance 
was set at α=0.05 (two-sided).

---

## Exact Definitions

**Bias:** risk_nonlab − risk_lab (percentage points)  
- Negative = Underestimation by non-lab  
- Positive = Overestimation by non-lab

**Missed High-Risk:** Lab Risk ≥{safe_thresh}% AND Non-Lab Risk <{safe_thresh}%

**Categorical Risk Bands:** <5%, 5-<10%, 10-<20%, 20-<30%, ≥30% (WHO 2019)

**Kappa Interpretation (Landis & Koch):**  
- <0.00: Poor  
- 0.00-0.20: Slight  
- 0.21-0.40: Fair  
- 0.41-0.60: Moderate  
- 0.61-0.80: Substantial  
- 0.81-1.00: Excellent
"""
        
        st.code(summary_text, language='markdown')
        
        st.download_button(
            label="📥 Download Summary as Markdown",
            data=summary_text,
            file_name="rq1_publication_summary.md",
            mime="text/markdown"
        )
