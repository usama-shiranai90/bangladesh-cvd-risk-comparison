import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import plotly.express as px  # Removed: Converted to Matplotlib
# import plotly.graph_objects as go  # Removed: Converted to Matplotlib
import scipy.stats as stats
from sklearn.metrics import cohen_kappa_score
from utils.helpers import get_risk_cat_4band, RISK_PALETTE
from utils.stats_summary import generate_html_table1
from utils.export_utils import add_download_button
from utils.risk_engines import add_all_risk_scores, compute_discordance_matrix

# Matplotlib/Seaborn setup (for consistency across modules)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import seaborn as sns
import io

# Import SciencePlots helpers for consistent styling
from utils.scienceplots_helpers import (
    setup_scienceplots_style,
    NATURE_COLORS, RISK_COLORS, RISK_LABELS, AGREEMENT_COLORS, COMPARISON_COLORS,
    get_figure_and_ax, apply_nature_style,
    create_grouped_bar_chart, create_line_chart, create_stacked_bar_chart,
    create_heatmap, create_confusion_matrix, create_pie_donut,
    create_sankey_alluvial,
    add_svg_download_button, add_png_download_button, add_dual_download_buttons
)

# Initialize SciencePlots styling globally
try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
    sns.set_theme(style="whitegrid", context="talk")
    SCIENCEPLOTS_AVAILABLE = True
except Exception:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    SCIENCEPLOTS_AVAILABLE = False

def calculate_wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion"""
    if n == 0: return 0.0, 0.0
    z = 1.96 # Approx for 95%
    p = k / n
    denom = 1 + z**2/n
    term1 = p + z**2/(2*n)
    term2 = z * np.sqrt((p*(1-p) + z**2/(4*n))/n)
    lower = (term1 - term2) / denom
    upper = (term1 + term2) / denom
    return lower * 100, upper * 100

def categorize_glycaemic_status(row):
    """
    Categorize glycaemic status based on diabetes diagnosis and blood glucose levels.
    
    Returns:
        str: 'Diabetes', 'IFG' (Impaired Fasting Glucose), 'Normoglycaemia', or 'Unknown'
    
    WHO Thresholds (mg/dL):
    - Fasting: Normoglycaemia <110, IFG 110-125, Diabetes ≥126
    - Random/Post-Prandial: Normoglycaemia <140, IGT 140-199, Diabetes ≥200
    """
    # Check diabetes status first
    if row.get('has_diabetes') == True or row.get('has_diabetes') == 'Yes' or row.get('diab_group') == 'with_diabetes':
        return 'Diabetes'
    
    bg = row.get('bg_mgdl')
    bstype = row.get('bstype')
    
    if pd.isna(bg):
        if row.get('diab_group') == 'no_diabetes':
            return 'Normoglycaemia'
        return 'Unknown'
    
    if bstype == 'fbs':
        if bg >= 126: return 'Diabetes'
        elif bg >= 110: return 'IFG'
        else: return 'Normoglycaemia'
    else:  # pbs or others
        if bg >= 200: return 'Diabetes'
        elif bg >= 140: return 'IFG'  # Labeling IGT as IFG for consistency
        else: return 'Normoglycaemia'


def calculate_nri(y_true, p_new, p_ref, threshold=0.20):
    """
    Calculate Net Reclassification Improvement (NRI) for event/non-event.
    y_true: binary outcome (1=High Risk, 0=Low Risk)
    p_new: probabilities/risk score of new model (Lab) - normalized to 0-1 if threshold is 0-1
    p_ref: probabilities/risk score of reference model (Non-Lab)
    threshold: risk threshold (e.g. 0.20 for 20%)
    
    If inputs are 0-100, threshold should be 20.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    p_new = np.array(p_new)
    p_ref = np.array(p_ref)
    
    # Events (High Risk by "True" Lab Standard) - wait, y_true is the ground truth.
    # Here we are comparing Lab vs Non-Lab classification. 
    # Usually NRI compares Model A vs Model B against an OUTCOME (Event).
    # Since we don't have prospective events, we treat "Lab High Risk" as the Gold Standard Outcome?
    # Or we just compare concordance.
    # Reclassification Table approach:
    # We want to see if Non-Lab -> Lab moves people to appropriate categories.
    # If Lab is the "Truth", then y_true = (Lab >= Threshold).
    # And we compare Non-Lab Prediction vs Random? No.
    # We compare Non-Lab Classification vs Lab Classification.
    # But usually NRI is (P(up|event) - P(down|event)) + (P(down|nonevent) - P(up|nonevent)).
    # Here "Event" = Lab High Risk. "Non-Event" = Lab Low Risk.
    # "Model" = Non-Lab Score.
    # But checking if Non-Lab classifies correctly involves just sensitivity/specificity.
    # Standard NRI might be checking if adding a feature improves prediction.
    # Here we are comparing two METHODS.
    # Let's calculate "Categorical NRI" treating Lab as Gold Standard.
    
    event_idx = (y_true == 1)
    nonevent_idx = (y_true == 0)
    
    # Risk Categories based on threshold
    # If p_new > p_ref and we are in event, that's "Up". 
    # Categorical NRI: did the category improve?
    
    # Let's simplify: 
    # We classify using Ref (Non-Lab) and New (Lab? - No, Lab is Truth).
    # Actually, we might want to compare "WHO Non-Lab" vs "ML Model" against Lab Truth.
    # But here in the paired tab, it's just Lab vs Non-Lab.
    # Effectively we are asking: "How much reclassification happens?"
    # We can just report the Reclassification Table (which we do).
    # But the user specifically asked for NRI.
    # Let's assume they want NRI for the ML model (RQ3) or just the discordance in RQ0.
    # If RQ0: Lab is Gold Standard. Non-Lab is Test.
    # There is no "New Model" vs "Old Model". It's "Test" vs "Reference".
    # Typically NRI is P(Risk_Test > Threshold | Event) - ...
    # Let's stick to the reclassification statistics we have, but if forced to calculate NRI:
    # It usually implies comparing two PREDICTORS against an outcome.
    # If Lab Risk is the OUTCOME, then we are comparing Non-Lab vs something else?
    # Or maybe we quantify the reclassification of Non-Lab relative to Lab.
    # Let's define: Event = Lab High Risk.
    # Evaluation: Does Non-Lab score "move" in the right direction?
    # This might be ill-posed for Method Comparison.
    # Better to calculate Kappa (Agreement) and Sensitivity/Specificity.
    # However, if we treat Non-Lab as the "Baseline Risk", and we want to see if "Lab" methodology 
    # improves it... but Lab is the truth.
    # Let's compute Kappa as requested.
    # And for NRI, maybe it's for RQ3 (ML vs baseline).
    # BUT the user asked for NRI in general.
    # I will add the function. It's useful for RQ3.
    pass

    events = np.sum(y_true)
    non_events = len(y_true) - events
    
    if events == 0 or non_events == 0:
        return np.nan

    # Reclassification statistics
    # Up: P_new > P_old (or category up)
    # Correct reclassification:
    # Event: New > Old
    # Non-Event: New < Old
    
    # Categorical NRI using threshold
    cat_ref = (p_ref >= threshold).astype(int)
    cat_new = (p_new >= threshold).astype(int)
    
    # For Events (y=1): We want New=1 (High)
    n_up_event = np.sum((cat_new == 1) & (cat_ref == 0) & event_idx)
    n_down_event = np.sum((cat_new == 0) & (cat_ref == 1) & event_idx)
    
    # For Non-Events (y=0): We want New=0 (Low)
    n_down_nonevent = np.sum((cat_new == 0) & (cat_ref == 1) & nonevent_idx)
    n_up_nonevent = np.sum((cat_new == 1) & (cat_ref == 0) & nonevent_idx)
    
    nri_e = (n_up_event - n_down_event) / events
    nri_ne = (n_down_nonevent - n_up_nonevent) / non_events
    nri = nri_e + nri_ne
    
    return nri, nri_e, nri_ne

def calculate_meta_analysis(sites_df):
    """
    DerSimonian-Laird Random Effects Meta-Analysis.
    Input df must have 'Prev' (0-100) and 'N'.
    """
    if sites_df.empty: return None

    # Calculate proportions (p) and SE
    p = sites_df['Prev'] / 100.0
    n = sites_df['N']
    se2 = p * (1 - p) / n
    se2 = se2.replace(0, 1e-6) # Avoid div by zero if p=0 or p=1

    # Fixed Effects Weights
    w_fe = 1.0 / se2
    sum_w = w_fe.sum()
    mu_fe = (w_fe * p).sum() / sum_w

    # Q statistic
    Q = (w_fe * (p - mu_fe)**2).sum()
    df = len(sites_df) - 1

    # Tau2 (Moment Estimator)
    denom = sum_w - (w_fe**2).sum() / sum_w
    tau2 = 0
    if denom > 0:
        tau2 = max(0, (Q - df) / denom)

    # I2
    I2 = max(0, (Q - df) / Q) * 100 if Q > df else 0

    # Random Effects Weights
    w_re = 1.0 / (se2 + tau2)
    mu_re = (w_re * p).sum() / w_re.sum()
    se_re = 1.0 / np.sqrt(w_re.sum())

    # 95% CI for pooled estimate
    lower_re = mu_re - 1.96 * se_re
    upper_re = mu_re + 1.96 * se_re

    return {
        'pooled_prev': mu_re * 100,
        'ci_lower': max(0, lower_re) * 100,
        'ci_upper': min(100, upper_re) * 100,
        'I2': I2,
        'tau2': tau2,
        'Q': Q,
        'p_val_Q': 1 - stats.chi2.cdf(Q, df),
        'weights': w_re
    }

def calculate_standardized_prevalence(df_cohort, site_col, age_col='age_band', risk_col='high_risk'):
    """
    Calculate age-standardized prevalence per site using direct standardization.
    Reference population: Total cohort age distribution.
    """
    # 1. Reference Weights (Global Age Dist)
    ref_counts = df_cohort[age_col].value_counts(normalize=True)

    results = []

    # 2. Per Site
    for site, group in df_cohort.groupby(site_col):
        # Age-specific prevalence in this site
        # We need to ensure we have rates for all age bands present in ref
        # Group by age band within site
        site_age_counts = group.groupby(age_col, observed=False)[risk_col].agg(['sum', 'count'])

        # Calculate rates (avoid div/0)
        rates = (site_age_counts['sum'] / site_age_counts['count']).fillna(0)

        # Standardize: Sum(Rate_i * Weight_i)
        # Only use bands that exist in both (though ref should have all)
        common_bands = ref_counts.index.intersection(rates.index)

        # If a site is missing an entire age band, we assume rate is 0.
        # Here we assume 0 for missing bands effectively if we just join, but better to be explicit.
        # Actually, if a site has 0 people in age 40-45, we can't estimate rate.
        # Simple strategy: weighted sum of available bands, re-normalized weights.

        std_rate = 0
        total_weight = 0

        for band in common_bands:
            if site_age_counts.loc[band, 'count'] > 0:
                w = ref_counts[band]
                r = rates[band]
                std_rate += r * w
                total_weight += w

        if total_weight > 0:
            final_std_rate = std_rate / total_weight
        else:
            final_std_rate = np.nan

        # Crude Rate
        crude = group[risk_col].mean()

        results.append({
            site_col: site,
            'Crude (%)': crude * 100,
            'Age-Std (%)': final_std_rate * 100,
            'N': len(group)
        })

    return pd.DataFrame(results)

def calculate_empirical_bayes_shrinkage(sites_df):
    """
    Beta-Binomial Empirical Bayes Shrinkage for prevalence.
    Shrinks small sites towards the global mean.
    Requires 'Cases' (k) and 'N' (n) in sites_df.
    """
    if sites_df.empty: return sites_df

    # Method of Moments to estimate hyperparameters alpha, beta of the Beta distribution
    # Mean and Variance of the observed proportions
    p = sites_df['Cases'] / sites_df['N']
    m = p.mean()
    v = p.var()
    if v == 0: v = 1e-6 # Avoid div zero

    # Common estimation for alpha/beta from mean/var
    # m = a / (a+b)
    # v = (a*b) / ((a+b)^2 * (a+b+1))
    # -> a = m * ((m*(1-m)/v) - 1)
    # -> b = (1-m) * ((m*(1-m)/v) - 1)

    common_factor = (m * (1 - m) / v) - 1
    if common_factor <= 0:
        # Variance is higher than binomial variance? Overdispersed significantly or just flat.
        # Fallback: very weak prior (uniform) or global mean
        alpha_hat = m * 10
        beta_hat = (1 - m) * 10
    else:
        alpha_hat = m * common_factor
        beta_hat = (1 - m) * common_factor

    # Posterior Estimates
    # Posterior Alpha = alpha_prior + successes
    # Posterior Beta = beta_prior + failures
    sites_df = sites_df.copy()
    sites_df['alpha_post'] = alpha_hat + sites_df['Cases']
    sites_df['beta_post'] = beta_hat + (sites_df['N'] - sites_df['Cases'])

    # Posterior Mean
    sites_df['Eb_Prev'] = (sites_df['alpha_post'] / (sites_df['alpha_post'] + sites_df['beta_post'])) * 100

    # Credible Interval (using Beta ppf)
    sites_df['Eb_Low'] = stats.beta.ppf(0.025, sites_df['alpha_post'], sites_df['beta_post']) * 100
    sites_df['Eb_High'] = stats.beta.ppf(0.975, sites_df['alpha_post'], sites_df['beta_post']) * 100

    return sites_df

def run_quantile_regression(df, quantile=0.9):
    """Run Quantile Regression on continuous risk score"""
    # Outcome: risk_nonlab
    model_df = df.dropna(subset=['risk_nonlab', 'age', 'gender']).copy()
    # Normalize risk to 0-1 for stability? Or keep as is (0-100).
    # QuantReg works on continuous.

    formula = "risk_nonlab ~ age + gender + urban_rural"
    try:
        mod = smf.quantreg(formula, model_df)
        res = mod.fit(q=quantile)
        return res
    except Exception as e:
        return str(e)

def run_glm_model(df, outcome_col='high_risk'):
    """Cluster-robust Logistic Regression"""
    # Prep data: specific columns
    model_df = df.dropna(subset=[outcome_col, 'age_band', 'gender', 'urban_rural', 'site_id']).copy()

    # Ensure categorical consistency
    model_df['urban_rural'] = model_df['urban_rural'].astype(str)

    # Check variation
    if model_df['urban_rural'].nunique() < 2:
        st.warning(f"Not enough variation in Urban/Rural for GLM (Found: {model_df['urban_rural'].unique()})")

    # Formula: Outcome ~ Predictors
    formula = f"{outcome_col} ~ age_band + gender + urban_rural"

    try:
        # Binomial families
        model = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial())

        # Cluster Robust SE
        # groups must be strictly aligned with data, fit() handles this via cov_kwds
        result = model.fit(cov_type='cluster', cov_kwds={'groups': model_df['site_id']})
        return result
    except Exception as e:
        st.error(f"GLM Fitting failed: {e}")
        return None

def render_rq0(df_merged, datasets=None, selected_dataset_name=None):
    st.title("RQ0: Baseline Risk Burden & Heterogeneity")

    # Display the selected dataset name prominently
    # if selected_dataset_name:
    #     st.success(f"📊 **Currently Analyzing:** `{selected_dataset_name}`")
    #     # st.dataframe(df_merged)
        
    st.info("Distribution of WHO 2019 non-lab 10-year CVD risk across age, sex, site, and urban/rural strata (Age 40-74).")

    if df_merged is None:
        st.error("No data loaded. Please select a dataset from the sidebar.")
        return

    # --- 1) Global Settings: Risk Threshold & Horizon ---
    st.subheader("⚙️ Analysis Settings")
    c_thresh, c_horiz, c_model = st.columns([1, 1, 1])
    with c_thresh:
        risk_threshold_opt = st.radio(
            "Primary Outcome Definitions:",
            ["High Risk (≥10%)", "Very High Risk (≥20%)"],
            horizontal=True
        )
    with c_horiz:
        horizon_opt = st.radio("Risk Horizon:", ["10-Year (Standard)", "5-Year (Derived)"], horizontal=True)
    
    with c_model:
        risk_model_opt = st.radio("Risk Model:", ["Non-Lab", "Lab", "Both (Compare)"], horizontal=True)

    # --- 2) Cohort Definition & Risk Calculation ---
    df_merged['age'] = pd.to_numeric(df_merged['age'], errors='coerce')
    df_merged['risk_nonlab'] = pd.to_numeric(df_merged['risk_nonlab'], errors='coerce')
    df_merged['risk_lab'] = pd.to_numeric(df_merged['risk_lab'], errors='coerce')

    if risk_model_opt == "Both (Compare)":
        # Intersection: valid data for BOTH
        mask = (df_merged['age'] >= 40) & (df_merged['age'] <= 74) & \
               (df_merged['risk_nonlab'].notna()) & (df_merged['risk_lab'].notna())
        df_cohort = df_merged[mask].copy()
        
        # Calculate Active Risks for BOTH
        if "5-Year" in horizon_opt:
            p10_nl = df_cohort['risk_nonlab'] / 100.0
            p10_l = df_cohort['risk_lab'] / 100.0
            
            df_cohort['active_risk_nonlab'] = (1 - np.power((1 - p10_nl).clip(0, 1), 0.5)) * 100.0
            df_cohort['active_risk_lab'] = (1 - np.power((1 - p10_l).clip(0, 1), 0.5)) * 100.0
            
            active_label = "5-Year Risk (Non-Lab vs Lab)"
        else:
            df_cohort['active_risk_nonlab'] = df_cohort['risk_nonlab']
            df_cohort['active_risk_lab'] = df_cohort['risk_lab']
            active_label = "10-Year Risk (Non-Lab vs Lab)"
            
        # Classify BOTH
        # Determine threshold logic active column for creating 'high_risk' (default to lab for 'truth' or nonlab? let's create separate)
        active_thresh = 10 if "10%" in risk_threshold_opt else 20
        df_cohort['high_risk_nonlab'] = (df_cohort['active_risk_nonlab'] >= active_thresh).astype(int)
        df_cohort['high_risk_lab'] = (df_cohort['active_risk_lab'] >= active_thresh).astype(int)
        
        # Default 'high_risk' and 'active_risk' to Non-Lab for simple tables compatibility
        df_cohort['active_risk'] = df_cohort['active_risk_nonlab']
        df_cohort['high_risk'] = df_cohort['high_risk_nonlab']

    else:
        # Single Model Logic
        risk_col = 'risk_lab' if risk_model_opt == 'Lab' else 'risk_nonlab'
        mask = (df_merged['age'] >= 40) & (df_merged['age'] <= 74) & (df_merged[risk_col].notna())
        df_cohort = df_merged[mask].copy()

        # Determine Active Risk Column
        if "5-Year" in horizon_opt:
            p10 = df_cohort[risk_col] / 100.0
            p5 = 1 - np.power((1 - p10).clip(0, 1), 0.5)
            df_cohort['active_risk'] = p5 * 100.0
            active_label = f"5-Year Risk ({risk_model_opt})"
        else:
            df_cohort['active_risk'] = df_cohort[risk_col]
            active_label = f"10-Year Risk ({risk_model_opt})"
            
        active_thresh = 10 if "10%" in risk_threshold_opt else 20
        df_cohort['high_risk'] = (df_cohort['active_risk'] >= active_thresh).astype(int)

    # Define target_risk_col for downstream compatibility
    target_risk_col = 'high_risk_20' if "20%" in risk_threshold_opt else 'high_risk_10'
    target_label = f"≥{active_thresh}% ({active_label})"

    # --- 3) Risk Categorization (5-band) ---
    RISK_BINS = [-np.inf, 5, 10, 20, 30, np.inf]
    RISK_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
    RISK_COLORS = {
        '<5%': 'green',
        '5% to <10%': 'gold',
        '10% to <20%': 'orange',
        '20% to <30%': 'red',
        '≥30%': 'darkred'
    }

    if risk_model_opt == "Both (Compare)":
        df_cohort['risk_cat_nonlab'] = pd.cut(df_cohort['active_risk_nonlab'], bins=RISK_BINS, labels=RISK_LABELS, right=False)
        df_cohort['risk_cat_lab'] = pd.cut(df_cohort['active_risk_lab'], bins=RISK_BINS, labels=RISK_LABELS, right=False)
        # Default risk_cat
        df_cohort['risk_cat'] = df_cohort['risk_cat_nonlab']
    else:
        df_cohort['risk_cat'] = pd.cut(df_cohort['active_risk'], bins=RISK_BINS, labels=RISK_LABELS, right=False)

    # Ensure Age Band logic
    if 'age_band' not in df_cohort.columns or df_cohort['age_band'].nunique() < 2:
        bins = [40, 45, 50, 55, 60, 65, 70, 75]
        labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
        df_cohort['age_band'] = pd.cut(df_cohort['age'], bins=bins, labels=labels, right=False)

    # Add dataset details in an expander
    with st.expander("📋 Dataset Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df_merged):,}")
            st.metric("Analyzed Cohort (Age 40-74)", f"{len(df_cohort):,}")
        with col2:
            if 'gender' in df_cohort.columns:
                gender_counts = df_cohort['gender'].value_counts()
                st.write("**Gender Distribution:**")
                for gender, count in gender_counts.items():
                    st.write(f"- {gender}: {count:,} ({count/len(df_cohort)*100:.1f}%)")
        with col3:
            if 'urban_rural' in df_cohort.columns:
                location_counts = df_cohort['urban_rural'].value_counts()
                st.write("**Location Distribution:**")
                for loc, count in location_counts.items():
                    st.write(f"- {loc}: {count:,} ({count/len(df_cohort)*100:.1f}%)")
        
        st.markdown("**Available Columns:**")
        st.write(", ".join(sorted(df_merged.columns.tolist())))

    # =============================================================================
    # MAIN TAB ORGANIZATION
    # =============================================================================
    # Tabs are organized by study/use-case:
    #   1. Descriptive Statistics: Baseline characteristics and distributions
    #   2. Publication Tables: Formatted tables for manuscript
    #   3. Prevalence Analysis: Stratified prevalence by demographics
    #   4. Site-Level Analysis: Heterogeneity and meta-analysis across sites  
    #   5. Statistical Modeling: GLM, quantile regression
    #   6-8. Paired Analysis: Comprehensive lab vs non-lab comparison
    # =============================================================================
    
    tab1, tab2, tab2a, tab2b, tab3, tab4, tab5, tab_paired, tab_htn, tab_multi_risk = st.tabs([
        "📋 1. Baseline Characteristics",
        "📊 2. Risk Distributions", 
        "📋 3. Table: Risk Categories",
        "📋 4. Table: Risk by Gender",
        "📈 5. Prevalence Analysis",
        "🏥 6. Site Heterogeneity",
        "🧮 7. Statistical Modeling",
        "🔬 8. Lab vs Non-Lab Comparison",
        "🩺 9. Hypertension Analysis",
        "🫀 10. Multi-Model Risk"
    ])

    # --- Tab 1: Baseline Table ---
    with tab1:
        st.subheader("Table 1: Baseline Characteristics")
        st.caption("Comparison using 10-year risk stratum (standard).")

        # Extract datasets from optional dict
        d_nonlab = datasets.get('nonlab') if datasets else None
        d_lab = datasets.get('lab') if datasets else None
        d_who_nonlab = datasets.get('who_nonlab') if datasets else df_cohort
        d_who_lab = datasets.get('who_lab') if datasets else None
        d_sites = datasets.get('sites') if datasets else None

        t1_html = generate_html_table1(
            df_nonlab=d_nonlab,
            df_lab=d_lab,
            df_who_nonlab_domain=d_who_nonlab,
            df_who_lab_domain=d_who_lab,
            df_sites=d_sites,
            site_col='site_id'
        )
        st.markdown(t1_html, unsafe_allow_html=True)
        
   

        # ==========================================
        # TABLE 2: 10-Year CVD Risk by Gender (Non-Lab)
        # ==========================================
        st.divider()
        st.subheader("Table 2: Ten-Year CVD Risk by Gender (WHO Non-Laboratory Charts)")
        st.caption("10-year risk of combined (fatal and non-fatal) cardiovascular disease using the 2019 WHO cardiovascular disease risk non-laboratory-based charts")
        
        # Use the WHO non-lab domain dataset for this table
        df_table2 = d_who_nonlab if d_who_nonlab is not None else df_cohort
        
        # Ensure risk categories exist
        RISK_CAT_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
        RISK_CAT_DISPLAY = ['Very low risk (<5%)', 'Low risk (5%–10%)', 'Moderate risk (10% to <20%)', 'High risk (20% to <30%)', 'Very high risk (≥30%)']
        
        # Determine risk column to use (default to non-lab if available, else standard)
        t2_risk_col = 'risk_nonlab_cat' if 'risk_nonlab_cat' in df_table2.columns else ('risk_cat' if 'risk_cat' in df_table2.columns else None)
        
        if t2_risk_col and 'gender' in df_table2.columns:
            # Build the table
            table2_data = []
            
            # Get gender groups
            male_df = df_table2[df_table2['gender'].isin(['M', 'Male', 'men'])]
            female_df = df_table2[df_table2['gender'].isin(['F', 'Female', 'women'])]
            
            n_male = len(male_df)
            n_female = len(female_df)
            n_total = len(df_table2)
            
            for cat_label, cat_display in zip(RISK_CAT_LABELS, RISK_CAT_DISPLAY):
                # Count for each gender
                male_count = (male_df[t2_risk_col] == cat_label).sum()
                female_count = (female_df[t2_risk_col] == cat_label).sum()
                total_count = (df_table2[t2_risk_col] == cat_label).sum()
                
                table2_data.append({
                    'Risk Category': cat_display,
                    'Male n (%)': f"{male_count} ({male_count/n_male*100:.1f}%)" if n_male > 0 else "0 (0.0%)",
                    'Female n (%)': f"{female_count} ({female_count/n_female*100:.1f}%)" if n_female > 0 else "0 (0.0%)",
                    'Total n (%)': f"{total_count} ({total_count/n_total*100:.1f}%)" if n_total > 0 else "0 (0.0%)"
                })
            
            # Add totals row
            table2_data.append({
                'Risk Category': 'Total',
                'Male n (%)': f"{n_male} (100.0%)",
                'Female n (%)': f"{n_female} (100.0%)",
                'Total n (%)': f"{n_total} (100.0%)"
            })
            
            table2_df = pd.DataFrame(table2_data)
            
            # Chi-square test
            try:
                contingency = pd.crosstab(df_table2['gender'], df_table2[t2_risk_col])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                p_str = f"P = {p_value:.4f}" if p_value >= 0.0001 else "P < 0.0001"
            except Exception:
                p_str = "P = N/A"
            
            # Display table
            st.dataframe(table2_df, use_container_width=True, hide_index=True)
            st.caption(f"*{p_str} based on χ² test comparing risk distribution between males and females.")
            
            # Summary metrics
            col_t2a, col_t2b, col_t2c = st.columns(3)
            with col_t2a:
                high_risk_male = ((male_df[t2_risk_col].isin(['20% to <30%', '≥30%'])).sum() / n_male * 100) if n_male > 0 else 0
                st.metric("High Risk (≥20%) - Male", f"{high_risk_male:.1f}%")
            with col_t2b:
                high_risk_female = ((female_df[t2_risk_col].isin(['20% to <30%', '≥30%'])).sum() / n_female * 100) if n_female > 0 else 0
                st.metric("High Risk (≥20%) - Female", f"{high_risk_female:.1f}%")
            with col_t2c:
                st.metric("Chi-Square Statistic", f"χ² = {chi2:.2f}" if 'chi2' in dir() else "N/A")
        else:
            st.warning("Required columns (risk_cat_nonlab, gender) not available for Table 2.")

            # ==========================================
            # TABLE 3: CVD Risk by Glycaemic/Diabetes Status
            # ==========================================
            st.divider()
            st.subheader("Table 3: CVD Risk Estimates by Glycaemic Status")
            st.caption("10-year cardiovascular disease risk stratified by glycaemic status (Normoglycaemia, Diabetes)")

            df_table3 = d_who_nonlab if d_who_nonlab is not None else df_cohort
            t3_risk_col = 'risk_nonlab_cat' if 'risk_nonlab_cat' in df_table3.columns else (
                'risk_cat' if 'risk_cat' in df_table3.columns else None)

            if 'gly_group' not in df_table3.columns:
                df_table3 = df_table3.copy()
                df_table3['gly_group'] = df_table3.apply(categorize_glycaemic_status, axis=1)

            if 'gly_group' in df_table3.columns and t3_risk_col:
                gly_order = ['Normoglycaemia', 'Diabetes']
                df_table3_filtered = df_table3[df_table3['gly_group'].isin(gly_order)]

                if len(df_table3_filtered) > 0:
                    table3_data = []

                    for cat_label, cat_display in zip(RISK_CAT_LABELS, RISK_CAT_DISPLAY):
                        row = {'Risk Category': cat_display}

                        for gly in gly_order:
                            gly_df = df_table3_filtered[df_table3_filtered['gly_group'] == gly]
                            n_gly = len(gly_df)
                            count = (gly_df[t3_risk_col] == cat_label).sum()
                            row[f'{gly} n (%)'] = f"{count} ({count / n_gly * 100:.1f}%)" if n_gly > 0 else "0 (0.0%)"

                        total_count = (df_table3_filtered[t3_risk_col] == cat_label).sum()
                        n_total_t3 = len(df_table3_filtered)
                        row[
                            'Total n (%)'] = f"{total_count} ({total_count / n_total_t3 * 100:.1f}%)" if n_total_t3 > 0 else "0 (0.0%)"

                        table3_data.append(row)

                    totals_row = {'Risk Category': 'Total'}
                    for gly in gly_order:
                        gly_df = df_table3_filtered[df_table3_filtered['gly_group'] == gly]
                        totals_row[f'{gly} n (%)'] = f"{len(gly_df)} (100.0%)"
                    totals_row['Total n (%)'] = f"{len(df_table3_filtered)} (100.0%)"
                    table3_data.append(totals_row)

                    table3_df = pd.DataFrame(table3_data)

                    # --- FIXED: Pre-initialize variables to prevent NameError/UnboundLocalError ---
                    chi2_t3 = None
                    dof_t3 = "N/A"
                    p_str_t3 = "P = N/A"
                    p_value_t3 = 1.0

                    try:
                        contingency_t3 = pd.crosstab(df_table3_filtered['gly_group'], df_table3_filtered[t3_risk_col])
                        # Check if table layout allows calculation
                        if contingency_t3.size > 0 and (contingency_t3.sum().sum() > 0):
                            chi2_t3, p_value_t3, dof_t3, expected_t3 = stats.chi2_contingency(contingency_t3)
                            p_str_t3 = f"P = {p_value_t3:.4f}" if p_value_t3 >= 0.0001 else "P < 0.0001"
                    except Exception:
                        chi2_t3 = None
                        dof_t3 = "N/A"
                        p_str_t3 = "P = N/A"

                    # Display table safely
                    st.dataframe(table3_df, use_container_width=True, hide_index=True)
                    st.caption(f"*{p_str_t3} based on χ² test comparing risk distribution across glycaemic groups.")

                    st.markdown("**High-Risk (≥20%) Prevalence by Glycaemic Status:**")
                    col_g1, col_g2, col_g3 = st.columns(3)

                    for i, gly in enumerate(gly_order):
                        gly_df = df_table3_filtered[df_table3_filtered['gly_group'] == gly]
                        n_gly = len(gly_df)
                        high_risk_count = gly_df[t3_risk_col].isin(['20% to <30%', '≥30%']).sum()
                        high_risk_pct = transatlantic_pct = (high_risk_count / n_gly * 100) if n_gly > 0 else 0

                        with [col_g1, col_g2, col_g3][i]:
                            st.metric(gly, f"{high_risk_pct:.1f}%", f"n={high_risk_count}/{n_gly}")

                    # Render summary conditionally using initialized parameters
                    if chi2_t3 is not None:
                        st.info(f"""
                            📊 **Statistical Summary:**
                            - Chi-Square Statistic: χ² = {chi2_t3:.2f}
                            - Degrees of Freedom: {dof_t3}
                            - {p_str_t3}

                            {'⚠️ Significant difference in CVD risk distribution across glycaemic groups.' if p_value_t3 < 0.05 else 'No statistically significant difference detected.'}
                            """)
                    else:
                        st.info(
                            "📊 **Statistical Summary:** Chi-Square test skipped (insufficient variance or structural missingness in data layers).")
                else:
                    st.warning("No records with valid glycaemic status found.")
            else:
                st.warning("Required columns (gly_group, risk_cat_nonlab) not available for Table 3.")

        # ==========================================
        # TABLE 4: Ten-Year CVD Risk by Gender (Lab AND Non-Lab Comparison)
        # ==========================================
        st.divider()
        st.subheader("Table 4: Ten-Year CVD Risk by Gender (WHO Non-Laboratory and Laboratory Charts)")
        st.caption("Comparison of 10-year CVD risk distribution between Laboratory and Non-Laboratory models, stratified by gender")
        
        # Check for non-lab and lab data
        df_nonlab_t4 = d_who_nonlab if d_who_nonlab is not None else None
        df_lab_t4 = d_who_lab if d_who_lab is not None else None
        
        has_nonlab_data = df_nonlab_t4 is not None and ('risk_nonlab_cat' in df_nonlab_t4.columns or 'risk_nonlab' in df_nonlab_t4.columns) and 'gender' in df_nonlab_t4.columns
        has_lab_data = df_lab_t4 is not None and ('risk_lab_cat' in df_lab_t4.columns or 'risk_lab' in df_lab_t4.columns) and 'gender' in df_lab_t4.columns
        
        if has_nonlab_data and has_lab_data:
            df_nl = df_nonlab_t4.copy()
            df_l = df_lab_t4.copy()
            
            # Ensure risk categories exist for both
            if 'risk_nonlab_cat' not in df_nl.columns and 'risk_nonlab' in df_nl.columns:
                df_nl['risk_nonlab_cat'] = pd.cut(df_nl['risk_nonlab'], 
                                                   bins=[-np.inf, 5, 10, 20, 30, np.inf], 
                                                   labels=RISK_CAT_LABELS, right=False)
            if 'risk_lab_cat' not in df_l.columns and 'risk_lab' in df_l.columns:
                df_l['risk_lab_cat'] = pd.cut(df_l['risk_lab'], 
                                                bins=[-np.inf, 5, 10, 20, 30, np.inf], 
                                                labels=RISK_CAT_LABELS, right=False)
            
            # Drop rows with missing data
            df_nl_valid = df_nl.dropna(subset=['risk_nonlab_cat', 'gender'])
            df_l_valid = df_l.dropna(subset=['risk_lab_cat', 'gender'])
            
            if len(df_nl_valid) > 0 and len(df_l_valid) > 0:
                # Get gender groups
                male_df_nl = df_nl_valid[df_nl_valid['gender'].isin(['M', 'Male', 'men'])]
                female_df_nl = df_nl_valid[df_nl_valid['gender'].isin(['F', 'Female', 'women'])]
                n_male_nl = len(male_df_nl)
                n_female_nl = len(female_df_nl)
                n_total_nl = len(df_nl_valid)
                
                male_df_l = df_l_valid[df_l_valid['gender'].isin(['M', 'Male', 'men'])]
                female_df_l = df_l_valid[df_l_valid['gender'].isin(['F', 'Female', 'women'])]
                n_male_l = len(male_df_l)
                n_female_l = len(female_df_l)
                n_total_l = len(df_l_valid)
                
                # Build comparison table
                table4_data = []
                
                for cat_label, cat_display in zip(RISK_CAT_LABELS, RISK_CAT_DISPLAY):
                    row = {'Risk Category': cat_display}
                    
                    # --- Non-Laboratory ---
                    nl_male = (male_df_nl['risk_nonlab_cat'] == cat_label).sum()
                    nl_female = (female_df_nl['risk_nonlab_cat'] == cat_label).sum()
                    nl_total = (df_nl_valid['risk_nonlab_cat'] == cat_label).sum()
                    
                    row['Non-Lab: Male n (%)'] = f"{nl_male} ({nl_male/n_male_nl*100:.1f}%)" if n_male_nl > 0 else "0 (0.0%)"
                    row['Non-Lab: Female n (%)'] = f"{nl_female} ({nl_female/n_female_nl*100:.1f}%)" if n_female_nl > 0 else "0 (0.0%)"
                    row['Non-Lab: Total n (%)'] = f"{nl_total} ({nl_total/n_total_nl*100:.1f}%)" if n_total_nl > 0 else "0 (0.0%)"
                    
                    # --- Laboratory ---
                    l_male = (male_df_l['risk_lab_cat'] == cat_label).sum()
                    l_female = (female_df_l['risk_lab_cat'] == cat_label).sum()
                    l_total = (df_l_valid['risk_lab_cat'] == cat_label).sum()
                    
                    row['Lab: Male n (%)'] = f"{l_male} ({l_male/n_male_l*100:.1f}%)" if n_male_l > 0 else "0 (0.0%)"
                    row['Lab: Female n (%)'] = f"{l_female} ({l_female/n_female_l*100:.1f}%)" if n_female_l > 0 else "0 (0.0%)"
                    row['Lab: Total n (%)'] = f"{l_total} ({l_total/n_total_l*100:.1f}%)" if n_total_l > 0 else "0 (0.0%)"
                    
                    table4_data.append(row)
                
                # Add totals row
                table4_data.append({
                    'Risk Category': 'Total',
                    'Non-Lab: Male n (%)': f"{n_male_nl} (100.0%)",
                    'Non-Lab: Female n (%)': f"{n_female_nl} (100.0%)",
                    'Non-Lab: Total n (%)': f"{n_total_nl} (100.0%)",
                    'Lab: Male n (%)': f"{n_male_l} (100.0%)",
                    'Lab: Female n (%)': f"{n_female_l} (100.0%)",
                    'Lab: Total n (%)': f"{n_total_l} (100.0%)"
                })
                
                table4_df = pd.DataFrame(table4_data)
                
                # Reorder columns for better display
                col_order = ['Risk Category', 
                             'Non-Lab: Male n (%)', 'Non-Lab: Female n (%)', 'Non-Lab: Total n (%)',
                             'Lab: Male n (%)', 'Lab: Female n (%)', 'Lab: Total n (%)']
                table4_df = table4_df[col_order]
                
                # Display table
                st.dataframe(table4_df, use_container_width=True, hide_index=True)
                
                # Chi-square tests
                chi_results = []
                
                # Chi-square for Non-Lab by gender
                try:
                    ct_nl = pd.crosstab(df_nl_valid['gender'], df_nl_valid['risk_nonlab_cat'])
                    chi2_nl, p_nl, dof_nl, _ = stats.chi2_contingency(ct_nl)
                    p_str_nl = f"P = {p_nl:.4f}" if p_nl >= 0.0001 else "P < 0.0001"
                    chi_results.append(('Non-Laboratory', chi2_nl, dof_nl, p_nl, p_str_nl))
                except Exception:
                    chi_results.append(('Non-Laboratory', None, None, None, 'N/A'))
                
                # Chi-square for Lab by gender
                try:
                    ct_l = pd.crosstab(df_l_valid['gender'], df_l_valid['risk_lab_cat'])
                    chi2_l, p_l, dof_l, _ = stats.chi2_contingency(ct_l)
                    p_str_l = f"P = {p_l:.4f}" if p_l >= 0.0001 else "P < 0.0001"
                    chi_results.append(('Laboratory', chi2_l, dof_l, p_l, p_str_l))
                except Exception:
                    chi_results.append(('Laboratory', None, None, None, 'N/A'))
                
                # Display chi-square summary
                st.markdown("**Statistical Comparison (χ² test by Gender):**")
                col_chi1, col_chi2 = st.columns(2)
                
                with col_chi1:
                    if chi_results[0][1] is not None:
                        st.metric("Non-Laboratory Model", chi_results[0][4], delta=f"χ² = {chi_results[0][1]:.2f}")
                    else:
                        st.metric("Non-Laboratory Model", "N/A")
                
                with col_chi2:
                    if chi_results[1][1] is not None:
                        st.metric("Laboratory Model", chi_results[1][4], delta=f"χ² = {chi_results[1][1]:.2f}")
                    else:
                        st.metric("Laboratory Model", "N/A")
                
                # Summary of high-risk by model and gender
                st.markdown("---")
                st.markdown("**High-Risk (≥20%) Prevalence Summary:**")
                
                summary_data = []
                for model, cat_col, male_df_used, female_df_used, n_male_used, n_female_used in [('Non-Laboratory', 'risk_nonlab_cat', male_df_nl, female_df_nl, n_male_nl, n_female_nl), ('Laboratory', 'risk_lab_cat', male_df_l, female_df_l, n_male_l, n_female_l)]:
                    for gender, g_df, n_g in [('Male', male_df_used, n_male_used), ('Female', female_df_used, n_female_used)]:
                        hr_count = g_df[cat_col].isin(['20% to <30%', '≥30%']).sum()
                        hr_pct = hr_count / n_g * 100 if n_g > 0 else 0
                        summary_data.append({
                            'Model': model,
                            'Gender': gender,
                            'n': n_g,
                            'High-Risk n': hr_count,
                            'High-Risk %': f"{hr_pct:.1f}%"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.caption("*High-risk defined as 10-year CVD risk ≥20% according to WHO 2019 guidelines.")
            else:
                st.warning("No valid data available after dropping missing values for Table 4.")
        else:
            st.warning("Table 4 requires both Laboratory and Non-Laboratory risk data with gender information.")


    # --- Tab 2: Visualizations ---
    with tab2:
        st.subheader(f"Risk Distribution: {active_label}")

        # target_risk_col
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Overall")
            
            if risk_model_opt == "Both (Compare)":
                # Prepare Comparative Data
                # 1. Non-Lab Counts
                c_nl = df_cohort['risk_cat_nonlab'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                c_nl.columns = ['risk_cat', 'percent']
                c_nl['Model'] = 'Non-Lab'
                
                # 2. Lab Counts
                c_l = df_cohort['risk_cat_lab'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                c_l.columns = ['risk_cat', 'percent']
                c_l['Model'] = 'Lab'
                
                # Combined
                counts_comb = pd.concat([c_nl, c_l], axis=0)
                
                # Enforce Order
                counts_comb['risk_cat'] = pd.Categorical(counts_comb['risk_cat'], categories=RISK_LABELS, ordered=True)
                counts_comb = counts_comb.sort_values(['risk_cat', 'Model'])
                
                # st.dataframe(counts_comb.pivot(index='risk_cat', columns='Model', values='percent'))
                
                # st.dataframe(counts_comb.pivot(index='risk_cat', columns='Model', values='percent'))
                
                # --- Matplotlib Comparison Plot (Bar + Line) ---
                # Ensure all categories are present for alignment
                def get_series(model_name):
                    sub = counts_comb[counts_comb['Model'] == model_name]
                    # Create Series indexed by risk_cat
                    s = sub.set_index('risk_cat')['percent']
                    # Reindex to force all RISK_LABELS limits
                    s = s.reindex(RISK_LABELS, fill_value=0.0)
                    return s

                s_nl = get_series('Non-Lab')
                s_l = get_series('Lab')
                
                cum_nl = s_nl.cumsum()
                cum_l = s_l.cumsum()
                
                x = np.arange(len(RISK_LABELS))
                width = 0.35  # width of the bars
                
                N = len(df_cohort)

                fig_ov, ax1 = plt.subplots(figsize=(10, 6))
                
                # Grouped Bars
                rects1 = ax1.bar(x - width/2, s_nl, width, label=f'Non-Lab (n={N})', color='#1f77b4', alpha=0.8)
                rects2 = ax1.bar(x + width/2, s_l, width, label=f'Lab (n={N})', color='#ff7f0e', alpha=0.8)
                
                # Add labels on bars
                ax1.bar_label(rects1, padding=3, fmt='%.1f%%', fontsize=8)
                ax1.bar_label(rects2, padding=3, fmt='%.1f%%', fontsize=8)
                
                ax1.set_ylabel('Prevalence (%)', fontweight='bold')
                ax1.set_title(f'Overall Risk Distribution (Non-Lab vs Lab)', fontweight='bold', pad=15)
                ax1.set_xticks(x)
                ax1.set_xticklabels(RISK_LABELS)
                ax1.set_ylim(0, max(s_nl.max(), s_l.max()) * 1.25)
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Secondary Axis for Cumulative
                ax2 = ax1.twinx()
                ax2.plot(x, cum_nl, color='#1f77b4', linestyle='--', marker='o', linewidth=2, label='Non-Lab (C)')
                ax2.plot(x, cum_l, color='#ff7f0e', linestyle='--', marker='s', linewidth=2, label='Lab (C)')
                
                ax2.set_ylabel('Cumulative (%)', fontweight='bold')
                ax2.set_ylim(0, 110)
                ax2.grid(False)
                
                # Combined Legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, fancybox=True)
                ax1.legend(lines1 + lines2, labels1 + labels2, 
                loc="center", bbox_to_anchor=(0.85, 0.7),
                 frameon=True, fancybox=True)

                plt.tight_layout()
                st.pyplot(fig_ov)
                add_download_button(fig_ov, "compare_risk_distribution", "matplotlib")
                
            else:
                # Single Model Logic (Existing)
                counts = df_cohort['risk_cat'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                counts.columns = ['risk_cat', 'percent']

                # Enforce Order
                counts['risk_cat'] = pd.Categorical(counts['risk_cat'], categories=RISK_LABELS, ordered=True)
                counts = counts.sort_values('risk_cat')

                # Cumulative Sum
                counts['cumulative'] = counts['percent'].cumsum()

                # Construct Combo Plot (Bar + Line) with matplotlib
                fig_ov, ax1 = plt.subplots(figsize=(10, 6))
                
                # Bar chart (Prevalence) on primary y-axis
                x_pos = np.arange(len(counts['risk_cat']))
                colors = [RISK_COLORS[c] for c in counts['risk_cat']]
                bars = ax1.bar(x_pos, counts['percent'], color=colors, alpha=0.8, label='Prevalence (%)')
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, counts['percent'])):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
                
                ax1.set_xlabel('Risk Category', fontweight='bold')
                ax1.set_ylabel('Prevalence (%)', fontweight='bold')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(counts['risk_cat'], rotation=0)
                ax1.set_ylim(0, max(counts['percent'].max()*1.2, 5))
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Line plot (Cumulative) on secondary y-axis
                ax2 = ax1.twinx()
                ax2.plot(x_pos, counts['cumulative'], color='#333333', linewidth=2.5, 
                        linestyle='--', marker='o', markersize=7, label='Cumulative %')
                ax2.set_ylabel('Cumulative (%)', fontweight='bold')
                ax2.set_ylim(0, 105)
                ax2.grid(False)
                
                # Combined legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # ax1.legend(lines1 + lines2, labels1 + labels2,
                #         loc='upper left', frameon=True, fancybox=True)
                ax1.legend(lines1 + lines2, labels1 + labels2,
                           loc="center", bbox_to_anchor=(0.85, 0.7),
                           frameon=True, fancybox=True)

                plt.title('Overall Risk Distribution', fontweight='bold', fontsize=12, pad=15)
                plt.tight_layout()
                st.pyplot(fig_ov)
                add_download_button(fig_ov, "overall_risk_distribution", "matplotlib")

        with c2:
            st.markdown("#### By Location Type")
            if 'urban_rural' in df_cohort.columns:
                
                if risk_model_opt == "Both (Compare)":
                    
                    # Prepare Data: Group by Location AND Model
                    # We need tidy data: urban_rural, Model, risk_cat, percent
                    
                    # Non-Lab
                    ur_nl = df_cohort.groupby('urban_rural', observed=True)['risk_cat_nonlab'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                    ur_nl.columns = ['urban_rural', 'risk_cat', 'percent'] # check rename
                    ur_nl['Model'] = 'Non-Lab'
                    
                    # Lab
                    ur_l = df_cohort.groupby('urban_rural', observed=True)['risk_cat_lab'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                    ur_l.columns = ['urban_rural', 'risk_cat', 'percent']
                    ur_l['Model'] = 'Lab'
                    
                    ur_comb = pd.concat([ur_nl, ur_l], axis=0)
                    
                    locations = sorted(ur_comb['urban_rural'].unique())
                    if not locations:
                        st.warning("No location data found.")
                    else:
                        # Create Single Unified Graph (Grouped by Location & Model)
                        fig_ur, ax = plt.subplots(figsize=(12, 7))
                        
                        x = np.arange(len(RISK_LABELS))
                        n_groups = len(locations) * 2  # Non-Lab, Lab per location
                        total_width = 0.8
                        bar_width = total_width / n_groups
                        
                        # Style Mapping
                        model_colors = {'Non-Lab': '#1f77b4', 'Lab': '#ff7f0e'}
                        loc_hatches = ['', '///', '..', 'xx']  # Distinct hatch per location
                        loc_markers = ['o', 's', '^', 'D']     # Distinct marker per location
                        line_styles = ['--', '-.', ':', '-']   # Distinct line style per location
                        
                        # Secondary axis for cumulative
                        ax2 = ax.twinx()
                        
                        legend_elements = []  # To store legend handles manually for cleaner order
                        max_val = 0
                        
                        for i_loc, loc in enumerate(locations):
                            # Filter
                            sub = ur_comb[ur_comb['urban_rural'] == loc]
                            
                            def get_series_loc(m):
                                s = sub[sub['Model'] == m].set_index('risk_cat')['percent']
                                return s.reindex(RISK_LABELS, fill_value=0.0)
                            
                            s_nl = get_series_loc('Non-Lab')
                            s_l = get_series_loc('Lab')
                            
                            max_val = max(max_val, s_nl.max(), s_l.max())
                            
                            # Plot Bars (Non-Lab then Lab)
                            # Calculation for position:
                            # Group center is x. 
                            # We have n_groups bars.
                            # Start from x - total_width/2
                            
                            # Location 'i' takes up 2 slots (NL, L)
                            # Slot 1: i_loc * 2
                            # Slot 2: i_loc * 2 + 1
                            
                            # Non-Lab Bar
                            pos_nl = x - total_width/2 + (i_loc * 2) * bar_width + bar_width/2
                            b1 = ax.bar(pos_nl, s_nl, bar_width, 
                                       color=model_colors['Non-Lab'], 
                                       hatch=loc_hatches[i_loc % len(loc_hatches)],
                                       alpha=0.9, edgecolor='white', label=f'{loc} Non-Lab')
                            
                            # Lab Bar
                            pos_l = x - total_width/2 + (i_loc * 2 + 1) * bar_width + bar_width/2
                            b2 = ax.bar(pos_l, s_l, bar_width, 
                                       color=model_colors['Lab'], 
                                       hatch=loc_hatches[i_loc % len(loc_hatches)],
                                       alpha=0.9, edgecolor='white', label=f'{loc} Lab')
                            
                            # Cumulative Lines
                            # Center lines on the specific bars? Or center on category x?
                            # Centering on x is cleaner for trend comparison, but distinguishing is hard.
                            # Let's offset slightly to match the "center of the location group"?
                            # Center of location group = (pos_nl + pos_l) / 2
                            loc_center = (pos_nl + pos_l) / 2
                            
                            l1 = ax2.plot(loc_center, s_nl.cumsum(), color=model_colors['Non-Lab'],
                                        linestyle=line_styles[i_loc % len(line_styles)],
                                        marker=loc_markers[i_loc % len(loc_markers)],
                                        markersize=6, linewidth=1.5, alpha=0.8,
                                        label=f'{loc} Non-Lab (Cum)')
                            
                            l2 = ax2.plot(loc_center, s_l.cumsum(), color=model_colors['Lab'],
                                        linestyle=line_styles[i_loc % len(line_styles)],
                                        marker=loc_markers[i_loc % len(loc_markers)],
                                        markersize=6, linewidth=1.5, alpha=0.8,
                                        label=f'{loc} Lab (Cum)')
                        
                        ax.set_title("Comparative Risk Distribution by Location", fontweight='bold')
                        ax.set_ylabel("Prevalence (%)", fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(RISK_LABELS)
                        ax.set_ylim(0, max_val * 1.3)
                        
                        ax2.set_ylabel("Cumulative (%)", fontweight='bold')
                        ax2.set_ylim(0, 110)
                        ax2.grid(False)
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Combined Legend below
                        # Get handles from axes
                        h1, l1 = ax.get_legend_handles_labels()
                        h2, l2 = ax2.get_legend_handles_labels()
                        ax.legend(h1 + h2, l1 + l2, loc='upper center', 
                                 bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
                        
                        plt.tight_layout()
                        st.pyplot(fig_ur)
                        add_download_button(fig_ur, "location_compare_risk_unified", "matplotlib")

                else:
                    # Original Stacked Bar Logic
                    ur_counts = df_cohort.groupby('urban_rural', observed=False)['risk_cat'].value_counts(normalize=True).mul(100).reset_index(name='percent')
                    
                    # Create matplotlib stacked bar chart
                    fig_ur, ax_ur = plt.subplots(figsize=(8, 6))
                    
                    # Pivot data for stacked bar
                    ur_pivot = ur_counts.pivot(index='urban_rural', columns='risk_cat', values='percent').fillna(0)
                    # Ensure correct column order
                    ur_pivot = ur_pivot.reindex(columns=RISK_LABELS, fill_value=0)
                    
                    # Create stacked bar chart
                    ur_pivot.plot(kind='bar', stacked=True, ax=ax_ur, 
                                 color=[RISK_COLORS.get(cat, '#cccccc') for cat in ur_pivot.columns],
                                 width=0.6, edgecolor='white', linewidth=0.5)
                    
                    ax_ur.set_ylabel('Percent (%)', fontweight='bold')
                    ax_ur.set_xlabel('', fontweight='bold')
                    ax_ur.set_title('Risk Distribution by Location Type', fontweight='bold', fontsize=11, pad=10)
                    ax_ur.set_xticklabels(ax_ur.get_xticklabels(), rotation=0)
                    ax_ur.legend(title='Risk Category', bbox_to_anchor=(1.05, 1), loc='upper left', 
                               frameon=True, fancybox=True)
                    ax_ur.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig_ur)
            else:
                st.warning("Urban/Rural data unavailable")



        st.subheader("Stratified Distribution (Age × Urban/Rural)")

        # st.dataframe(df_cohort[['urban_rural', 'age_band', 'risk_cat']])
        if {'urban_rural', 'age_band', 'risk_cat'}.issubset(df_cohort.columns):
            if risk_model_opt == "Both (Compare)":
                strat_nl = (df_cohort.groupby(['age_band', 'urban_rural'], observed=False)['risk_cat_nonlab']
                            .value_counts(normalize=True).mul(100).reset_index(name='percent'))
                strat_nl['Model'] = 'Non-Lab'
                strat_nl.rename(columns={'risk_cat_nonlab': 'risk_cat'}, inplace=True)
                
                strat_l = (df_cohort.groupby(['age_band', 'urban_rural'], observed=False)['risk_cat_lab']
                           .value_counts(normalize=True).mul(100).reset_index(name='percent'))
                strat_l['Model'] = 'Lab'
                strat_l.rename(columns={'risk_cat_lab': 'risk_cat'}, inplace=True)
                
                strat_comb = pd.concat([strat_nl, strat_l], axis=0)

                locs = sorted(strat_comb['urban_rural'].unique())
                
                if not locs:
                    st.warning("No stratified data available.")
                else:
                    # Create Single Unified Figure: Horizontal Facets for Locations
                    # X-Axis = Age Bands (Pure)
                    # Facets = Locations (Columns)
                    # Groups = Models (Side-by-Side within Age)
                    
                    age_bands = sorted(strat_comb['age_band'].astype(str).unique())
                    n_locs = len(locs)
                    
                    # One row, N columns (Visually contiguous)
                    fig_s, axes = plt.subplots(1, n_locs, figsize=(5 * n_locs, 6), sharey=True)
                    if n_locs == 1: axes = [axes]
                    
                    # Style
                    model_hatches = {'Non-Lab': '', 'Lab': '///'}
                    # Bar params
                    x = np.arange(len(age_bands))
                    width = 0.35
                    
                    for i, loc in enumerate(locs):
                        ax = axes[i]
                        sub_loc = strat_comb[strat_comb['urban_rural'] == loc]
                        
                        # We need to iterate over Age Bands to ensure alignment and missing data handling
                        # But simpler: Pivot [Age, Model] -> [Risk1, Risk2...]
                        # Actually we need grouped bars: Model A at x-w/2, Model B at x+w/2.
                        
                        for m_idx, model in enumerate(['Non-Lab', 'Lab']):
                            # Filter Model
                            sub_m = sub_loc[sub_loc['Model'] == model]
                            
                            # Pivot to shape: Index=AgeBand, Cols=RiskCat
                            # Ensure all age bands exist
                            if sub_m.empty:
                                df_plot = pd.DataFrame(0, index=age_bands, columns=RISK_LABELS)
                            else:
                                df_plot = sub_m.pivot(index='age_band', columns='risk_cat', values='percent')
                                df_plot = df_plot.reindex(index=age_bands, columns=RISK_LABELS, fill_value=0.0)
                            
                            # Plot Stacked Bar
                            # Calculate Bottoms
                            bottoms = np.zeros(len(age_bands))
                            bar_x = x - (width/2) if m_idx == 0 else x + (width/2)
                            
                            for risk in RISK_LABELS:
                                heights = df_plot[risk].values
                                ax.bar(bar_x, heights, width, bottom=bottoms,
                                       color=RISK_COLORS[risk],
                                       hatch=model_hatches[model],
                                       edgecolor='white', linewidth=0.5, alpha=0.9)
                                bottoms += heights
                        
                        # Titles & Grid
                        ax.set_title(f"{loc} Setting", fontweight='bold', fontsize=11)
                        ax.set_xlabel("Age Band", fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(age_bands, rotation=45, ha='right')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        if i == 0:
                            ax.set_ylabel("Prevalence (%)", fontweight='bold')
                    
                    # Unified Legend
                    # 1. Risk Colors
                    risk_patches = [plt.Rectangle((0,0),1,1, color=RISK_COLORS[c]) for c in RISK_LABELS]
                    # 2. Model Patterns
                    hatch_patches = [
                        plt.Rectangle((0,0),1,1, facecolor='#cccccc', edgecolor='black', hatch='', label='Non-Lab'),
                        plt.Rectangle((0,0),1,1, facecolor='#cccccc', edgecolor='black', hatch='///', label='Lab')
                    ]
                    
                    # Combine legend? Or two legends?
                    # Let's put Model legend on top, Risk legend on right?
                    # Or both on top.
                    
                    fig_s.legend(risk_patches, RISK_LABELS, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(RISK_LABELS), title="Risk Categories", frameon=False)
                    fig_s.legend(handles=hatch_patches, labels=['Non-Lab', 'Lab'], loc='upper left', bbox_to_anchor=(0.02, 1.05), ncol=2, frameon=False)

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85, wspace=0.05) # Compact look
                    
                    st.pyplot(fig_s)
                    add_download_button(fig_s, "stratified_compare_facets", "matplotlib")

            else:
                # Existing Single Model Logic (Grouped by Location) - Matplotlib version
                strat = (
                    df_cohort
                    .groupby(['age_band', 'urban_rural'], observed=False)['risk_cat']
                    .value_counts(normalize=True)
                    .mul(100)
                    .reset_index(name='percent')
                )

                # Keep consistent ordering
                age_order = sorted(strat['age_band'].dropna().unique().tolist())
                possible_locs = ['Rural', 'Urban', 'Semi-urban']
                unique_locs = set(strat['urban_rural'].unique())
                
                if set(['Rural','Urban']).issubset(unique_locs):
                     loc_order = [l for l in possible_locs if l in unique_locs] + [l for l in unique_locs if l not in possible_locs]
                else:
                     loc_order = sorted(unique_locs)

                # Create Matplotlib grouped stacked bar chart
                n_locs = len(loc_order)
                n_ages = len(age_order)
                
                fig_strat, ax_strat = plt.subplots(figsize=(12, 6))
                
                x_base = np.arange(n_ages)
                width = 0.8 / n_locs  # Width for each location's bar
                
                for j, loc in enumerate(loc_order):
                    # Calculate offset for this location group
                    offset = (j - n_locs/2 + 0.5) * width
                    
                    # Get data for this location
                    loc_data = strat[strat['urban_rural'] == loc]
                    
                    # Build stacked bars
                    bottoms = np.zeros(n_ages)
                    
                    for risk in RISK_LABELS:
                        heights = []
                        for age in age_order:
                            val = loc_data[(loc_data['age_band'] == age) & (loc_data['risk_cat'] == risk)]['percent'].values
                            heights.append(val[0] if len(val) > 0 else 0)
                        heights = np.array(heights)
                        
                        # Different hatch patterns for different locations
                        hatch = '' if j == 0 else '///' if j == 1 else '...'
                        
                        ax_strat.bar(x_base + offset, heights, width, bottom=bottoms,
                                   color=RISK_COLORS.get(risk, '#888888'),
                                   hatch=hatch,
                                   edgecolor='white', linewidth=0.5,
                                   label=risk if j == 0 else "")  # Label only first set
                        bottoms += heights
                
                # Customize plot
                ax_strat.set_xlabel('Age Band', fontsize=12, fontfamily='serif')
                ax_strat.set_ylabel('Prevalence (%)', fontsize=12, fontfamily='serif')
                ax_strat.set_title(f'Stratified Risk Distribution by Age and Location', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_strat.set_xticks(x_base)
                ax_strat.set_xticklabels(age_order, rotation=45, ha='right', fontsize=10)
                ax_strat.spines['top'].set_visible(False)
                ax_strat.spines['right'].set_visible(False)
                ax_strat.grid(True, alpha=0.3, axis='y')
                
                # Create legend for risk categories
                risk_patches = [mpatches.Patch(facecolor=RISK_COLORS[c], edgecolor='black', label=c) for c in RISK_LABELS]
                
                # Create legend for locations (using hatches)
                loc_handles = []
                for j, loc in enumerate(loc_order):
                    hatch = '' if j == 0 else '///' if j == 1 else '...'
                    loc_handles.append(mpatches.Patch(facecolor='#cccccc', edgecolor='black', hatch=hatch, label=loc))
                
                # Add legends
                legend1 = ax_strat.legend(handles=risk_patches, loc='upper right', title='Risk Category', frameon=True)
                ax_strat.add_artist(legend1)
                ax_strat.legend(handles=loc_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                              ncol=len(loc_order), title='Location', frameon=True)
                
                plt.tight_layout()
                st.pyplot(fig_strat)
                add_svg_download_button(fig_strat, "stratified_risk_distribution", key="strat_dist_svg")
                add_download_button(fig_strat, "stratified_risk_distribution", "matplotlib")
                plt.close(fig_strat)
        else:
            st.warning("Data for Stratified Plot missing.")

        st.divider()
        st.subheader("Distribution by Age & Gender")
        if 'gender' in df_cohort.columns:
            
            if risk_model_opt == "Both (Compare)":
                # Comparative Logic: Facet by Gender, X=Age, Grouped(Non-Lab, Lab)
                # Prepare grouped data
                ag_nl = (df_cohort.groupby(['age_band', 'gender'], observed=False)['risk_cat_nonlab']
                            .value_counts(normalize=True).mul(100).reset_index(name='percent'))
                ag_nl['Model'] = 'Non-Lab'
                ag_nl.rename(columns={'risk_cat_nonlab': 'risk_cat'}, inplace=True)
                
                ag_l = (df_cohort.groupby(['age_band', 'gender'], observed=False)['risk_cat_lab']
                           .value_counts(normalize=True).mul(100).reset_index(name='percent'))
                ag_l['Model'] = 'Lab'
                ag_l.rename(columns={'risk_cat_lab': 'risk_cat'}, inplace=True)
                
                ag_comb = pd.concat([ag_nl, ag_l], axis=0)
                
                genders = sorted(ag_comb['gender'].unique())
                n_genders = len(genders)
                
                # Create Horizontal Facets (1 Row, N Columns)
                fig_ag, axes = plt.subplots(1, n_genders, figsize=(6 * n_genders, 6), sharey=True)
                if n_genders == 1: axes = [axes]
                
                # Style Params
                model_hatches = {'Non-Lab': '', 'Lab': '///'}
                x = np.arange(len(sorted(ag_comb['age_band'].unique().astype(str)))) # Assume aligned age bands
                age_bands = sorted(ag_comb['age_band'].unique().astype(str))
                width = 0.35
                
                for i, gender in enumerate(genders):
                    ax = axes[i]
                    sub_gen = ag_comb[ag_comb['gender'] == gender]
                    
                    for m_idx, model in enumerate(['Non-Lab', 'Lab']):
                        sub_m = sub_gen[sub_gen['Model'] == model]
                        
                        if sub_m.empty:
                            df_plot = pd.DataFrame(0, index=age_bands, columns=RISK_LABELS)
                        else:
                            df_plot = sub_m.pivot(index='age_band', columns='risk_cat', values='percent')
                            df_plot = df_plot.reindex(index=age_bands, columns=RISK_LABELS, fill_value=0.0)
                            
                        bottoms = np.zeros(len(age_bands))
                        bar_x = x - (width/2) if m_idx == 0 else x + (width/2)
                        
                        for risk in RISK_LABELS:
                            heights = df_plot[risk].values
                            ax.bar(bar_x, heights, width, bottom=bottoms,
                                   color=RISK_COLORS[risk],
                                   hatch=model_hatches[model],
                                   edgecolor='white', linewidth=0.5, alpha=0.9)
                            bottoms += heights
                            
                    ax.set_title(f"{gender}", fontweight='bold', fontsize=11)
                    ax.set_xlabel("Age Band", fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(age_bands, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    if i == 0:
                        ax.set_ylabel("Prevalence (%)", fontweight='bold')

                # Legends
                risk_patches = [plt.Rectangle((0,0),1,1, color=RISK_COLORS[c]) for c in RISK_LABELS]
                hatch_patches = [
                    plt.Rectangle((0,0),1,1, facecolor='#cccccc', edgecolor='black', hatch='', label='Non-Lab'),
                    plt.Rectangle((0,0),1,1, facecolor='#cccccc', edgecolor='black', hatch='///', label='Lab')
                ]
                
                fig_ag.legend(risk_patches, RISK_LABELS, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(RISK_LABELS), title="Risk Categories", frameon=False)
                fig_ag.legend(handles=hatch_patches, labels=['Non-Lab', 'Lab'], loc='upper left', bbox_to_anchor=(0.02, 1.05), ncol=2, frameon=False)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.85, wspace=0.1)
                st.pyplot(fig_ag)
                add_download_button(fig_ag, "gender_compare_facets", "matplotlib")

            else:
                # Existing Single Model Logic
                ag_counts = df_cohort.groupby(['age_band', 'gender'], observed=False)['risk_cat'].value_counts(normalize=True).mul(100).reset_index(name='percent')
                
                # Create matplotlib subplots for each gender
                genders = ag_counts['gender'].unique()
                fig_ag, axes = plt.subplots(1, len(genders), figsize=(14, 5), sharey=True)
                
                # Handle single gender case
                if len(genders) == 1:
                    axes = [axes]
                
                for idx, (gender, ax) in enumerate(zip(genders, axes)):
                    gender_data = ag_counts[ag_counts['gender'] == gender]
                    
                    # Pivot for stacked bar
                    pivot_data = gender_data.pivot(index='age_band', columns='risk_cat', values='percent').fillna(0)
                    pivot_data = pivot_data.reindex(columns=RISK_LABELS, fill_value=0)
                    
                    # Create stacked bar
                    pivot_data.plot(kind='bar', stacked=True, ax=ax,
                                   color=[RISK_COLORS.get(cat, '#cccccc') for cat in pivot_data.columns],
                                   width=0.7, edgecolor='white', linewidth=0.5,
                                   legend=(idx == len(genders) - 1))  # Only show legend on last subplot
                    
                    ax.set_title(f'{gender}', fontweight='bold', fontsize=11, pad=10)
                    ax.set_xlabel('Age Band', fontweight='bold')
                    ax.set_ylabel('Percent (%)' if idx == 0 else '', fontweight='bold')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    if idx == len(genders) - 1:
                        ax.legend(title='Risk Category', bbox_to_anchor=(1.05, 1), loc='upper left',
                                 frameon=True, fancybox=True)
                
                fig_ag.suptitle('Distribution by Age & Gender', fontweight='bold', fontsize=13, y=1.02)
                plt.tight_layout()
                st.pyplot(fig_ag)
            
        st.divider()
        st.subheader("Stratified Distribution (Age × Gender × Location)")
        st.caption("Combined interaction of Age, Sex, and Geographic Setting on Risk Distribution")

        if {'urban_rural', 'age_band', 'risk_cat', 'gender'}.issubset(df_cohort.columns):
            # Group by all 3
            inter_counts = df_cohort.groupby(['age_band', 'gender', 'urban_rural'], observed=False)['risk_cat'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
            
            # Use Plotly Express for Faceted Chart
            # X=Age, Y=Percent, Color=Risk, FacetCol=Gender, FacetRow=UrbanRural? Or FacetCol=Gender, X=Age, Color=Risk...
            # User wants to see Variation. 
            # Let's show: X=Age, Y=Percent, Color=Risk, Facet_Col=Gender, Facet_Row=Urban/Rural
            
            # Ensure order
            inter_counts['risk_cat'] = pd.Categorical(inter_counts['risk_cat'], categories=RISK_LABELS, ordered=True)
            inter_counts = inter_counts.sort_values('risk_cat')
            
            # Create Matplotlib faceted figure
            genders = sorted(inter_counts['gender'].astype(str).unique())
            locations = sorted(inter_counts['urban_rural'].astype(str).unique())
            age_bands = sorted(inter_counts['age_band'].astype(str).unique())
            
            n_rows = len(locations)
            n_cols = len(genders)
            
            fig_inter, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharey=True)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            x = np.arange(len(age_bands))
            width = 0.8
            
            for r, loc in enumerate(locations):
                for c, gen in enumerate(genders):
                    ax = axes[r, c]
                    sub = inter_counts[(inter_counts['urban_rural'].astype(str) == loc) & 
                                      (inter_counts['gender'].astype(str) == gen)]
                    
                    # Build stacked bars
                    bottoms = np.zeros(len(age_bands))
                    for risk in RISK_LABELS:
                        heights = []
                        for age in age_bands:
                            val = sub[(sub['age_band'].astype(str) == age) & (sub['risk_cat'] == risk)]['percent'].values
                            heights.append(val[0] if len(val) > 0 else 0)
                        heights = np.array(heights)
                        
                        ax.bar(x, heights, width, bottom=bottoms,
                              color=RISK_COLORS.get(risk, '#888888'),
                              edgecolor='white', linewidth=0.5)
                        bottoms += heights
                    
                    ax.set_title(f'{gen} - {loc}', fontsize=11, fontweight='bold', fontfamily='serif')
                    ax.set_xticks(x)
                    ax.set_xticklabels(age_bands, rotation=45, ha='right', fontsize=9)
                    ax.set_ylim(0, 105)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    if c == 0:
                        ax.set_ylabel('Prevalence (%)', fontsize=10, fontfamily='serif')
                    if r == n_rows - 1:
                        ax.set_xlabel('Age Group', fontsize=10, fontfamily='serif')
            
            # Legend
            risk_patches = [mpatches.Patch(facecolor=RISK_COLORS[c], edgecolor='black', label=c) for c in RISK_LABELS]
            fig_inter.legend(handles=risk_patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                           ncol=len(RISK_LABELS), frameon=True, fontsize=9)
            
            fig_inter.suptitle('Risk Distribution by Age, Gender, and Location', fontsize=14, fontweight='bold', y=1.06)
            plt.tight_layout()
            st.pyplot(fig_inter)
            add_svg_download_button(fig_inter, "risk_by_age_gender_location", key="inter_facet_svg")
            plt.close(fig_inter)


        st.subheader("Stratified Distribution (Age × Gender × Location)")
        st.caption("Combined interaction of Age, Sex, and Geographic Setting on Risk Distribution")

        if {'urban_rural', 'age_band', 'risk_cat', 'gender'}.issubset(df_cohort.columns):
            # Prepare hierarchical data - Group by all strata
            stratified_counts = df_cohort.groupby(
                ['urban_rural', 'gender', 'age_band', 'risk_cat'], 
                observed=False
            ).size().reset_index(name='count')
            
            # Calculate percentages within each group (Location x Gender x Age)
            stratified_counts['percentage'] = stratified_counts.groupby(
                ['urban_rural', 'gender', 'age_band']
            )['count'].transform(lambda x: x / x.sum() * 100)
            
            # Map labels for cleaner display
            stratified_counts['location_label'] = stratified_counts['urban_rural'].map({
                'Urban': 'Urban',
                'Rural': 'Rural'
            }).fillna(stratified_counts['urban_rural'])
            
            stratified_counts['gender_label'] = stratified_counts['gender'].map({
                'M': 'Male',
                'F': 'Female'
            }).fillna(stratified_counts['gender'])
            
            # Create Matplotlib Faceted Stacked Bar Plot (2x2 grid: Gender x Location)
            genders = ['Male', 'Female']
            locations = ['Urban', 'Rural']
            age_bands = sorted(stratified_counts['age_band'].unique())
            
            fig_stratified, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
            
            for r, gender in enumerate(genders):
                for c, location in enumerate(locations):
                    ax = axes[r, c]
                    
                    sub = stratified_counts[
                        (stratified_counts['gender_label'] == gender) & 
                        (stratified_counts['location_label'] == location)
                    ]
                    
                    x = np.arange(len(age_bands))
                    width = 0.7
                    bottoms = np.zeros(len(age_bands))
                    
                    for risk in RISK_LABELS:
                        heights = []
                        for age in age_bands:
                            val = sub[(sub['age_band'] == age) & (sub['risk_cat'] == risk)]['percentage'].values
                            heights.append(val[0] if len(val) > 0 else 0)
                        heights = np.array(heights)
                        
                        ax.bar(x, heights, width, bottom=bottoms,
                              color=RISK_COLORS.get(risk, '#888888'),
                              edgecolor='white', linewidth=0.5)
                        bottoms += heights
                    
                    ax.set_title(f'{gender} - {location}', fontsize=12, fontweight='bold', fontfamily='serif')
                    ax.set_xticks(x)
                    ax.set_xticklabels(age_bands, rotation=45, ha='right', fontsize=9)
                    ax.set_ylim(0, 105)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    if c == 0:
                        ax.set_ylabel('Percentage (%)', fontsize=11, fontfamily='serif')
                    if r == 1:
                        ax.set_xlabel('Age Group', fontsize=11, fontfamily='serif')
            
            # Add legend
            risk_patches = [mpatches.Patch(facecolor=RISK_COLORS[c], edgecolor='black', label=c) for c in RISK_LABELS]
            fig_stratified.legend(handles=risk_patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                                 ncol=len(RISK_LABELS), frameon=True, fontsize=9, title='Risk Category')
            
            fig_stratified.suptitle('Stratified Risk Distribution\nRisk Categories by Age Group, split by Location and Gender', 
                                   fontsize=14, fontweight='bold', y=1.06, fontfamily='serif')
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            st.pyplot(fig_stratified)
            add_svg_download_button(fig_stratified, "stratified_age_gender_location", key="strat_full_svg")
            plt.close(fig_stratified)
            
            with st.expander("📊 How to Read This Chart"):
                st.markdown("""
                - **Facets**: The chart is divided into a grid. 
                    - **Columns**: Urban vs Rural
                    - **Rows**: Male vs Female
                - **X-Axis**: Age groups (ascending order).
                - **Y-Axis**: Percentage of individuals in each risk category (0-100%).
                - **Colors**: Indicate the CVD risk level (from <5% to ≥30%).
                - **Bars**: Each bar represents 100% of the population within that specific age/gender/location group. The stacking shows the proportion of each risk category.
                """)
        
        


        
        st.divider()
        st.subheader("Risk Factor Interactions")
        
        # Check for necessary columns
        col_diabetes = 'has_diabetes' if 'has_diabetes' in df_cohort.columns else 'diabetes'
        col_smoker = 'smoker_who' if 'smoker_who' in df_cohort.columns else 'smoker'
        has_lab_risk = 'risk_lab' in df_cohort.columns or 'risk_lab_cat' in df_cohort.columns
        
        c_p1, c_p2 = st.columns(2)
        
        # 1. Smoker vs Diabetes
        with c_p1:
            st.markdown("#### Smoker Distribution by Diabetes Status")
            if col_diabetes in df_cohort.columns and col_smoker in df_cohort.columns:
                # Prepare data for cleaner labels
                plot_df = df_cohort[[col_diabetes, col_smoker]].copy().dropna()
                
                # st.dataframe(plot_df[['has_diabetes', 'smoker_who']])
                # st.dataframe(plot_df['has_diabetes'].value_counts())
                # st.dataframe(plot_df['smoker_who'].value_counts())
                
                # Clean Diabetes Labels
                # Handle boolean or 0/1 or strings
                plot_df['Diabetes'] = plot_df[col_diabetes].astype(str).map({
                    '0': 'Non-Diabetic', '0.0': 'Non-Diabetic', 'False': 'Non-Diabetic', 'no_diabetes': 'Non-Diabetic', False: 'Non-Diabetic',
                    '1': 'Diabetic', '1.0': 'Diabetic', 'True': 'Diabetic', 'with_diabetes': 'Diabetic', True: 'Diabetic'
                }).fillna(plot_df[col_diabetes].astype(str))
                
                # Clean Smoker Labels
                plot_df['Smoker'] = plot_df[col_smoker].astype(str).map({
                    '0': 'Non-Smoker', 'No': 'Non-Smoker', 'Non-smoker': 'Non-Smoker', 'no_smoker': 'Non-Smoker',
                    '1': 'Smoker', 'Yes': 'Smoker', 'Smoker': 'Smoker', 'yes': 'Smoker'
                }).fillna(plot_df[col_smoker].astype(str))
                
                # Matplotlib grouped bar chart for Smoker vs Diabetes
                cross_tab = pd.crosstab(plot_df['Diabetes'], plot_df['Smoker'])
                
                fig_sd, ax_sd = plt.subplots(figsize=(8, 5))
                
                x = np.arange(len(cross_tab.index))
                width = 0.35
                
                smoker_cols = [c for c in cross_tab.columns]
                colors = ['#2ca02c', '#d62728']  # Green for Non-Smoker, Red for Smoker
                
                for i, col in enumerate(smoker_cols):
                    offset = (i - len(smoker_cols)/2 + 0.5) * width
                    bars = ax_sd.bar(x + offset, cross_tab[col], width, label=col,
                                    color=colors[i] if i < len(colors) else None,
                                    edgecolor='black', linewidth=0.5)
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax_sd.text(bar.get_x() + bar.get_width()/2, height + 5,
                                  f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontfamily='serif')
                
                ax_sd.set_ylabel('Count', fontsize=12, fontfamily='serif')
                ax_sd.set_xlabel('Diabetes Status', fontsize=12, fontfamily='serif')
                ax_sd.set_title('Smoker Status within Diabetes Groups', fontsize=13, fontweight='bold', fontfamily='serif')
                ax_sd.set_xticks(x)
                ax_sd.set_xticklabels(cross_tab.index, fontsize=10)
                ax_sd.legend(title='Smoker', frameon=True)
                ax_sd.spines['top'].set_visible(False)
                ax_sd.spines['right'].set_visible(False)
                ax_sd.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig_sd)
                add_svg_download_button(fig_sd, "smoker_vs_diabetes", key="smoker_diab_svg")
                plt.close(fig_sd)
            else:
                st.warning(f"Diabetes ({col_diabetes}) or Smoker ({col_smoker}) data missing.")

        # 2. Diabetes vs Lab Risk
        with c_p2:
            st.markdown("#### Diabetes Prevalence by Lab Risk Category")
            if col_diabetes in df_cohort.columns and has_lab_risk:
                # Need Lab Risk Categories. 
                plot_df2 = df_cohort.copy()
                
                if 'risk_lab_cat' in df_cohort.columns:
                    plot_df2['Lab Risk'] = df_cohort['risk_lab_cat']
                elif 'risk_lab' in df_cohort.columns:
                     plot_df2['Lab Risk'] = pd.cut(df_cohort['risk_lab'], bins=RISK_BINS, labels=RISK_LABELS, right=False)
                else:
                    plot_df2['Lab Risk'] = None
                
                # Reuse cleaned labels logic if possible, or just map again
                plot_df2['Diabetes'] = plot_df2[col_diabetes].astype(str).map({
                    '0': 'Non-Diabetic', '0.0': 'Non-Diabetic', 'False': 'Non-Diabetic', 'no_diabetes': 'Non-Diabetic',
                    '1': 'Diabetic', '1.0': 'Diabetic', 'True': 'Diabetic', 'with_diabetes': 'Diabetic'
                }).fillna(plot_df2[col_diabetes].astype(str))
                
                plot_df2 = plot_df2.dropna(subset=['Lab Risk', 'Diabetes'])
                
                # Plot: X=Lab Risk, Color=Diabetes - Matplotlib stacked bar
                cross_tab2 = pd.crosstab(plot_df2['Lab Risk'], plot_df2['Diabetes']).reindex(RISK_LABELS, fill_value=0)
                
                fig_ld, ax_ld = plt.subplots(figsize=(8, 5))
                
                x = np.arange(len(cross_tab2.index))
                width = 0.6
                
                diab_cols = [c for c in cross_tab2.columns]
                colors = ['#1f77b4', '#ff7f0e']  # Blue for Non-Diabetic, Orange for Diabetic
                
                bottoms = np.zeros(len(cross_tab2.index))
                for i, col in enumerate(diab_cols):
                    heights = cross_tab2[col].values
                    bars = ax_ld.bar(x, heights, width, bottom=bottoms, label=col,
                                    color=colors[i] if i < len(colors) else None,
                                    edgecolor='white', linewidth=0.5)
                    # Add value labels for significant values
                    for bar, h, b in zip(bars, heights, bottoms):
                        if h > 0:
                            ax_ld.text(bar.get_x() + bar.get_width()/2, b + h/2,
                                      f'{int(h):,}', ha='center', va='center', fontsize=8, 
                                      fontweight='bold', color='white', fontfamily='serif')
                    bottoms += heights
                
                ax_ld.set_ylabel('Count', fontsize=12, fontfamily='serif')
                ax_ld.set_xlabel('Lab Risk Category', fontsize=12, fontfamily='serif')
                ax_ld.set_title('Diabetes Status across Lab Risk Bands', fontsize=13, fontweight='bold', fontfamily='serif')
                ax_ld.set_xticks(x)
                ax_ld.set_xticklabels(cross_tab2.index, rotation=45, ha='right', fontsize=9)
                ax_ld.legend(title='Diabetes', frameon=True)
                ax_ld.spines['top'].set_visible(False)
                ax_ld.spines['right'].set_visible(False)
                ax_ld.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig_ld)
                add_svg_download_button(fig_ld, "diabetes_vs_lab_risk", key="diab_lab_risk_svg")
                plt.close(fig_ld)

            else:
                st.warning(f"Diabetes or Lab Risk data missing. (Cols: {df_cohort.columns.tolist()})")

        # ---------------------------------------------
        # 3. Advanced Comparative Analysis (New Request)
        # ---------------------------------------------
        if risk_model_opt == "Both (Compare)":
            st.divider()
            st.subheader("Advanced Risk Factor Analysis")
            
            t_adv1, t_adv2, t_adv3, t_adv4 = st.tabs([
                "🌊 Risk Score Distribution",
                "twisted_rightwards_arrows: Transition (Sankey)",
                "📈 Change Vectors",
                "🎯 Individual Concordance"
            ])
            
            # --- 1. Ridgeline / Violin (Distribution) ---
            with t_adv1:
                st.caption("Comparison of continuous risk score distributions (Non-Lab vs Lab) across Age Bands.")
                
                # Prepare Long Format Data for Plotly Violin
                # We need: AgeBand, Model, RiskScore
                # Helper to melt
                d_violin = df_cohort[['age_band', 'active_risk_nonlab', 'active_risk_lab']].copy()
                d_violin.columns = ['Age Band', 'Non-Lab', 'Lab']
                d_violin = d_violin.melt(id_vars='Age Band', var_name='Model', value_name='Risk Score')
                
                # Matplotlib/Seaborn Split Violin Plot
                fig_vio, ax_vio = plt.subplots(figsize=(12, 6))
                
                # Create split violin using seaborn
                palette = {'Non-Lab': NATURE_COLORS['blue'], 'Lab': NATURE_COLORS['orange']}
                
                sns.violinplot(
                    data=d_violin, 
                    x='Age Band', 
                    y='Risk Score', 
                    hue='Model',
                    split=True,
                    inner='quartile',
                    palette=palette,
                    ax=ax_vio
                )
                
                ax_vio.set_xlabel('Age Band', fontsize=12, fontfamily='serif')
                ax_vio.set_ylabel('Risk Score (%)', fontsize=12, fontfamily='serif')
                ax_vio.set_title('Risk Score Distribution by Age Band (Split Violin)', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_vio.spines['top'].set_visible(False)
                ax_vio.spines['right'].set_visible(False)
                ax_vio.grid(True, alpha=0.3, axis='y')
                ax_vio.legend(title='Model', frameon=True)
                
                # Rotate x-axis labels if many age bands
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig_vio)
                add_svg_download_button(fig_vio, "risk_score_violin", key="violin_svg")
                plt.close(fig_vio)
                
            # --- 2. Sankey Diagram (Transition) ---
            with t_adv2:
                st.caption("Flow of participants between Non-Lab and Lab risk categories.")
                
                # Count transitions
                transitions = df_cohort.groupby(['risk_cat_nonlab', 'risk_cat_lab'], observed=True).size().reset_index(name='count')
                
                # Create flow matrix
                flow_matrix = {}
                for _, row in transitions.iterrows():
                    src = row['risk_cat_nonlab']
                    tgt = row['risk_cat_lab']
                    if src in RISK_LABELS and tgt in RISK_LABELS:
                        if src not in flow_matrix:
                            flow_matrix[src] = {}
                        flow_matrix[src][tgt] = row['count']
                
                # Get left and right value counts
                left_values = df_cohort['risk_cat_nonlab'].value_counts().to_dict()
                right_values = df_cohort['risk_cat_lab'].value_counts().to_dict()
                
                # Use the create_sankey_alluvial helper from scienceplots_helpers
                fig_sankey = create_sankey_alluvial(
                    left_values=left_values,
                    right_values=right_values,
                    flow_matrix=flow_matrix,
                    left_labels=RISK_LABELS,
                    right_labels=RISK_LABELS,
                    left_title='Non-Lab',
                    right_title='Lab',
                    title='Risk Category Reclassification (Non-Lab → Lab)',
                    figsize=(12, 8),
                    left_color=NATURE_COLORS['purple'],
                    right_color=NATURE_COLORS['blue']
                )
                
                st.pyplot(fig_sankey)
                add_svg_download_button(fig_sankey, "sankey_reclassification", key="sankey_svg")
                plt.close(fig_sankey)
            
            # --- 3. Slopegraphs (Change Vectors) ---
            with t_adv3:
                st.caption("Change in mean risk score by Age Band.")
                
                # Calculate means
                means = df_cohort.groupby('age_band', observed=True)[['active_risk_nonlab', 'active_risk_lab']].mean().reset_index()
                
                # Matplotlib Slopegraph
                fig_slope, ax_slope = plt.subplots(figsize=(8, 6))
                
                for i, row in means.iterrows():
                    age = str(row['age_band'])
                    val_nl = row['active_risk_nonlab']
                    val_l = row['active_risk_lab']
                    
                    # Color based on direction (green = decrease, red = increase)
                    color = NATURE_COLORS['green'] if val_l < val_nl else NATURE_COLORS['red']
                    
                    ax_slope.plot([0, 1], [val_nl, val_l], marker='o', markersize=8, 
                                 linewidth=2, color=color)
                    
                    # Add left label
                    ax_slope.text(-0.05, val_nl, f'{age}: {val_nl:.1f}%', 
                                 ha='right', va='center', fontsize=9, fontfamily='serif')
                    
                    # Add right label
                    ax_slope.text(1.05, val_l, f'{age}: {val_l:.1f}%', 
                                 ha='left', va='center', fontsize=9, fontfamily='serif')
                
                ax_slope.set_xlim(-0.3, 1.3)
                ax_slope.set_xticks([0, 1])
                ax_slope.set_xticklabels(['Non-Lab', 'Lab'], fontsize=12, fontfamily='serif')
                ax_slope.set_ylabel('Mean Risk Score (%)', fontsize=12, fontfamily='serif')
                ax_slope.set_title('Mean Risk Score Trajectory by Age Band', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_slope.spines['top'].set_visible(False)
                ax_slope.spines['right'].set_visible(False)
                ax_slope.spines['bottom'].set_visible(False)
                ax_slope.grid(True, alpha=0.3, axis='y')
                
                # Add legend for direction
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color=NATURE_COLORS['green'], linewidth=2, label='Decrease (Lab < Non-Lab)'),
                    Line2D([0], [0], color=NATURE_COLORS['red'], linewidth=2, label='Increase (Lab ≥ Non-Lab)')
                ]
                ax_slope.legend(handles=legend_elements, loc='upper right', frameon=True)
                
                plt.tight_layout()
                st.pyplot(fig_slope)
                add_svg_download_button(fig_slope, "slopegraph_trajectories", key="slope_svg")
                plt.close(fig_slope)
            
            # --- 4. Density Overlay (Gold Standard) ---
            with t_adv4:
                st.caption("Individual-level concordance. Points on diagonal = perfect agreement.")
                
                # Sample data if too large
                plot_data = df_cohort if len(df_cohort) < 5000 else df_cohort.sample(5000)
                
                # Matplotlib Scatter + Density Contour
                fig_dens, ax_dens = plt.subplots(figsize=(8, 8))
                
                # Scatter plot with low opacity
                ax_dens.scatter(plot_data["active_risk_nonlab"], plot_data["active_risk_lab"],
                               s=10, c='black', alpha=0.1)
                
                # Add KDE contours using seaborn
                try:
                    sns.kdeplot(data=plot_data, x="active_risk_nonlab", y="active_risk_lab",
                               ax=ax_dens, levels=5, linewidths=1.5, 
                               color=NATURE_COLORS['blue'])
                except Exception:
                    pass  # Skip KDE if it fails (e.g., not enough data)
                
                # Add Diagonal Line (perfect agreement)
                max_val = max(plot_data["active_risk_nonlab"].max(), plot_data["active_risk_lab"].max())
                ax_dens.plot([0, max_val], [0, max_val], '--', color='gray', linewidth=2, label='Perfect Agreement')
                
                # Add Risk Threshold Lines (10%, 20%)
                for th in [10, 20]:
                    ax_dens.axvline(x=th, color=NATURE_COLORS['red'], linestyle=':', linewidth=1, alpha=0.7)
                    ax_dens.axhline(y=th, color=NATURE_COLORS['red'], linestyle=':', linewidth=1, alpha=0.7)
                
                ax_dens.set_xlabel('Non-Lab Risk (%)', fontsize=12, fontfamily='serif')
                ax_dens.set_ylabel('Lab Risk (%)', fontsize=12, fontfamily='serif')
                ax_dens.set_title('Risk Concordance Density', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_dens.set_xlim(0, max_val * 1.1)
                ax_dens.set_ylim(0, max_val * 1.1)
                ax_dens.spines['top'].set_visible(False)
                ax_dens.spines['right'].set_visible(False)
                ax_dens.grid(True, alpha=0.3)
                ax_dens.legend(loc='lower right', frameon=True)
                ax_dens.set_aspect('equal')
                
                plt.tight_layout()
                st.pyplot(fig_dens)
                add_svg_download_button(fig_dens, "concordance_density", key="dens_svg")
                plt.close(fig_dens)
            
            # ========================================================================
            # NEW: Model Discordance Deep Dive
            # ========================================================================
            st.divider()
            st.subheader("🔬 Model Discordance Deep Dive")
            st.caption("Mechanistic analysis of when and why Non-Lab underestimates risk compared to Lab")
            
            # Create derived metrics
            df_cohort['risk_diff'] = df_cohort['active_risk_nonlab'] - df_cohort['active_risk_lab']
            df_cohort['underestimates'] = (df_cohort['risk_diff'] < -2).astype(int)  # Non-Lab at least 2% lower
            df_cohort['missed_highrisk_20'] = ((df_cohort['active_risk_lab'] >= 20) & (df_cohort['active_risk_nonlab'] < 20)).astype(int)
            df_cohort['missed_highrisk_10'] = ((df_cohort['active_risk_lab'] >= 10) & (df_cohort['active_risk_nonlab'] < 10)).astype(int)
            
            # Check for measurement flags (if available)
            has_chol_measured = 'cholesterol_measured' in df_cohort.columns or 'has_cholesterol' in df_cohort.columns
            has_bg_measured = 'bg_mgdl_measured' in df_cohort.columns or 'has_bg' in df_cohort.columns
            
            # BMI bands (if BMI available)
            if 'bmi' in df_cohort.columns:
                df_cohort['bmi_band'] = pd.cut(df_cohort['bmi'], 
                                              bins=[0, 18.5, 25, 30, 100], 
                                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                              right=False)
            
            # SBP bands
            if 'sbp' in df_cohort.columns:
                df_cohort['sbp_band'] = pd.cut(df_cohort['sbp'],
                                              bins=[0, 120, 140, 160, 300],
                                              labels=['Normal', 'Elevated', 'Stage 1 HTN', 'Stage 2+ HTN'],
                                              right=False)
            
            t_disc1, t_disc2, t_disc3, t_disc4 = st.tabs([
                "⚠️ Underestimation Map",
                "🚨 Dangerous Discordance",
                "📏 Measurement Bias",
                "📊 Risk Difference Distributions"
            ])
            
            # ---- TAB: Underestimation Map ----
            with t_disc1:
                st.markdown("**When does Non-Lab underestimate?** (P(underestimate) by strata)")
                
                strata_vars = []
                if 'has_diabetes' in df_cohort.columns or col_diabetes in df_cohort.columns:
                    strata_vars.append(col_diabetes)
                if 'age_band' in df_cohort.columns:
                    strata_vars.append('age_band')
                if 'gender' in df_cohort.columns:
                    strata_vars.append('gender')
                if 'urban_rural' in df_cohort.columns:
                    strata_vars.append('urban_rural')
                if 'bmi_band' in df_cohort.columns:
                    strata_vars.append('bmi_band')
                if 'sbp_band' in df_cohort.columns:
                    strata_vars.append('sbp_band')
                
                if strata_vars:
                    # Compute underestimation rates
                    under_summary = []
                    for var in strata_vars:
                        grp = df_cohort.groupby(var, observed=True)['underestimates'].agg(['sum', 'count', 'mean'])
                        grp['rate_pct'] = grp['mean'] * 100
                        grp['stratum'] = var
                        grp['level'] = grp.index.astype(str)
                        under_summary.append(grp.reset_index(drop=True))
                    
                    under_df = pd.concat(under_summary, ignore_index=True)
                    under_df = under_df.sort_values('rate_pct', ascending=False)
                    
                    st.dataframe(under_df[['stratum', 'level', 'count', 'sum', 'rate_pct']].rename(columns={
                        'stratum': 'Stratification',
                        'level': 'Category',
                        'count': 'N',
                        'sum': 'Underestimations',
                        'rate_pct': 'Rate (%)'
                    }), use_container_width=True)
                    
                    # Heatmap of underestimation by Diabetes x Age
                    if 'age_band' in strata_vars and col_diabetes in strata_vars:
                        heat_data = df_cohort.groupby(['age_band', col_diabetes], observed=True)['underestimates'].mean().unstack() * 100
                        
                        # Matplotlib/Seaborn heatmap
                        fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
                        
                        sns.heatmap(heat_data.T, ax=ax_heat, cmap='Reds', 
                                   annot=True, fmt='.1f', linewidths=0.5, linecolor='white',
                                   cbar_kws={'label': 'Underestimation Rate (%)'})
                        
                        ax_heat.set_xlabel('Age Band', fontsize=12, fontfamily='serif')
                        ax_heat.set_ylabel('Diabetes Status', fontsize=12, fontfamily='serif')
                        ax_heat.set_title('Underestimation Rate Heatmap', fontsize=14, fontweight='bold', fontfamily='serif')
                        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_heat)
                        add_svg_download_button(fig_heat, "underestimation_heatmap", key="underest_heat_svg")
                        plt.close(fig_heat)
                else:
                    st.info("Insufficient stratification variables available.")
            
            # ---- TAB: Dangerous Discordance ----
            with t_disc2:
                st.markdown("**Missed High-Risk Cases** (Lab ≥20% but Non-Lab <20%)")
                
                miss_rate_20 = df_cohort['missed_highrisk_20'].mean() * 100
                total_lab_highrisk = (df_cohort['active_risk_lab'] >= 20).sum()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Miss Rate (≥20%)", f"{miss_rate_20:.1f}%")
                col_m2.metric("Total Lab High-Risk", f"{total_lab_highrisk:,}")
                col_m3.metric("Missed Count", f"{df_cohort['missed_highrisk_20'].sum():,}")
                
                # Stratified miss rates
                miss_strata = []
                for var in strata_vars:
                    grp = df_cohort[df_cohort['active_risk_lab'] >= 20].groupby(var, observed=True)['missed_highrisk_20'].agg(['sum', 'count', 'mean'])
                    grp['miss_rate_pct'] = grp['mean'] * 100
                    grp['stratum'] = var
                    grp['level'] = grp.index.astype(str)
                    miss_strata.append(grp.reset_index(drop=True))
                
                if miss_strata:
                    miss_df = pd.concat(miss_strata, ignore_index=True)
                    miss_df = miss_df.sort_values('miss_rate_pct', ascending=False)
                    
                    st.markdown("**Miss Rate by Stratum** (among Lab ≥20%)")
                    st.dataframe(miss_df[['stratum', 'level', 'count', 'sum', 'miss_rate_pct']].head(20).rename(columns={
                        'stratum': 'Stratification',
                        'level': 'Category',
                        'count': 'Lab High-Risk N',
                        'sum': 'Missed by Non-Lab',
                        'miss_rate_pct': 'Miss Rate (%)'
                    }), use_container_width=True)
                    
                    # Bar chart: Miss rate by setting - Matplotlib version
                    if 'urban_rural' in strata_vars:
                        miss_by_setting = df_cohort[df_cohort['active_risk_lab'] >= 20].groupby('urban_rural', observed=True)['missed_highrisk_20'].mean() * 100
                        
                        fig_miss, ax_miss = plt.subplots(figsize=(8, 5))
                        
                        x = np.arange(len(miss_by_setting))
                        bars = ax_miss.bar(x, miss_by_setting.values, 
                                          color=NATURE_COLORS['red'],
                                          edgecolor='black', linewidth=0.5)
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            ax_miss.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontfamily='serif')
                        
                        ax_miss.set_xticks(x)
                        ax_miss.set_xticklabels(miss_by_setting.index, fontsize=11)
                        ax_miss.set_xlabel('Setting', fontsize=12, fontfamily='serif')
                        ax_miss.set_ylabel('Miss Rate (%)', fontsize=12, fontfamily='serif')
                        ax_miss.set_title('High-Risk Miss Rate by Location Setting', fontsize=14, fontweight='bold', fontfamily='serif')
                        ax_miss.spines['top'].set_visible(False)
                        ax_miss.spines['right'].set_visible(False)
                        ax_miss.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_miss)
                        add_svg_download_button(fig_miss, "miss_rate_by_setting", key="miss_rate_svg")
                        plt.close(fig_miss)
            
            # ---- TAB: Measurement Bias ----
            with t_disc3:
                st.markdown("**Measurement Selection Bias Check**")
                st.caption("Do patients with lab measurements differ systematically?")
                
                if has_chol_measured or has_bg_measured:
                    bias_results = []
                    
                    # Determine measurement flags
                    chol_flag = 'cholesterol_measured' if 'cholesterol_measured' in df_cohort.columns else 'has_cholesterol'
                    bg_flag = 'bg_mgdl_measured' if 'bg_mgdl_measured' in df_cohort.columns else 'has_bg'
                    
                    for flag in [chol_flag, bg_flag]:
                        if flag not in df_cohort.columns:
                            continue
                        
                        st.markdown(f"**{flag} Comparison:**")
                        
                        # Compare measured vs not
                        comp_vars = ['age', 'gender', 'urban_rural']
                        if 'bmi' in df_cohort.columns:
                            comp_vars.append('bmi')
                        
                        comp_data = []
                        for var in comp_vars:
                            if var not in df_cohort.columns:
                                continue
                            
                            if var in ['age', 'bmi']:
                                # Continuous variable
                                measured = df_cohort[df_cohort[flag] == 1][var].mean()
                                not_measured = df_cohort[df_cohort[flag] == 0][var].mean()
                                comp_data.append({'Variable': var, 'Measured': f"{measured:.1f}", 'Not Measured': f"{not_measured:.1f}"})
                            else:
                                # Categorical - show distribution
                                meas_dist = df_cohort[df_cohort[flag] == 1][var].value_counts(normalize=True).to_dict()
                                not_meas_dist = df_cohort[df_cohort[flag] == 0][var].value_counts(normalize=True).to_dict()
                                
                                for cat in set(list(meas_dist.keys()) + list(not_meas_dist.keys())):
                                    comp_data.append({
                                        'Variable': f"{var}:{cat}",
                                        'Measured': f"{meas_dist.get(cat, 0)*100:.1f}%",
                                        'Not Measured': f"{not_meas_dist.get(cat, 0)*100:.1f}%"
                                    })
                        
                        if comp_data:
                            comp_df = pd.DataFrame(comp_data)
                            st.dataframe(comp_df, use_container_width=True)
                else:
                    st.info("No measurement flag columns detected in dataset.")
            
            # ---- TAB: Risk Difference Distributions ----
            with t_disc4:
                st.markdown("**Risk Difference Distributions** (Non-Lab minus Lab)")
                st.caption("Negative values = Non-Lab underestimates")
                
                # Overall distribution - Matplotlib histogram
                fig_diff_hist, ax_diff = plt.subplots(figsize=(10, 5))
                
                ax_diff.hist(df_cohort['risk_diff'].dropna(), bins=50, 
                            color=NATURE_COLORS['blue'], edgecolor='white', linewidth=0.5, alpha=0.8)
                ax_diff.axvline(x=0, color=NATURE_COLORS['red'], linestyle='--', linewidth=2, label='Zero (Perfect Agreement)')
                
                ax_diff.set_xlabel('Risk Diff (Non-Lab - Lab, %)', fontsize=12, fontfamily='serif')
                ax_diff.set_ylabel('Frequency', fontsize=12, fontfamily='serif')
                ax_diff.set_title('Overall Risk Difference Distribution', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_diff.spines['top'].set_visible(False)
                ax_diff.spines['right'].set_visible(False)
                ax_diff.grid(True, alpha=0.3, axis='y')
                ax_diff.legend(loc='upper right', frameon=True)
                
                plt.tight_layout()
                st.pyplot(fig_diff_hist)
                add_svg_download_button(fig_diff_hist, "risk_diff_histogram", key="diff_hist_svg")
                plt.close(fig_diff_hist)
                
                # Violin by Diabetes Status - Seaborn
                if col_diabetes in df_cohort.columns:
                    fig_diff_vio, ax_vio = plt.subplots(figsize=(8, 6))
                    
                    sns.violinplot(data=df_cohort, x=col_diabetes, y='risk_diff', 
                                  inner='box', palette=[NATURE_COLORS['blue'], NATURE_COLORS['orange']],
                                  ax=ax_vio)
                    ax_vio.axhline(y=0, color=NATURE_COLORS['red'], linestyle='--', linewidth=2)
                    
                    ax_vio.set_xlabel('Diabetes Status', fontsize=12, fontfamily='serif')
                    ax_vio.set_ylabel('Risk Diff (%)', fontsize=12, fontfamily='serif')
                    ax_vio.set_title('Risk Difference by Diabetes Status', fontsize=14, fontweight='bold', fontfamily='serif')
                    ax_vio.spines['top'].set_visible(False)
                    ax_vio.spines['right'].set_visible(False)
                    ax_vio.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig_diff_vio)
                    add_svg_download_button(fig_diff_vio, "risk_diff_violin", key="diff_vio_svg")
                    plt.close(fig_diff_vio)
                
                # Faceted by Location - Seaborn violin
                if 'urban_rural' in df_cohort.columns:
                    fig_diff_facet, ax_facet = plt.subplots(figsize=(10, 6))
                    
                    sns.violinplot(data=df_cohort, x='urban_rural', y='risk_diff',
                                  inner='box', palette='Set2', ax=ax_facet)
                    ax_facet.axhline(y=0, color=NATURE_COLORS['red'], linestyle='--', linewidth=2)
                    
                    ax_facet.set_xlabel('Setting', fontsize=12, fontfamily='serif')
                    ax_facet.set_ylabel('Risk Diff (%)', fontsize=12, fontfamily='serif')
                    ax_facet.set_title('Risk Difference by Location Setting', fontsize=14, fontweight='bold', fontfamily='serif')
                    ax_facet.spines['top'].set_visible(False)
                    ax_facet.spines['right'].set_visible(False)
                    ax_facet.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig_diff_facet)
                    add_svg_download_button(fig_diff_facet, "risk_diff_by_location", key="diff_loc_svg")
                    plt.close(fig_diff_facet)
                
                # Summary stats
                st.markdown("**Distribution Summary:**")
                diff_stats = df_cohort['risk_diff'].describe()
                st.write(diff_stats)
                
                # Identify extreme discordance cases
                extreme_under = df_cohort[df_cohort['risk_diff'] < -10].shape[0]
                extreme_over = df_cohort[df_cohort['risk_diff'] > 10].shape[0]
                
                st.metric("Extreme Underestimation (>10% lower)", f"{extreme_under:,} ({extreme_under/len(df_cohort)*100:.1f}%)")
                st.metric("Extreme Overestimation (>10% higher)", f"{extreme_over:,} ({extreme_over/len(df_cohort)*100:.1f}%)")


    # --- Tab 2a: Table 2 - CVD Risk Distribution ---
    with tab2a:
        # st.subheader("Table 2: Distribution of Participants by CVD Risk Score")
        # st.caption(f"WHO {active_label} 10-year risk categories (N={len(df_cohort):,})")

        # # Calculate distribution
        # risk_dist = df_cohort['risk_cat'].value_counts().reindex(RISK_LABELS, fill_value=0)
        # risk_pct = (risk_dist / len(df_cohort) * 100).round(1)

        # st.dataframe(df_cohort['risk_cat'].value_counts())
        # st.dataframe(df_cohort['risk_lab_cat'].value_counts())
        # st.dataframe(df_cohort['risk_nonlab_cat'].value_counts())


        # # Create table
        # table2_data = pd.DataFrame({
        #     'Risk Category': RISK_LABELS,
        #     'N': risk_dist.values,
        #     '%': risk_pct.values
        # })

        # # Add total row
        # total_row = pd.DataFrame({
        #     'Risk Category': ['Total'],
        #     'N': [len(df_cohort)],
        #     '%': [100.0]
        # })

        # table2 = pd.concat([table2_data, total_row], ignore_index=True)

        # # Add cumulative percentage
        # table2['Cumulative %'] = table2['%'].cumsum()

        # # Display styled table
        # st.dataframe(
        #     table2.style.format({
        #         'N': '{:,}',
        #         '%': '{:.1f}%',
        #         'Cumulative %': '{:.1f}%'
        #     }).background_gradient(cmap='Blues', subset=['N']),
        #     use_container_width=True
        # )

        st.subheader("Table 2: Distribution of Participants by CVD Risk Score")
        st.caption(f"WHO {active_label} risk categories (N={len(df_cohort):,})")

        # ---------------------------------------------------------------------
        # KEEP this for your existing plots/metrics (DO NOT BREAK downstream code)
        # table2_data = ACTIVE distribution (whatever df_cohort['risk_cat'] represents)
        # ---------------------------------------------------------------------
        risk_dist_active = df_cohort["risk_cat"].value_counts().reindex(RISK_LABELS, fill_value=0)
        risk_pct_active  = (risk_dist_active / len(df_cohort) * 100).round(1)

        table2_data = pd.DataFrame({
            "Risk Category": RISK_LABELS,
            "N": risk_dist_active.values,
            "%": risk_pct_active.values,
        })

        # ---------------------------------------------------------------------
        # NEW: Side-by-side Lab vs Non-Lab table (this is what you asked for)
        # ---------------------------------------------------------------------
        nonlab_dist = df_cohort["risk_nonlab_cat"].value_counts().reindex(RISK_LABELS, fill_value=0)
        lab_dist    = df_cohort["risk_lab_cat"].value_counts().reindex(RISK_LABELS, fill_value=0)

        nonlab_pct  = (nonlab_dist / len(df_cohort) * 100).round(1)
        lab_pct     = (lab_dist / len(df_cohort) * 100).round(1)

        table2 = pd.DataFrame({
            "Risk Category": RISK_LABELS,

            "Non-Lab N": nonlab_dist.values,
            "Non-Lab %": nonlab_pct.values,
            "Non-Lab Cumulative %": nonlab_pct.cumsum().values,

            "Lab N": lab_dist.values,
            "Lab %": lab_pct.values,
            "Lab Cumulative %": lab_pct.cumsum().values,
        })

        # Total row (optional but consistent)
        table2 = pd.concat([table2, pd.DataFrame([{
            "Risk Category": "Total",
            "Non-Lab N": int(nonlab_dist.sum()),
            "Non-Lab %": 100.0,
            "Non-Lab Cumulative %": 100.0,
            "Lab N": int(lab_dist.sum()),
            "Lab %": 100.0,
            "Lab Cumulative %": 100.0,
        }])], ignore_index=True)

        st.dataframe(
            table2.style.format({
                "Non-Lab N": "{:,}",
                "Lab N": "{:,}",
                "Non-Lab %": "{:.1f}%",
                "Lab %": "{:.1f}%",
                "Non-Lab Cumulative %": "{:.1f}%",
                "Lab Cumulative %": "{:.1f}%",
            }),
            use_container_width=True
        )




        # Visualization
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            # Bar chart with matplotlib
            fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
            
            x_pos = np.arange(len(table2_data['Risk Category']))
            colors = [RISK_COLORS.get(cat, '#cccccc') for cat in table2_data['Risk Category']]
            bars = ax_bar.bar(x_pos, table2_data['N'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, table2_data['N']):
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax_bar.set_xlabel('Risk Category', fontweight='bold')
            ax_bar.set_ylabel('Number of Participants', fontweight='bold')
            ax_bar.set_title('Absolute Count by Risk Category', fontweight='bold', fontsize=11, pad=10)
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(table2_data['Risk Category'], rotation=45, ha='right')
            ax_bar.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig_bar)

        with col_viz2:
            # Pie chart with matplotlib
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            
            colors = [RISK_COLORS.get(cat, '#cccccc') for cat in table2_data['Risk Category']]
            wedges, texts, autotexts = ax_pie.pie(
                table2_data['N'], 
                labels=table2_data['Risk Category'],
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 9, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
            )
            
            # Make percentage text white for better visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
            
            ax_pie.set_title('Percentage Distribution', fontweight='bold', fontsize=11, pad=10)
            
            plt.tight_layout()
            st.pyplot(fig_pie)

        # Summary statistics
        st.markdown("---")
        st.markdown("### Summary Statistics")

        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

        low_risk = table2_data[table2_data['Risk Category'] == '<5%']['N'].sum()
        moderate = table2_data[table2_data['Risk Category'].isin(['5% to <10%', '10% to <20%'])]['N'].sum()
        high_risk = table2_data[table2_data['Risk Category'].isin(['20% to <30%', '≥30%'])]['N'].sum()
        very_high = table2_data[table2_data['Risk Category'] == '≥30%']['N'].sum()

        col_sum1.metric("Low Risk (<5%)", f"{low_risk:,}", f"{low_risk/len(df_cohort)*100:.1f}%")
        col_sum2.metric("Moderate (5-<20%)", f"{moderate:,}", f"{moderate/len(df_cohort)*100:.1f}%")
        col_sum3.metric("High (≥20%)", f"{high_risk:,}", f"{high_risk/len(df_cohort)*100:.1f}%")
        col_sum4.metric("Very High (≥30%)", f"{very_high:,}", f"{very_high/len(df_cohort)*100:.1f}%")

    # --- Tab 2b: Table 3 - Risk by Gender ---
    with tab2b:
        st.subheader("Table 3: 10-Year CVD Risk by Gender")
        st.caption(f"WHO {active_label} laboratory-based charts (Fatal & Non-Fatal MI/Stroke)")

        if 'gender' not in df_cohort.columns:
            st.warning("⚠️ Gender variable not available in dataset")
        else:
            # Calculate risk distribution by gender
            gender_risk = pd.crosstab(
                df_cohort['gender'],
                df_cohort['risk_cat'],
                normalize='index'
            ) * 100

            # Reorder columns
            gender_risk = gender_risk.reindex(columns=RISK_LABELS, fill_value=0)

            # Add counts
            gender_counts = pd.crosstab(
                df_cohort['gender'],
                df_cohort['risk_cat']
            ).reindex(columns=RISK_LABELS, fill_value=0)

            # Create combined table with N (%)
            table3_data = []

            for gender in gender_risk.index:
                row = {'Gender': gender, 'Total N': len(df_cohort[df_cohort['gender'] == gender])}

                for risk_cat in RISK_LABELS:
                    n = gender_counts.loc[gender, risk_cat] if risk_cat in gender_counts.columns else 0
                    pct = gender_risk.loc[gender, risk_cat] if risk_cat in gender_risk.columns else 0
                    row[risk_cat] = f"{int(n)} ({pct:.1f}%)"

                table3_data.append(row)

            table3 = pd.DataFrame(table3_data)

            # Display table
            st.dataframe(table3, use_container_width=True)

            # Stratified visualization
            st.markdown("---")
            st.markdown("### Risk Distribution by Gender")

            st.info(gender_risk)
            # Prepare data for grouped bar chart
            plot_data = []
            for gender in gender_risk.index:
                for risk_cat in RISK_LABELS:
                    pct = gender_risk.loc[gender, risk_cat] if risk_cat in gender_risk.columns else 0
                    plot_data.append({
                        'Gender': gender,
                        'Risk Category': risk_cat,
                        'Percentage': pct
                    })

            plot_df = pd.DataFrame(plot_data)

            # Matplotlib Grouped bar chart
            fig_gender, ax_gender = plt.subplots(figsize=(10, 6))
            
            genders = plot_df['Gender'].unique()
            n_genders = len(genders)
            x = np.arange(len(RISK_LABELS))
            width = 0.8 / n_genders
            
            gender_colors = [NATURE_COLORS['blue'], NATURE_COLORS['red'], NATURE_COLORS['green']]
            
            for i, gender in enumerate(genders):
                gen_data = plot_df[plot_df['Gender'] == gender]
                # Reindex to ensure all risk labels are present
                pcts = []
                for risk in RISK_LABELS:
                    val = gen_data[gen_data['Risk Category'] == risk]['Percentage'].values
                    pcts.append(val[0] if len(val) > 0 else 0)
                
                offset = (i - n_genders/2 + 0.5) * width
                bars = ax_gender.bar(x + offset, pcts, width, label=gender,
                                    color=gender_colors[i] if i < len(gender_colors) else None,
                                    edgecolor='black', linewidth=0.5)
            
            ax_gender.set_ylabel('Percentage (%)', fontsize=12, fontfamily='serif')
            ax_gender.set_xlabel('Risk Category', fontsize=12, fontfamily='serif')
            ax_gender.set_title('CVD Risk Distribution Stratified by Gender', fontsize=14, fontweight='bold', fontfamily='serif')
            ax_gender.set_xticks(x)
            ax_gender.set_xticklabels(RISK_LABELS, rotation=45, ha='right', fontsize=9)
            ax_gender.legend(title='Gender', frameon=True)
            ax_gender.spines['top'].set_visible(False)
            ax_gender.spines['right'].set_visible(False)
            ax_gender.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig_gender)
            add_svg_download_button(fig_gender, "risk_by_gender", key="gender_risk_svg")
            plt.close(fig_gender)

            # Statistical comparison
            st.markdown("---")
            st.markdown("### Gender-Specific Summary")

            # Calculate high-risk rates by gender
            gender_high_risk = df_cohort.groupby('gender')['high_risk'].agg(['sum', 'count', 'mean'])

            cols_gender = st.columns(len(gender_high_risk))

            for idx, (gender, row) in enumerate(gender_high_risk.iterrows()):
                with cols_gender[idx]:
                    st.metric(
                        f"{gender}",
                        f"{int(row['sum']):,} / {int(row['count']):,}",
                        f"{row['mean']*100:.1f}% high-risk"
                    )

            # Chi-square test if more than one gender
            if len(gender_high_risk) > 1:
                from scipy.stats import chi2_contingency

                contingency = pd.crosstab(df_cohort['gender'], df_cohort['high_risk'])
                chi2, pval, dof, expected = chi2_contingency(contingency)

                st.info(f"📊 **Chi-Square Test:** χ² = {chi2:.2f}, p = {pval:.4f} ({'Significant' if pval < 0.05 else 'Not significant'} at α=0.05)")

    # --- Tab 3: Burden Estimation ---
    with tab3:
        st.subheader(f"{risk_threshold_opt} Prevalence Estimates")

        def render_prev_table(dim_col):
            if dim_col not in df_cohort.columns: return

            stats = df_cohort.groupby(dim_col).agg(
                N=('high_risk', 'count'),
                Cases=('high_risk', 'sum')
            ).reset_index()

            # CI
            stats[['CI_Lower', 'CI_Upper']] = stats.apply(
                lambda x: pd.Series(calculate_wilson_ci(x['Cases'], x['N'])), axis=1
            )
            stats['Prevalence (%)'] = (stats['Cases'] / stats['N'] * 100)

            st.markdown(f"**Stratified by {dim_col}**")
            st.dataframe(stats.style.format({
                'Prevalence (%)': '{:.1f}',
                'CI_Lower': '{:.1f}',
                'CI_Upper': '{:.1f}'
            }), use_container_width=True)

        c_a, c_b = st.columns(2)
        with c_a: render_prev_table('urban_rural')
        with c_b: render_prev_table('gender')

        render_prev_table('age_band')

    # --- Tab 4: Site Ranking ---
    with tab4:
        st.subheader(f"Site Ranking & Meta-Analysis")

        # Decide Name column - fallback to project_title if site_title is null
        if 'site_title' in df_cohort.columns and 'project_title' in df_cohort.columns:
            df_cohort['_name'] = df_cohort['site_title'].combine_first(df_cohort['project_title'])
        elif 'site_title' in df_cohort.columns:
            df_cohort['_name'] = df_cohort['site_title']
        elif 'project_title' in df_cohort.columns:
            df_cohort['_name'] = df_cohort['project_title']
        else:
            df_cohort['_name'] = df_cohort['site_id']

        name_col = '_name'

        site_agg = df_cohort.groupby('site_id').agg(
            N=('high_risk', 'count'),
            Cases=('high_risk', 'sum'),
            label=(name_col, 'first')
        ).reset_index()

        # Filter - Set Default to 50 as requested
        min_n = st.number_input("Minimum Patients per Site", 10, 500, 50, step=10)
        site_agg = site_agg[site_agg['N'] >= min_n].copy()

        if site_agg.empty:
            st.warning("No sites meet the criteria.")
        else:
            # --- 1) Advanced Metrics Calculation ---
            # A) Standard Wilson
            site_agg[['CI_Low', 'CI_High']] = site_agg.apply(
                lambda x: pd.Series(calculate_wilson_ci(x['Cases'], x['N'])), axis=1
            )
            site_agg['Prev'] = site_agg['Cases'] / site_agg['N'] * 100

            # B) Age Standardization - Pass risk_col explicity
            std_res = calculate_standardized_prevalence(df_cohort, 'site_id', risk_col='high_risk')
            site_agg = site_agg.merge(std_res[['site_id', 'Age-Std (%)']], on='site_id', how='left')

            # C) Bayesian POoling
            site_agg = calculate_empirical_bayes_shrinkage(site_agg)

            # --- 2) User Controls ---
            c_mode = st.radio("Metric to Display", ["Crude Prevalence", "Age-Standardized", "Bayesian Smoothed"], horizontal=True)

            # Determine Y and X
            y_col = 'Prev'
            ci_lo = 'CI_Low'
            ci_hi = 'CI_High'
            title_prefix = "Crude"

            if c_mode == "Age-Standardized":
                y_col = 'Age-Std (%)'
                ci_lo, ci_hi = None, None
                title_prefix = "Age-Standardized"
            elif c_mode == "Bayesian Smoothed":
                y_col = 'Eb_Prev'
                ci_lo = 'Eb_Low'
                ci_hi = 'Eb_High'
                title_prefix = "Bayesian Smoothed (Empirical Bayes)"

            site_agg = site_agg.sort_values(y_col, ascending=True)

            # --- 3) Matplotlib Dot Plot with Error Bars ---
            fig_dot, ax_dot = plt.subplots(figsize=(10, max(6, len(site_agg) * 0.3)))
            
            labels = site_agg['label'].fillna(site_agg['site_id'].astype(str)).values
            y_pos = np.arange(len(labels))
            x_vals = site_agg[y_col].values
            
            # Error bars (if available)
            if ci_lo and ci_hi and ci_lo in site_agg.columns and ci_hi in site_agg.columns:
                xerr_lower = site_agg[y_col].values - site_agg[ci_lo].values
                xerr_upper = site_agg[ci_hi].values - site_agg[y_col].values
                xerr = [xerr_lower, xerr_upper]
            else:
                xerr = None
            
            ax_dot.errorbar(x_vals, y_pos, xerr=xerr, fmt='o', 
                           color=NATURE_COLORS['green'], markersize=8,
                           ecolor='gray', elinewidth=1, capsize=3)
            
            # Add median line
            median_val = site_agg[y_col].median()
            ax_dot.axvline(x=median_val, color=NATURE_COLORS['red'], linestyle=':', linewidth=2)
            ax_dot.text(median_val, len(labels) + 0.5, f'Median: {median_val:.1f}%', 
                       ha='center', fontsize=10, color=NATURE_COLORS['red'], fontfamily='serif')
            
            ax_dot.set_yticks(y_pos)
            ax_dot.set_yticklabels(labels, fontsize=9)
            ax_dot.set_xlabel('Prevalence (%)', fontsize=12, fontfamily='serif')
            ax_dot.set_title(f'{title_prefix} Prevalence ({target_label}) per Site (N≥{min_n})', 
                           fontsize=14, fontweight='bold', fontfamily='serif')
            ax_dot.spines['top'].set_visible(False)
            ax_dot.spines['right'].set_visible(False)
            ax_dot.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            st.pyplot(fig_dot)
            add_svg_download_button(fig_dot, "site_prevalence_dotplot", key="site_dot_svg")
            plt.close(fig_dot)

            # --- 4) Meta-Analysis ---
            st.markdown("---")
            st.subheader(f"Random-Effects Meta-Analysis ({target_label})")
            ma_res = calculate_meta_analysis(site_agg)
            if ma_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Pooled Prev", f"{ma_res['pooled_prev']:.1f}%", f"95% CI: {ma_res['ci_lower']:.1f}-{ma_res['ci_upper']:.1f}")
                c2.metric("Heterogeneity (I²)", f"{ma_res['I2']:.1f}%", f"p(Q) = {ma_res['p_val_Q']:.3f}")
                c3.metric("Sites", len(site_agg))

                st.info(f"Using {target_label} cutoff. I² = {ma_res['I2']:.1f}%.")

    with tab5:
        st.subheader("Advanced Statistical Modeling")

        model_type = st.selectbox("Select Method", [
            "Cluster-Robust Logistic Regression (Binary Outcome)",
            "Quantile Regression (Continuous Risk Score)"
        ])

        if model_type == "Cluster-Robust Logistic Regression (Binary Outcome)":
            st.markdown(f"""
            **Method:** Logistic Regression of `{risk_threshold_opt}` ~ Age + Gender + Urban/Rural.
            **Standard Errors:** Clustered by `site_id`.
            """)

            if target_risk_col == 'high_risk_20':
                st.warning("⚠️ Warning: Modeling events for ≥20% risk may be unstable due to low event counts. ≥10% is recommended.")

            if st.button(f"Run GLM Model ({target_label})"):
                with st.spinner("Fitting model..."):
                    # Use the active 'high_risk' column which matches the selection
                    res = run_glm_model(df_cohort, outcome_col='high_risk')
                    if res:
                        st.success("Model Converged.")

                        # Extract Odds Ratios
                        params = res.params
                        conf = res.conf_int()
                        conf['OR'] = np.exp(params)
                        conf['Lower'] = np.exp(conf[0])
                        conf['Upper'] = np.exp(conf[1])
                        conf['P-value'] = res.pvalues

                        results_df = conf[['OR', 'Lower', 'Upper', 'P-value']]
                        st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

                        # Matplotlib Forest Plot
                        plot_df = results_df.drop('Intercept', errors='ignore')
                        
                        fig_forest, ax_forest = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.6)))
                        
                        y_pos = np.arange(len(plot_df))
                        
                        # Error bars
                        xerr_lower = plot_df['OR'].values - plot_df['Lower'].values
                        xerr_upper = plot_df['Upper'].values - plot_df['OR'].values
                        
                        ax_forest.errorbar(plot_df['OR'].values, y_pos, 
                                          xerr=[xerr_lower, xerr_upper],
                                          fmt='o', color='black', markersize=10,
                                          ecolor='gray', elinewidth=1.5, capsize=4)
                        
                        # Add vertical line at OR=1
                        ax_forest.axvline(x=1, color=NATURE_COLORS['red'], linestyle='--', linewidth=2)
                        
                        ax_forest.set_yticks(y_pos)
                        ax_forest.set_yticklabels(plot_df.index, fontsize=10)
                        ax_forest.set_xscale('log')
                        ax_forest.set_xlabel('Odds Ratio (Log Scale)', fontsize=12, fontfamily='serif')
                        ax_forest.set_title(f'Adjusted Odds Ratios ({target_label})', 
                                           fontsize=14, fontweight='bold', fontfamily='serif')
                        ax_forest.spines['top'].set_visible(False)
                        ax_forest.spines['right'].set_visible(False)
                        ax_forest.grid(True, alpha=0.3, axis='x')
                        
                        plt.tight_layout()
                        st.pyplot(fig_forest)
                        add_svg_download_button(fig_forest, "forest_plot_or", key="forest_or_svg")
                        plt.close(fig_forest)

                        with st.expander("Full Model Summary"):
                            st.text(res.summary())

        elif model_type == "Quantile Regression (Continuous Risk Score)":
            st.markdown("""
            **Method:** Quantile Regression estimating the effect of covariates on a specific *quantile* of the risk distribution (e.g., Median Risk).
            Useful for understanding shifts in the distribution tails.
            """)

            q = st.slider("Quantile (tau)", 0.1, 0.95, 0.5, 0.05)

            if st.button(f"Run Quantile Regression (tau={q})"):
                with st.spinner("Fitting model..."):
                    res = run_quantile_regression(df_cohort, quantile=q)
                    if isinstance(res, str): # Error
                        st.error(f"Failed: {res}")
                    else:
                        st.success("Model Converged.")
                        st.text(res.summary())

                        # Plot coefficients
                        params = res.params
                        conf = res.conf_int()
                        conf.columns = ['Lower', 'Upper']
                        conf['Coef'] = params

                        st.subheader("Coefficients")
                        st.dataframe(conf.style.format("{:.3f}"), use_container_width=True)

                        # Matplotlib Coefficient Plot
                        plot_df = conf.drop('Intercept', errors='ignore')
                        
                        fig_q, ax_q = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.6)))
                        
                        y_pos = np.arange(len(plot_df))
                        
                        # Error bars
                        xerr_lower = plot_df['Coef'].values - plot_df['Lower'].values
                        xerr_upper = plot_df['Upper'].values - plot_df['Coef'].values
                        
                        ax_q.errorbar(plot_df['Coef'].values, y_pos, 
                                     xerr=[xerr_lower, xerr_upper],
                                     fmt='o', color=NATURE_COLORS['blue'], markersize=10,
                                     ecolor='gray', elinewidth=1.5, capsize=4)
                        
                        # Add vertical line at 0
                        ax_q.axvline(x=0, color=NATURE_COLORS['red'], linestyle='--', linewidth=2)
                        
                        ax_q.set_yticks(y_pos)
                        ax_q.set_yticklabels(plot_df.index, fontsize=10)
                        ax_q.set_xlabel('Coefficient Value (Change in Risk %)', fontsize=12, fontfamily='serif')
                        ax_q.set_title(f'Regression Coefficients (Quantile: {q})', 
                                      fontsize=14, fontweight='bold', fontfamily='serif')
                        ax_q.spines['top'].set_visible(False)
                        ax_q.spines['right'].set_visible(False)
                        ax_q.grid(True, alpha=0.3, axis='x')
                        
                        plt.tight_layout()
                        st.pyplot(fig_q)
                        add_svg_download_button(fig_q, "quantile_coef_plot", key="quantile_svg")
                        plt.close(fig_q)

    # =============================================================================
    # TAB 8: PAIRED DATA ANALYSIS (Lab vs Non-Lab Comparison)
    # =============================================================================
    # This tab contains comprehensive analyses comparing lab-based and non-lab-based
    # WHO CVD risk assessments using paired data. Sections are organized as:
    #   0. Agreement Flow (Sankey Diagram)
    #   1. Risk Distribution by Age Group
    #   2. Age-Stratified Risk Trends
    #   3. Agreement & Discordance Analysis
    #   4. Risk Prevalence Thresholds (1-30%)
    #   5. Stratified Analysis (Gender, BMI, BP, Diabetes, Smoking)
    #   6. Multi-Variable Heatmap
    #   7. TABLE: Participant Characteristics by Sex
    #   8. TABLE: CVD Risk by Glycaemic Status
    # =============================================================================
    with tab_paired:
        st.title("🔬 Lab vs Non-Lab Risk Comparison")
        st.markdown("""
        **Comprehensive validation of non-laboratory-based CVD risk assessment against laboratory-based reference standard.**
        
        This analysis uses **paired data** where both lab-based and non-lab-based risk scores are available for the same individuals.
        """)
        
        st.info(f"Total merged records having diabetes: {len(df_merged[df_merged['has_diabetes'] == 1])} out of {len(df_merged)} records")
        # Load paired data from datasets if available
        if datasets and 'paired' in datasets and datasets['paired'] is not None:
            df_paired_full = datasets['paired']

            # Filter for eligible paired data
            if 'eligible_paired' in df_paired_full.columns:
                df_paired = df_paired_full[df_paired_full['eligible_paired']].reset_index(drop=True)
                st.success(f"✅ Using {len(df_paired):,} paired records with both lab and non-lab assessments")
            else:
                df_paired = df_paired_full.copy()
                st.warning("⚠️ 'eligible_paired' column not found, using all records")
            
            st.info(f"Total paired records having diabetes: {len(df_paired[df_paired['has_diabetes'] == 1])} out of {len(df_paired)} records")
            # Ensure required columns exist
            if 'age' not in df_paired.columns or 'risk_lab' not in df_paired.columns or 'risk_nonlab' not in df_paired.columns:
                st.error("❌ Required columns (age, risk_lab, risk_nonlab) not found in paired dataset")
            else:
                # Filter age 40-74
                df_paired = df_paired[(df_paired['age'] >= 40) & (df_paired['age'] <= 74)].copy()
                
                # Create age bands if not present
                if 'age_band' not in df_paired.columns or df_paired['age_band'].nunique() < 2:
                    bins = [40, 45, 50, 55, 60, 65, 70, 75]
                    labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                    df_paired['age_band'] = pd.cut(df_paired['age'], bins=bins, labels=labels, right=False)
                
                # Create risk categories for both lab and non-lab
                RISK_BINS = [-np.inf, 5, 10, 20, 30, np.inf]
                RISK_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
                RISK_COLORS = {
                    '<5%': '#2E7D32',
                    '5% to <10%': '#FDD835',
                    '10% to <20%': '#FB8C00',
                    '20% to <30%': '#E53935',
                    '≥30%': '#8E24AA'
                }
                
                df_paired['risk_cat_lab'] = pd.cut(
                    df_paired['risk_lab'],
                    bins=RISK_BINS,
                    labels=RISK_LABELS,
                    right=False
                )
                
                df_paired['risk_cat_nonlab'] = pd.cut(
                    df_paired['risk_nonlab'],
                    bins=RISK_BINS,
                    labels=RISK_LABELS,
                    right=False
                )

                # --- Derive Glycaemic Groups for Stratification ---
                df_paired['gly_group'] = df_paired.apply(categorize_glycaemic_status, axis=1)


                # ==========================================
                # PHASE 1 FEATURE ENGINEERING (Enhanced Analysis)
                # ==========================================
                
                # --- 1. Hemodynamic Features ---
                # Pulse Pressure: Strong predictor of arterial stiffness
                if 'sbp' in df_paired.columns and 'dbp' in df_paired.columns:
                    df_paired['pulse_pressure'] = df_paired['sbp'] - df_paired['dbp']
                    # Mean Arterial Pressure: Organ perfusion indicator
                    df_paired['map'] = df_paired['dbp'] + (df_paired['sbp'] - df_paired['dbp']) / 3
                    # PP bands for stratification
                    df_paired['pp_band'] = pd.cut(
                        df_paired['pulse_pressure'],
                        bins=[0, 40, 50, 60, 200],
                        labels=['<40', '40-49', '50-59', '≥60'],
                        right=False
                    )
                
                # --- 2. Central Obesity Features ---
                # Waist-to-Height Ratio (better discriminator than BMI)
                if 'waist' in df_paired.columns and 'height' in df_paired.columns:
                    df_paired['whtr'] = df_paired['waist'] / df_paired['height']
                    # Central obesity flag: WHtR ≥ 0.5 indicates elevated cardiometabolic risk
                    df_paired['central_obesity'] = df_paired['whtr'] >= 0.5
                    # WHtR bands
                    df_paired['whtr_band'] = pd.cut(
                        df_paired['whtr'],
                        bins=[0, 0.4, 0.5, 0.6, 10],
                        labels=['<0.4 (Slim)', '0.4-0.5 (Healthy)', '0.5-0.6 (Elevated)', '≥0.6 (High)'],
                        right=False
                    )
                
                # --- 3. Temporal/Longitudinal Features ---
                if 'pid' in df_paired.columns and 'date' in df_paired.columns:
                    # Convert date to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df_paired['date']):
                        df_paired['date'] = pd.to_datetime(df_paired['date'], errors='coerce')
                    
                    # Sort by patient and date for correct sequencing
                    df_paired = df_paired.sort_values(['pid', 'date']).reset_index(drop=True)
                    
                    # Visit sequence number per patient
                    df_paired['visit_num'] = df_paired.groupby('pid').cumcount() + 1
                    
                    # Total visits per patient
                    df_paired['total_visits'] = df_paired.groupby('pid')['visit_num'].transform('max')
                    
                    # Days since first visit for this patient
                    df_paired['first_visit_date'] = df_paired.groupby('pid')['date'].transform('min')
                    df_paired['days_since_first'] = (df_paired['date'] - df_paired['first_visit_date']).dt.days
                    
                    # Days since previous visit
                    df_paired['prev_visit_date'] = df_paired.groupby('pid')['date'].shift(1)
                    df_paired['days_since_last'] = (df_paired['date'] - df_paired['prev_visit_date']).dt.days
                    
                    # Year of visit (for trend analysis)
                    df_paired['visit_year'] = df_paired['date'].dt.year
                    
                    # Clean up temp columns
                    df_paired.drop(['first_visit_date', 'prev_visit_date'], axis=1, inplace=True, errors='ignore')
                
                # --- 4. Risk Reclassification Features ---
                # Risk category indices for calculating shift magnitude
                risk_cat_order = {cat: i for i, cat in enumerate(RISK_LABELS)}
                df_paired['risk_idx_lab'] = df_paired['risk_cat_lab'].map(risk_cat_order).astype(float)
                df_paired['risk_idx_nonlab'] = df_paired['risk_cat_nonlab'].map(risk_cat_order).astype(float)
                
                # Category shift: Positive = NonLab overestimates, Negative = NonLab underestimates
                df_paired['risk_cat_shift'] = df_paired['risk_idx_nonlab'] - df_paired['risk_idx_lab']
                
                # Agreement status (already exists but ensure consistency)
                df_paired['agree_status'] = df_paired['risk_cat_shift'].apply(
                    lambda x: 'Concordance' if x == 0 else ('Underestimation' if x < 0 else 'Overestimation')
                )
                
                # Magnitude of disagreement (absolute shift)
                df_paired['disagree_magnitude'] = df_paired['risk_cat_shift'].abs()
                
                # --- 5. Treatment Misclassification Flags (Clinical Actionability) ---
                # These capture scenarios where non-lab method leads to WRONG treatment decisions
                
                # At 10% threshold (common statin initiation threshold)
                df_paired['lab_ge10'] = df_paired['risk_lab'] >= 10
                df_paired['nonlab_ge10'] = df_paired['risk_nonlab'] >= 10
                df_paired['misclass_10_under'] = df_paired['lab_ge10'] & ~df_paired['nonlab_ge10']  # Missed statin candidates
                df_paired['misclass_10_over'] = ~df_paired['lab_ge10'] & df_paired['nonlab_ge10']  # Unnecessary statin candidates
                
                # At 20% threshold (high-risk, intensive treatment threshold)
                df_paired['lab_ge20'] = df_paired['risk_lab'] >= 20
                df_paired['nonlab_ge20'] = df_paired['risk_nonlab'] >= 20
                df_paired['misclass_20_under'] = df_paired['lab_ge20'] & ~df_paired['nonlab_ge20']  # Missed high-risk
                df_paired['misclass_20_over'] = ~df_paired['lab_ge20'] & df_paired['nonlab_ge20']  # Over-treated
                
                # Severe misclassification: Lab ≥20% but Non-Lab <10% (two category jump)
                df_paired['severe_underest'] = df_paired['lab_ge20'] & (df_paired['risk_nonlab'] < 10)
                
                # --- 6. Partial Metabolic Syndrome Score ---
                # Using available components only (3 of 5 criteria)
                mets_score = 0
                if 'sbp' in df_paired.columns and 'dbp' in df_paired.columns:
                    df_paired['mets_bp'] = ((df_paired['sbp'] >= 130) | (df_paired['dbp'] >= 85)).astype(int)
                    mets_score = df_paired['mets_bp']
                
                if 'bg_mgdl' in df_paired.columns:
                    # Using 100 mg/dL as threshold (IFG criterion)
                    df_paired['mets_glucose'] = (df_paired['bg_mgdl'] >= 100).astype(int)
                    mets_score = mets_score + df_paired['mets_glucose']
                
                if 'waist' in df_paired.columns and 'gender' in df_paired.columns:
                    # Gender-specific waist thresholds
                    df_paired['mets_waist'] = (
                        ((df_paired['gender'].isin(['M', 'Male', 'men'])) & (df_paired['waist'] > 102)) |
                        ((df_paired['gender'].isin(['F', 'Female', 'women'])) & (df_paired['waist'] > 88))
                    ).astype(int)
                    mets_score = mets_score + df_paired['mets_waist']
                
                df_paired['mets_partial_score'] = mets_score
                df_paired['mets_partial_positive'] = df_paired['mets_partial_score'] >= 2  # 2 of 3 available criteria
                
                # Log feature engineering summary
                st.caption(f"📊 Enhanced features: PP, MAP, WHtR, visit sequences, treatment misclassification flags, partial MetS score")

                # ==========================================
                # SECTION 8.1: AGREEMENT FLOW (SANKEY - MATPLOTLIB SCIENCEPLOT)
                # ==========================================
                st.header("8.1 📊 Agreement Flow: Lab vs Non-Lab Risk Categories")
                st.markdown("Visualizing the flow of participants between **Lab-based** (Reference) and **Non-lab-based** risk categories.")
                
                def plot_risk_sankey(data, title_suffix=""):
                    """
                    Matplotlib implementation of Sankey (Alluvial) diagram for SciencePlots compatibility.
                    Ensures vector-quality fonts and labels.
                    """
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    from matplotlib.path import Path
                    import matplotlib.colors as mcolors
                    
                    # Configuration
                    CATS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
                    
                    # Colors (SciencePlots / Nature)
                    COLOR_LAB_BAR = "#0C5DA5"      # Blue
                    COLOR_NONLAB_BAR = "#845B97"   # Purple
                    
                    # Flow Colors
                    COLOR_CONCORDANT = "#C8C8C8"  # Grey
                    COLOR_UNDER = "#FF2C00"       # Red
                    COLOR_OVER = "#FF9500"        # Orange
                    ALPHA_RIBBON = 0.6
                    
                    # Data Preparation
                    left_counts = data['risk_cat_lab'].value_counts().reindex(CATS, fill_value=0)
                    right_counts = data['risk_cat_nonlab'].value_counts().reindex(CATS, fill_value=0)
                    
                    # Crosstab for flows
                    flows = pd.crosstab(
                        data['risk_cat_lab'], 
                        data['risk_cat_nonlab']
                    ).reindex(index=CATS, columns=CATS, fill_value=0)
                    
                    # Layout Parameters
                    GAP_PROPORTION = 0.05
                    total_n = len(data)
                    gap_height = total_n * GAP_PROPORTION
                    
                    # Calculate Y positions
                    # Coordinate system: 0 at bottom, Total height at top
                    # We stack from Top to Bottom to match standard visual order (<5% at top usually?) 
                    # Actually standard Sankey usually puts largest at bottom or preserves order.
                    # Let's preserve index order: <5% at Top.
                    
                    # Calculate total drawing height including gaps
                    total_drawing_height = total_n + (len(CATS) - 1) * gap_height
                    
                    # Bar Positions
                    # Left Bars
                    y_left_start = {}
                    current_y = total_drawing_height
                    for cat in CATS:
                        h = left_counts[cat]
                        y_left_start[cat] = current_y - h
                        current_y -= (h + gap_height)
                        
                    # Right Bars
                    y_right_start = {}
                    current_y = total_drawing_height
                    for cat in CATS:
                        h = right_counts[cat]
                        y_right_start[cat] = current_y - h
                        current_y -= (h + gap_height)
                        
                    # Initialize Plot
                    # Use SciencePlots-like context manually to ensure consistency
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman'],
                        'font.size': 12,
                        'axes.titlesize': 14,
                        'axes.labelsize': 12,
                        'text.color': 'black',
                        'axes.labelcolor': 'black',
                        'xtick.color': 'black',
                        'ytick.color': 'black'
                    })
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, total_drawing_height)
                    ax.axis('off')
                    
                    # Draw Ribbons
                    # We need to track the "current used height" within each bar to stack ribbons correctly
                    # Left side: stack outgoing based on Target order
                    # Right side: stack incoming based on Source order
                    
                    left_offsets = {c: 0 for c in CATS}
                    right_offsets = {c: 0 for c in CATS}
                    
                    X_LEFT_BAR = 0.05
                    X_RIGHT_BAR = 0.95
                    BAR_WIDTH = 0.03
                    
                    # Draw Ribbons
                    for src in CATS:
                        for tgt in CATS:
                            val = flows.loc[src, tgt]
                            if val > 0:
                                # Start Y interval (Left)
                                y1_top = y_left_start[src] + left_counts[src] - left_offsets[src]
                                y1_bot = y1_top - val
                                left_offsets[src] += val
                                
                                # End Y interval (Right)
                                y2_top = y_right_start[tgt] + right_counts[tgt] - right_offsets[tgt]
                                y2_bot = y2_top - val
                                right_offsets[tgt] += val
                                
                                # Determine Color
                                if src == tgt:
                                    color = COLOR_CONCORDANT
                                    zorder = 1
                                elif CATS.index(src) > CATS.index(tgt):
                                    # Lab Higher -> NonLab Lower (Underestimation)
                                    color = COLOR_UNDER
                                    zorder = 2
                                else:
                                    # Lab Lower -> NonLab Higher (Overestimation)
                                    color = COLOR_OVER
                                    zorder = 2
                                    
                                # Draw Ribbon using Bezier curves
                                # Path: (xL, y1_bot) -> (xL, y1_top) -> curve -> (xR, y2_top) -> (xR, y2_bot) -> curve -> close
                                
                                verts = [
                                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),      # Start Bottom Left
                                    (X_LEFT_BAR + BAR_WIDTH, y1_top),      # Start Top Left
                                    (X_RIGHT_BAR - BAR_WIDTH, y2_top),     # End Top Right
                                    (X_RIGHT_BAR - BAR_WIDTH, y2_bot),     # End Bottom Right
                                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),      # Close
                                ]
                                
                                # Bezier Control Points for smooth sigmoid
                                # CP1 matches start tangent (horizontal), CP2 matches end tangent
                                midpoint = (X_LEFT_BAR + BAR_WIDTH + X_RIGHT_BAR - BAR_WIDTH) / 2
                                
                                codes = [
                                    Path.MOVETO,
                                    Path.LINETO,
                                    Path.CURVE4,
                                    Path.CURVE4,
                                    Path.CURVE4,
                                    Path.LINETO,
                                    Path.CURVE4,
                                    Path.CURVE4,
                                    Path.CURVE4
                                ]
                                
                                # Redefine verts for CURVE4
                                verts = [
                                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),      # Start BL
                                    (X_LEFT_BAR + BAR_WIDTH, y1_top),      # Start TL
                                    
                                    # Curve Top: Start TL -> End TR
                                    (midpoint, y1_top),                    # CP1
                                    (midpoint, y2_top),                    # CP2
                                    (X_RIGHT_BAR - BAR_WIDTH, y2_top),     # End TR
                                    
                                    (X_RIGHT_BAR - BAR_WIDTH, y2_bot),     # End BR
                                    
                                    # Curve Bottom: End BR -> Start BL
                                    (midpoint, y2_bot),                    # CP1
                                    (midpoint, y1_bot),                    # CP2
                                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),      # Start BL
                                ]
                                
                                path = Path(verts, codes)
                                patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none', alpha=ALPHA_RIBBON, zorder=zorder)
                                ax.add_patch(patch)
                                
                    # Draw Bars and Labels
                    # Left Side
                    for cat in CATS:
                        if left_counts[cat] > 0:
                            rect = mpatches.Rectangle(
                                (X_LEFT_BAR, y_left_start[cat]), 
                                BAR_WIDTH, 
                                left_counts[cat], 
                                facecolor=COLOR_LAB_BAR,
                                edgecolor='black',
                                linewidth=0.5,
                                zorder=3
                            )
                            ax.add_patch(rect)
                            
                            # Label
                            y_center = y_left_start[cat] + left_counts[cat]/2
                            pct = (left_counts[cat] / total_n) * 100
                            label_str = f"{cat}\n({pct:.1f}%)" if pct > 3 else f"{cat}"
                            
                            ax.text(
                                X_LEFT_BAR - 0.01, y_center, 
                                label_str, 
                                ha='right', va='center', 
                                fontsize=10, 
                                color='black',
                                fontfamily='serif'
                            )

                    # Right Side
                    for cat in CATS:
                        if right_counts[cat] > 0:
                            rect = mpatches.Rectangle(
                                (X_RIGHT_BAR - BAR_WIDTH, y_right_start[cat]), 
                                BAR_WIDTH, 
                                right_counts[cat], 
                                facecolor=COLOR_NONLAB_BAR,
                                edgecolor='black',
                                linewidth=0.5,
                                zorder=3
                            )
                            ax.add_patch(rect)
                            
                            # Label
                            y_center = y_right_start[cat] + right_counts[cat]/2
                            pct = (right_counts[cat] / total_n) * 100
                            label_str = f"{cat}\n({pct:.1f}%)" if pct > 3 else f"{cat}"
                            
                            ax.text(
                                X_RIGHT_BAR + 0.01, y_center, 
                                label_str, 
                                ha='left', va='center', 
                                fontsize=10, 
                                color='black',
                                fontfamily='serif'
                            )
                            
                    # Main Titles & Headers
                    ax.text(X_LEFT_BAR + BAR_WIDTH/2, total_drawing_height + gap_height, "Laboratory Based", 
                            ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLOR_LAB_BAR)
                    
                    ax.text(X_RIGHT_BAR - BAR_WIDTH/2, total_drawing_height + gap_height, "Non-Laboratory Based", 
                            ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLOR_NONLAB_BAR)
                            
                    plt.title(f"Risk Reclassification Flow {title_suffix} (N={total_n})", fontsize=16, pad=30, fontfamily='serif')
                    
                    return fig

                # Controls for Sankey
                sankey_mode = st.radio(
                    "View Mode:", 
                    ["Overall Flow", "Stratified by Glycaemic Group (Side-by-Side)"], 
                    horizontal=True,
                    key="sankey_mode_radio"
                )
                
                if sankey_mode == "Overall Flow":
                    fig = plot_risk_sankey(df_paired, "")
                    st.pyplot(fig)
                    
                    # Legend for flow colors
                    st.caption(
                        "**Flow Colors:** <span style='color:grey'>■ Concordance</span> | "
                        "<span style='color:orange'>■ Overestimation (Non-Lab > Lab)</span> | "
                        "<span style='color:red'>■ Underestimation (Non-Lab < Lab)</span>", 
                        unsafe_allow_html=True
                    )
                    
                else:
                    # Stratified View
                    groups = ['Normoglycaemia', 'IFG', 'Diabetes']
                    cols = st.columns(3)
                    
                    for i, group in enumerate(groups):
                        subset = df_paired[df_paired['gly_group'] == group]
                        with cols[i]:
                            st.markdown(f"**{group}**")
                            if len(subset) > 0:
                                fig = plot_risk_sankey(subset, f"\n({group})")
                                st.pyplot(fig)
                            else:
                                st.info(f"No data for {group}")

                st.divider()
                
                # ==========================================
                # SECTION 8.2: COMPREHENSIVE RISK DISTRIBUTION COMPARISON
                # ==========================================
                st.header("8.2 📈 Risk Distribution by Age Group")
                st.markdown("**Comparing lab-based and non-lab based risk distributions across age groups for all risk categories**")
                
                # Calculate distribution by age group
                lab_dist = df_paired.groupby('age_band')['risk_cat_lab'].apply(
                    lambda x: x.value_counts(normalize=True).mul(100)
                ).reset_index()
                lab_dist.columns = ['age_band', 'risk_category', 'percentage']
                lab_dist['method'] = 'Lab-based'
                
                nonlab_dist = df_paired.groupby('age_band')['risk_cat_nonlab'].apply(
                    lambda x: x.value_counts(normalize=True).mul(100)
                ).reset_index()
                nonlab_dist.columns = ['age_band', 'risk_category', 'percentage']
                nonlab_dist['method'] = 'Non-lab based'
                
                # Combine data
                combined_dist = pd.concat([lab_dist, nonlab_dist], ignore_index=True)
                
                # Ensure all combinations exist (fill missing with 0)
                all_ages = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                all_methods = ['Lab-based', 'Non-lab based']
                
                # Create complete grid
                complete_grid = []
                for age in all_ages:
                    for cat in RISK_LABELS:
                        for method in all_methods:
                            complete_grid.append({
                                'age_band': age,
                                'risk_category': cat,
                                'method': method,
                                'percentage': 0.0
                            })
                
                complete_df = pd.DataFrame(complete_grid)
                
                # Merge with actual data
                combined_dist_complete = complete_df.merge(
                    combined_dist,
                    on=['age_band', 'risk_category', 'method'],
                    how='left',
                    suffixes=('_default', '_actual')
                )
                combined_dist_complete['percentage'] = combined_dist_complete['percentage_actual'].fillna(
                    combined_dist_complete['percentage_default']
                )
                combined_dist_complete = combined_dist_complete[['age_band', 'risk_category', 'method', 'percentage']]
                
                # Create a comprehensive grouped bar chart with ALL risk categories visible
                st.markdown("### Single Comprehensive View: All Risk Categories by Age Group")
                
                # Create figure with subplots for each age group
                from plotly.subplots import make_subplots
                
         
                # Color mapping for risk categories
                RISK_COLORS_DETAILED = {
                    '<5%': '#2E7D32',
                    '5% to <10%': '#FDD835',
                    '10% to <20%': '#FB8C00',
                    '20% to <30%': '#E53935',
                    '≥30%': '#8E24AA'
                }
                
                # Single unified graph with all data
                st.markdown("---")
                st.markdown("### Comprehensive Single Graph: Lab vs Non-Lab")
                st.caption("All age groups and risk categories in one unified visualization")
                
                # Create the unified Matplotlib grouped bar chart
                fig_unified, ax_unified = plt.subplots(figsize=(16, 8))
                
                # Get unique values
                age_groups = sorted(combined_dist_complete['age_band'].unique())
                methods = ['Lab-based', 'Non-lab based']
                n_ages = len(age_groups)
                n_risk = len(RISK_LABELS)
                n_methods = len(methods)
                
                # Create bar positions
                bar_width = 0.08
                x_base = np.arange(n_ages)
                
                # Color mapping
                RISK_COLORS_DETAILED = {
                    '<5%': '#2E7D32',
                    '5% to <10%': '#FDD835',
                    '10% to <20%': '#FB8C00',
                    '20% to <30%': '#E53935',
                    '≥30%': '#8E24AA'
                }
                
                # Different hatch patterns for methods
                hatches = {'Lab-based': '', 'Non-lab based': '//'}
                
                for i_risk, risk in enumerate(RISK_LABELS):
                    for i_method, method in enumerate(methods):
                        offset = (i_risk * n_methods + i_method - (n_risk * n_methods) / 2 + 0.5) * bar_width
                        
                        pcts = []
                        for age in age_groups:
                            val = combined_dist_complete[
                                (combined_dist_complete['age_band'] == age) &
                                (combined_dist_complete['risk_category'] == risk) &
                                (combined_dist_complete['method'] == method)
                            ]['percentage'].values
                            pcts.append(val[0] if len(val) > 0 else 0)
                        
                        label = f'{risk} ({method})' if i_method == 0 else None
                        ax_unified.bar(x_base + offset, pcts, bar_width,
                                      color=RISK_COLORS_DETAILED.get(risk, '#888888'),
                                      hatch=hatches[method],
                                      edgecolor='black', linewidth=0.3,
                                      label=label if i_method == 0 else None)
                
                ax_unified.set_xticks(x_base)
                ax_unified.set_xticklabels(age_groups, rotation=45, ha='right', fontsize=10)
                ax_unified.set_xlabel('Age Group', fontsize=12, fontfamily='serif')
                ax_unified.set_ylabel('Percentage (%)', fontsize=12, fontfamily='serif')
                ax_unified.set_title('Comprehensive Risk Distribution: Lab-based vs Non-lab based Across All Age Groups',
                                    fontsize=14, fontweight='bold', fontfamily='serif')
                ax_unified.spines['top'].set_visible(False)
                ax_unified.spines['right'].set_visible(False)
                ax_unified.grid(True, alpha=0.3, axis='y')
                
                # Legend
                risk_patches = [mpatches.Patch(facecolor=RISK_COLORS_DETAILED.get(c, '#888888'), edgecolor='black', label=c) for c in RISK_LABELS]
                pattern_patches = [
                    mpatches.Patch(facecolor='white', edgecolor='black', label='Lab-based'),
                    mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Non-lab based')
                ]
                ax_unified.legend(handles=risk_patches + pattern_patches, loc='upper right', frameon=True, fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig_unified)
                add_svg_download_button(fig_unified, "comprehensive_risk_dist", key="comprehensive_svg")
                plt.close(fig_unified)
                
                # Add helpful legend explanation
                st.info("""
                📊 **How to read this graph:**
                - Each age group shows all combinations of risk categories and assessment methods
                - **Solid bars** = Lab-based assessment
                - **Patterned bars (with //)** = Non-lab based assessment  
                - Colors represent different risk categories (green=low risk, red/purple=high risk)
                - Bars are grouped by age for easy comparison
                """)
                
                # Alternative view: Side-by-side for each age group
                with st.expander("📊 Alternative View: Grouped by Age Band"):
                    # Matplotlib Faceted Bar Chart
                    age_bands = sorted(combined_dist['age_band'].unique())
                    n_cols = 4
                    n_rows = int(np.ceil(len(age_bands) / n_cols))
                    
                    fig_bar_alt, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
                    axes = axes.flatten()
                    
                    method_colors = {'Lab-based': '#2E86AB', 'Non-lab based': '#A23B72'}
                    
                    for idx, age in enumerate(age_bands):
                        ax = axes[idx]
                        age_data = combined_dist[combined_dist['age_band'] == age]
                        
                        x = np.arange(len(RISK_LABELS))
                        width = 0.35
                        
                        for i, method in enumerate(['Lab-based', 'Non-lab based']):
                            pcts = []
                            for risk in RISK_LABELS:
                                val = age_data[(age_data['risk_category'] == risk) & 
                                              (age_data['method'] == method)]['percentage'].values
                                pcts.append(val[0] if len(val) > 0 else 0)
                            
                            offset = (i - 0.5) * width
                            ax.bar(x + offset, pcts, width, label=method, color=method_colors[method],
                                  edgecolor='black', linewidth=0.5)
                        
                        ax.set_title(f'Age {age}', fontsize=11, fontweight='bold', fontfamily='serif')
                        ax.set_xticks(x)
                        ax.set_xticklabels(RISK_LABELS, rotation=45, ha='right', fontsize=7)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        if idx == 0:
                            ax.legend(fontsize=8, frameon=True)
                    
                    # Hide empty subplots
                    for idx in range(len(age_bands), len(axes)):
                        axes[idx].set_visible(False)
                    
                    fig_bar_alt.suptitle('Risk Distribution Comparison by Age Group', fontsize=14, fontweight='bold', y=1.02)
                    plt.tight_layout()
                    st.pyplot(fig_bar_alt)
                    add_svg_download_button(fig_bar_alt, "risk_by_age_alt", key="risk_age_alt_svg")
                    plt.close(fig_bar_alt)
                
                # Summary statistics table
                st.markdown("### Summary Statistics by Age Group")
                
                summary_stats = []
                for age_band in ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]:
                    age_data = df_paired[df_paired['age_band'] == age_band]
                    
                    summary_stats.append({
                        'Age Group': age_band,
                        'N': len(age_data),
                        'Lab Mean Risk (%)': age_data['risk_lab'].mean(),
                        'Non-Lab Mean Risk (%)': age_data['risk_nonlab'].mean(),
                        'Lab ≥20% (%)': (age_data['risk_lab'] >= 20).mean() * 100,
                        'Non-Lab ≥20% (%)': (age_data['risk_nonlab'] >= 20).mean() * 100
                    })
                
                summary_df = pd.DataFrame(summary_stats)
                st.dataframe(
                    summary_df.style.format({
                        'N': '{:,}',
                        'Lab Mean Risk (%)': '{:.2f}',
                        'Non-Lab Mean Risk (%)': '{:.2f}',
                        'Lab ≥20% (%)': '{:.1f}',
                        'Non-Lab ≥20% (%)': '{:.1f}'
                    }).background_gradient(cmap='YlOrRd', subset=['Lab Mean Risk (%)', 'Non-Lab Mean Risk (%)']),
                    use_container_width=True
                )
                
                st.divider()
                
                # ==========================================
                # SECTION 8.3: AGE-STRATIFIED TRENDS
                # ==========================================
                st.header("8.3 📈 Age-Stratified Risk Trends")
                st.markdown("Multi-series comparative line chart showing cardiovascular risk escalation across age groups")
                
                # Calculate mean risk and confidence intervals by age group
                age_risk_stats = []

                for age_band in ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]:
                    age_data = df_paired[df_paired['age_band'] == age_band]
                    
                    if len(age_data) > 0:
                        # Lab-based stats
                        lab_mean = age_data['risk_lab'].mean()
                        lab_sem = age_data['risk_lab'].sem()
                        lab_ci_lower = lab_mean - 1.96 * lab_sem
                        lab_ci_upper = lab_mean + 1.96 * lab_sem
                        
                        # Non-lab stats
                        nonlab_mean = age_data['risk_nonlab'].mean()
                        nonlab_sem = age_data['risk_nonlab'].sem()
                        nonlab_ci_lower = nonlab_mean - 1.96 * nonlab_sem
                        nonlab_ci_upper = nonlab_mean + 1.96 * nonlab_sem
                        
                        age_risk_stats.append({
                            'age_band': str(age_band),
                            'age_midpoint': age_data['age'].mean(),
                            'n': len(age_data),
                            'lab_mean': lab_mean,
                            'lab_ci_lower': lab_ci_lower,
                            'lab_ci_upper': lab_ci_upper,
                            'nonlab_mean': nonlab_mean,
                            'nonlab_ci_lower': nonlab_ci_lower,
                            'nonlab_ci_upper': nonlab_ci_upper
                        })
                
                stats_df = pd.DataFrame(age_risk_stats)
                
                # Create Matplotlib multi-series line chart with CI
                fig_line, ax_line = plt.subplots(figsize=(12, 7))
                
                x = np.arange(len(stats_df))
                
                # Lab-based line with CI fill
                ax_line.plot(x, stats_df['lab_mean'], 'o-', color='#2E86AB', linewidth=3, 
                            markersize=10, label='Lab-based')
                ax_line.fill_between(x, stats_df['lab_ci_lower'], stats_df['lab_ci_upper'], 
                                    color='#2E86AB', alpha=0.2)
                
                # Non-lab based line with CI fill
                ax_line.plot(x, stats_df['nonlab_mean'], 'd--', color='#A23B72', linewidth=3, 
                            markersize=10, label='Non-lab based')
                ax_line.fill_between(x, stats_df['nonlab_ci_lower'], stats_df['nonlab_ci_upper'], 
                                    color='#A23B72', alpha=0.2)
                
                # Add reference lines for risk thresholds
                ax_line.axhline(y=10, color='orange', linestyle=':', linewidth=2, alpha=0.8)
                ax_line.text(len(x) - 0.5, 10.5, '10% Threshold', fontsize=9, color='orange', fontfamily='serif')
                
                ax_line.axhline(y=20, color='red', linestyle=':', linewidth=2, alpha=0.8)
                ax_line.text(len(x) - 0.5, 20.5, '20% Threshold (High Risk)', fontsize=9, color='red', fontfamily='serif')
                
                ax_line.set_xticks(x)
                ax_line.set_xticklabels(stats_df['age_band'], rotation=45, ha='right', fontsize=10)
                ax_line.set_xlabel('Age Group', fontsize=12, fontfamily='serif')
                ax_line.set_ylabel('Mean 10-Year CVD Risk (%)', fontsize=12, fontfamily='serif')
                ax_line.set_title('Age-Stratified Cardiovascular Risk Escalation: Lab vs Non-Lab Assessment',
                                 fontsize=14, fontweight='bold', fontfamily='serif')
                ax_line.spines['top'].set_visible(False)
                ax_line.spines['right'].set_visible(False)
                ax_line.grid(True, alpha=0.3)
                ax_line.legend(loc='upper left', frameon=True, fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig_line)
                add_svg_download_button(fig_line, "age_risk_trends", key="age_risk_trends_svg")
                plt.close(fig_line)
                
                # Statistical comparison table
                with st.expander("📊 Detailed Age-Stratified Statistics"):
                    display_stats = stats_df.copy()
                    display_stats['Lab 95% CI'] = display_stats.apply(
                        lambda x: f"{x['lab_ci_lower']:.2f} - {x['lab_ci_upper']:.2f}", axis=1
                    )
                    display_stats['Non-Lab 95% CI'] = display_stats.apply(
                        lambda x: f"{x['nonlab_ci_lower']:.2f} - {x['nonlab_ci_upper']:.2f}", axis=1
                    )
                    display_stats['Difference'] = display_stats['lab_mean'] - display_stats['nonlab_mean']
                    
                    display_table = display_stats[[
                        'age_band', 'n', 'lab_mean', 'Lab 95% CI', 
                        'nonlab_mean', 'Non-Lab 95% CI', 'Difference'
                    ]].copy()
                    
                    display_table.columns = [
                        'Age Group', 'N', 'Lab Mean (%)', 'Lab 95% CI',
                        'Non-Lab Mean (%)', 'Non-Lab 95% CI', 'Difference (%)'
                    ]
                    
                    st.dataframe(
                        display_table.style.format({
                            'N': '{:,}',
                            'Lab Mean (%)': '{:.2f}',
                            'Non-Lab Mean (%)': '{:.2f}',
                            'Difference (%)': '{:.2f}'
                        }).background_gradient(cmap='RdYlGn_r', subset=['Difference (%)']),
                        use_container_width=True
                    )
                
                # Key Insights
                st.markdown("### Key Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                overall_lab_mean = df_paired['risk_lab'].mean()
                overall_nonlab_mean = df_paired['risk_nonlab'].mean()
                correlation = df_paired[['risk_lab', 'risk_nonlab']].corr().iloc[0, 1]
                
                col1.metric("Overall Lab Mean", f"{overall_lab_mean:.2f}%")
                col2.metric("Overall Non-Lab Mean", f"{overall_nonlab_mean:.2f}%")
                col3.metric("Mean Difference", f"{overall_lab_mean - overall_nonlab_mean:.2f}%")
                col4.metric("Correlation (r)", f"{correlation:.3f}")
                
                # Paired t-test
                from scipy.stats import ttest_rel
                t_stat, p_val = ttest_rel(df_paired['risk_lab'], df_paired['risk_nonlab'])
                
                st.info(f"""
                📊 **Paired t-test Results:**
                - t-statistic: {t_stat:.3f}
                - p-value: {p_val:.4f}
                - {'Significant' if p_val < 0.05 else 'Not significant'} difference at α=0.05
                """)
                
                # ==========================================
                # SECTION 8.3A: HIGH-RISK PREVALENCE TRENDS
                # ==========================================
                st.markdown("---")
                st.markdown("### 📈 High-Risk Prevalence Trends Across Age Groups")
                st.caption("Comparing how cardiovascular high-risk prevalence increases with age between lab-based and non-lab based assessments")
                
                # Calculate prevalence for different risk thresholds by age group
                age_prevalence_stats = []
                
                for age_band in ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]:
                    age_data = df_paired[df_paired['age_band'] == age_band]
                    
                    if len(age_data) > 0:
                        n = len(age_data)
                        
                        # Lab-based prevalence (≥10% and ≥20%)
                        lab_10_count = (age_data['risk_lab'] >= 10).sum()
                        lab_20_count = (age_data['risk_lab'] >= 20).sum()
                        lab_10_prev = (lab_10_count / n) * 100
                        lab_20_prev = (lab_20_count / n) * 100
                        
                        # 95% CI using Wilson score interval for lab ≥10%
                        lab_10_ci_low, lab_10_ci_high = calculate_wilson_ci(lab_10_count, n)
                        lab_20_ci_low, lab_20_ci_high = calculate_wilson_ci(lab_20_count, n)
                        
                        # Non-lab based prevalence (≥10% and ≥20%)
                        nonlab_10_count = (age_data['risk_nonlab'] >= 10).sum()
                        nonlab_20_count = (age_data['risk_nonlab'] >= 20).sum()
                        nonlab_10_prev = (nonlab_10_count / n) * 100
                        nonlab_20_prev = (nonlab_20_count / n) * 100
                        
                        # 95% CI for non-lab ≥10%
                        nonlab_10_ci_low, nonlab_10_ci_high = calculate_wilson_ci(nonlab_10_count, n)
                        nonlab_20_ci_low, nonlab_20_ci_high = calculate_wilson_ci(nonlab_20_count, n)
                        
                        age_prevalence_stats.append({
                            'age_band': str(age_band),
                            'n': n,
                            'lab_10_prev': lab_10_prev,
                            'lab_10_ci_low': lab_10_ci_low,
                            'lab_10_ci_high': lab_10_ci_high,
                            'lab_20_prev': lab_20_prev,
                            'lab_20_ci_low': lab_20_ci_low,
                            'lab_20_ci_high': lab_20_ci_high,
                            'nonlab_10_prev': nonlab_10_prev,
                            'nonlab_10_ci_low': nonlab_10_ci_low,
                            'nonlab_10_ci_high': nonlab_10_ci_high,
                            'nonlab_20_prev': nonlab_20_prev,
                            'nonlab_20_ci_low': nonlab_20_ci_low,
                            'nonlab_20_ci_high': nonlab_20_ci_high
                        })
                
                prev_df = pd.DataFrame(age_prevalence_stats)
                
                # Create Matplotlib figure with two subplots (≥10% and ≥20%)
                fig_prevalence, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                x = np.arange(len(prev_df))
                
                # ===== SUBPLOT 1: ≥10% Prevalence =====
                ax1 = axes[0]
                
                # Lab-based ≥10% with CI fill
                ax1.plot(x, prev_df['lab_10_prev'], 'o-', color='#2E86AB', linewidth=3, 
                        markersize=10, label='Lab-based ≥10%')
                ax1.fill_between(x, prev_df['lab_10_ci_low'], prev_df['lab_10_ci_high'], 
                               color='#2E86AB', alpha=0.2)
                
                # Non-lab based ≥10% with CI fill
                ax1.plot(x, prev_df['nonlab_10_prev'], 'd--', color='#A23B72', linewidth=3, 
                        markersize=10, label='Non-lab ≥10%')
                ax1.fill_between(x, prev_df['nonlab_10_ci_low'], prev_df['nonlab_10_ci_high'], 
                               color='#A23B72', alpha=0.2)
                
                ax1.set_title('Moderate Risk: ≥10% Prevalence', fontsize=12, fontweight='bold', fontfamily='serif')
                ax1.set_xticks(x)
                ax1.set_xticklabels(prev_df['age_band'], rotation=45, ha='right', fontsize=9)
                ax1.set_xlabel('Age Group', fontsize=11, fontfamily='serif')
                ax1.set_ylabel('Prevalence (%)', fontsize=11, fontfamily='serif')
                ax1.set_ylim(0, 105)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper left', frameon=True, fontsize=9)
                
                # ===== SUBPLOT 2: ≥20% Prevalence =====
                ax2 = axes[1]
                
                # Lab-based ≥20% with CI fill
                ax2.plot(x, prev_df['lab_20_prev'], 'o-', color='#2E86AB', linewidth=3, 
                        markersize=10, label='Lab-based ≥20%')
                ax2.fill_between(x, prev_df['lab_20_ci_low'], prev_df['lab_20_ci_high'], 
                               color='#2E86AB', alpha=0.2)
                
                # Non-lab based ≥20% with CI fill
                ax2.plot(x, prev_df['nonlab_20_prev'], 'd--', color='#A23B72', linewidth=3, 
                        markersize=10, label='Non-lab ≥20%')
                ax2.fill_between(x, prev_df['nonlab_20_ci_low'], prev_df['nonlab_20_ci_high'], 
                               color='#A23B72', alpha=0.2)
                
                ax2.set_title('High Risk: ≥20% Prevalence', fontsize=12, fontweight='bold', fontfamily='serif')
                ax2.set_xticks(x)
                ax2.set_xticklabels(prev_df['age_band'], rotation=45, ha='right', fontsize=9)
                ax2.set_xlabel('Age Group', fontsize=11, fontfamily='serif')
                ax2.set_ylabel('Prevalence (%)', fontsize=11, fontfamily='serif')
                ax2.set_ylim(0, 105)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left', frameon=True, fontsize=9)
                
                fig_prevalence.suptitle('High-Risk Prevalence Trends: Age-Specific Comparison (Lab vs Non-Lab)',
                                       fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig_prevalence)
                add_svg_download_button(fig_prevalence, "prevalence_trends", key="prevalence_trends_svg")
                plt.close(fig_prevalence)
                
                # Summary table
                with st.expander("📊 Age-Specific Prevalence Statistics Table"):
                    prev_table = prev_df.copy()
                    prev_table['Lab ≥10% (95% CI)'] = prev_table.apply(
                        lambda x: f"{x['lab_10_prev']:.1f}% ({x['lab_10_ci_low']:.1f}-{x['lab_10_ci_high']:.1f})", axis=1
                    )
                    prev_table['Non-Lab ≥10% (95% CI)'] = prev_table.apply(
                        lambda x: f"{x['nonlab_10_prev']:.1f}% ({x['nonlab_10_ci_low']:.1f}-{x['nonlab_10_ci_high']:.1f})", axis=1
                    )
                    prev_table['Lab ≥20% (95% CI)'] = prev_table.apply(
                        lambda x: f"{x['lab_20_prev']:.1f}% ({x['lab_20_ci_low']:.1f}-{x['lab_20_ci_high']:.1f})", axis=1
                    )
                    prev_table['Non-Lab ≥20% (95% CI)'] = prev_table.apply(
                        lambda x: f"{x['nonlab_20_prev']:.1f}% ({x['nonlab_20_ci_low']:.1f}-{x['nonlab_20_ci_high']:.1f})", axis=1
                    )
                    
                    display_prev_table = prev_table[['age_band', 'n', 'Lab ≥10% (95% CI)', 'Non-Lab ≥10% (95% CI)', 
                                                      'Lab ≥20% (95% CI)', 'Non-Lab ≥20% (95% CI)']].copy()
                    display_prev_table.columns = ['Age Group', 'N', 'Lab ≥10%', 'Non-Lab ≥10%', 'Lab ≥20%', 'Non-Lab ≥20%']
                    
                    st.dataframe(display_prev_table, use_container_width=True)
                    
                    # Calculate overall prevalence
                    st.markdown("**Overall Prevalence (All Ages):**")
                    overall_cols = st.columns(4)
                    
                    overall_lab_10 = (df_paired['risk_lab'] >= 10).mean() * 100
                    overall_nonlab_10 = (df_paired['risk_nonlab'] >= 10).mean() * 100
                    overall_lab_20 = (df_paired['risk_lab'] >= 20).mean() * 100
                    overall_nonlab_20 = (df_paired['risk_nonlab'] >= 20).mean() * 100
                    
                    overall_cols[0].metric("Lab ≥10%", f"{overall_lab_10:.1f}%")
                    overall_cols[1].metric("Non-Lab ≥10%", f"{overall_nonlab_10:.1f}%")
                    overall_cols[2].metric("Lab ≥20%", f"{overall_lab_20:.1f}%")
                    overall_cols[3].metric("Non-Lab ≥20%", f"{overall_nonlab_20:.1f}%")
                
                st.divider()
                
                # ==========================================
                # SECTION 8.3B: COHORT COMPARISON
                # ==========================================
                st.markdown("---")
                st.markdown("### 📊 Age-Based Risk Escalation: Main Cohort vs Validation Cohort")
                st.caption("Comparing entire non-lab cohort (main) with paired lab-based validation subset")
                
                # Helper function to calculate age prevalence statistics
                def calculate_age_prevalence_stats(df, score_col, age_bands):
                    """Calculate prevalence of ≥10% and ≥20% risk per age band"""
                    stats = []
                    for age_band in age_bands:
                        age_data = df[df['age_band'] == age_band]
                        n = len(age_data)
                        
                        if n > 0:
                            risk_scores = pd.to_numeric(age_data[score_col], errors='coerce')
                            high_10 = (risk_scores >= 10).sum()
                            high_20 = (risk_scores >= 20).sum()
                            
                            pct_10 = (high_10 / n) * 100
                            pct_20 = (high_20 / n) * 100
                            
                            # Standard error for binomial proportions (in percentage points)
                            se_10 = np.sqrt((pct_10/100 * (1 - pct_10/100)) / n) * 100
                            se_20 = np.sqrt((pct_20/100 * (1 - pct_20/100)) / n) * 100
                            
                            # 95% CI
                            ci_10_low, ci_10_high = calculate_wilson_ci(high_10, n)
                            ci_20_low, ci_20_high = calculate_wilson_ci(high_20, n)
                            
                            stats.append({
                                'age_band': age_band,
                                'n': n,
                                'n_10': high_10,
                                'n_20': high_20,
                                'pct_10': pct_10,
                                'pct_20': pct_20,
                                'se_10': se_10,
                                'se_20': se_20,
                                'ci_10_low': ci_10_low,
                                'ci_10_high': ci_10_high,
                                'ci_20_low': ci_20_low,
                                'ci_20_high': ci_20_high
                            })
                    
                    return pd.DataFrame(stats)
                
                # Calculate statistics for both cohorts
                age_bands_list = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                
                # Main cohort stats (using df_cohort from parent scope - all non-lab data)
                stats_main = calculate_age_prevalence_stats(df_cohort, 'active_risk', age_bands_list)
                
                # Validation cohort stats (using df_paired - lab-based scores)
                stats_valid = calculate_age_prevalence_stats(df_paired, 'risk_lab', age_bands_list)
                
                # Create Matplotlib comparison plot
                fig_main_vs_valid, ax_mv = plt.subplots(figsize=(12, 7))
                
                x = np.arange(len(stats_main))
                
                # Main cohort ≥10% with CI fill
                ax_mv.plot(x, stats_main['pct_10'], 'o-', color='#d62828', linewidth=3, 
                          markersize=10, label='Main Cohort (≥10% Risk)')
                ax_mv.fill_between(x, 
                                  stats_main['pct_10'] - 1.96 * stats_main['se_10'], 
                                  stats_main['pct_10'] + 1.96 * stats_main['se_10'], 
                                  color='#d62828', alpha=0.15)
                
                # Main cohort ≥20% (dashed line)
                ax_mv.plot(x, stats_main['pct_20'], 's--', color='#f4a261', linewidth=3, 
                          markersize=9, label='Main Cohort (≥20% Risk)')
                
                # Validation cohort ≥10% (scatter overlay)
                ax_mv.scatter(x, stats_valid['pct_10'], s=150, marker='X', c='#2a9d8f',
                             edgecolors='black', linewidth=1.5, label='Validation (≥10% Risk)', zorder=5)
                
                # Validation cohort ≥20% (scatter overlay)
                ax_mv.scatter(x, stats_valid['pct_20'], s=150, marker='D', c='#06668d',
                             edgecolors='black', linewidth=1.5, label='Validation (≥20% Risk)', zorder=5)
                
                # Add sample size annotations
                for i, row in stats_main.iterrows():
                    ax_mv.annotate(f"n={int(row['n'])}", (i, 5), fontsize=9, color='gray', 
                                  ha='center', fontfamily='serif')
                
                ax_mv.set_xticks(x)
                ax_mv.set_xticklabels(stats_main['age_band'], rotation=45, ha='right', fontsize=10)
                ax_mv.set_xlabel('Age Band (Years)', fontsize=12, fontfamily='serif')
                ax_mv.set_ylabel('Prevalence of High Risk (%)', fontsize=12, fontfamily='serif')
                ax_mv.set_title('Age-Based Escalation of Cardiovascular Risk\nMain Cohort (Non-Lab) vs Validation Cohort (Lab-Based)',
                               fontsize=14, fontweight='bold', fontfamily='serif')
                ax_mv.set_ylim(0, max(100, float(stats_main[['pct_10', 'pct_20']].max().max()) * 1.15))
                ax_mv.spines['top'].set_visible(False)
                ax_mv.spines['right'].set_visible(False)
                ax_mv.grid(True, alpha=0.3)
                ax_mv.legend(loc='upper left', frameon=True, fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig_main_vs_valid)
                add_svg_download_button(fig_main_vs_valid, "age_risk_escalation", key="age_esc_svg")
                plt.close(fig_main_vs_valid)
                
                # Summary statistics table
                with st.expander("📊 Detailed Prevalence Comparison Table"):
                    # Combine stats for display
                    comparison_table = pd.DataFrame({
                        'Age Band': stats_main['age_band'],
                        'Main N': stats_main['n'].astype(int),
                        'Main ≥10%': stats_main['pct_10'].round(1).astype(str) + '%',
                        'Main ≥20%': stats_main['pct_20'].round(1).astype(str) + '%',
                        'Valid N': stats_valid['n'].astype(int),
                        'Valid ≥10%': stats_valid['pct_10'].round(1).astype(str) + '%',
                        'Valid ≥20%': stats_valid['pct_20'].round(1).astype(str) + '%',
                    })
                    
                    st.markdown("**Main Cohort:** All non-lab risk assessments (n={:,})".format(len(df_cohort)))
                    st.markdown("**Validation Cohort:** Paired subset with lab-based risk (n={:,})".format(len(df_paired)))
                    st.dataframe(comparison_table, use_container_width=True)
                    
                    # Key insights
                    st.markdown("---")
                    st.markdown("**Key Insights:**")
                    
                    col_insight1, col_insight2 = st.columns(2)
                    
                    with col_insight1:
                        overall_main_10 = (stats_main['n_10'].sum() / stats_main['n'].sum()) * 100
                        overall_main_20 = (stats_main['n_20'].sum() / stats_main['n'].sum()) * 100
                        
                        st.metric("Main Cohort ≥10%", f"{overall_main_10:.1f}%", 
                                 help="Overall prevalence across all age groups")
                        st.metric("Main Cohort ≥20%", f"{overall_main_20:.1f}%")
                    
                    with col_insight2:
                        overall_valid_10 = (stats_valid['n_10'].sum() / stats_valid['n'].sum()) * 100
                        overall_valid_20 = (stats_valid['n_20'].sum() / stats_valid['n'].sum()) * 100
                        
                        st.metric("Validation Cohort ≥10%", f"{overall_valid_10:.1f}%",
                                 help="Lab-based prevalence in paired subset")
                        st.metric("Validation Cohort ≥20%", f"{overall_valid_20:.1f}%")
                
                st.divider()
                
                # ==========================================
                # SECTION 8.4: AGREEMENT & DISCORDANCE ANALYSIS
                # ==========================================
                st.header("8.4 🎯 Agreement & Discordance Analysis")
                st.markdown("Detailed comparison of risk category distributions, stratified agreement heatmaps, and systematic bias analysis.")
                
                # Overall distribution comparison
                st.markdown("### Overall Risk Distribution")
                
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    st.markdown("#### Lab-based Distribution")
                    lab_dist_overall = df_paired['risk_cat_lab'].value_counts(normalize=True).mul(100).reindex(RISK_LABELS, fill_value=0)
                    
                    # Matplotlib Donut Chart for Lab-based
                    fig_lab_pie, ax_lab = plt.subplots(figsize=(8, 8))
                    colors_lab = [RISK_COLORS.get(cat, '#888888') for cat in lab_dist_overall.index]
                    wedges_lab, texts_lab, autotexts_lab = ax_lab.pie(
                        lab_dist_overall.values,
                        labels=lab_dist_overall.index,
                        colors=colors_lab,
                        autopct='%1.1f%%',
                        pctdistance=0.75,
                        startangle=90,
                        wedgeprops=dict(width=0.7, edgecolor='white')
                    )
                    for text in texts_lab:
                        text.set_fontfamily('serif')
                        text.set_fontsize(10)
                    for autotext in autotexts_lab:
                        autotext.set_fontfamily('serif')
                        autotext.set_fontsize(9)
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    ax_lab.set_title('Lab-based Risk Categories', fontsize=14, fontweight='bold', fontfamily='serif')
                    plt.tight_layout()
                    st.pyplot(fig_lab_pie)
                    add_svg_download_button(fig_lab_pie, "lab_risk_distribution", key="lab_pie_svg")
                    plt.close(fig_lab_pie)
                
                with col_dist2:
                    st.markdown("#### Non-lab based Distribution")
                    nonlab_dist_overall = df_paired['risk_cat_nonlab'].value_counts(normalize=True).mul(100).reindex(RISK_LABELS, fill_value=0)
                    
                    # Matplotlib Donut Chart for Non-Lab
                    fig_nonlab_pie, ax_nonlab = plt.subplots(figsize=(8, 8))
                    colors_nonlab = [RISK_COLORS.get(cat, '#888888') for cat in nonlab_dist_overall.index]
                    wedges_nonlab, texts_nonlab, autotexts_nonlab = ax_nonlab.pie(
                        nonlab_dist_overall.values,
                        labels=nonlab_dist_overall.index,
                        colors=colors_nonlab,
                        autopct='%1.1f%%',
                        pctdistance=0.75,
                        startangle=90,
                        wedgeprops=dict(width=0.7, edgecolor='white')
                    )
                    for text in texts_nonlab:
                        text.set_fontfamily('serif')
                        text.set_fontsize(10)
                    for autotext in autotexts_nonlab:
                        autotext.set_fontfamily('serif')
                        autotext.set_fontsize(9)
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    ax_nonlab.set_title('Non-lab based Risk Categories', fontsize=14, fontweight='bold', fontfamily='serif')
                    plt.tight_layout()
                    st.pyplot(fig_nonlab_pie)
                    add_svg_download_button(fig_nonlab_pie, "nonlab_risk_distribution", key="nonlab_pie_svg")
                    plt.close(fig_nonlab_pie)
                
                # Comparative bar chart - Matplotlib version
                st.markdown("### Side-by-Side Comparison")
                
                # Create grouped bar chart with Matplotlib
                fig_comparison, ax_comp = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(RISK_LABELS))
                width = 0.35
                
                bars_lab = ax_comp.bar(x - width/2, lab_dist_overall.values, width, 
                                       label='Lab-based', color=NATURE_COLORS['blue'], 
                                       edgecolor='black', linewidth=0.5)
                bars_nonlab = ax_comp.bar(x + width/2, nonlab_dist_overall.values, width, 
                                          label='Non-lab based', color=NATURE_COLORS['purple'],
                                          edgecolor='black', linewidth=0.5)
                
                # Add value labels on bars
                for bar, val in zip(bars_lab, lab_dist_overall.values):
                    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontfamily='serif')
                for bar, val in zip(bars_nonlab, nonlab_dist_overall.values):
                    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontfamily='serif')
                
                ax_comp.set_ylabel('Percentage (%)', fontsize=12, fontfamily='serif')
                ax_comp.set_xlabel('Risk Category', fontsize=12, fontfamily='serif')
                ax_comp.set_title('Risk Distribution: Lab-based vs Non-lab based', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_comp.set_xticks(x)
                ax_comp.set_xticklabels(RISK_LABELS, rotation=45, ha='right', fontsize=10)
                ax_comp.legend(frameon=True, framealpha=0.9, edgecolor='gray')
                ax_comp.spines['top'].set_visible(False)
                ax_comp.spines['right'].set_visible(False)
                ax_comp.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig_comparison)
                add_svg_download_button(fig_comparison, "risk_comparison_bar", key="comparison_bar_svg")
                plt.close(fig_comparison)
                
                # Legacy Plotly chart removed - Matplotlib version now used instead
                
                # --- STRATIFIED AGREEMENT HEATMAP ---
                st.divider()
                st.subheader("Stratified Agreement Heatmap")
                st.caption("Compare Lab (Rows) vs Non-Lab (Cols) risk categories. Diagonal cells indicate agreement.")

                # Controls
                c_hm1, c_hm2 = st.columns(2)
                with c_hm1:
                    hm_strat = st.selectbox("Stratify Heatmap By:", ["None (Overall)", "Glycaemic Group", "Gender", "Age Band"], key="hm_strat")
                with c_hm2:
                    hm_norm = st.selectbox("Normalize Heatmap Cells:", ["Count (N)", "% of Total", "% of Row (Lab Reference)", "% of Col (Non-Lab)"], key="hm_norm")

                norm_map = {"Count (N)": False, "% of Total": "all", "% of Row (Lab Reference)": "index", "% of Col (Non-Lab)": "columns"}
                strat_col_map = {"Glycaemic Group": "gly_group", "Gender": "gender", "Age Band": "age_band"}

                def plot_confusion_heatmap_mpl(data, title, key_suffix=""):
                    """Matplotlib version of confusion heatmap with SciencePlots styling"""
                    cm = pd.crosstab(data['risk_cat_lab'], data['risk_cat_nonlab'], 
                                    normalize=norm_map[hm_norm], dropna=False).reindex(
                                        index=RISK_LABELS, columns=RISK_LABELS, fill_value=0)
                    
                    if norm_map[hm_norm] is False:
                        annot_fmt = '.0f'
                        title_sub = "(N)"
                    else:
                        cm = cm * 100
                        annot_fmt = '.1f'
                        title_sub = "(%)"
                    
                    # Create Matplotlib heatmap
                    fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                    
                    sns.heatmap(cm, ax=ax_hm, cmap='Blues', annot=True, fmt=annot_fmt,
                               linewidths=0.5, linecolor='white',
                               cbar_kws={'label': title_sub, 'shrink': 0.8})
                    
                    ax_hm.set_xlabel('Non-Lab Category', fontsize=12, fontfamily='serif')
                    ax_hm.set_ylabel('Lab Category', fontsize=12, fontfamily='serif')
                    ax_hm.set_title(f"{title} {title_sub}", fontsize=14, fontweight='bold', fontfamily='serif')
                    
                    # Rotate x-axis labels
                    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=9)
                    
                    plt.tight_layout()
                    return fig_hm

                if hm_strat == "None (Overall)":
                    fig_overall_hm = plot_confusion_heatmap_mpl(df_paired, "Overall Agreement")
                    st.pyplot(fig_overall_hm)
                    add_svg_download_button(fig_overall_hm, "agreement_heatmap_overall", key="hm_overall_svg")
                    plt.close(fig_overall_hm)
                else:
                    scol = strat_col_map[hm_strat]
                    if scol in df_paired.columns:
                        groups = sorted(df_paired[scol].astype(str).unique())
                        rows = (len(groups) + 1) // 2
                        for r in range(rows):
                            cols = st.columns(2)
                            for c in range(2):
                                idx = r * 2 + c
                                if idx < len(groups):
                                    g = groups[idx]
                                    with cols[c]:
                                        fig_grp_hm = plot_confusion_heatmap_mpl(
                                            df_paired[df_paired[scol].astype(str) == g], 
                                            f"{g}", 
                                            key_suffix=f"_{g}"
                                        )
                                        st.pyplot(fig_grp_hm)
                                        add_svg_download_button(fig_grp_hm, f"agreement_heatmap_{g.replace(' ', '_').lower()}", 
                                                               key=f"hm_{g.replace(' ', '_')}_svg")
                                        plt.close(fig_grp_hm)
                    else:
                         st.warning(f"Column {scol} not found.")

                # --- AGREEMENT DIRECTIONALITY (DIVERGING BAR) ---
                st.divider()
                st.subheader("Agreement Directionality by Group")
                st.markdown("**Visualizing bias direction:** Are discordant cases mostly underestimated or overestimated?")

                # Logic
                cat_map = {c: i for i, c in enumerate(RISK_LABELS)}
                df_paired['cat_idx_lab'] = df_paired['risk_cat_lab'].map(cat_map)
                df_paired['cat_idx_nonlab'] = df_paired['risk_cat_nonlab'].map(cat_map)
                
                def classify_agreement(row):
                    if pd.isna(row['cat_idx_lab']) or pd.isna(row['cat_idx_nonlab']): return "Unknown"
                    if row['cat_idx_nonlab'] < row['cat_idx_lab']: return "Underestimation"
                    if row['cat_idx_nonlab'] > row['cat_idx_lab']: return "Overestimation"
                    return "Concordance"
                
                df_paired['agree_status'] = df_paired.apply(classify_agreement, axis=1)

                div_strat = st.selectbox("Group Directionality By:", ["Glycaemic Group", "Age Band", "Gender", "BMI Category"], key="div_strat")
                div_col_map = {"Glycaemic Group": "gly_group", "Age Band": "age_band", "Gender": "gender", "BMI Category": "bmi_band"}
                target_col = div_col_map.get(div_strat, "gly_group")

                if target_col in df_paired.columns:
                    ct = pd.crosstab(df_paired[target_col], df_paired['agree_status'], normalize='index') * 100
                    ct = ct.reset_index()
                    for c in ["Underestimation", "Concordance", "Overestimation"]:
                        if c not in ct.columns: ct[c] = 0.0
                    counts = df_paired[target_col].value_counts()
                    ct['N'] = ct[target_col].map(counts)
                    ct = ct.sort_values("Underestimation", ascending=True)  # Reversed for horizontal bars (top to bottom)
                    
                    # Matplotlib Horizontal Stacked Bar Chart
                    fig_div, ax_div = plt.subplots(figsize=(10, max(4, len(ct) * 0.8)))
                    
                    y_labels = ct[target_col].astype(str) + " (n=" + ct['N'].astype(str) + ")"
                    y_pos = np.arange(len(ct))
                    
                    # Stacking bars
                    bars_under = ax_div.barh(y_pos, ct['Underestimation'], 
                                            label='Underestimation (Non-Lab < Lab)', 
                                            color=AGREEMENT_COLORS['Underestimation'],
                                            edgecolor='white', linewidth=0.5)
                    bars_conc = ax_div.barh(y_pos, ct['Concordance'], left=ct['Underestimation'],
                                           label='Concordance', 
                                           color=AGREEMENT_COLORS['Concordance'],
                                           edgecolor='white', linewidth=0.5)
                    bars_over = ax_div.barh(y_pos, ct['Overestimation'], 
                                           left=ct['Underestimation'] + ct['Concordance'],
                                           label='Overestimation (Non-Lab > Lab)', 
                                           color=AGREEMENT_COLORS['Overestimation'],
                                           edgecolor='white', linewidth=0.5)
                    
                    # Add value labels inside bars
                    def add_bar_labels(bars, values, left_offset):
                        for bar, val, left in zip(bars, values, left_offset):
                            if val > 5:  # Only show if wide enough
                                ax_div.text(left + val/2, bar.get_y() + bar.get_height()/2,
                                           f'{val:.1f}%', ha='center', va='center',
                                           fontsize=9, fontweight='bold', color='white', fontfamily='serif')
                    
                    add_bar_labels(bars_under, ct['Underestimation'].values, np.zeros(len(ct)))
                    add_bar_labels(bars_conc, ct['Concordance'].values, ct['Underestimation'].values)
                    add_bar_labels(bars_over, ct['Overestimation'].values, 
                                  (ct['Underestimation'] + ct['Concordance']).values)
                    
                    ax_div.set_yticks(y_pos)
                    ax_div.set_yticklabels(y_labels, fontsize=10, fontfamily='serif')
                    ax_div.set_xlabel('Percentage (%)', fontsize=12, fontfamily='serif')
                    ax_div.set_title(f'Directionality by {div_strat}', fontsize=14, fontweight='bold', fontfamily='serif')
                    ax_div.set_xlim(0, 100)
                    ax_div.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, 
                                 frameon=True, framealpha=0.9, fontsize=9)
                    ax_div.spines['top'].set_visible(False)
                    ax_div.spines['right'].set_visible(False)
                    ax_div.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    st.pyplot(fig_div)
                    add_svg_download_button(fig_div, f"directionality_by_{div_strat.replace(' ', '_').lower()}", 
                                           key="directionality_svg")
                    plt.close(fig_div)
                    
                    with st.expander("Show Data Table"):
                        st.dataframe(ct.style.format("{:.1f}", subset=["Underestimation", "Concordance", "Overestimation"]))

                
                # Agreement metrics
                st.markdown("### Agreement Metrics")
                
                # Calculate exact agreement
                exact_agreement = (df_paired['risk_cat_lab'] == df_paired['risk_cat_nonlab']).mean() * 100

                # Calculate agreement within 1 category (FIXED)
                cat_mapping = {cat: i for i, cat in enumerate(RISK_LABELS)}

                lab_numeric = df_paired['risk_cat_lab'].astype(str).map(cat_mapping)
                nonlab_numeric = df_paired['risk_cat_nonlab'].astype(str).map(cat_mapping)

                # drop rows where mapping failed (e.g., NaN categories)
                valid = lab_numeric.notna() & nonlab_numeric.notna()
                within_one = ((lab_numeric[valid] - nonlab_numeric[valid]).abs() <= 1).mean() * 100

                # High risk agreement (≥20%)
                lab_high_risk = df_paired['risk_lab'] >= 20
                nonlab_high_risk = df_paired['risk_nonlab'] >= 20
                high_risk_agreement = (lab_high_risk == nonlab_high_risk).mean() * 100
                
                # Cohen's Kappa
                # Use weighted kappa for ordinal data
                try:
                    kappa = cohen_kappa_score(
                        lab_numeric[valid], 
                        nonlab_numeric[valid], 
                        weights='quadratic'
                    )
                except:
                    kappa = np.nan
                
                # Categorical NRI (Lab High Risk as Truth, evaluating Non-Lab performance? 
                # Or improvement of Lab over Non-Lab? 
                # Since Lab is the Gold Standard, NRI is interpreted as:
                # "Net benefit of using Lab over Non-Lab" relative to the Truth (Lab).
                # Which is tautological: Lab is always right. 
                # So NRI would be comparing "Non-Lab" vs "Random"? No.
                # Actually, NRI is valid if we consider Lab Risk as the 'Event' outcome.
                # And we compare 'Non-Lab' prediction vs null?
                # Let's just report Kappa here as it's the standard for Concordance.
                # I'll add Sensitivity/Specificity of Non-Lab for detecting Lab High Risk instead of NRI for this tab.
                # NRI is better for RQ3 (ML Model vs Non-Lab).
                
                # Sensitivity / Specificity for ≥20%
                tp = ((lab_high_risk == 1) & (nonlab_high_risk == 1)).sum()
                fn = ((lab_high_risk == 1) & (nonlab_high_risk == 0)).sum()
                tn = ((lab_high_risk == 0) & (nonlab_high_risk == 0)).sum()
                fp = ((lab_high_risk == 0) & (nonlab_high_risk == 1)).sum()
                
                sens = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Exact Agreement", f"{exact_agreement:.1f}%")
                col_m2.metric("Within 1 Category", f"{within_one:.1f}%")
                col_m3.metric("Cohen's Kappa (Weighted)", f"{kappa:.3f}", help="quadratic weights")
                col_m4.metric("High-Risk Agreement", f"{high_risk_agreement:.1f}%")
                
                col_s1, col_s2 = st.columns(2)
                col_s1.metric("Non-Lab Sensitivity (for Lab High-Risk)", f"{sens:.1f}%")
                col_s2.metric("Non-Lab Specificity", f"{spec:.1f}%")
                
                # Summary statistics table
                with st.expander("📊 Summary Statistics by Risk Category"):
                    summary_by_cat = []
                    for cat in RISK_LABELS:
                        lab_count = (df_paired['risk_cat_lab'] == cat).sum()
                        nonlab_count = (df_paired['risk_cat_nonlab'] == cat).sum()
                        
                        summary_by_cat.append({
                            'Risk Category': cat,
                            'Lab Count': lab_count,
                            'Lab %': lab_count / len(df_paired) * 100,
                            'Non-Lab Count': nonlab_count,
                            'Non-Lab %': nonlab_count / len(df_paired) * 100,
                            'Difference (Count)': lab_count - nonlab_count
                        })
                    
                    summary_cat_df = pd.DataFrame(summary_by_cat)
                    
                    st.dataframe(
                        summary_cat_df.style.format({
                            'Lab Count': '{:,}',
                            'Lab %': '{:.1f}',
                            'Non-Lab Count': '{:,}',
                            'Non-Lab %': '{:.1f}',
                            'Difference (Count)': '{:+,}'
                        }).background_gradient(cmap='RdYlGn', subset=['Difference (Count)']),
                        use_container_width=True
                    )

                # --- 3E: NET RECLASSIFICATION IMPROVEMENT (NRI) ---
                st.divider()
                st.subheader("3D. Net Reclassification Improvement (NRI)")
                st.caption(
                    "Quantifying the 'net benefit' of using the Reference (Lab) over the Test (Non-Lab). "
                    "Positive NRI implies Lab classification improves upon Non-Lab. "
                    "(Note: Since Lab is the Gold Standard, this measures how 'wrong' Non-Lab is relative to Lab)."
                )

                # NRI Calculation Controls
                c_nri1, c_nri2 = st.columns(2)
                with c_nri1:
                    nri_thresh = st.selectbox("High Risk Threshold for NRI:", [10, 20], index=1)
                
                # Logic: Comparison of Non-Lab (Ref for improvement?) -> Lab (New)
                # Usually NRI tests if New Model improves over Old.
                # Here: Does LAB improve over NON-LAB? (Yes, obviously, it's the truth).
                # So we calculate NRI of Lab vs Non-Lab.
                # Event = Lab >= Threshold.
                # Input to nri function: y_true, p_new (Lab), p_ref (Non-Lab).
                # But p_new IS y_true effectively.
                # calculate_nri expects probabilities, creates categories inside.
                
                y_true = (df_paired['risk_lab'] >= nri_thresh).astype(int)
                p_new = df_paired['risk_lab']      # Lab (Gold Standard)
                p_ref = df_paired['risk_nonlab']   # Non-Lab (Baseline)
                
                nri_res, nri_e, nri_ne = calculate_nri(y_true, p_new, p_ref, threshold=nri_thresh)
                
                # Metric Display
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Overall NRI", f"{nri_res:.3f}", help="NRI > 0 means Lab classification is better")
                c_m2.metric("Event NRI (Sensitivity Gain)", f"{nri_e:.3f}", help="Net gain in classifying High Risk correctly")
                c_m3.metric("Non-Event NRI (Specificity Gain)", f"{nri_ne:.3f}", help="Net gain in classifying Low Risk correctly")
                
                # Visualization: Waterfall / Breakdown
                # Shows: Correct Up, Incorrect Down (for Events) | Correct Down, Incorrect Up (for Non-Events)
                cat_ref_bin = (p_ref >= nri_thresh).astype(int)
                cat_new_bin = (p_new >= nri_thresh).astype(int) # This matches y_true
                
                # Breakdown values
                nonlab_agree_high = ((cat_ref_bin==1) & (y_true==1)).sum()
                nonlab_agree_low = ((cat_ref_bin==0) & (y_true==0)).sum()
                lab_correct_high = ((cat_ref_bin==0) & (y_true==1)).sum()  # Non-Lab missed, Lab caught
                lab_correct_low = ((cat_ref_bin==1) & (y_true==0)).sum()   # Non-Lab false alarm, Lab cleared
                
                # Matplotlib grouped bar chart
                fig_nri_bar, ax_nri = plt.subplots(figsize=(10, 6))
                
                x = np.arange(2)
                width = 0.35
                
                bars_agree = ax_nri.bar(x - width/2, [nonlab_agree_high, nonlab_agree_low], width,
                                       label='Non-Lab Agreement', color='lightgrey',
                                       edgecolor='black', linewidth=0.5)
                bars_correct = ax_nri.bar(x + width/2, [lab_correct_high, lab_correct_low], width,
                                         label='Lab Correction (Improvement)', color=NATURE_COLORS['green'],
                                         edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for bar in bars_agree:
                    height = bar.get_height()
                    ax_nri.text(bar.get_x() + bar.get_width()/2, height + 5,
                               f'n={int(height):,}', ha='center', va='bottom', fontsize=10, fontfamily='serif')
                for bar, label_ in zip(bars_correct, ['Restored High Risk', 'Corrected False Alarm']):
                    height = bar.get_height()
                    ax_nri.text(bar.get_x() + bar.get_width()/2, height + 5,
                               f'{label_}\n(n={int(height):,})', ha='center', va='bottom', fontsize=9, fontfamily='serif')
                
                ax_nri.set_ylabel('Count', fontsize=12, fontfamily='serif')
                ax_nri.set_title('Reclassification Benefit of Lab Assessment', fontsize=14, fontweight='bold', fontfamily='serif')
                ax_nri.set_xticks(x)
                ax_nri.set_xticklabels(['True High Risk', 'True Low Risk'], fontsize=11, fontfamily='serif')
                ax_nri.legend(loc='upper right', frameon=True, framealpha=0.9)
                ax_nri.spines['top'].set_visible(False)
                ax_nri.spines['right'].set_visible(False)
                ax_nri.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig_nri_bar)
                add_svg_download_button(fig_nri_bar, "nri_reclassification_benefit", key="nri_bar_svg")
                plt.close(fig_nri_bar)

                # --- 3F: SMALL MULTIPLES (Misclassification by Stratum) ---
                st.divider()
                st.subheader("3E. Misclassification by Risk Stratum (Small Multiples)")
                st.caption("For each **True Lab-Based Risk Stratum**, how did Non-Lab classify participants?")
                
                # Logic: Facet by Lab Category (x-axis), Show stacked bars of Agreement Status
                strat_counts = df_paired.groupby(['risk_cat_lab', 'agree_status'], observed=False).size().reset_index(name='count')
                strat_totals = strat_counts.groupby('risk_cat_lab', observed=False)['count'].transform('sum')
                strat_counts['pct'] = (strat_counts['count'] / strat_totals) * 100
                
                status_order = ['Underestimation', 'Concordance', 'Overestimation']
                status_colors = {'Underestimation': AGREEMENT_COLORS['Underestimation'], 
                                'Concordance': AGREEMENT_COLORS['Concordance'], 
                                'Overestimation': AGREEMENT_COLORS['Overestimation']}
                
                # Matplotlib faceted bar chart (Small Multiples)
                n_cats = len(RISK_LABELS)
                fig_small, axes = plt.subplots(2, 3, figsize=(14, 8))
                axes = axes.flatten()
                
                for i, cat in enumerate(RISK_LABELS):
                    ax = axes[i]
                    cat_data = strat_counts[strat_counts['risk_cat_lab'] == cat]
                    
                    # Get percentages for each status
                    pcts = []
                    for status in status_order:
                        val = cat_data[cat_data['agree_status'] == status]['pct'].values
                        pcts.append(val[0] if len(val) > 0 else 0)
                    
                    x_pos = np.arange(len(status_order))
                    bars = ax.bar(x_pos, pcts, 
                                 color=[status_colors[s] for s in status_order],
                                 edgecolor='black', linewidth=0.5)
                    
                    # Add value labels
                    for bar, pct in zip(bars, pcts):
                        if pct > 2:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontfamily='serif')
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(['Under', 'Conc', 'Over'], fontsize=9)
                    ax.set_ylim(0, 100)
                    ax.set_title(cat, fontsize=11, fontweight='bold', fontfamily='serif')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    if i % 3 == 0:
                        ax.set_ylabel('Percentage (%)', fontsize=10, fontfamily='serif')
                
                # Hide unused subplot
                axes[-1].set_visible(False)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [mpatches.Patch(facecolor=status_colors[s], edgecolor='black', 
                                                  label=s) for s in status_order]
                fig_small.legend(handles=legend_elements, loc='upper center', 
                               bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=10)
                
                fig_small.suptitle('Misclassification Profile by True Risk Level', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout(rect=[0, 0.05, 1, 1])
                st.pyplot(fig_small)
                add_svg_download_button(fig_small, "misclassification_small_multiples", key="small_mult_svg")
                plt.close(fig_small)

                # --- 3F: FACETED RESIDUAL HEATMAP (MOSAIC PROXY) ---
                st.divider()
                st.subheader("3F. Faceted Residual Heatmap (Association Analysis)")
                st.caption("Visualizing **Pearson Residuals** to identify cell combinations that are significantly over-represented (Blue) or under-represented (Red) compared to what would be expected by random chance (Independence).")
                
                # Controls
                resid_strat = st.selectbox("Stratify Residuals By:", ["Glycaemic Group", "Gender", "Age Band"], key="resid_strat")
                strat_col_resid = {"Glycaemic Group": "gly_group", "Gender": "gender", "Age Band": "age_band"}[resid_strat]
                
                # Logic: Calculate Expectation and Residuals for each Stratum
                if strat_col_resid in df_paired.columns:
                    groups = sorted(df_paired[strat_col_resid].astype(str).unique())
                    cols = st.columns(min(len(groups), 3))
                    
                    for i, grp in enumerate(groups):
                        sub_df = df_paired[df_paired[strat_col_resid].astype(str) == grp]
                        if len(sub_df) > 0:
                            # Crosstab Obs
                            obs = pd.crosstab(sub_df['risk_cat_lab'], sub_df['risk_cat_nonlab']).reindex(index=RISK_LABELS, columns=RISK_LABELS, fill_value=0)
                            # Chi2 Exp
                            chi2, p, dof, exp = stats.chi2_contingency(obs + 0.0001) # Avoid zero division
                            exp_df = pd.DataFrame(exp, index=obs.index, columns=obs.columns)
                            # Residuals: (Obs - Exp) / sqrt(Exp)
                            resid = (obs - exp_df) / np.sqrt(exp_df)
                            
                            # Plot with Matplotlib
                            max_bal = max(abs(resid.min().min()), abs(resid.max().max()), 3)
                            
                            fig_res, ax_res = plt.subplots(figsize=(6, 5))
                            
                            # Create heatmap with annotations showing counts
                            sns.heatmap(resid, ax=ax_res, cmap='RdBu', center=0,
                                       vmin=-max_bal, vmax=max_bal,
                                       annot=obs.values, fmt='d',
                                       linewidths=0.5, linecolor='white',
                                       cbar_kws={'label': 'Residual', 'shrink': 0.8})
                            
                            ax_res.set_xlabel('Non-Lab Category', fontsize=10, fontfamily='serif')
                            ax_res.set_ylabel('Lab Category', fontsize=10, fontfamily='serif')
                            ax_res.set_title(f'{grp} (p={p:.3g})', fontsize=12, fontweight='bold', fontfamily='serif')
                            ax_res.set_xticklabels(ax_res.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                            ax_res.set_yticklabels(ax_res.get_yticklabels(), rotation=0, fontsize=8)
                            
                            plt.tight_layout()
                            
                            col_idx = i % 3
                            with cols[col_idx]:
                                st.pyplot(fig_res)
                                add_svg_download_button(fig_res, f"residual_heatmap_{grp.replace(' ', '_').lower()}", 
                                                       key=f"resid_{grp.replace(' ', '_')}_svg")
                            plt.close(fig_res)
                    
                    # Common Colorbar explanation
                    st.markdown("**Interpretation:** **Blue** = More cases than expected (Strong Association). **Red** = Fewer cases than expected. **White** = As expected.")
                
                # --- 3G: PARALLEL CATEGORIES PLOT ---
                st.divider()
                st.subheader("3G. Multivariate Flow (Parallel Categories)")
                st.caption("Trace individual paths through demographic and clinical factors to agreement outcome.")
                
                # Prep Data
                # Bin continuous vars if needed or use existing bands
                # Ensure we have: Age Band, Gender, BMI Band (if avail), Lab Risk, Non-Lab Risk, Agree Status
                
                # Check for BMI Band or create
                if 'bmi_band' not in df_paired.columns:
                    # Create simple BMI 3-levels
                    def get_bmi_cat(b):
                        if pd.isna(b): return 'Unknown'
                        if b < 25: return 'Normal'
                        if b < 30: return 'Overweight'
                        return 'Obese'
                    df_paired['bmi_simple'] = df_paired['bmi'].apply(get_bmi_cat)
                    bmi_node = 'bmi_simple'
                else:
                    bmi_node = 'bmi_band'

                # Limit data size for performance if needed
                pc_df = df_paired.copy()
                if len(pc_df) > 2000:
                    pc_df = pc_df.sample(2000, random_state=42)
                    st.caption(f"Visualizing random sample of 2,000 participants (Total: {len(df_paired)})")
                
                # Data preparation is sufficient for the Matplotlib charts below
                
                # Replace Parcats with Stacked Bar Charts of Outcome by Key Factors
                st.markdown("### Misclassification Patterns by Demographic Factors")
                
                factors = ['gender', 'age_band', bmi_node]
                factor_labels = {'gender': 'Gender', 'age_band': 'Age Group', bmi_node: 'BMI Category'}
                
                fig_multi, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
                
                outcomes = ['Concordance', 'Underestimation', 'Overestimation']
                colors_outcome = {'Concordance': '#27ae60', 'Underestimation': '#d32f2f', 'Overestimation': '#fbc02d'}
                
                for idx, col in enumerate(factors):
                    ax = axes[idx]
                    # Cross-tabulate and normalize
                    xtab = pd.crosstab(pc_df[col], pc_df['agree_status'], normalize='index') * 100
                    # Reorder columns if needed
                    xtab = xtab[outcomes] if all(o in xtab.columns for o in outcomes) else xtab
                    
                    x = np.arange(len(xtab))
                    bottom = np.zeros(len(xtab))
                    
                    for outcome in outcomes:
                        if outcome in xtab.columns:
                            y = xtab[outcome].values
                            ax.bar(x, y, bottom=bottom, label=outcome, color=colors_outcome[outcome], 
                                   edgecolor='black', linewidth=0.5, width=0.7)
                            bottom += y
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(xtab.index, rotation=45, ha='right', fontsize=9)
                    ax.set_xlabel(factor_labels[col], fontsize=11, fontfamily='serif')
                    if idx == 0:
                        ax.set_ylabel('Percentage (%)', fontsize=11, fontfamily='serif')
                    ax.set_title(f'By {factor_labels[col]}', fontsize=12, fontweight='bold', fontfamily='serif')
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    if idx == 0:
                        ax.legend(loc='upper center', bbox_to_anchor=(1.7, 1.15), ncol=3, frameon=False)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.85) # Make room for legend
                st.pyplot(fig_multi)
                add_svg_download_button(fig_multi, "misclass_factors_stacked", key="misclass_factors_svg")
                plt.close(fig_multi)
                
                # ==========================================
                # SECTION 8.5: RISK PREVALENCE THRESHOLDS
                # ==========================================
                st.divider()
                st.header("8.5 📊 Risk Prevalence by Threshold (1%-30%)")
                st.markdown("Prevalence of individuals exceeding various risk thresholds from 1% to 30%")
                
                # Calculate prevalence at each threshold (1% to 30%)
                risk_thresholds = list(range(1, 31))
                prevalence_data = []
                
                for threshold in risk_thresholds:
                    lab_above = (df_paired['risk_lab'] >= threshold).sum()
                    nonlab_above = (df_paired['risk_nonlab'] >= threshold).sum()
                    n_total = len(df_paired)
                    
                    lab_pct = (lab_above / n_total) * 100
                    nonlab_pct = (nonlab_above / n_total) * 100
                    
                    # Wilson CI
                    lab_ci_low, lab_ci_high = calculate_wilson_ci(lab_above, n_total)
                    nonlab_ci_low, nonlab_ci_high = calculate_wilson_ci(nonlab_above, n_total)
                    
                    prevalence_data.append({
                        'threshold': threshold,
                        'lab_pct': lab_pct,
                        'lab_ci_low': lab_ci_low,
                        'lab_ci_high': lab_ci_high,
                        'lab_n': lab_above,
                        'nonlab_pct': nonlab_pct,
                        'nonlab_ci_low': nonlab_ci_low,
                        'nonlab_ci_high': nonlab_ci_high,
                        'nonlab_n': nonlab_above
                    })
                
                prev_threshold_df = pd.DataFrame(prevalence_data)
                
                # Create Matplotlib visualization
                fig_threshold, ax_thresh = plt.subplots(figsize=(10, 6))
                
                # Lab-based line with CI fill
                ax_thresh.plot(prev_threshold_df['threshold'], prev_threshold_df['lab_pct'], 
                              'o-', color='#2E86AB', linewidth=3, markersize=6, label='Lab-based')
                ax_thresh.fill_between(prev_threshold_df['threshold'], 
                                      prev_threshold_df['lab_ci_low'], prev_threshold_df['lab_ci_high'], 
                                      color='#2E86AB', alpha=0.2)
                
                # Non-lab based line with CI fill
                ax_thresh.plot(prev_threshold_df['threshold'], prev_threshold_df['nonlab_pct'], 
                              'd--', color='#A23B72', linewidth=3, markersize=6, label='Non-lab based')
                ax_thresh.fill_between(prev_threshold_df['threshold'], 
                                      prev_threshold_df['nonlab_ci_low'], prev_threshold_df['nonlab_ci_high'], 
                                      color='#A23B72', alpha=0.2)
                
                # Add vertical reference lines for key thresholds
                for thresh_val, color, name in [(10, 'orange', '10% Threshold'), (20, 'red', '20% Threshold')]:
                    ax_thresh.axvline(x=thresh_val, color=color, linestyle=':', linewidth=2)
                    ax_thresh.text(thresh_val + 0.5, 95, name, color=color, fontsize=9, 
                                  rotation=90, verticalalignment='top', fontfamily='serif')
                
                ax_thresh.set_xticks(np.arange(0, 31, 5))
                ax_thresh.set_xlabel('Risk Threshold (%)', fontsize=12, fontfamily='serif')
                ax_thresh.set_ylabel('Prevalence (%)', fontsize=12, fontfamily='serif')
                ax_thresh.set_title('Prevalence of Risk ≥ Threshold: Lab-based vs Non-lab based (1% to 30%)',
                                   fontsize=14, fontweight='bold', fontfamily='serif')
                ax_thresh.set_ylim(0, 105)
                ax_thresh.set_xlim(0, 31)
                ax_thresh.spines['top'].set_visible(False)
                ax_thresh.spines['right'].set_visible(False)
                ax_thresh.grid(True, alpha=0.3)
                ax_thresh.legend(loc='lower left', frameon=True, fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig_threshold)
                add_svg_download_button(fig_threshold, "prevalence_by_threshold", key="prev_thresh_svg")
                plt.close(fig_threshold)
                
                # Summary table for key thresholds
                with st.expander("📊 Detailed Prevalence by Risk Threshold"):
                    key_thresholds = [5, 10, 15, 20, 25, 30]
                    key_prev_df = prev_threshold_df[prev_threshold_df['threshold'].isin(key_thresholds)].copy()
                    
                    key_prev_df['Lab Prevalence (95% CI)'] = key_prev_df.apply(
                        lambda x: f"{x['lab_pct']:.1f}% ({x['lab_ci_low']:.1f}-{x['lab_ci_high']:.1f})", axis=1
                    )
                    key_prev_df['Non-Lab Prevalence (95% CI)'] = key_prev_df.apply(
                        lambda x: f"{x['nonlab_pct']:.1f}% ({x['nonlab_ci_low']:.1f}-{x['nonlab_ci_high']:.1f})", axis=1
                    )
                    key_prev_df['Difference'] = key_prev_df['lab_pct'] - key_prev_df['nonlab_pct']
                    
                    display_threshold_df = key_prev_df[['threshold', 'lab_n', 'Lab Prevalence (95% CI)', 
                                                         'nonlab_n', 'Non-Lab Prevalence (95% CI)', 'Difference']].copy()
                    display_threshold_df.columns = ['Threshold (%)', 'Lab N', 'Lab Prevalence (95% CI)', 
                                                    'Non-Lab N', 'Non-Lab Prevalence (95% CI)', 'Difference (%)']
                    
                    st.dataframe(
                        display_threshold_df.style.format({
                            'Lab N': '{:,}',
                            'Non-Lab N': '{:,}',
                            'Difference (%)': '{:+.1f}%'
                        }).background_gradient(cmap='RdYlGn_r', subset=['Difference (%)']),
                        use_container_width=True
                    )
                
                # Key metrics for important thresholds
                st.markdown("### Key Threshold Summary")
                col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                
                with col_t1:
                    lab_5 = prev_threshold_df[prev_threshold_df['threshold'] == 5]['lab_pct'].values[0]
                    nonlab_5 = prev_threshold_df[prev_threshold_df['threshold'] == 5]['nonlab_pct'].values[0]
                    st.metric("≥5% Risk", f"Lab: {lab_5:.1f}%", f"Non-Lab: {nonlab_5:.1f}%")
                
                with col_t2:
                    lab_10 = prev_threshold_df[prev_threshold_df['threshold'] == 10]['lab_pct'].values[0]
                    nonlab_10 = prev_threshold_df[prev_threshold_df['threshold'] == 10]['nonlab_pct'].values[0]
                    st.metric("≥10% Risk", f"Lab: {lab_10:.1f}%", f"Non-Lab: {nonlab_10:.1f}%")
                
                with col_t3:
                    lab_20 = prev_threshold_df[prev_threshold_df['threshold'] == 20]['lab_pct'].values[0]
                    nonlab_20 = prev_threshold_df[prev_threshold_df['threshold'] == 20]['nonlab_pct'].values[0]
                    st.metric("≥20% Risk", f"Lab: {lab_20:.1f}%", f"Non-Lab: {nonlab_20:.1f}%")
                
                with col_t4:
                    lab_30 = prev_threshold_df[prev_threshold_df['threshold'] == 30]['lab_pct'].values[0]
                    nonlab_30 = prev_threshold_df[prev_threshold_df['threshold'] == 30]['nonlab_pct'].values[0]
                    st.metric("≥30% Risk", f"Lab: {lab_30:.1f}%", f"Non-Lab: {nonlab_30:.1f}%")
                
                # ==========================================
                # SECTION 8.6: STRATIFIED ANALYSIS
                # ==========================================
                st.divider()
                st.header("8.6 👥 Stratified Analysis by Demographics")
                st.markdown("Risk distribution comparison stratified by key demographic and clinical variables")
                
                # Create sub-tabs for different stratification analyses
                strat_tab1, strat_tab2, strat_tab3, strat_tab4, strat_tab5 = st.tabs([
                    "👤 Gender", "📏 BMI Category", "💓 Blood Pressure", "🩺 Diabetes", "🚬 Smoking Status"
                ])
                
                # Helper function for stratified analysis
                def create_stratified_risk_plot(df, strat_col, strat_label, category_order=None):
                    """Create a stratified risk distribution plot for lab vs non-lab"""
                    if strat_col not in df.columns:
                        return None, None, f"Column '{strat_col}' not found in dataset"
                    
                    # Calculate mean risk and prevalence by stratum
                    strat_stats = []
                    
                    categories = category_order if category_order else df[strat_col].dropna().unique().tolist()
                    
                    for cat in categories:
                        cat_data = df[df[strat_col] == cat]
                        if len(cat_data) > 0:
                            n = len(cat_data)
                            
                            # Mean risk
                            lab_mean = cat_data['risk_lab'].mean()
                            nonlab_mean = cat_data['risk_nonlab'].mean()
                            lab_sem = cat_data['risk_lab'].sem()
                            nonlab_sem = cat_data['risk_nonlab'].sem()
                            
                            # High risk prevalence (≥20%)
                            lab_high = (cat_data['risk_lab'] >= 20).sum()
                            nonlab_high = (cat_data['risk_nonlab'] >= 20).sum()
                            lab_high_pct = (lab_high / n) * 100
                            nonlab_high_pct = (nonlab_high / n) * 100
                            
                            # CIs
                            lab_ci_low, lab_ci_high = calculate_wilson_ci(lab_high, n)
                            nonlab_ci_low, nonlab_ci_high = calculate_wilson_ci(nonlab_high, n)
                            
                            strat_stats.append({
                                'category': str(cat),
                                'n': n,
                                'lab_mean': lab_mean,
                                'lab_sem': lab_sem,
                                'nonlab_mean': nonlab_mean,
                                'nonlab_sem': nonlab_sem,
                                'lab_high_pct': lab_high_pct,
                                'lab_high_ci_low': lab_ci_low,
                                'lab_high_ci_high': lab_ci_high,
                                'nonlab_high_pct': nonlab_high_pct,
                                'nonlab_high_ci_low': nonlab_ci_low,
                                'nonlab_high_ci_high': nonlab_ci_high
                            })
                    
                    if not strat_stats:
                        return None, None, "No valid data for stratification"
                    
                    strat_df = pd.DataFrame(strat_stats)
                    
                    # Create Matplotlib grouped bar chart for mean risk
                    fig_mean, ax_mean = plt.subplots(figsize=(7, 5))
                    
                    x = np.arange(len(strat_df))
                    width = 0.35
                    
                    # Mean Risk Bars
                    ax_mean.bar(x - width/2, strat_df['lab_mean'], width, label='Lab-based', 
                                yerr=1.96 * strat_df['lab_sem'], capsize=3,
                                color='#2E86AB', edgecolor='black', linewidth=0.5)
                    ax_mean.bar(x + width/2, strat_df['nonlab_mean'], width, label='Non-lab based', 
                                yerr=1.96 * strat_df['nonlab_sem'], capsize=3,
                                color='#A23B72', edgecolor='black', linewidth=0.5)
                    
                    ax_mean.set_xticks(x)
                    ax_mean.set_xticklabels(strat_df['category'], rotation=45, ha='right', fontsize=9)
                    ax_mean.set_xlabel(strat_label, fontsize=11, fontfamily='serif')
                    ax_mean.set_ylabel('Mean Risk (%)', fontsize=11, fontfamily='serif')
                    ax_mean.set_title(f'Mean 10-Year CVD Risk by {strat_label}', fontsize=12, fontweight='bold', fontfamily='serif')
                    
                    # Add reference lines
                    ax_mean.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='10% Threshold')
                    ax_mean.axhline(y=20, color='red', linestyle=':', linewidth=1.5, alpha=0.8, label='20% Threshold')
                    
                    ax_mean.spines['top'].set_visible(False)
                    ax_mean.spines['right'].set_visible(False)
                    ax_mean.grid(True, alpha=0.3, axis='y')
                    ax_mean.legend(loc='upper right', frameon=True, fontsize=8)
                    plt.close(fig_mean) # Close to prevent auto-display in some envs, standard practice before returning
                    
                    # Create Matplotlib grouped bar chart for high-risk prevalence (≥20%)
                    fig_high, ax_high = plt.subplots(figsize=(7, 5))
                    
                    # High Risk Bars
                    # Calculate error bar heights for CIs
                    lab_err = [
                        strat_df['lab_high_pct'] - strat_df['lab_high_ci_low'],
                        strat_df['lab_high_ci_high'] - strat_df['lab_high_pct']
                    ]
                    nonlab_err = [
                        strat_df['nonlab_high_pct'] - strat_df['nonlab_high_ci_low'],
                        strat_df['nonlab_high_ci_high'] - strat_df['nonlab_high_pct']
                    ]
                    
                    ax_high.bar(x - width/2, strat_df['lab_high_pct'], width, label='Lab-based ≥20%', 
                                yerr=lab_err, capsize=3,
                                color='#2E86AB', edgecolor='black', linewidth=0.5)
                    ax_high.bar(x + width/2, strat_df['nonlab_high_pct'], width, label='Non-lab based ≥20%', 
                                yerr=nonlab_err, capsize=3,
                                color='#A23B72', edgecolor='black', linewidth=0.5)
                    
                    ax_high.set_xticks(x)
                    ax_high.set_xticklabels(strat_df['category'], rotation=45, ha='right', fontsize=9)
                    ax_high.set_xlabel(strat_label, fontsize=11, fontfamily='serif')
                    ax_high.set_ylabel('Prevalence (%)', fontsize=11, fontfamily='serif')
                    ax_high.set_title(f'High-Risk (≥20%) Prevalence by {strat_label}', fontsize=12, fontweight='bold', fontfamily='serif')
                    
                    ax_high.spines['top'].set_visible(False)
                    ax_high.spines['right'].set_visible(False)
                    ax_high.grid(True, alpha=0.3, axis='y')
                    ax_high.legend(loc='upper right', frameon=True, fontsize=8)
                    plt.close(fig_high)

                    return fig_mean, fig_high, strat_df
                
                # --- Gender Stratification ---
                with strat_tab1:
                    st.subheader("Risk Distribution by Gender")
                    
                    if 'gender' in df_paired.columns:
                        fig_gender_mean, fig_gender_high, gender_stats = create_stratified_risk_plot(
                            df_paired, 'gender', 'Gender', category_order=['Male', 'Female']
                        )
                        
                        if fig_gender_mean:
                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                st.pyplot(fig_gender_mean)
                                add_svg_download_button(fig_gender_mean, "gender_risk_mean", key="gen_mean_svg")
                            with col_g2:
                                st.pyplot(fig_gender_high)
                                add_svg_download_button(fig_gender_high, "gender_risk_high", key="gen_high_svg")
                            
                            # Statistics table
                            with st.expander("📊 Gender-Stratified Statistics"):
                                if isinstance(gender_stats, pd.DataFrame):
                                    display_gender = gender_stats[['category', 'n', 'lab_mean', 'nonlab_mean', 
                                                                    'lab_high_pct', 'nonlab_high_pct']].copy()
                                    display_gender.columns = ['Gender', 'N', 'Lab Mean (%)', 'Non-Lab Mean (%)',
                                                              'Lab ≥20% (%)', 'Non-Lab ≥20% (%)']
                                    st.dataframe(
                                        display_gender.style.format({
                                            'N': '{:,}',
                                            'Lab Mean (%)': '{:.1f}',
                                            'Non-Lab Mean (%)': '{:.1f}',
                                            'Lab ≥20% (%)': '{:.1f}',
                                            'Non-Lab ≥20% (%)': '{:.1f}'
                                        }),
                                        use_container_width=True
                                    )
                    else:
                        st.warning("Gender column not available in dataset")
                
                # --- BMI Category Stratification ---
                with strat_tab2:
                    st.subheader("Risk Distribution by BMI Category")
                    
                    if 'bmi_band' in df_paired.columns:
                        # Define BMI categories order
                        bmi_order = df_paired['bmi_band'].dropna().unique().tolist()
                        bmi_order.sort()
                        
                        fig_bmi_mean, fig_bmi_high, bmi_stats = create_stratified_risk_plot(
                            df_paired, 'bmi_band', 'BMI Category', category_order=bmi_order
                        )
                        
                        if fig_bmi_mean:
                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                st.pyplot(fig_bmi_mean)
                                add_svg_download_button(fig_bmi_mean, "bmi_risk_mean", key="bmi_mean_svg")
                            with col_b2:
                                st.pyplot(fig_bmi_high)
                                add_svg_download_button(fig_bmi_high, "bmi_risk_high", key="bmi_high_svg")
                            
                            with st.expander("📊 BMI-Stratified Statistics"):
                                if isinstance(bmi_stats, pd.DataFrame):
                                    display_bmi = bmi_stats[['category', 'n', 'lab_mean', 'nonlab_mean', 
                                                             'lab_high_pct', 'nonlab_high_pct']].copy()
                                    display_bmi.columns = ['BMI Category', 'N', 'Lab Mean (%)', 'Non-Lab Mean (%)',
                                                           'Lab ≥20% (%)', 'Non-Lab ≥20% (%)']
                                    st.dataframe(
                                        display_bmi.style.format({
                                            'N': '{:,}',
                                            'Lab Mean (%)': '{:.1f}',
                                            'Non-Lab Mean (%)': '{:.1f}',
                                            'Lab ≥20% (%)': '{:.1f}',
                                            'Non-Lab ≥20% (%)': '{:.1f}'
                                        }),
                                        use_container_width=True
                                    )
                    else:
                        st.warning("BMI band column not available in dataset")
                
                # --- Blood Pressure Category Stratification ---
                with strat_tab3:
                    st.subheader("Risk Distribution by Blood Pressure")
                    
                    if 'sbp_band' in df_paired.columns:
                        sbp_order = df_paired['sbp_band'].dropna().unique().tolist()
                        sbp_order.sort()
                        
                        fig_sbp_mean, fig_sbp_high, sbp_stats = create_stratified_risk_plot(
                            df_paired, 'sbp_band', 'SBP Category', category_order=sbp_order
                        )
                        
                        if fig_sbp_mean:
                            col_s1, col_s2 = st.columns(2)
                            with col_s1:
                                st.pyplot(fig_sbp_mean)
                                add_svg_download_button(fig_sbp_mean, "sbp_risk_mean", key="sbp_mean_svg")
                            with col_s2:
                                st.pyplot(fig_sbp_high)
                                add_svg_download_button(fig_sbp_high, "sbp_risk_high", key="sbp_high_svg")
                            
                            with st.expander("📊 SBP-Stratified Statistics"):
                                if isinstance(sbp_stats, pd.DataFrame):
                                    display_sbp = sbp_stats[['category', 'n', 'lab_mean', 'nonlab_mean', 
                                                             'lab_high_pct', 'nonlab_high_pct']].copy()
                                    display_sbp.columns = ['SBP Category', 'N', 'Lab Mean (%)', 'Non-Lab Mean (%)',
                                                           'Lab ≥20% (%)', 'Non-Lab ≥20% (%)']
                                    st.dataframe(
                                        display_sbp.style.format({
                                            'N': '{:,}',
                                            'Lab Mean (%)': '{:.1f}',
                                            'Non-Lab Mean (%)': '{:.1f}',
                                            'Lab ≥20% (%)': '{:.1f}',
                                            'Non-Lab ≥20% (%)': '{:.1f}'
                                        }),
                                        use_container_width=True
                                    )
                    elif 'bp_category' in df_paired.columns:
                        bp_order = ['Normal', 'Elevated', 'Stage1', 'Stage2', 'Crisis']
                        available_bp = [bp for bp in bp_order if bp in df_paired['bp_category'].unique()]
                        
                        fig_bp_mean, fig_bp_high, bp_stats = create_stratified_risk_plot(
                            df_paired, 'bp_category', 'BP Category', category_order=available_bp
                        )
                        
                        if fig_bp_mean:
                            col_bp1, col_bp2 = st.columns(2)
                            with col_bp1:
                                st.pyplot(fig_bp_mean)
                                add_svg_download_button(fig_bp_mean, "bp_risk_mean", key="bp_mean_svg")
                            with col_bp2:
                                st.pyplot(fig_bp_high)
                                add_svg_download_button(fig_bp_high, "bp_risk_high", key="bp_high_svg")
                    else:
                        st.warning("Blood pressure category column not available in dataset")
                
                # --- Diabetes Stratification ---
                with strat_tab4:
                    st.subheader("Risk Distribution by Diabetes Status")
                    
                    if 'has_diabetes' in df_paired.columns:
                        # Create readable labels
                        df_paired_diab = df_paired.copy()
                        df_paired_diab['diabetes_label'] = df_paired_diab['has_diabetes'].map({
                            True: 'Diabetic', False: 'Non-Diabetic', 1: 'Diabetic', 0: 'Non-Diabetic',
                            'Yes': 'Diabetic', 'No': 'Non-Diabetic'
                        })
                        
                        fig_diab_mean, fig_diab_high, diab_stats = create_stratified_risk_plot(
                            df_paired_diab, 'diabetes_label', 'Diabetes Status', 
                            category_order=['Non-Diabetic', 'Diabetic']
                        )
                        
                        if fig_diab_mean:
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.pyplot(fig_diab_mean)
                                add_svg_download_button(fig_diab_mean, "diab_risk_mean", key="diab_mean_svg")
                            with col_d2:
                                st.pyplot(fig_diab_high)
                                add_svg_download_button(fig_diab_high, "diab_risk_high", key="diab_high_svg")
                            
                            with st.expander("📊 Diabetes-Stratified Statistics"):
                                if isinstance(diab_stats, pd.DataFrame):
                                    display_diab = diab_stats[['category', 'n', 'lab_mean', 'nonlab_mean', 
                                                               'lab_high_pct', 'nonlab_high_pct']].copy()
                                    display_diab.columns = ['Diabetes Status', 'N', 'Lab Mean (%)', 'Non-Lab Mean (%)',
                                                            'Lab ≥20% (%)', 'Non-Lab ≥20% (%)']
                                    st.dataframe(
                                        display_diab.style.format({
                                            'N': '{:,}',
                                            'Lab Mean (%)': '{:.1f}',
                                            'Non-Lab Mean (%)': '{:.1f}',
                                            'Lab ≥20% (%)': '{:.1f}',
                                            'Non-Lab ≥20% (%)': '{:.1f}'
                                        }),
                                        use_container_width=True
                                    )
                    elif 'diab_group' in df_paired.columns:
                        fig_diab_mean, fig_diab_high, diab_stats = create_stratified_risk_plot(
                            df_paired, 'diab_group', 'Diabetes Group'
                        )
                        
                        if fig_diab_mean:
                            col_d1, col_d2 = st.columns(2)
                            with col_d1:
                                st.pyplot(fig_diab_mean)
                                add_svg_download_button(fig_diab_mean, "diab_group_mean", key="diab_grp_mean_svg")
                            with col_d2:
                                st.pyplot(fig_diab_high)
                                add_svg_download_button(fig_diab_high, "diab_group_high", key="diab_grp_high_svg")
                    else:
                        st.warning("Diabetes status column not available in dataset")
                
                # --- Smoking Status Stratification ---
                with strat_tab5:
                    st.subheader("Risk Distribution by Smoking Status")
                    
                    if 'smoker_who' in df_paired.columns:
                        smoker_order = ['Non-smoker', 'Smoker']
                        available_smoker = [s for s in smoker_order if s in df_paired['smoker_who'].unique()]
                        
                        fig_smoke_mean, fig_smoke_high, smoke_stats = create_stratified_risk_plot(
                            df_paired, 'smoker_who', 'Smoking Status', category_order=available_smoker
                        )
                        
                        if fig_smoke_mean:
                            col_sm1, col_sm2 = st.columns(2)
                            with col_sm1:
                                st.pyplot(fig_smoke_mean)
                                add_svg_download_button(fig_smoke_mean, "smoke_risk_mean", key="smoke_mean_svg")
                            with col_sm2:
                                st.pyplot(fig_smoke_high)
                                add_svg_download_button(fig_smoke_high, "smoke_risk_high", key="smoke_high_svg")
                            
                            with st.expander("📊 Smoking-Stratified Statistics"):
                                if isinstance(smoke_stats, pd.DataFrame):
                                    display_smoke = smoke_stats[['category', 'n', 'lab_mean', 'nonlab_mean', 
                                                                  'lab_high_pct', 'nonlab_high_pct']].copy()
                                    display_smoke.columns = ['Smoking Status', 'N', 'Lab Mean (%)', 'Non-Lab Mean (%)',
                                                             'Lab ≥20% (%)', 'Non-Lab ≥20% (%)']
                                    st.dataframe(
                                        display_smoke.style.format({
                                            'N': '{:,}',
                                            'Lab Mean (%)': '{:.1f}',
                                            'Non-Lab Mean (%)': '{:.1f}',
                                            'Lab ≥20% (%)': '{:.1f}',
                                            'Non-Lab ≥20% (%)': '{:.1f}'
                                        }),
                                        use_container_width=True
                                    )
                    elif 'smoker' in df_paired.columns:
                        df_paired_smk = df_paired.copy()
                        df_paired_smk['smoker_label'] = df_paired_smk['smoker'].map({
                            True: 'Smoker', False: 'Non-Smoker', 1: 'Smoker', 0: 'Non-Smoker',
                            'Yes': 'Smoker', 'No': 'Non-Smoker'
                        })
                        
                        fig_smoke_mean, fig_smoke_high, smoke_stats = create_stratified_risk_plot(
                            df_paired_smk, 'smoker_label', 'Smoking Status',
                            category_order=['Non-Smoker', 'Smoker']
                        )
                        
                        if fig_smoke_mean:
                            col_sm1, col_sm2 = st.columns(2)
                            with col_sm1:
                                st.pyplot(fig_smoke_mean)
                                add_svg_download_button(fig_smoke_mean, "smoke_risk_mean_v2", key="smoke_mean_v2_svg")
                            with col_sm2:
                                st.pyplot(fig_smoke_high)
                                add_svg_download_button(fig_smoke_high, "smoke_risk_high_v2", key="smoke_high_v2_svg")
                    else:
                        st.warning("Smoking status column not available in dataset")
                
                # ==========================================
                # SECTION 8.7: MULTI-VARIABLE HEATMAP
                # ==========================================
                st.divider()
                st.header("8.7 🔥 Multi-Variable Risk Heatmap")
                st.markdown("Interactive heatmap showing mean risk across multiple stratification variables")
                
                # Create a combined analysis heatmap
                if 'gender' in df_paired.columns and 'age_band' in df_paired.columns:
                    # Create pivot table for gender × age
                    heatmap_data_lab = df_paired.pivot_table(
                        values='risk_lab', 
                        index='gender', 
                        columns='age_band', 
                        aggfunc='mean'
                    )
                    
                    heatmap_data_nonlab = df_paired.pivot_table(
                        values='risk_nonlab', 
                        index='gender', 
                        columns='age_band', 
                        aggfunc='mean'
                    )
                    
                    # Reorder columns
                    age_order = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
                    available_ages = [a for a in age_order if a in heatmap_data_lab.columns]
                    heatmap_data_lab = heatmap_data_lab[available_ages]
                    heatmap_data_nonlab = heatmap_data_nonlab[available_ages]
                    
                    col_h1, col_h2 = st.columns(2)
                    
                    with col_h1:
                        st.markdown("#### Lab-based Mean Risk by Gender × Age")
                        fig_heat_lab, ax_hl = plt.subplots(figsize=(8, 4))
                        sns.heatmap(heatmap_data_lab, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax_hl,
                                   cbar_kws={'label': 'Mean Risk (%)'})
                        ax_hl.set_xlabel('Age Group', fontsize=10, fontfamily='serif')
                        ax_hl.set_ylabel('Gender', fontsize=10, fontfamily='serif')
                        ax_hl.set_title("Lab-based Mean Risk", fontsize=11, fontweight='bold', fontfamily='serif')
                        plt.tight_layout()
                        st.pyplot(fig_heat_lab)
                        add_svg_download_button(fig_heat_lab, "lab_heatmap", key="lab_heat_svg")
                        plt.close(fig_heat_lab)
                    
                    with col_h2:
                        st.markdown("#### Non-Lab Mean Risk by Gender × Age")
                        fig_heat_nonlab, ax_hn = plt.subplots(figsize=(8, 4))
                        sns.heatmap(heatmap_data_nonlab, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax_hn,
                                   cbar_kws={'label': 'Mean Risk (%)'})
                        ax_hn.set_xlabel('Age Group', fontsize=10, fontfamily='serif')
                        ax_hn.set_ylabel('Gender', fontsize=10, fontfamily='serif')
                        ax_hn.set_title("Non-Lab Mean Risk", fontsize=11, fontweight='bold', fontfamily='serif')
                        plt.tight_layout()
                        st.pyplot(fig_heat_nonlab)
                        add_svg_download_button(fig_heat_nonlab, "nonlab_heatmap", key="nonlab_heat_svg")
                        plt.close(fig_heat_nonlab)
                    
                    # Difference heatmap
                    st.markdown("#### Risk Difference (Lab - Non-Lab) by Gender × Age")
                    heatmap_diff = heatmap_data_lab - heatmap_data_nonlab
                    
                    fig_heat_diff, ax_hd = plt.subplots(figsize=(8, 4))
                    sns.heatmap(heatmap_diff, annot=True, fmt='.1f', cmap='RdBu', center=0, ax=ax_hd,
                               cbar_kws={'label': 'Difference (%)'})
                    ax_hd.set_xlabel('Age Group', fontsize=10, fontfamily='serif')
                    ax_hd.set_ylabel('Gender', fontsize=10, fontfamily='serif')
                    ax_hd.set_title("Risk Difference (Lab - Non-Lab)", fontsize=11, fontweight='bold', fontfamily='serif')
                    plt.tight_layout()
                    st.pyplot(fig_heat_diff)
                    add_svg_download_button(fig_heat_diff, "diff_heatmap", key="diff_heat_svg")
                    plt.close(fig_heat_diff)
                    
                    st.info("""
                    📊 **Interpretation:**
                    - **Positive values (blue):** Lab-based assessment yields higher risk estimates
                    - **Negative values (red):** Non-lab based assessment yields higher risk estimates
                    - The difference varies by age and gender, reflecting the different inputs used by each method
                    """)
                
                # ==========================================
                # SECTION 8.7A: TREATMENT MISCLASSIFICATION ANALYSIS
                # ==========================================
                st.divider()
                st.header("8.7A 💊 Treatment Misclassification (Clinical Actionability)")
                st.markdown("""
                **Key Question:** "If we used Non-Lab scores for treatment decisions, how many patients would receive **wrong** treatment?"
                
                This section quantifies the clinical impact of using non-laboratory-based risk assessment, focusing on 
                treatment thresholds (10% and 20%) where statin therapy and intensive interventions are typically initiated.
                """)
                
                # --- Summary Metrics ---
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                n_total = len(df_paired)
                
                with col_m1:
                    n_miss_10 = df_paired['misclass_10_under'].sum()
                    pct_miss_10 = n_miss_10 / n_total * 100
                    st.metric(
                        "Missed at ≥10%", 
                        f"{n_miss_10:,}", 
                        f"{pct_miss_10:.1f}% of all",
                        help="Patients where Lab ≥10% but Non-Lab <10% (missed for statin consideration)"
                    )
                
                with col_m2:
                    n_over_10 = df_paired['misclass_10_over'].sum()
                    pct_over_10 = n_over_10 / n_total * 100
                    st.metric(
                        "Over-flagged at ≥10%", 
                        f"{n_over_10:,}", 
                        f"{pct_over_10:.1f}% of all",
                        help="Patients where Lab <10% but Non-Lab ≥10% (unnecessary statin consideration)"
                    )
                
                with col_m3:
                    n_miss_20 = df_paired['misclass_20_under'].sum()
                    pct_miss_20 = n_miss_20 / n_total * 100
                    st.metric(
                        "Missed at ≥20%", 
                        f"{n_miss_20:,}", 
                        f"{pct_miss_20:.1f}% of all",
                        delta_color="inverse",
                        help="Patients where Lab ≥20% but Non-Lab <20% (missed for intensive treatment)"
                    )
                
                with col_m4:
                    n_severe = df_paired['severe_underest'].sum()
                    pct_severe = n_severe / n_total * 100
                    st.metric(
                        "⚠️ Severe Underest.", 
                        f"{n_severe:,}", 
                        f"{pct_severe:.1f}% of all",
                        delta_color="inverse",
                        help="Lab ≥20% but Non-Lab <10% (two category jump - most dangerous)"
                    )
                
                st.markdown("---")
                
                # --- Misclassification by Subgroup ---
                st.subheader("Misclassification by Demographic Subgroup")
                
                misclass_strat = st.selectbox(
                    "Stratify by:",
                    ["Glycaemic Group", "Gender", "Age Band", "Pulse Pressure Band", "Central Obesity"],
                    key="misclass_strat_select"
                )
                
                strat_col_map = {
                    "Glycaemic Group": "gly_group",
                    "Gender": "gender",
                    "Age Band": "age_band",
                    "Pulse Pressure Band": "pp_band",
                    "Central Obesity": "central_obesity"
                }
                strat_col = strat_col_map.get(misclass_strat, "gly_group")
                
                if strat_col in df_paired.columns:
                    # Calculate misclassification rates by stratum
                    misclass_by_strat = []
                    for grp in df_paired[strat_col].dropna().unique():
                        sub = df_paired[df_paired[strat_col] == grp]
                        n_sub = len(sub)
                        if n_sub > 0:
                            misclass_by_strat.append({
                                'Stratum': str(grp),
                                'N': n_sub,
                                'Missed ≥10% (n)': sub['misclass_10_under'].sum(),
                                'Missed ≥10% (%)': sub['misclass_10_under'].mean() * 100,
                                'Missed ≥20% (n)': sub['misclass_20_under'].sum(),
                                'Missed ≥20% (%)': sub['misclass_20_under'].mean() * 100,
                                'Severe Underest (n)': sub['severe_underest'].sum(),
                                'Severe Underest (%)': sub['severe_underest'].mean() * 100,
                                'Over-flagged ≥10% (%)': sub['misclass_10_over'].mean() * 100
                            })
                    
                    misclass_df = pd.DataFrame(misclass_by_strat)
                    
                    if len(misclass_df) > 0:
                        # Sort by severity (missed at 10%)
                        misclass_df = misclass_df.sort_values('Missed ≥10% (%)', ascending=False)
                        
                        # Visualization: Grouped Bar Chart
                        fig_misclass, ax_mc = plt.subplots(figsize=(10, 6))
                        
                        x = np.arange(len(misclass_df))
                        width = 0.25
                        
                        rects1 = ax_mc.bar(x - width, misclass_df['Missed ≥10% (%)'], width, label='Missed at ≥10%', 
                                          color='#E53935', edgecolor='black', linewidth=0.5)
                        rects2 = ax_mc.bar(x, misclass_df['Missed ≥20% (%)'], width, label='Missed at ≥20%', 
                                          color='#8E24AA', edgecolor='black', linewidth=0.5)
                        rects3 = ax_mc.bar(x + width, misclass_df['Over-flagged ≥10% (%)'], width, label='Over-flagged at ≥10%', 
                                          color='#FBC02D', edgecolor='black', linewidth=0.5)
                        
                        ax_mc.set_xticks(x)
                        ax_mc.set_xticklabels(misclass_df['Stratum'], rotation=45, ha='right', fontsize=9)
                        ax_mc.set_xlabel(misclass_strat, fontsize=11, fontfamily='serif')
                        ax_mc.set_ylabel('Misclassification Rate (%)', fontsize=11, fontfamily='serif')
                        ax_mc.set_title(f'Treatment Misclassification Rates by {misclass_strat}', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        # Add value labels
                        for rects in [rects1, rects2, rects3]:
                            ax_mc.bar_label(rects, fmt='%.1f%%', padding=3, fontsize=8)
                        
                        ax_mc.spines['top'].set_visible(False)
                        ax_mc.spines['right'].set_visible(False)
                        ax_mc.grid(True, alpha=0.3, axis='y')
                        ax_mc.legend(loc='upper right', frameon=True, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_misclass)
                        add_svg_download_button(fig_misclass, "misclass_rates", key="misclass_svg")
                        plt.close(fig_misclass)
                        
                        # Display table
                        with st.expander("📋 Detailed Misclassification Table"):
                            st.dataframe(
                                misclass_df.style.format({
                                    'Missed ≥10% (%)': '{:.1f}',
                                    'Missed ≥20% (%)': '{:.1f}',
                                    'Severe Underest (%)': '{:.1f}',
                                    'Over-flagged ≥10% (%)': '{:.1f}'
                                }).background_gradient(
                                    subset=['Missed ≥10% (%)', 'Missed ≥20% (%)'],
                                    cmap='Reds'
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                
                # --- Who Gets Missed? Profile Analysis ---
                st.markdown("---")
                st.subheader("Profile of Missed High-Risk Patients")
                st.caption("Characteristics of patients where Non-Lab fails to identify them as high-risk (Lab ≥10%)")
                
                missed_df = df_paired[df_paired['misclass_10_under']].copy()
                
                if len(missed_df) > 0:
                    col_prof1, col_prof2 = st.columns(2)
                    
                    with col_prof1:
                        # Age distribution of missed patients
                        age_dist_all = df_paired['age_band'].value_counts(normalize=True).sort_index() * 100
                        age_dist_missed = missed_df['age_band'].value_counts(normalize=True).sort_index() * 100
                        
                        age_compare = pd.DataFrame({
                            'All Patients': age_dist_all,
                            'Missed High-Risk': age_dist_missed
                        }).reset_index()
                        age_compare.columns = ['Age Band', 'All Patients', 'Missed High-Risk']
                        
                        fig_age_miss, ax_am = plt.subplots(figsize=(6, 4))
                        x_age = np.arange(len(age_compare))
                        width_age = 0.35
                        
                        ax_am.bar(x_age - width_age/2, age_compare['All Patients'], width_age, label='All Patients', color='#90CAF9')
                        ax_am.bar(x_age + width_age/2, age_compare['Missed High-Risk'], width_age, label='Missed High-Risk', color='#E53935')
                        
                        ax_am.set_xticks(x_age)
                        ax_am.set_xticklabels(age_compare['Age Band'], rotation=45, ha='right', fontsize=9)
                        ax_am.set_title('Age Distribution', fontsize=11, fontweight='bold', fontfamily='serif')
                        ax_am.set_ylabel('Percentage (%)', fontsize=10)
                        
                        ax_am.spines['top'].set_visible(False)
                        ax_am.spines['right'].set_visible(False)
                        ax_am.legend(loc='upper right', frameon=True, fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig_age_miss)
                        add_svg_download_button(fig_age_miss, "age_miss_profile", key="age_miss_svg")
                        plt.close(fig_age_miss)
                    
                    with col_prof2:
                        # Glycaemic group distribution
                        if 'gly_group' in df_paired.columns:
                            gly_dist_all = df_paired['gly_group'].value_counts(normalize=True) * 100
                            gly_dist_missed = missed_df['gly_group'].value_counts(normalize=True) * 100
                            
                            gly_compare = pd.DataFrame({
                                'All Patients': gly_dist_all,
                                'Missed High-Risk': gly_dist_missed
                            }).reset_index()
                            gly_compare.columns = ['Glycaemic Group', 'All Patients', 'Missed High-Risk']
                            
                            fig_gly_miss, ax_gm = plt.subplots(figsize=(6, 4))
                            x_gly = np.arange(len(gly_compare))
                            
                            ax_gm.bar(x_gly - width_age/2, gly_compare['All Patients'], width_age, label='All Patients', color='#90CAF9')
                            ax_gm.bar(x_gly + width_age/2, gly_compare['Missed High-Risk'], width_age, label='Missed High-Risk', color='#E53935')
                            
                            ax_gm.set_xticks(x_gly)
                            ax_gm.set_xticklabels(gly_compare['Glycaemic Group'], rotation=45, ha='right', fontsize=9)
                            ax_gm.set_title('Glycaemic Status Distribution', fontsize=11, fontweight='bold', fontfamily='serif')
                            ax_gm.set_ylabel('Percentage (%)', fontsize=10)
                            
                            ax_gm.spines['top'].set_visible(False)
                            ax_gm.spines['right'].set_visible(False)
                            ax_gm.legend(loc='upper right', frameon=True, fontsize=8)
                            
                            plt.tight_layout()
                            st.pyplot(fig_gly_miss)
                            add_svg_download_button(fig_gly_miss, "gly_miss_profile", key="gly_miss_svg")
                            plt.close(fig_gly_miss)
                    
                    # Key insight
                    st.info(f"""
                    📋 **Missed High-Risk Profile (n={len(missed_df)})**:
                    - Mean Age: **{missed_df['age'].mean():.1f}** years (vs {df_paired['age'].mean():.1f} overall)
                    - Mean Lab Risk: **{missed_df['risk_lab'].mean():.1f}%** (vs {df_paired['risk_lab'].mean():.1f}% overall)
                    - Mean Non-Lab Risk: **{missed_df['risk_nonlab'].mean():.1f}%** (demonstrating the gap)
                    - Diabetes Prevalence: **{missed_df['has_diabetes'].mean()*100:.1f}%** (vs {df_paired['has_diabetes'].mean()*100:.1f}% overall)
                    """)
                else:
                    st.success("✅ No missed high-risk patients at the 10% threshold.")
                
                st.divider()
                st.header("8.8 🔗 Risk Factor Interactions")
                st.markdown("""
                **Advanced analysis of how multiple cardiovascular risk factors interact** to influence:
                1. Overall CVD risk levels
                2. Agreement/discordance between Lab and Non-Lab risk assessment
                3. Synergistic risk amplification patterns
                """)
                
                # Create risk factor indicators if not present
                # Diabetes
                if 'diab_flag' not in df_paired.columns:
                    if 'has_diabetes' in df_paired.columns:
                        df_paired['diab_flag'] = df_paired['has_diabetes'].apply(lambda x: 1 if x in [True, 1, 'Yes', 'True'] else 0)
                    elif 'diab_group' in df_paired.columns:
                        df_paired['diab_flag'] = (df_paired['diab_group'] == 'with_diabetes').astype(int)
                    else:
                        df_paired['diab_flag'] = 0
                
                # Hypertension (SBP ≥140 or DBP ≥90)
                if 'htn_flag' not in df_paired.columns:
                    if 'sbp' in df_paired.columns and 'dbp' in df_paired.columns:
                        df_paired['htn_flag'] = ((df_paired['sbp'] >= 140) | (df_paired['dbp'] >= 90)).astype(int)
                    elif 'bp_category' in df_paired.columns:
                        df_paired['htn_flag'] = df_paired['bp_category'].isin(['HTN Stage 1', 'HTN Stage 2']).astype(int)
                    else:
                        df_paired['htn_flag'] = 0
                
                # Smoking
                if 'smoke_flag' not in df_paired.columns:
                    if 'smoker_who' in df_paired.columns:
                        df_paired['smoke_flag'] = (df_paired['smoker_who'] == 'Smoker').astype(int)
                    else:
                        df_paired['smoke_flag'] = 0
                
                # Obesity (BMI ≥ 30)
                if 'obesity_flag' not in df_paired.columns:
                    if 'bmi' in df_paired.columns:
                        df_paired['obesity_flag'] = (df_paired['bmi'] >= 30).astype(int)
                    elif 'bmi_band' in df_paired.columns:
                        df_paired['obesity_flag'] = df_paired['bmi_band'].isin(['30-34', '>=35']).astype(int)
                    else:
                        df_paired['obesity_flag'] = 0
                
                # Dyslipidemia (Cholesterol ≥ 6 mmol/L or ≥ 232 mg/dL)
                if 'dyslip_flag' not in df_paired.columns:
                    if 'cholesterol_mmolL' in df_paired.columns:
                        df_paired['dyslip_flag'] = (df_paired['cholesterol_mmolL'] >= 6).astype(int)
                    elif 'chol_band' in df_paired.columns:
                        df_paired['dyslip_flag'] = df_paired['chol_band'].isin(['6-6.9', '>=7']).astype(int)
                    else:
                        df_paired['dyslip_flag'] = 0
                
                # Total risk factor count
                df_paired['rf_count'] = df_paired['diab_flag'] + df_paired['htn_flag'] + df_paired['smoke_flag'] + df_paired['obesity_flag'] + df_paired['dyslip_flag']
                
                # Create sub-tabs for different interaction analyses
                int_tab1, int_tab2, int_tab3, int_tab4 = st.tabs([
                    "📊 Multi-Factor Heatmap",
                    "📈 Synergistic Risk Amplification", 
                    "🔄 Co-occurrence Patterns",
                    "🎯 Model Agreement by RF Burden"
                ])
                
                # === TAB 1: Multi-Factor Risk Heatmap ===
                with int_tab1:
                    st.markdown("### Risk by Combinations of Key Risk Factors")
                    st.markdown("Mean 10-year CVD risk stratified by **two risk factors** at a time")
                    
                    # Select which factors to cross-tabulate
                    rf_options = {
                        'Diabetes': 'diab_flag',
                        'Hypertension': 'htn_flag', 
                        'Smoking': 'smoke_flag',
                        'Obesity (BMI≥30)': 'obesity_flag',
                        'Dyslipidemia': 'dyslip_flag'
                    }
                    
                    col_rf1, col_rf2 = st.columns(2)
                    with col_rf1:
                        rf1_name = st.selectbox("Row Factor:", list(rf_options.keys()), index=0, key="rf1_select")
                    with col_rf2:
                        rf2_name = st.selectbox("Column Factor:", list(rf_options.keys()), index=1, key="rf2_select")
                    
                    rf1_col = rf_options[rf1_name]
                    rf2_col = rf_options[rf2_name]
                    
                    # Create crosstab of mean lab risk
                    pivot_lab = df_paired.pivot_table(
                        values='risk_lab',
                        index=rf1_col,
                        columns=rf2_col,
                        aggfunc='mean'
                    )
                    
                    pivot_nonlab = df_paired.pivot_table(
                        values='risk_nonlab',
                        index=rf1_col,
                        columns=rf2_col,
                        aggfunc='mean'
                    )
                    
                    pivot_n = df_paired.pivot_table(
                        values='risk_lab',
                        index=rf1_col,
                        columns=rf2_col,
                        aggfunc='count'
                    )
                    
                    # Rename indices for clarity
                    rf1_labels = {0: f'No {rf1_name}', 1: rf1_name}
                    rf2_labels = {0: f'No {rf2_name}', 1: rf2_name}
                    
                    pivot_lab.index = pivot_lab.index.map(rf1_labels)
                    pivot_lab.columns = pivot_lab.columns.map(rf2_labels)
                    pivot_nonlab.index = pivot_nonlab.index.map(rf1_labels)
                    pivot_nonlab.columns = pivot_nonlab.columns.map(rf2_labels)
                    pivot_n.index = pivot_n.index.map(rf1_labels)
                    pivot_n.columns = pivot_n.columns.map(rf2_labels)
                    
                    col_hm1, col_hm2 = st.columns(2)
                    
                    with col_hm1:
                        st.markdown("**Lab-Based Mean Risk (%)**")
                        # Create combined text for hoverdata
                        text_lab = []
                        for i, row_idx in enumerate(pivot_lab.index):
                            row_text = []
                            for j, col_idx in enumerate(pivot_lab.columns):
                                val = pivot_lab.iloc[i, j]
                                n = pivot_n.iloc[i, j] if not pd.isna(pivot_n.iloc[i, j]) else 0
                                row_text.append(f'{val:.1f}% (n={int(n)})')
                            text_lab.append(row_text)
                        
                        fig_hm_lab, ax_lab = plt.subplots(figsize=(8, 4))
                        sns.heatmap(pivot_lab, annot=np.array(text_lab), fmt='', cmap='Reds', ax=ax_lab,
                                   cbar_kws={'label': 'Risk %'})
                        ax_lab.set_xlabel(rf2_name, fontsize=10, fontfamily='serif')
                        ax_lab.set_ylabel(rf1_name, fontsize=10, fontfamily='serif')
                        plt.tight_layout()
                        st.pyplot(fig_hm_lab)
                        add_svg_download_button(fig_hm_lab, "lab_interaction_heatmap", key="lab_inter_svg")
                        plt.close(fig_hm_lab)
                    
                    with col_hm2:
                        st.markdown("**Non-Lab Mean Risk (%)**")
                        text_nonlab = []
                        for i, row_idx in enumerate(pivot_nonlab.index):
                            row_text = []
                            for j, col_idx in enumerate(pivot_nonlab.columns):
                                val = pivot_nonlab.iloc[i, j]
                                n = pivot_n.iloc[i, j] if not pd.isna(pivot_n.iloc[i, j]) else 0
                                row_text.append(f'{val:.1f}% (n={int(n)})')
                            text_nonlab.append(row_text)
                        
                        fig_hm_nonlab, ax_nonlab = plt.subplots(figsize=(8, 4))
                        sns.heatmap(pivot_nonlab, annot=np.array(text_nonlab), fmt='', cmap='Blues', ax=ax_nonlab,
                                   cbar_kws={'label': 'Risk %'})
                        ax_nonlab.set_xlabel(rf2_name, fontsize=10, fontfamily='serif')
                        ax_nonlab.set_ylabel(rf1_name, fontsize=10, fontfamily='serif')
                        plt.tight_layout()
                        st.pyplot(fig_hm_nonlab)
                        add_svg_download_button(fig_hm_nonlab, "nonlab_interaction_heatmap", key="nonlab_inter_svg")
                        plt.close(fig_hm_nonlab)
                    
                    # Calculate synergy metrics
                    if len(pivot_lab) >= 2 and len(pivot_lab.columns) >= 2:
                        baseline = pivot_lab.iloc[0, 0]  # Neither factor
                        rf1_only = pivot_lab.iloc[1, 0] if len(pivot_lab) > 1 else baseline
                        rf2_only = pivot_lab.iloc[0, 1] if len(pivot_lab.columns) > 1 else baseline
                        both = pivot_lab.iloc[1, 1] if len(pivot_lab) > 1 and len(pivot_lab.columns) > 1 else baseline
                        
                        expected_additive = baseline + (rf1_only - baseline) + (rf2_only - baseline)
                        synergy = both - expected_additive
                        
                        st.markdown(f"""
                        **Synergy Analysis:**
                        - Baseline risk (neither factor): **{baseline:.1f}%**
                        - {rf1_name} only: **{rf1_only:.1f}%** (+{rf1_only - baseline:.1f}%)
                        - {rf2_name} only: **{rf2_only:.1f}%** (+{rf2_only - baseline:.1f}%)
                        - Both factors: **{both:.1f}%**
                        - Expected (additive): **{expected_additive:.1f}%**
                        - **Synergy/Interaction**: **{synergy:+.1f}%** {'⚠️ Super-additive' if synergy > 0.5 else '📉 Sub-additive' if synergy < -0.5 else '➡️ Approximately additive'}
                        """)
                
                # === TAB 2: Synergistic Risk Amplification ===
                with int_tab2:
                    st.markdown("### Risk Amplification by Number of Risk Factors")
                    st.markdown("How does CVD risk increase as risk factors accumulate?")
                    
                    # Calculate mean risk and high-risk prevalence by RF count
                    rf_summary = []
                    for rf_n in range(6):  # 0 to 5 factors
                        subset = df_paired[df_paired['rf_count'] == rf_n]
                        n = len(subset)
                        if n > 0:
                            lab_mean = subset['risk_lab'].mean()
                            lab_median = subset['risk_lab'].median()
                            nonlab_mean = subset['risk_nonlab'].mean()
                            nonlab_median = subset['risk_nonlab'].median()
                            lab_high_10 = (subset['risk_lab'] >= 10).sum() / n * 100
                            lab_high_20 = (subset['risk_lab'] >= 20).sum() / n * 100
                            nonlab_high_10 = (subset['risk_nonlab'] >= 10).sum() / n * 100
                            nonlab_high_20 = (subset['risk_nonlab'] >= 20).sum() / n * 100
                            
                            rf_summary.append({
                                'Risk Factors': rf_n,
                                'N': n,
                                'Lab Mean (%)': lab_mean,
                                'Lab Median (%)': lab_median,
                                'Non-Lab Mean (%)': nonlab_mean,
                                'Non-Lab Median (%)': nonlab_median,
                                'Lab ≥10% (%)': lab_high_10,
                                'Lab ≥20% (%)': lab_high_20,
                                'Non-Lab ≥10% (%)': nonlab_high_10,
                                'Non-Lab ≥20% (%)': nonlab_high_20
                            })
                    
                    rf_df = pd.DataFrame(rf_summary)
                    
                    if len(rf_df) > 0:
                        # Create dual-axis plot
                        # Create dual-axis plot
                        fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
                        ax_rf2 = ax_rf.twinx()
                        
                        x = np.arange(len(rf_df))
                        width = 0.35
                        
                        # Bar chart for mean risk
                        bars1 = ax_rf.bar(x - width/2, rf_df['Lab Mean (%)'], width, label='Lab Mean Risk', 
                                         color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
                        bars2 = ax_rf.bar(x + width/2, rf_df['Non-Lab Mean (%)'], width, label='Non-Lab Mean Risk', 
                                         color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.5)
                        
                        # Add line for sample size
                        line1 = ax_rf2.plot(x, rf_df['N'], 'o--', color='gray', label='Sample Size (N)', 
                                           linewidth=1.5, markersize=6)
                        
                        # Annotations
                        ax_rf.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=8)
                        ax_rf.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=8)
                        for i, n in enumerate(rf_df['N']):
                            ax_rf2.annotate(f'n={n}', (i, n), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                        
                        ax_rf.set_xticks(x)
                        ax_rf.set_xticklabels(rf_df['Risk Factors'], fontsize=9)
                        ax_rf.set_xlabel('Number of Risk Factors (Diabetes, HTN, Smoking, Obesity, Dyslipidemia)', fontsize=10, fontfamily='serif')
                        ax_rf.set_ylabel('Mean 10-Year CVD Risk (%)', fontsize=10, fontfamily='serif')
                        ax_rf2.set_ylabel('Sample Size', fontsize=10, fontfamily='serif', color='gray')
                        
                        ax_rf.set_title('Mean CVD Risk by Number of Risk Factors', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        ax_rf.spines['top'].set_visible(False)
                        ax_rf2.spines['top'].set_visible(False)
                        ax_rf.grid(True, alpha=0.3, axis='y')
                        
                        # Combine legends
                        lines, labels = ax_rf.get_legend_handles_labels()
                        lines2, labels2 = ax_rf2.get_legend_handles_labels()
                        ax_rf.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_rf)
                        add_svg_download_button(fig_rf, "rf_count_risk_mean", key="rf_mean_svg")
                        plt.close(fig_rf)
                        
                        # High-risk prevalence by RF count
                        st.markdown("### High-Risk Prevalence by Risk Factor Burden")
                        
                        fig_high, ax_high = plt.subplots(figsize=(10, 6))
                        
                        # Plot lines
                        ax_high.plot(rf_df['Risk Factors'], rf_df['Lab ≥10% (%)'], 'o-', color='#1f77b4', linewidth=2, label='Lab ≥10%')
                        ax_high.plot(rf_df['Risk Factors'], rf_df['Non-Lab ≥10% (%)'], 'o-', color='#ff7f0e', linewidth=2, label='Non-Lab ≥10%')
                        ax_high.plot(rf_df['Risk Factors'], rf_df['Lab ≥20% (%)'], 's--', color='#1f77b4', linewidth=2, label='Lab ≥20%')
                        ax_high.plot(rf_df['Risk Factors'], rf_df['Non-Lab ≥20% (%)'], 's--', color='#ff7f0e', linewidth=2, label='Non-Lab ≥20%')
                        
                        ax_high.set_xlabel('Number of Risk Factors', fontsize=10, fontfamily='serif')
                        ax_high.set_ylabel('Prevalence (%)', fontsize=10, fontfamily='serif')
                        ax_high.set_title('High-Risk Prevalence by Risk Factor Count', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        ax_high.spines['top'].set_visible(False)
                        ax_high.spines['right'].set_visible(False)
                        ax_high.grid(True, alpha=0.3)
                        ax_high.legend(loc='upper left', frameon=True, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_high)
                        add_svg_download_button(fig_high, "rf_count_risk_prev", key="rf_prev_svg")
                        plt.close(fig_high)
                        
                        # Display summary table
                        st.markdown("### Summary Table")
                        st.dataframe(rf_df.style.format({
                            'Lab Mean (%)': '{:.1f}',
                            'Lab Median (%)': '{:.1f}',
                            'Non-Lab Mean (%)': '{:.1f}',
                            'Non-Lab Median (%)': '{:.1f}',
                            'Lab ≥10% (%)': '{:.1f}',
                            'Lab ≥20% (%)': '{:.1f}',
                            'Non-Lab ≥10% (%)': '{:.1f}',
                            'Non-Lab ≥20% (%)': '{:.1f}'
                        }), use_container_width=True, hide_index=True)
                
                # === TAB 3: Co-occurrence Patterns ===
                with int_tab3:
                    st.markdown("### Risk Factor Co-occurrence Patterns")
                    st.markdown("Visualizing how risk factors cluster together in the population")
                    
                    # Create co-occurrence matrix
                    rf_cols = ['diab_flag', 'htn_flag', 'smoke_flag', 'obesity_flag', 'dyslip_flag']
                    rf_names = ['Diabetes', 'Hypertension', 'Smoking', 'Obesity', 'Dyslipidemia']
                    
                    cooccur_matrix = np.zeros((5, 5))
                    for i, rf1 in enumerate(rf_cols):
                        for j, rf2 in enumerate(rf_cols):
                            if i == j:
                                cooccur_matrix[i, j] = df_paired[rf1].sum()
                            else:
                                cooccur_matrix[i, j] = ((df_paired[rf1] == 1) & (df_paired[rf2] == 1)).sum()
                    
                    # Convert to percentage of total
                    total_n = len(df_paired)
                    cooccur_pct = cooccur_matrix / total_n * 100
                    
                    fig_cooccur, ax_co = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cooccur_pct, annot=True, fmt='.1f', cmap='viridis', ax=ax_co,
                               cbar_kws={'label': '% of Population'})
                    ax_co.set_xlabel('Risk Factor', fontsize=10, fontfamily='serif')
                    ax_co.set_ylabel('Risk Factor', fontsize=10, fontfamily='serif')
                    ax_co.set_title('Risk Factor Co-occurrence Matrix (% of Population)', fontsize=12, fontweight='bold', fontfamily='serif')
                    plt.tight_layout()
                    st.pyplot(fig_cooccur)
                    add_svg_download_button(fig_cooccur, "rf_cooccurrence", key="rf_cooccur_svg")
                    plt.close(fig_cooccur)
                    
                    # Bar chart of individual RF prevalence
                    rf_prev = [df_paired[col].sum() / total_n * 100 for col in rf_cols]
                    
                    fig_prev, ax_prev = plt.subplots(figsize=(8, 4))
                    colors_prev = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
                    bars_prev = ax_prev.bar(rf_names, rf_prev, color=colors_prev[:len(rf_names)], edgecolor='black', linewidth=0.5)
                    
                    ax_prev.bar_label(bars_prev, fmt='%.1f%%', padding=3, fontsize=9)
                    ax_prev.set_xlabel('Risk Factor', fontsize=10, fontfamily='serif')
                    ax_prev.set_ylabel('Prevalence (%)', fontsize=10, fontfamily='serif')
                    ax_prev.set_title('Individual Risk Factor Prevalence', fontsize=12, fontweight='bold', fontfamily='serif')
                    
                    ax_prev.spines['top'].set_visible(False)
                    ax_prev.spines['right'].set_visible(False)
                    ax_prev.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig_prev)
                    add_svg_download_button(fig_prev, "rf_indiv_prev", key="rf_indiv_prev_svg")
                    plt.close(fig_prev)
                    
                    # Distribution of RF count
                    rf_count_dist = df_paired['rf_count'].value_counts().sort_index()
                    
                    fig_dist, ax_dist = plt.subplots(figsize=(7, 5))
                    colors_dist = ['#27ae60', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
                    
                    # Ensure enough colors
                    if len(rf_count_dist) > len(colors_dist):
                        colors_dist = plt.cm.tab10.colors
                    
                    wedges, texts, autotexts = ax_dist.pie(
                        rf_count_dist.values, 
                        labels=[f'{i} RFs' for i in rf_count_dist.index],
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=colors_dist[:len(rf_count_dist)],
                        wedgeprops=dict(width=0.6, edgecolor='w'),
                        textprops=dict(color="black")
                    )
                    
                    ax_dist.set_title('Distribution of Risk Factor Burden', fontsize=12, fontweight='bold', fontfamily='serif')
                    
                    plt.tight_layout()
                    st.pyplot(fig_dist)
                    add_svg_download_button(fig_dist, "rf_burden_dist", key="rf_burden_svg")
                    plt.close(fig_dist)
                
                # === TAB 4: Model Agreement by RF Burden ===
                with int_tab4:
                    st.markdown("### Lab vs Non-Lab Agreement by Risk Factor Burden")
                    st.markdown("How does the concordance between methods vary with cardiovascular risk factor accumulation?")
                    
                    # Calculate agreement metrics by RF count
                    agree_by_rf = []
                    for rf_n in range(6):
                        subset = df_paired[df_paired['rf_count'] == rf_n]
                        n = len(subset)
                        if n >= 10:  # Minimum sample size
                            # Classification agreement at 10% threshold
                            lab_high = (subset['risk_lab'] >= 10)
                            nonlab_high = (subset['risk_nonlab'] >= 10)
                            
                            concordant = (lab_high == nonlab_high).sum()
                            underest = (lab_high & ~nonlab_high).sum()  # Lab high but nonlab low
                            overest = (~lab_high & nonlab_high).sum()  # Lab low but nonlab high
                            
                            # Risk difference
                            risk_diff = (subset['risk_lab'] - subset['risk_nonlab']).mean()
                            risk_diff_std = (subset['risk_lab'] - subset['risk_nonlab']).std()
                            
                            # Correlation
                            corr = subset['risk_lab'].corr(subset['risk_nonlab'])
                            
                            agree_by_rf.append({
                                'Risk Factors': rf_n,
                                'N': n,
                                'Concordance (%)': concordant / n * 100,
                                'Underestimation (%)': underest / n * 100,
                                'Overestimation (%)': overest / n * 100,
                                'Mean Diff (Lab-NonLab)': risk_diff,
                                'SD of Diff': risk_diff_std,
                                'Correlation': corr
                            })
                    
                    agree_df = pd.DataFrame(agree_by_rf)
                    
                    if len(agree_df) > 0:
                        # Stacked bar for concordance
                        # Stacked bar for concordance
                        fig_agree, ax_agree = plt.subplots(figsize=(10, 6))
                        x = np.arange(len(agree_df))
                        width = 0.5
                        
                        p1 = ax_agree.bar(x, agree_df['Concordance (%)'], width, label='Concordant', 
                                         color='#27ae60', edgecolor='black', linewidth=0.5)
                        p2 = ax_agree.bar(x, agree_df['Underestimation (%)'], width, bottom=agree_df['Concordance (%)'], 
                                         label='Underestimation (NonLab < Lab)', color='#e74c3c', edgecolor='black', linewidth=0.5)
                        p3 = ax_agree.bar(x, agree_df['Overestimation (%)'], width, 
                                         bottom=agree_df['Concordance (%)'] + agree_df['Underestimation (%)'],
                                         label='Overestimation (NonLab > Lab)', color='#f39c12', edgecolor='black', linewidth=0.5)
                        
                        # Add labels (optional filters for clarity)
                        ax_agree.bar_label(p1, fmt='%.1f%%', label_type='center', fontsize=8, color='white')
                        ax_agree.bar_label(p2, fmt='%.1f%%', label_type='center', fontsize=8, color='white')
                        ax_agree.bar_label(p3, fmt='%.1f%%', label_type='center', fontsize=8, color='black')
                        
                        ax_agree.set_xticks(x)
                        ax_agree.set_xticklabels(agree_df['Risk Factors'])
                        ax_agree.set_xlabel('Number of Risk Factors', fontsize=10, fontfamily='serif')
                        ax_agree.set_ylabel('Percentage', fontsize=10, fontfamily='serif')
                        ax_agree.set_title('Classification Agreement (≥10% threshold) by Risk Factor Burden', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        ax_agree.spines['top'].set_visible(False)
                        ax_agree.spines['right'].set_visible(False)
                        ax_agree.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3, frameon=False, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_agree)
                        add_svg_download_button(fig_agree, "rf_agree_stacked", key="rf_agree_svg")
                        plt.close(fig_agree)
                        
                        # Risk difference plot
                        # Risk difference plot
                        fig_diff, ax_diff = plt.subplots(figsize=(8, 5))
                        
                        ax_diff.errorbar(agree_df['Risk Factors'], agree_df['Mean Diff (Lab-NonLab)'], 
                                        yerr=agree_df['SD of Diff'], fmt='-o', color='#3498db', 
                                        linewidth=2, markersize=6, capsize=4, label='Mean Difference')
                        
                        ax_diff.axhline(0, color='red', linestyle='--', label='No difference')
                        
                        ax_diff.set_xlabel('Number of Risk Factors', fontsize=10, fontfamily='serif')
                        ax_diff.set_ylabel('Mean Difference (%)', fontsize=10, fontfamily='serif')
                        ax_diff.set_title('Mean Risk Difference (Lab - Non-Lab)', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        ax_diff.spines['top'].set_visible(False)
                        ax_diff.spines['right'].set_visible(False)
                        ax_diff.grid(True, alpha=0.3)
                        ax_diff.legend(loc='upper left', frameon=True, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig_diff)
                        add_svg_download_button(fig_diff, "rf_risk_diff", key="rf_diff_svg")
                        plt.close(fig_diff)
                        
                        # Correlation by RF count
                        # Correlation by RF count
                        fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
                        
                        bars_corr = ax_corr.bar(agree_df['Risk Factors'], agree_df['Correlation'], 
                                               color='#9b59b6', edgecolor='black', linewidth=0.5)
                        
                        ax_corr.bar_label(bars_corr, fmt='%.3f', padding=3, fontsize=9)
                        
                        ax_corr.set_xlabel('Number of Risk Factors', fontsize=10, fontfamily='serif')
                        ax_corr.set_ylabel('Pearson Correlation', fontsize=10, fontfamily='serif')
                        ax_corr.set_title('Correlation (Lab vs Non-Lab) by Risk Factor Burden', fontsize=12, fontweight='bold', fontfamily='serif')
                        ax_corr.set_ylim(0, 1.05)
                        
                        ax_corr.spines['top'].set_visible(False)
                        ax_corr.spines['right'].set_visible(False)
                        ax_corr.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        add_svg_download_button(fig_corr, "rf_correlation", key="rf_corr_svg")
                        plt.close(fig_corr)
                        
                        # Summary table
                        st.markdown("### Detailed Agreement Metrics")
                        st.dataframe(agree_df.style.format({
                            'Concordance (%)': '{:.1f}',
                            'Underestimation (%)': '{:.1f}',
                            'Overestimation (%)': '{:.1f}',
                            'Mean Diff (Lab-NonLab)': '{:+.2f}',
                            'SD of Diff': '{:.2f}',
                            'Correlation': '{:.3f}'
                        }), use_container_width=True, hide_index=True)
                        
                        # Key findings
                        max_underest_rf = agree_df.loc[agree_df['Underestimation (%)'].idxmax(), 'Risk Factors']
                        max_underest_pct = agree_df['Underestimation (%)'].max()
                        
                        st.info(f"""
                        📊 **Key Findings:**
                        - **Maximum underestimation** occurs at **{max_underest_rf} risk factors** ({max_underest_pct:.1f}% of cases)
                        - Non-lab method tends to {'underestimate' if agree_df['Mean Diff (Lab-NonLab)'].mean() > 0 else 'overestimate'} risk in individuals with multiple risk factors
                        - Correlation between methods: **{agree_df['Correlation'].mean():.3f}** (average across RF burden levels)
                        """)
                
                
                # ==========================================
                # SECTION 8.9: PUBLICATION TABLE - CHARACTERISTICS BY SEX
                # ==========================================
                st.divider()
                st.markdown("---")
                st.markdown("## 📋 Publication-Ready Tables")
                st.header("8.9 📋 Table: Participant Characteristics by Sex")
                st.markdown("Data are presented as **median (25th–75th percentiles)** or **n (%)**. Statistical comparison using **Mann-Whitney U Test** for continuous variables and **Chi-square test** for categorical variables.")
                st.caption("BP: Blood Pressure; BMI: Body Mass Index; CVD: Cardiovascular Disease; FPG: Fasting Plasma Glucose")
                
                # Define variables for the table
                continuous_vars = [
                    ('age', 'Age (years)'),
                    ('bmi', 'BMI (kg/m²)'),
                    ('sbp', 'Systolic BP (mmHg)'),
                    ('dbp', 'Diastolic BP (mmHg)'),
                    ('bg_mgdl', 'Blood Glucose (mg/dL)'),
                    ('cholesterol_mmolL', 'Total Cholesterol (mmol/L)'),
                    ('risk_lab', '10-Year CVD Risk - Lab (%)'),
                    ('risk_nonlab', '10-Year CVD Risk - Non-Lab (%)')
                ]
                
                categorical_vars = [
                    ('smoker_who', 'Smoking Status', ['Smoker', 'Non-smoker']),
                    ('has_diabetes', 'Diabetes', [True, False]),
                    ('bp_category', 'Hypertension (≥140/90)', None)  # Will be computed
                ]
                
                # Build the characteristics table
                char_table_data = []
                
                # Get male and female subgroups
                df_male = df_paired[df_paired['gender'] == 'M'].copy() if 'M' in df_paired['gender'].values else df_paired[df_paired['gender'].str.lower() == 'male'].copy()
                df_female = df_paired[df_paired['gender'] == 'F'].copy() if 'F' in df_paired['gender'].values else df_paired[df_paired['gender'].str.lower() == 'female'].copy()
                
                n_male = len(df_male)
                n_female = len(df_female)
                n_total = len(df_paired)
                
                # Add sample size row
                char_table_data.append({
                    'Characteristic': 'N',
                    'Total': f'{n_total:,}',
                    'Male': f'{n_male:,}',
                    'Female': f'{n_female:,}',
                    'P-value': ''
                })
                
                # Process continuous variables
                for col, label in continuous_vars:
                    if col in df_paired.columns:
                        # Total statistics
                        total_median = df_paired[col].median()
                        total_q25 = df_paired[col].quantile(0.25)
                        total_q75 = df_paired[col].quantile(0.75)
                        
                        # Male statistics
                        male_vals = df_male[col].dropna()
                        male_median = male_vals.median() if len(male_vals) > 0 else np.nan
                        male_q25 = male_vals.quantile(0.25) if len(male_vals) > 0 else np.nan
                        male_q75 = male_vals.quantile(0.75) if len(male_vals) > 0 else np.nan
                        
                        # Female statistics
                        female_vals = df_female[col].dropna()
                        female_median = female_vals.median() if len(female_vals) > 0 else np.nan
                        female_q25 = female_vals.quantile(0.25) if len(female_vals) > 0 else np.nan
                        female_q75 = female_vals.quantile(0.75) if len(female_vals) > 0 else np.nan
                        
                        # Mann-Whitney U Test
                        try:
                            if len(male_vals) > 0 and len(female_vals) > 0:
                                stat, p_value = stats.mannwhitneyu(male_vals, female_vals, alternative='two-sided')
                                p_str = f'{p_value:.3f}' if p_value >= 0.001 else '<0.001'
                            else:
                                p_str = 'N/A'
                        except Exception:
                            p_str = 'N/A'
                        
                        char_table_data.append({
                            'Characteristic': label,
                            'Total': f'{total_median:.1f} ({total_q25:.1f}–{total_q75:.1f})',
                            'Male': f'{male_median:.1f} ({male_q25:.1f}–{male_q75:.1f})' if not pd.isna(male_median) else 'N/A',
                            'Female': f'{female_median:.1f} ({female_q25:.1f}–{female_q75:.1f})' if not pd.isna(female_median) else 'N/A',
                            'P-value': p_str
                        })
                
                # Process categorical variables
                # Smoking
                if 'smoker_who' in df_paired.columns:
                    smoker_total = (df_paired['smoker_who'] == 'Smoker').sum()
                    smoker_male = (df_male['smoker_who'] == 'Smoker').sum()
                    smoker_female = (df_female['smoker_who'] == 'Smoker').sum()
                    
                    # Chi-square test
                    try:
                        contingency = pd.crosstab(df_paired['gender'], df_paired['smoker_who'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        p_str = f'{p_val:.3f}' if p_val >= 0.001 else '<0.001'
                    except Exception:
                        p_str = 'N/A'
                    
                    char_table_data.append({
                        'Characteristic': 'Current Smoker, n (%)',
                        'Total': f'{smoker_total:,} ({100*smoker_total/n_total:.1f}%)',
                        'Male': f'{smoker_male:,} ({100*smoker_male/n_male:.1f}%)' if n_male > 0 else 'N/A',
                        'Female': f'{smoker_female:,} ({100*smoker_female/n_female:.1f}%)' if n_female > 0 else 'N/A',
                        'P-value': p_str
                    })
                
                # Diabetes
                if 'has_diabetes' in df_paired.columns:
                    diab_total = df_paired['has_diabetes'].sum() if df_paired['has_diabetes'].dtype == bool else (df_paired['has_diabetes'] == True).sum()
                    diab_male = df_male['has_diabetes'].sum() if df_male['has_diabetes'].dtype == bool else (df_male['has_diabetes'] == True).sum()
                    diab_female = df_female['has_diabetes'].sum() if df_female['has_diabetes'].dtype == bool else (df_female['has_diabetes'] == True).sum()
                    
                    try:
                        contingency = pd.crosstab(df_paired['gender'], df_paired['has_diabetes'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        p_str = f'{p_val:.3f}' if p_val >= 0.001 else '<0.001'
                    except Exception:
                        p_str = 'N/A'
                    
                    char_table_data.append({
                        'Characteristic': 'Diabetes, n (%)',
                        'Total': f'{diab_total:,} ({100*diab_total/n_total:.1f}%)',
                        'Male': f'{diab_male:,} ({100*diab_male/n_male:.1f}%)' if n_male > 0 else 'N/A',
                        'Female': f'{diab_female:,} ({100*diab_female/n_female:.1f}%)' if n_female > 0 else 'N/A',
                        'P-value': p_str
                    })
                
                # Hypertension (SBP ≥140 or DBP ≥90)
                if 'sbp' in df_paired.columns and 'dbp' in df_paired.columns:
                    df_paired['hypertension'] = (df_paired['sbp'] >= 140) | (df_paired['dbp'] >= 90)
                    df_male['hypertension'] = (df_male['sbp'] >= 140) | (df_male['dbp'] >= 90)
                    df_female['hypertension'] = (df_female['sbp'] >= 140) | (df_female['dbp'] >= 90)
                    
                    htn_total = df_paired['hypertension'].sum()
                    htn_male = df_male['hypertension'].sum()
                    htn_female = df_female['hypertension'].sum()
                    
                    try:
                        contingency = pd.crosstab(df_paired['gender'], df_paired['hypertension'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        p_str = f'{p_val:.3f}' if p_val >= 0.001 else '<0.001'
                    except Exception:
                        p_str = 'N/A'
                    
                    char_table_data.append({
                        'Characteristic': 'Hypertension (≥140/90 mmHg), n (%)',
                        'Total': f'{htn_total:,} ({100*htn_total/n_total:.1f}%)',
                        'Male': f'{htn_male:,} ({100*htn_male/n_male:.1f}%)' if n_male > 0 else 'N/A',
                        'Female': f'{htn_female:,} ({100*htn_female/n_female:.1f}%)' if n_female > 0 else 'N/A',
                        'P-value': p_str
                    })
                
                # High Risk Categories
                for threshold, label in [(10, '≥10% CVD Risk'), (20, '≥20% CVD Risk')]:
                    # Lab-based
                    high_lab_total = (df_paired['risk_lab'] >= threshold).sum()
                    high_lab_male = (df_male['risk_lab'] >= threshold).sum()
                    high_lab_female = (df_female['risk_lab'] >= threshold).sum()
                    
                    try:
                        df_paired['high_lab_temp'] = df_paired['risk_lab'] >= threshold
                        contingency = pd.crosstab(df_paired['gender'], df_paired['high_lab_temp'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        p_str = f'{p_val:.3f}' if p_val >= 0.001 else '<0.001'
                    except Exception:
                        p_str = 'N/A'
                    
                    char_table_data.append({
                        'Characteristic': f'{label} (Lab-based), n (%)',
                        'Total': f'{high_lab_total:,} ({100*high_lab_total/n_total:.1f}%)',
                        'Male': f'{high_lab_male:,} ({100*high_lab_male/n_male:.1f}%)' if n_male > 0 else 'N/A',
                        'Female': f'{high_lab_female:,} ({100*high_lab_female/n_female:.1f}%)' if n_female > 0 else 'N/A',
                        'P-value': p_str
                    })
                    
                    # Non-Lab based
                    high_nonlab_total = (df_paired['risk_nonlab'] >= threshold).sum()
                    high_nonlab_male = (df_male['risk_nonlab'] >= threshold).sum()
                    high_nonlab_female = (df_female['risk_nonlab'] >= threshold).sum()
                    
                    try:
                        df_paired['high_nonlab_temp'] = df_paired['risk_nonlab'] >= threshold
                        contingency = pd.crosstab(df_paired['gender'], df_paired['high_nonlab_temp'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        p_str = f'{p_val:.3f}' if p_val >= 0.001 else '<0.001'
                    except Exception:
                        p_str = 'N/A'
                    
                    char_table_data.append({
                        'Characteristic': f'{label} (Non-Lab), n (%)',
                        'Total': f'{high_nonlab_total:,} ({100*high_nonlab_total/n_total:.1f}%)',
                        'Male': f'{high_nonlab_male:,} ({100*high_nonlab_male/n_male:.1f}%)' if n_male > 0 else 'N/A',
                        'Female': f'{high_nonlab_female:,} ({100*high_nonlab_female/n_female:.1f}%)' if n_female > 0 else 'N/A',
                        'P-value': p_str
                    })
                
                # Create and display the table
                char_df = pd.DataFrame(char_table_data)
                char_df.columns = ['Characteristic', f'Total (N={n_total:,})', f'Male (n={n_male:,})', f'Female (n={n_female:,})', 'P-value']
                
                # Style the table
                def highlight_significant(val):
                    if val == '<0.001' or (isinstance(val, str) and val.replace('.', '').replace('0', '').isdigit()):
                        try:
                            if val == '<0.001' or float(val) < 0.05:
                                return 'font-weight: bold; color: #D32F2F'
                        except:
                            pass
                    return ''
                
                st.dataframe(
                    char_df.style.applymap(highlight_significant, subset=['P-value']),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add download button for the table
                csv_char = char_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Table (CSV)",
                    data=csv_char,
                    file_name="participant_characteristics_by_sex.csv",
                    mime="text/csv",
                    key="download_char_table"
                )
                
                # ==========================================
                # SECTION 8.10: PUBLICATION TABLE - RISK BY GLYCAEMIC STATUS
                # ==========================================
                st.divider()
                st.header("8.10 📋 Table: CVD Risk by Glycaemic Status")
                st.markdown("""
                Comparison of **laboratory-based** and **non-laboratory-based** CVD risk estimation approaches across three glycaemic groups:
                - **Normoglycaemia**: FPG <110 mg/dL (fasting) or <140 mg/dL (random)
                - **Impaired Fasting Glucose (IFG)**: FPG 110-125 mg/dL (fasting) or 140-199 mg/dL (random)
                - **Diabetes**: FPG ≥126 mg/dL (fasting) or ≥200 mg/dL (random), or known diabetes
                """)
                
                # Ensure gly_group exists (should already be created earlier)
                if 'gly_group' not in df_paired.columns:
                    def categorize_glycaemic_local(row):
                        if row.get('has_diabetes') == True or row.get('diab_group') == 'with_diabetes':
                            return 'Diabetes'
                        bg = row.get('bg_mgdl')
                        bstype = row.get('bstype')
                        if pd.isna(bg):
                            if row.get('diab_group') == 'no_diabetes':
                                return 'Normoglycaemia'
                            return 'Unknown'
                        if bstype == 'fbs':
                            if bg >= 126: return 'Diabetes'
                            elif bg >= 110: return 'IFG'
                            else: return 'Normoglycaemia'
                        else:
                            if bg >= 200: return 'Diabetes'
                            elif bg >= 140: return 'IFG'
                            else: return 'Normoglycaemia'
                    df_paired['gly_group'] = df_paired.apply(categorize_glycaemic_local, axis=1)
                
                # Filter out Unknown glycaemic status for this analysis
                df_gly = df_paired[df_paired['gly_group'].isin(['Normoglycaemia', 'IFG', 'Diabetes'])].copy()
                gly_order = ['Normoglycaemia', 'IFG', 'Diabetes']
                
                # Build glycaemic risk table
                gly_table_data = []
                
                for gly_group in gly_order:
                    df_g = df_gly[df_gly['gly_group'] == gly_group]
                    n_g = len(df_g)
                    
                    if n_g > 0:
                        # Lab-based metrics
                        lab_median = df_g['risk_lab'].median()
                        lab_q25 = df_g['risk_lab'].quantile(0.25)
                        lab_q75 = df_g['risk_lab'].quantile(0.75)
                        lab_ge10 = (df_g['risk_lab'] >= 10).sum()
                        lab_ge20 = (df_g['risk_lab'] >= 20).sum()
                        
                        # Non-Lab metrics
                        nonlab_median = df_g['risk_nonlab'].median()
                        nonlab_q25 = df_g['risk_nonlab'].quantile(0.25)
                        nonlab_q75 = df_g['risk_nonlab'].quantile(0.75)
                        nonlab_ge10 = (df_g['risk_nonlab'] >= 10).sum()
                        nonlab_ge20 = (df_g['risk_nonlab'] >= 20).sum()
                        
                        # Risk difference (Lab - NonLab)
                        diff_median = df_g['risk_lab'].median() - df_g['risk_nonlab'].median()
                        
                        # Concordance metrics
                        concordant = ((df_g['risk_lab'] >= 10) == (df_g['risk_nonlab'] >= 10)).sum()
                        underest = ((df_g['risk_lab'] >= 10) & (df_g['risk_nonlab'] < 10)).sum()  # NonLab misses high-risk
                        overest = ((df_g['risk_lab'] < 10) & (df_g['risk_nonlab'] >= 10)).sum()  # NonLab over-calls
                        
                        gly_table_data.append({
                            'Glycaemic Status': gly_group,
                            'N': n_g,
                            'Lab Median (IQR)': f'{lab_median:.1f} ({lab_q25:.1f}–{lab_q75:.1f})',
                            'Lab ≥10% n (%)': f'{lab_ge10} ({100*lab_ge10/n_g:.1f}%)',
                            'Lab ≥20% n (%)': f'{lab_ge20} ({100*lab_ge20/n_g:.1f}%)',
                            'Non-Lab Median (IQR)': f'{nonlab_median:.1f} ({nonlab_q25:.1f}–{nonlab_q75:.1f})',
                            'Non-Lab ≥10% n (%)': f'{nonlab_ge10} ({100*nonlab_ge10/n_g:.1f}%)',
                            'Non-Lab ≥20% n (%)': f'{nonlab_ge20} ({100*nonlab_ge20/n_g:.1f}%)',
                            'Median Difference': f'{diff_median:+.1f}',
                            'Concordance (≥10%)': f'{concordant} ({100*concordant/n_g:.1f}%)',
                            'Underestimation': f'{underest} ({100*underest/n_g:.1f}%)',
                            'Overestimation': f'{overest} ({100*overest/n_g:.1f}%)'
                        })
                
                # Add total row
                n_total_gly = len(df_gly)
                if n_total_gly > 0:
                    lab_median_t = df_gly['risk_lab'].median()
                    lab_q25_t = df_gly['risk_lab'].quantile(0.25)
                    lab_q75_t = df_gly['risk_lab'].quantile(0.75)
                    lab_ge10_t = (df_gly['risk_lab'] >= 10).sum()
                    lab_ge20_t = (df_gly['risk_lab'] >= 20).sum()
                    
                    nonlab_median_t = df_gly['risk_nonlab'].median()
                    nonlab_q25_t = df_gly['risk_nonlab'].quantile(0.25)
                    nonlab_q75_t = df_gly['risk_nonlab'].quantile(0.75)
                    nonlab_ge10_t = (df_gly['risk_nonlab'] >= 10).sum()
                    nonlab_ge20_t = (df_gly['risk_nonlab'] >= 20).sum()
                    
                    diff_median_t = lab_median_t - nonlab_median_t
                    concordant_t = ((df_gly['risk_lab'] >= 10) == (df_gly['risk_nonlab'] >= 10)).sum()
                    underest_t = ((df_gly['risk_lab'] >= 10) & (df_gly['risk_nonlab'] < 10)).sum()
                    overest_t = ((df_gly['risk_lab'] < 10) & (df_gly['risk_nonlab'] >= 10)).sum()
                    
                    gly_table_data.append({
                        'Glycaemic Status': '**Total**',
                        'N': n_total_gly,
                        'Lab Median (IQR)': f'{lab_median_t:.1f} ({lab_q25_t:.1f}–{lab_q75_t:.1f})',
                        'Lab ≥10% n (%)': f'{lab_ge10_t} ({100*lab_ge10_t/n_total_gly:.1f}%)',
                        'Lab ≥20% n (%)': f'{lab_ge20_t} ({100*lab_ge20_t/n_total_gly:.1f}%)',
                        'Non-Lab Median (IQR)': f'{nonlab_median_t:.1f} ({nonlab_q25_t:.1f}–{nonlab_q75_t:.1f})',
                        'Non-Lab ≥10% n (%)': f'{nonlab_ge10_t} ({100*nonlab_ge10_t/n_total_gly:.1f}%)',
                        'Non-Lab ≥20% n (%)': f'{nonlab_ge20_t} ({100*nonlab_ge20_t/n_total_gly:.1f}%)',
                        'Median Difference': f'{diff_median_t:+.1f}',
                        'Concordance (≥10%)': f'{concordant_t} ({100*concordant_t/n_total_gly:.1f}%)',
                        'Underestimation': f'{underest_t} ({100*underest_t/n_total_gly:.1f}%)',
                        'Overestimation': f'{overest_t} ({100*overest_t/n_total_gly:.1f}%)'
                    })
                
                gly_risk_df = pd.DataFrame(gly_table_data)
                
                # Display the table in two sections for better readability
                st.subheader("8A. Risk Distribution by Glycaemic Status")
                
                # Part 1: Risk Distribution
                gly_dist_cols = ['Glycaemic Status', 'N', 'Lab Median (IQR)', 'Lab ≥10% n (%)', 'Lab ≥20% n (%)', 
                                 'Non-Lab Median (IQR)', 'Non-Lab ≥10% n (%)', 'Non-Lab ≥20% n (%)']
                st.dataframe(
                    gly_risk_df[gly_dist_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.subheader("8B. Classification Agreement by Glycaemic Status")
                st.caption("Underestimation: Non-Lab <10% when Lab ≥10% (missed high-risk). Overestimation: Non-Lab ≥10% when Lab <10%.")
                
                # Part 2: Agreement Analysis
                gly_agree_cols = ['Glycaemic Status', 'N', 'Median Difference', 'Concordance (≥10%)', 
                                  'Underestimation', 'Overestimation']
                st.dataframe(
                    gly_risk_df[gly_agree_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Statistical comparison across glycaemic groups
                st.subheader("8C. Statistical Comparison Across Glycaemic Groups")
                
                # Kruskal-Wallis test for risk scores across glycaemic groups
                normo_lab = df_gly[df_gly['gly_group'] == 'Normoglycaemia']['risk_lab'].dropna()
                ifg_lab = df_gly[df_gly['gly_group'] == 'IFG']['risk_lab'].dropna()
                diab_lab = df_gly[df_gly['gly_group'] == 'Diabetes']['risk_lab'].dropna()
                
                normo_nonlab = df_gly[df_gly['gly_group'] == 'Normoglycaemia']['risk_nonlab'].dropna()
                ifg_nonlab = df_gly[df_gly['gly_group'] == 'IFG']['risk_nonlab'].dropna()
                diab_nonlab = df_gly[df_gly['gly_group'] == 'Diabetes']['risk_nonlab'].dropna()
                
                stat_results = []
                
                # Kruskal-Wallis for Lab-based risk
                try:
                    h_lab, p_lab = stats.kruskal(normo_lab, ifg_lab, diab_lab)
                    stat_results.append({
                        'Comparison': 'Lab-based Risk across Glycaemic Groups',
                        'Test': 'Kruskal-Wallis H',
                        'Statistic': f'{h_lab:.2f}',
                        'P-value': f'{p_lab:.4f}' if p_lab >= 0.0001 else '<0.0001'
                    })
                except Exception as e:
                    stat_results.append({
                        'Comparison': 'Lab-based Risk across Glycaemic Groups',
                        'Test': 'Kruskal-Wallis H',
                        'Statistic': 'N/A',
                        'P-value': 'N/A'
                    })
                
                # Kruskal-Wallis for Non-Lab risk
                try:
                    h_nonlab, p_nonlab = stats.kruskal(normo_nonlab, ifg_nonlab, diab_nonlab)
                    stat_results.append({
                        'Comparison': 'Non-Lab Risk across Glycaemic Groups',
                        'Test': 'Kruskal-Wallis H',
                        'Statistic': f'{h_nonlab:.2f}',
                        'P-value': f'{p_nonlab:.4f}' if p_nonlab >= 0.0001 else '<0.0001'
                    })
                except Exception as e:
                    stat_results.append({
                        'Comparison': 'Non-Lab Risk across Glycaemic Groups',
                        'Test': 'Kruskal-Wallis H',
                        'Statistic': 'N/A',
                        'P-value': 'N/A'
                    })
                
                # Chi-square for high-risk classification
                for threshold in [10, 20]:
                    try:
                        df_gly[f'high_lab_{threshold}'] = df_gly['risk_lab'] >= threshold
                        contingency = pd.crosstab(df_gly['gly_group'], df_gly[f'high_lab_{threshold}'])
                        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                        stat_results.append({
                            'Comparison': f'Lab ≥{threshold}% Classification by Glycaemic Group',
                            'Test': 'Chi-square',
                            'Statistic': f'{chi2:.2f}',
                            'P-value': f'{p_val:.4f}' if p_val >= 0.0001 else '<0.0001'
                        })
                    except Exception:
                        stat_results.append({
                            'Comparison': f'Lab ≥{threshold}% Classification by Glycaemic Group',
                            'Test': 'Chi-square',
                            'Statistic': 'N/A',
                            'P-value': 'N/A'
                        })
                
                stat_df = pd.DataFrame(stat_results)
                st.dataframe(stat_df, use_container_width=True, hide_index=True)
                
                # Key findings summary
                st.markdown("### Key Findings")
                
                # Calculate key metrics for the summary
                if len(gly_table_data) > 0:
                    diab_row = next((r for r in gly_table_data if r['Glycaemic Status'] == 'Diabetes'), None)
                    normo_row = next((r for r in gly_table_data if r['Glycaemic Status'] == 'Normoglycaemia'), None)
                    
                    if diab_row and normo_row:
                        st.info(f"""
                        📊 **Risk Distribution Patterns:**
                        - **Diabetes group** shows the highest median CVD risk for both assessment methods
                        - Median difference (Lab - Non-Lab) indicates systematic {'underestimation' if '+' in str(diab_row.get('Median Difference', '')) else 'overestimation'} by Non-Lab method in diabetics
                        
                        🎯 **Classification Implications:**
                        - Underestimation (missed high-risk cases) is particularly concerning in diabetic patients
                        - Non-lab methods may miss patients who would benefit from preventive pharmacotherapy
                        - Lab-based assessment is especially important for patients with glycaemic abnormalities
                        """)
                
                # Download button for glycaemic table
                csv_gly = gly_risk_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Glycaemic Risk Table (CSV)",
                    data=csv_gly,
                    file_name="cvd_risk_by_glycaemic_status.csv",
                    mime="text/csv",
                    key="download_gly_table"
                )
                
                # ==========================================
                # SECTION 8.11: LONGITUDINAL ANALYSIS
                # ==========================================
                st.divider()
                st.header("8.11 📅 Longitudinal Analysis (Repeat Visits)")
                st.markdown("""
                **Analyzing patterns across multiple visits** for patients with repeated measurements.
                This reveals risk trajectory, measurement stability, and concordance consistency over time.
                """)
                
                # Check if temporal features exist
                if 'visit_num' in df_paired.columns and 'total_visits' in df_paired.columns:
                    
                    # --- Visit Distribution ---
                    col_long1, col_long2 = st.columns(2)
                    
                    with col_long1:
                        # Unique patients vs total observations
                        n_unique = df_paired['pid'].nunique()
                        n_obs = len(df_paired)
                        avg_visits = n_obs / n_unique
                        
                        st.metric("Unique Patients", f"{n_unique:,}")
                        st.metric("Total Observations", f"{n_obs:,}")
                        st.metric("Avg Visits/Patient", f"{avg_visits:.1f}")
                    
                    with col_long2:
                        # Distribution of visits per patient
                        visit_dist = df_paired.groupby('pid')['visit_num'].max().value_counts().sort_index()
                        
                        fig_visit, ax_vis = plt.subplots(figsize=(7, 4))
                        
                        bars = ax_vis.bar(visit_dist.index.astype(str), visit_dist.values, color='#3498db', edgecolor='black', linewidth=0.5)
                        ax_vis.bar_label(bars, fmt='%d', padding=3)
                        
                        ax_vis.set_xlabel('Number of Visits', fontsize=10, fontfamily='serif')
                        ax_vis.set_ylabel('Number of Patients', fontsize=10, fontfamily='serif')
                        ax_vis.set_title('Distribution of Visits per Patient', fontsize=12, fontweight='bold', fontfamily='serif')
                        
                        ax_vis.spines['top'].set_visible(False)
                        ax_vis.spines['right'].set_visible(False)
                        ax_vis.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_visit)
                        add_svg_download_button(fig_visit, "patient_visits_dist", key="visits_dist_svg")
                        plt.close(fig_visit)
                    
                    st.markdown("---")
                    
                    # --- Risk Over Time ---
                    st.subheader("Risk Trajectory Over Time")
                    
                    if 'visit_year' in df_paired.columns and df_paired['visit_year'].nunique() > 1:
                        # Mean risk by calendar year
                        risk_by_year = df_paired.groupby('visit_year').agg({
                            'risk_lab': ['mean', 'std', 'count'],
                            'risk_nonlab': ['mean', 'std'],
                            'misclass_10_under': 'mean'
                        }).reset_index()
                        risk_by_year.columns = ['Year', 'Lab Mean', 'Lab SD', 'N', 'NonLab Mean', 'NonLab SD', 'Misclass Rate']
                        risk_by_year = risk_by_year[risk_by_year['N'] >= 10]  # Filter years with few observations
                        
                        if len(risk_by_year) > 1:
                            fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
                            ax_trend2 = ax_trend.twinx()
                            
                            # Sample size bars (on secondary axis, plotted first)
                            ax_trend2.bar(risk_by_year['Year'], risk_by_year['N'], color='gray', alpha=0.2, label='Sample Size')
                            
                            # Lab risk trend
                            ax_trend.errorbar(risk_by_year['Year'], risk_by_year['Lab Mean'], yerr=risk_by_year['Lab SD']/2,
                                             fmt='-o', color='#1f77b4', linewidth=2, markersize=8, capsize=3, label='Lab-based Risk')
                            
                            # Non-lab risk trend
                            ax_trend.errorbar(risk_by_year['Year'], risk_by_year['NonLab Mean'], yerr=risk_by_year['NonLab SD']/2,
                                             fmt='-o', color='#ff7f0e', linewidth=2, markersize=8, capsize=3, label='Non-Lab Risk')
                            
                            ax_trend.set_xlabel('Year', fontsize=10, fontfamily='serif')
                            ax_trend.set_ylabel('Mean 10-Year CVD Risk (%)', fontsize=10, fontfamily='serif')
                            ax_trend2.set_ylabel('Sample Size', fontsize=10, fontfamily='serif', color='gray')
                            ax_trend.set_title('Mean CVD Risk Trend Over Calendar Years', fontsize=12, fontweight='bold', fontfamily='serif')
                            
                            ax_trend.spines['top'].set_visible(False)
                            ax_trend2.spines['top'].set_visible(False)
                            ax_trend.grid(True, alpha=0.3)
                            
                            # Combine legends
                            lines, labels = ax_trend.get_legend_handles_labels()
                            lines2, labels2 = ax_trend2.get_legend_handles_labels()
                            ax_trend.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig_trend)
                            add_svg_download_button(fig_trend, "risk_trend_years", key="risk_trend_svg")
                            plt.close(fig_trend)
                    
                    # --- Within-Patient Concordance Stability ---
                    st.markdown("---")
                    st.subheader("Within-Patient Concordance Stability")
                    st.caption("For patients with multiple visits, does the Lab vs Non-Lab agreement remain consistent?")
                    
                    # Only analyze patients with 2+ visits
                    multi_visit_pids = df_paired[df_paired['total_visits'] >= 2]['pid'].unique()
                    
                    if len(multi_visit_pids) > 0:
                        # Calculate concordance consistency
                        stability_data = []
                        for pid in multi_visit_pids:
                            patient_df = df_paired[df_paired['pid'] == pid].sort_values('date')
                            n_visits = len(patient_df)
                            n_concordant = (patient_df['agree_status'] == 'Concordance').sum()
                            n_under = (patient_df['agree_status'] == 'Underestimation').sum()
                            n_over = (patient_df['agree_status'] == 'Overestimation').sum()
                            
                            # Categorize patient's consistency
                            if n_concordant == n_visits:
                                consistency = 'Always Concordant'
                            elif n_under == n_visits:
                                consistency = 'Always Underestimated'
                            elif n_over == n_visits:
                                consistency = 'Always Overestimated'
                            else:
                                consistency = 'Variable'
                            
                            stability_data.append({
                                'pid': pid,
                                'n_visits': n_visits,
                                'concordant_visits': n_concordant,
                                'concordance_rate': n_concordant / n_visits * 100,
                                'consistency': consistency
                            })
                        
                        stability_df = pd.DataFrame(stability_data)
                        
                        col_stab1, col_stab2 = st.columns(2)
                        
                        with col_stab1:
                            # Consistency distribution
                            consist_dist = stability_df['consistency'].value_counts()
                            
                            fig_consist, ax_const = plt.subplots(figsize=(7, 5))
                            
                            wedges, texts, autotexts = ax_const.pie(
                                consist_dist.values, 
                                labels=consist_dist.index,
                                autopct='%1.1f%%', 
                                startangle=90,
                                colors=['#27ae60', '#e74c3c', '#f39c12', '#95a5a6'],
                                wedgeprops=dict(width=0.6, edgecolor='w'),
                                textprops=dict(color="black")
                            )
                            
                            ax_const.set_title(f'Agreement Consistency (n={len(stability_df)} patients)', fontsize=12, fontweight='bold', fontfamily='serif')
                            
                            plt.tight_layout()
                            st.pyplot(fig_consist)
                            add_svg_download_button(fig_consist, "agree_consistency_pie", key="agree_pie_svg")
                            plt.close(fig_consist)
                        
                        with col_stab2:
                            # Distribution of concordance rates
                            fig_conc_hist, ax_hist = plt.subplots(figsize=(7, 5))
                            
                            ax_hist.hist(stability_df['concordance_rate'], bins=20, color='#3498db', edgecolor='black', linewidth=0.5)
                            
                            ax_hist.set_xlabel('Concordance Rate (%)', fontsize=10, fontfamily='serif')
                            ax_hist.set_ylabel('Number of Patients', fontsize=10, fontfamily='serif')
                            ax_hist.set_title('Distribution of Patient-Level Concordance Rate', fontsize=12, fontweight='bold', fontfamily='serif')
                            
                            ax_hist.spines['top'].set_visible(False)
                            ax_hist.spines['right'].set_visible(False)
                            ax_hist.grid(True, alpha=0.3, axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig_conc_hist)
                            add_svg_download_button(fig_conc_hist, "patient_concordance_hist", key="conc_hist_svg")
                            plt.close(fig_conc_hist)
                        
                        # Summary metrics
                        always_conc = (stability_df['consistency'] == 'Always Concordant').sum()
                        always_under = (stability_df['consistency'] == 'Always Underestimated').sum()
                        variable = (stability_df['consistency'] == 'Variable').sum()
                        
                        st.info(f"""
                        📊 **Longitudinal Concordance Summary** (n={len(stability_df)} patients with repeat visits):
                        - **Always Concordant**: {always_conc} ({always_conc/len(stability_df)*100:.1f}%) - Agreement holds across all visits
                        - **Always Underestimated**: {always_under} ({always_under/len(stability_df)*100:.1f}%) - Consistent underestimation pattern
                        - **Variable**: {variable} ({variable/len(stability_df)*100:.1f}%) - Agreement status changes between visits
                        - Mean concordance rate: **{stability_df['concordance_rate'].mean():.1f}%**
                        """)
                    else:
                        st.info("No patients with multiple visits available for longitudinal analysis.")
                else:
                    st.info("Temporal features not available. Ensure data includes 'date' and 'pid' columns.")
                
        else:
            st.warning("⚠️ Paired dataset not available. Please ensure cvd_paired.csv is loaded.")

    # =========================================================================
    # TAB 9: HYPERTENSION ANALYSIS
    # =========================================================================
    with tab_htn:
        st.subheader("🩺 Hypertension Analysis")
        st.caption("Analysis of blood pressure categories and hypertension prevalence across demographics")
        
        # Check for required columns
        has_bp_category = 'bp_category' in df_cohort.columns
        has_sbp = 'sbp' in df_cohort.columns
        has_dbp = 'dbp' in df_cohort.columns
        
        if not has_bp_category and not (has_sbp and has_dbp):
            st.warning("⚠️ Blood pressure data not available. Required columns: 'bp_category' or ('sbp' and 'dbp').")
        else:
            # Create BP category if not present
            df_htn = df_cohort.copy()
            
            if not has_bp_category and has_sbp and has_dbp:
                # Define BP categories based on AHA/ACC 2017 guidelines
                def classify_bp(row):
                    sbp = row.get('sbp', np.nan)
                    dbp = row.get('dbp', np.nan)
                    if pd.isna(sbp) or pd.isna(dbp):
                        return 'Unknown'
                    if sbp < 120 and dbp < 80:
                        return 'Normal'
                    elif sbp < 130 and dbp < 80:
                        return 'Elevated'
                    elif sbp < 140 or dbp < 90:
                        return 'HTN Stage 1'
                    elif sbp >= 140 or dbp >= 90:
                        return 'HTN Stage 2'
                    else:
                        return 'Unknown'
                df_htn['bp_category'] = df_htn.apply(classify_bp, axis=1)
            
            # Define BP category order and colors
            BP_ORDER = ['Normal', 'Elevated', 'HTN Stage 1', 'HTN Stage 2']
            BP_COLORS = {
                'Normal': '#27ae60',
                'Elevated': '#f1c40f',
                'HTN Stage 1': '#e67e22',
                'HTN Stage 2': '#c0392b',
                'Unknown': '#95a5a6'
            }
            
            # Filter out Unknown for analysis
            df_htn_valid = df_htn[df_htn['bp_category'].isin(BP_ORDER)]
            
            if len(df_htn_valid) == 0:
                st.warning("No valid blood pressure data for analysis.")
            else:
                # Create hypertension indicator (HTN Stage 1 or 2)
                df_htn_valid = df_htn_valid.copy()
                df_htn_valid['has_hypertension'] = df_htn_valid['bp_category'].isin(['HTN Stage 1', 'HTN Stage 2']).astype(int)
                
                # ========================================
                # SECTION 1: Summary Metrics
                # ========================================
                st.markdown("### 📊 Summary Metrics")
                
                total_n = len(df_htn_valid)
                htn_n = df_htn_valid['has_hypertension'].sum()
                htn_prev = htn_n / total_n * 100 if total_n > 0 else 0
                
                # Calculate Wilson CI for hypertension prevalence
                htn_low, htn_high = calculate_wilson_ci(htn_n, total_n)
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total Analyzed", f"{total_n:,}")
                with col_m2:
                    st.metric("With Hypertension", f"{htn_n:,}")
                with col_m3:
                    st.metric("HTN Prevalence", f"{htn_prev:.1f}%", 
                              delta=f"95% CI: {htn_low:.1f}-{htn_high:.1f}%")
                with col_m4:
                    # Calculate mean SBP/DBP if available
                    if has_sbp and has_dbp:
                        mean_sbp = df_htn_valid['sbp'].mean()
                        mean_dbp = df_htn_valid['dbp'].mean()
                        st.metric("Mean BP", f"{mean_sbp:.0f}/{mean_dbp:.0f} mmHg")
                    else:
                        st.metric("BP Data", "Category Only")
                
                st.divider()
                
                # ========================================
                # SECTION 2: BP Category Distribution
                # ========================================
                st.markdown("### 📈 Blood Pressure Category Distribution")
                
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    # Pie chart of BP categories
                    bp_counts = df_htn_valid['bp_category'].value_counts()
                    bp_counts = bp_counts.reindex(BP_ORDER, fill_value=0)
                    
                    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
                    colors = [BP_COLORS[cat] for cat in bp_counts.index]
                    
                    wedges, texts, autotexts = ax_pie.pie(
                        bp_counts.values,
                        labels=bp_counts.index,
                        autopct=lambda pct: f'{pct:.1f}%\n(n={int(pct/100*total_n)})',
                        startangle=90,
                        colors=colors,
                        wedgeprops=dict(edgecolor='white', linewidth=2),
                        textprops=dict(fontsize=10)
                    )
                    
                    for autotext in autotexts:
                        autotext.set_fontsize(9)
                        autotext.set_weight('bold')
                    
                    ax_pie.set_title('Blood Pressure Category Distribution', fontsize=12, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig_pie)
                    add_download_button(fig_pie, "bp_category_pie", "matplotlib")
                    plt.close(fig_pie)
                
                with col_v2:
                    # Bar chart with cumulative line
                    bp_counts_pct = bp_counts / total_n * 100
                    
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                    
                    x_pos = np.arange(len(BP_ORDER))
                    colors_bar = [BP_COLORS[cat] for cat in BP_ORDER]
                    
                    bars = ax_bar.bar(x_pos, bp_counts_pct, color=colors_bar, edgecolor='black', linewidth=0.5)
                    
                    # Add value labels
                    for bar, val, count in zip(bars, bp_counts_pct, bp_counts):
                        height = bar.get_height()
                        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{val:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)
                    
                    # Cumulative line
                    ax_bar2 = ax_bar.twinx()
                    cumulative = bp_counts_pct.cumsum()
                    ax_bar2.plot(x_pos, cumulative, color='#2c3e50', linestyle='--', marker='o', 
                                 linewidth=2, markersize=8, label='Cumulative %')
                    
                    ax_bar.set_xlabel('Blood Pressure Category', fontweight='bold')
                    ax_bar.set_ylabel('Prevalence (%)', fontweight='bold')
                    ax_bar.set_xticks(x_pos)
                    ax_bar.set_xticklabels(BP_ORDER, rotation=15, ha='right')
                    ax_bar.set_ylim(0, max(bp_counts_pct) * 1.3)
                    ax_bar.grid(True, alpha=0.3, axis='y')
                    
                    ax_bar2.set_ylabel('Cumulative (%)', fontweight='bold')
                    ax_bar2.set_ylim(0, 110)
                    ax_bar2.legend(loc='upper left')
                    
                    ax_bar.set_title('Blood Pressure Category Prevalence', fontsize=12, fontweight='bold', pad=15)
                    ax_bar.spines['top'].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    add_download_button(fig_bar, "bp_category_bar", "matplotlib")
                    plt.close(fig_bar)
                
                st.divider()
                
                # ========================================
                # SECTION 3: Hypertension by Demographics
                # ========================================
                st.markdown("### 👥 Hypertension Prevalence by Demographics")
                
                # --- By Gender ---
                col_d1, col_d2 = st.columns(2)
                
                with col_d1:
                    st.markdown("#### By Gender")
                    if 'gender' in df_htn_valid.columns:
                        gender_htn = df_htn_valid.groupby('gender').agg(
                            n=('has_hypertension', 'count'),
                            htn_count=('has_hypertension', 'sum')
                        ).reset_index()
                        gender_htn['prevalence'] = gender_htn['htn_count'] / gender_htn['n'] * 100
                        
                        # Calculate CI
                        gender_htn['ci_low'] = gender_htn.apply(lambda r: calculate_wilson_ci(r['htn_count'], r['n'])[0], axis=1)
                        gender_htn['ci_high'] = gender_htn.apply(lambda r: calculate_wilson_ci(r['htn_count'], r['n'])[1], axis=1)
                        
                        fig_gender, ax_gender = plt.subplots(figsize=(7, 5))
                        
                        x_g = np.arange(len(gender_htn))
                        bars_g = ax_gender.bar(x_g, gender_htn['prevalence'], color=['#3498db', '#e74c3c'][:len(gender_htn)], 
                                               edgecolor='black', linewidth=0.5)
                        
                        # Add error bars
                        ax_gender.errorbar(x_g, gender_htn['prevalence'], 
                                           yerr=[gender_htn['prevalence'] - gender_htn['ci_low'], 
                                                 gender_htn['ci_high'] - gender_htn['prevalence']],
                                           fmt='none', color='black', capsize=5, capthick=2)
                        
                        # Labels
                        for bar, val, n in zip(bars_g, gender_htn['prevalence'], gender_htn['n']):
                            ax_gender.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                                           f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)
                        
                        ax_gender.set_ylabel('Hypertension Prevalence (%)', fontweight='bold')
                        ax_gender.set_xticks(x_g)
                        ax_gender.set_xticklabels(gender_htn['gender'])
                        ax_gender.set_ylim(0, max(gender_htn['ci_high']) * 1.2)
                        ax_gender.set_title('Hypertension Prevalence by Gender', fontweight='bold', fontsize=11)
                        ax_gender.spines['top'].set_visible(False)
                        ax_gender.spines['right'].set_visible(False)
                        ax_gender.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_gender)
                        add_download_button(fig_gender, "htn_by_gender", "matplotlib")
                        plt.close(fig_gender)
                    else:
                        st.info("Gender data not available.")
                
                with col_d2:
                    st.markdown("#### By Age Band")
                    if 'age_band' in df_htn_valid.columns:
                        age_htn = df_htn_valid.groupby('age_band', observed=False).agg(
                            n=('has_hypertension', 'count'),
                            htn_count=('has_hypertension', 'sum')
                        ).reset_index()
                        age_htn['prevalence'] = age_htn['htn_count'] / age_htn['n'] * 100
                        age_htn = age_htn[age_htn['n'] > 0]
                        
                        fig_age, ax_age = plt.subplots(figsize=(8, 5))
                        
                        x_a = np.arange(len(age_htn))
                        bars_a = ax_age.bar(x_a, age_htn['prevalence'], color='#9b59b6', edgecolor='black', linewidth=0.5)
                        
                        # Trend line
                        if len(age_htn) > 1:
                            z = np.polyfit(x_a, age_htn['prevalence'], 1)
                            p = np.poly1d(z)
                            ax_age.plot(x_a, p(x_a), '--', color='#c0392b', linewidth=2, label='Trend')
                            ax_age.legend(loc='upper left')
                        
                        for bar, val in zip(bars_a, age_htn['prevalence']):
                            ax_age.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
                        
                        ax_age.set_ylabel('Hypertension Prevalence (%)', fontweight='bold')
                        ax_age.set_xlabel('Age Band', fontweight='bold')
                        ax_age.set_xticks(x_a)
                        ax_age.set_xticklabels(age_htn['age_band'], rotation=45, ha='right')
                        ax_age.set_title('Hypertension Prevalence by Age', fontweight='bold', fontsize=11)
                        ax_age.spines['top'].set_visible(False)
                        ax_age.spines['right'].set_visible(False)
                        ax_age.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_age)
                        add_download_button(fig_age, "htn_by_age", "matplotlib")
                        plt.close(fig_age)
                    else:
                        st.info("Age band data not available.")
                
                # --- By Location Type ---
                if 'urban_rural' in df_htn_valid.columns:
                    st.markdown("#### By Location Type")
                    
                    loc_htn = df_htn_valid.groupby('urban_rural').agg(
                        n=('has_hypertension', 'count'),
                        htn_count=('has_hypertension', 'sum')
                    ).reset_index()
                    loc_htn['prevalence'] = loc_htn['htn_count'] / loc_htn['n'] * 100
                    loc_htn['ci_low'] = loc_htn.apply(lambda r: calculate_wilson_ci(r['htn_count'], r['n'])[0], axis=1)
                    loc_htn['ci_high'] = loc_htn.apply(lambda r: calculate_wilson_ci(r['htn_count'], r['n'])[1], axis=1)
                    
                    fig_loc, ax_loc = plt.subplots(figsize=(8, 5))
                    
                    x_l = np.arange(len(loc_htn))
                    colors_loc = ['#1abc9c', '#16a085', '#2c3e50'][:len(loc_htn)]
                    bars_l = ax_loc.bar(x_l, loc_htn['prevalence'], color=colors_loc, edgecolor='black', linewidth=0.5)
                    
                    ax_loc.errorbar(x_l, loc_htn['prevalence'],
                                    yerr=[loc_htn['prevalence'] - loc_htn['ci_low'],
                                          loc_htn['ci_high'] - loc_htn['prevalence']],
                                    fmt='none', color='black', capsize=5, capthick=2)
                    
                    for bar, val, n in zip(bars_l, loc_htn['prevalence'], loc_htn['n']):
                        ax_loc.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                                    f'{val:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)
                    
                    ax_loc.set_ylabel('Hypertension Prevalence (%)', fontweight='bold')
                    ax_loc.set_xticks(x_l)
                    ax_loc.set_xticklabels(loc_htn['urban_rural'])
                    ax_loc.set_title('Hypertension Prevalence by Location Type', fontweight='bold', fontsize=11)
                    ax_loc.spines['top'].set_visible(False)
                    ax_loc.spines['right'].set_visible(False)
                    ax_loc.grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    st.pyplot(fig_loc)
                    add_download_button(fig_loc, "htn_by_location", "matplotlib")
                    plt.close(fig_loc)
                
                st.divider()
                
                # ========================================
                # SECTION 4: Summary Table
                # ========================================
                st.markdown("### 📋 Hypertension Summary Table")
                st.caption("Blood pressure category distribution by gender with chi-square test")
                
                # Build the summary table
                if 'gender' in df_htn_valid.columns:
                    table_data = []
                    
                    male_df = df_htn_valid[df_htn_valid['gender'].isin(['M', 'Male', 'men'])]
                    female_df = df_htn_valid[df_htn_valid['gender'].isin(['F', 'Female', 'women'])]
                    n_male = len(male_df)
                    n_female = len(female_df)
                    n_total_htn = len(df_htn_valid)
                    
                    for bp_cat in BP_ORDER:
                        male_count = (male_df['bp_category'] == bp_cat).sum()
                        female_count = (female_df['bp_category'] == bp_cat).sum()
                        total_count = (df_htn_valid['bp_category'] == bp_cat).sum()
                        
                        table_data.append({
                            'BP Category': bp_cat,
                            'Male n (%)': f"{male_count} ({male_count/n_male*100:.1f}%)" if n_male > 0 else "0 (0.0%)",
                            'Female n (%)': f"{female_count} ({female_count/n_female*100:.1f}%)" if n_female > 0 else "0 (0.0%)",
                            'Total n (%)': f"{total_count} ({total_count/n_total_htn*100:.1f}%)" if n_total_htn > 0 else "0 (0.0%)"
                        })
                    
                    # Add hypertension row (HTN Stage 1 + 2)
                    male_htn = male_df['has_hypertension'].sum()
                    female_htn = female_df['has_hypertension'].sum()
                    total_htn = df_htn_valid['has_hypertension'].sum()
                    
                    table_data.append({
                        'BP Category': 'Any Hypertension (Stage 1+2)',
                        'Male n (%)': f"{male_htn} ({male_htn/n_male*100:.1f}%)" if n_male > 0 else "0 (0.0%)",
                        'Female n (%)': f"{female_htn} ({female_htn/n_female*100:.1f}%)" if n_female > 0 else "0 (0.0%)",
                        'Total n (%)': f"{total_htn} ({total_htn/n_total_htn*100:.1f}%)" if n_total_htn > 0 else "0 (0.0%)"
                    })
                    
                    # Add totals row
                    table_data.append({
                        'BP Category': 'Total',
                        'Male n (%)': f"{n_male} (100.0%)",
                        'Female n (%)': f"{n_female} (100.0%)",
                        'Total n (%)': f"{n_total_htn} (100.0%)"
                    })
                    
                    htn_table_df = pd.DataFrame(table_data)
                    
                    # Chi-square test
                    try:
                        contingency_htn = pd.crosstab(df_htn_valid['gender'], df_htn_valid['bp_category'])
                        chi2_htn, p_value_htn, dof_htn, expected_htn = stats.chi2_contingency(contingency_htn)
                        p_str_htn = f"P = {p_value_htn:.4f}" if p_value_htn >= 0.0001 else "P < 0.0001"
                    except Exception:
                        chi2_htn = None
                        p_str_htn = "P = N/A"
                    
                    # Style and display
                    st.dataframe(htn_table_df, use_container_width=True, hide_index=True)
                    st.caption(f"*{p_str_htn} based on χ² test comparing BP category distribution between genders.")
                    
                    if chi2_htn is not None:
                        st.info(f"""
                        📊 **Chi-Square Test Results:**
                        - χ² = {chi2_htn:.2f}
                        - Degrees of Freedom: {dof_htn}
                        - {p_str_htn}
                        
                        {'⚠️ Significant difference in BP category distribution between genders.' if p_value_htn < 0.05 else 'No statistically significant difference detected.'}
                        """)
                else:
                    # No gender data - simpler table
                    table_data = []
                    for bp_cat in BP_ORDER:
                        count = (df_htn_valid['bp_category'] == bp_cat).sum()
                        table_data.append({
                            'BP Category': bp_cat,
                            'n': count,
                            '%': f"{count/total_n*100:.1f}%"
                        })
                    
                    # Add hypertension row
                    table_data.append({
                        'BP Category': 'Any Hypertension',
                        'n': htn_n,
                        '%': f"{htn_prev:.1f}%"
                    })
                    
                    htn_table_df = pd.DataFrame(table_data)
                    st.dataframe(htn_table_df, use_container_width=True, hide_index=True)
                
                # ========================================
                # SECTION 5: BP & CVD Risk Relationship
                # ========================================
                st.divider()
                st.markdown("### 🔗 Hypertension & CVD Risk Relationship")
                
                if 'active_risk' in df_htn_valid.columns or 'risk_nonlab' in df_htn_valid.columns:
                    risk_col_htn = 'active_risk' if 'active_risk' in df_htn_valid.columns else 'risk_nonlab'
                    
                    # Mean CVD risk by BP category
                    risk_by_bp = df_htn_valid.groupby('bp_category')[risk_col_htn].agg(['mean', 'std', 'count']).reset_index()
                    risk_by_bp.columns = ['BP Category', 'Mean Risk (%)', 'SD', 'n']
                    risk_by_bp['BP Category'] = pd.Categorical(risk_by_bp['BP Category'], categories=BP_ORDER, ordered=True)
                    risk_by_bp = risk_by_bp.sort_values('BP Category')
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        fig_risk_bp, ax_risk_bp = plt.subplots(figsize=(8, 5))
                        
                        x_r = np.arange(len(risk_by_bp))
                        colors_r = [BP_COLORS.get(cat, '#95a5a6') for cat in risk_by_bp['BP Category']]
                        bars_r = ax_risk_bp.bar(x_r, risk_by_bp['Mean Risk (%)'], color=colors_r, edgecolor='black', linewidth=0.5)
                        
                        # Error bars (standard error)
                        se = risk_by_bp['SD'] / np.sqrt(risk_by_bp['n'])
                        ax_risk_bp.errorbar(x_r, risk_by_bp['Mean Risk (%)'], yerr=se, fmt='none', color='black', capsize=4)
                        
                        for bar, val in zip(bars_r, risk_by_bp['Mean Risk (%)']):
                            ax_risk_bp.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                            f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
                        
                        ax_risk_bp.set_ylabel('Mean CVD Risk (%)', fontweight='bold')
                        ax_risk_bp.set_xticks(x_r)
                        ax_risk_bp.set_xticklabels(risk_by_bp['BP Category'], rotation=15, ha='right')
                        ax_risk_bp.set_title('Mean CVD Risk by BP Category', fontweight='bold', fontsize=11)
                        ax_risk_bp.spines['top'].set_visible(False)
                        ax_risk_bp.spines['right'].set_visible(False)
                        ax_risk_bp.grid(True, alpha=0.3, axis='y')
                        
                        plt.tight_layout()
                        st.pyplot(fig_risk_bp)
                        add_download_button(fig_risk_bp, "cvd_risk_by_bp", "matplotlib")
                        plt.close(fig_risk_bp)
                    
                    with col_r2:
                        # High risk prevalence by BP category
                        high_risk_col = 'high_risk' if 'high_risk' in df_htn_valid.columns else None
                        if high_risk_col:
                            hr_by_bp = df_htn_valid.groupby('bp_category').agg(
                                n=('has_hypertension', 'count'),
                                high_risk_n=(high_risk_col, 'sum')
                            ).reset_index().rename(columns={'bp_category': 'BP Category'})
                            hr_by_bp['HR Prevalence (%)'] = hr_by_bp['high_risk_n'] / hr_by_bp['n'] * 100
                            hr_by_bp['BP Category'] = pd.Categorical(hr_by_bp['BP Category'], categories=BP_ORDER, ordered=True)
                            hr_by_bp = hr_by_bp.sort_values('BP Category')
                            
                            fig_hr_bp, ax_hr_bp = plt.subplots(figsize=(8, 5))
                            
                            x_hr = np.arange(len(hr_by_bp))
                            colors_hr = [BP_COLORS.get(cat, '#95a5a6') for cat in hr_by_bp['BP Category']]
                            bars_hr = ax_hr_bp.bar(x_hr, hr_by_bp['HR Prevalence (%)'], color=colors_hr, edgecolor='black', linewidth=0.5)
                            
                            for bar, val in zip(bars_hr, hr_by_bp['HR Prevalence (%)']):
                                ax_hr_bp.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                              f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
                            
                            ax_hr_bp.set_ylabel(f'High CVD Risk Prevalence (%)', fontweight='bold')
                            ax_hr_bp.set_xticks(x_hr)
                            ax_hr_bp.set_xticklabels(hr_by_bp['BP Category'], rotation=15, ha='right')
                            ax_hr_bp.set_title('High CVD Risk (≥10%) by BP Category', fontweight='bold', fontsize=11)
                            ax_hr_bp.spines['top'].set_visible(False)
                            ax_hr_bp.spines['right'].set_visible(False)
                            ax_hr_bp.grid(True, alpha=0.3, axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig_hr_bp)
                            add_download_button(fig_hr_bp, "high_cvd_risk_by_bp", "matplotlib")
                            plt.close(fig_hr_bp)
                        else:
                            st.info("High risk indicator not available.")
                else:
                    st.info("CVD risk data not available for relationship analysis.")

    # ==========================================================================
    # TAB 10 — Multi-Model CVD Risk  (FRS · SCORE2 Asia-Pacific · Globorisk)
    # ==========================================================================
    with tab_multi_risk:
        st.subheader("Multi-Model CVD Risk Comparison")
        st.info(
            "Framingham Risk Score (FRS), SCORE2 Asia-Pacific (2025), and "
            "Globorisk (Bangladesh-calibrated) computed on PHC data alongside "
            "WHO/ISH 2019 non-lab and lab scores."
        )

        # ── compute once and cache in session_state ──
        _mr_key = f"_mr_{id(df_cohort)}_{len(df_cohort)}"
        if _mr_key not in st.session_state:
            with st.spinner("Computing FRS, SCORE2-AP & Globorisk on the cohort …"):
                st.session_state[_mr_key] = add_all_risk_scores(df_cohort)
        df_mr = st.session_state[_mr_key]

        # ── model registry ──
        MR_MODELS = {
            "WHO Non-Lab": "risk_nonlab",
            "FRS (Non-Lab)": "risk_frs_nonlab",
            "SCORE2-AP": "risk_score2_ap",
            "Globorisk": "risk_globorisk",
        }
        if "risk_lab" in df_mr.columns and df_mr["risk_lab"].notna().sum() > 0:
            MR_MODELS["WHO Lab"] = "risk_lab"
        if "risk_frs_lab" in df_mr.columns and df_mr["risk_frs_lab"].notna().sum() > 0:
            MR_MODELS["FRS (Lab)"] = "risk_frs_lab"

        MR_CAT_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']
        MR_CAT_DISPLAY = [
            'Very low (<5%)', 'Low (5–10%)',
            'Moderate (10–20%)', 'High (20–30%)',
            'Very high (≥30%)'
        ]
        MR_BINS = [-np.inf, 5, 10, 20, 30, np.inf]

        # -------------------------------------------------------------------
        # TABLE A – Descriptive Summary
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table A: Descriptive Summary of Multi-Model 10-Year CVD Risk")
        st.caption(
            "Mean, SD, median, IQR and prevalence of elevated risk (≥10 % and ≥20 %) "
            "for every scoring system applied to the WHO-domain eligible cohort (age 40–74)."
        )

        desc_rows = []
        for lbl, col in MR_MODELS.items():
            if col not in df_mr.columns:
                continue
            s = pd.to_numeric(df_mr[col], errors='coerce').dropna()
            if len(s) == 0:
                continue
            desc_rows.append({
                "Model": lbl,
                "N": f"{len(s):,}",
                "Mean (SD)": f"{s.mean():.2f} ({s.std():.2f})",
                "Median [IQR]": f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]",
                "Min": f"{s.min():.1f}",
                "Max": f"{s.max():.1f}",
                "≥10 % n (%)": f"{(s>=10).sum()} ({(s>=10).mean()*100:.1f}%)",
                "≥20 % n (%)": f"{(s>=20).sum()} ({(s>=20).mean()*100:.1f}%)",
            })
        if desc_rows:
            st.dataframe(pd.DataFrame(desc_rows).set_index("Model"), use_container_width=True)

        # -------------------------------------------------------------------
        # TABLE B – Risk-Category Distribution per Model
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table B: Risk Category Distribution by Scoring System")
        st.caption("Number and percentage of individuals in each WHO 5-band risk category, per model.")

        for lbl, col in MR_MODELS.items():
            cc = f"_mr_cat_{col}"
            if col in df_mr.columns:
                df_mr[cc] = pd.cut(
                    pd.to_numeric(df_mr[col], errors='coerce'),
                    bins=MR_BINS, labels=MR_CAT_LABELS, right=False,
                )

        cat_rows = []
        for cl, cd in zip(MR_CAT_LABELS, MR_CAT_DISPLAY):
            row = {"Risk Category": cd}
            for lbl, col in MR_MODELS.items():
                cc = f"_mr_cat_{col}"
                if cc in df_mr.columns:
                    nv = df_mr[cc].notna().sum()
                    cnt = (df_mr[cc] == cl).sum()
                    row[f"{lbl} n (%)"] = f"{cnt} ({cnt/nv*100:.1f}%)" if nv else "—"
                else:
                    row[f"{lbl} n (%)"] = "—"
            cat_rows.append(row)
        # totals
        tot = {"Risk Category": "Total"}
        for lbl, col in MR_MODELS.items():
            cc = f"_mr_cat_{col}"
            nv = df_mr[cc].notna().sum() if cc in df_mr.columns else 0
            tot[f"{lbl} n (%)"] = f"{nv} (100.0%)"
        cat_rows.append(tot)
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        # -------------------------------------------------------------------
        # TABLE C – Multi-Model Risk by Gender
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table C: Multi-Model Risk by Gender")
        st.caption("Mean 10-year risk and high-risk (≥20 %) prevalence stratified by sex.")

        sex_mr = "gender_key" if "gender_key" in df_mr.columns else "gender"
        if sex_mr in df_mr.columns:
            g_rows = []
            for lbl, col in MR_MODELS.items():
                if col not in df_mr.columns:
                    continue
                for sv, sd in [("men", "Male"), ("women", "Female")]:
                    sub = df_mr[df_mr[sex_mr] == sv]
                    s = pd.to_numeric(sub[col], errors='coerce').dropna()
                    if len(s) == 0:
                        continue
                    g_rows.append({
                        "Model": lbl, "Sex": sd,
                        "N": f"{len(s):,}",
                        "Mean (SD)": f"{s.mean():.2f} ({s.std():.2f})",
                        "Median": f"{s.median():.1f}",
                        "≥10 % n (%)": f"{(s>=10).sum()} ({(s>=10).mean()*100:.1f}%)",
                        "≥20 % n (%)": f"{(s>=20).sum()} ({(s>=20).mean()*100:.1f}%)",
                    })
            if g_rows:
                st.dataframe(pd.DataFrame(g_rows).set_index(["Model","Sex"]), use_container_width=True)

        # -------------------------------------------------------------------
        # TABLE D – Multi-Model Risk by Age Band
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table D: Multi-Model Risk by Age Band")
        st.caption("Mean 10-year risk and high-risk prevalence across age strata for each model.")

        if "age_band" in df_mr.columns:
            ab_order = ["40-44","45-49","50-54","55-59","60-64","65-69","70-74"]
            a_rows = []
            for lbl, col in MR_MODELS.items():
                if col not in df_mr.columns:
                    continue
                for ab in ab_order:
                    sub = df_mr[df_mr["age_band"] == ab]
                    s = pd.to_numeric(sub[col], errors='coerce').dropna()
                    if len(s) == 0:
                        continue
                    a_rows.append({
                        "Model": lbl, "Age Band": ab,
                        "N": f"{len(s):,}",
                        "Mean": f"{s.mean():.2f}",
                        "Median": f"{s.median():.1f}",
                        "≥10 %": f"{(s>=10).mean()*100:.1f}%",
                        "≥20 %": f"{(s>=20).mean()*100:.1f}%",
                    })
            if a_rows:
                st.dataframe(pd.DataFrame(a_rows).set_index(["Model","Age Band"]), use_container_width=True)

        # -------------------------------------------------------------------
        # TABLE E – Pairwise Discordance vs WHO Non-Lab
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table E: Discordance Matrix — Each Model vs. WHO Non-Lab")
        st.caption(
            "Agreement / discordance in high-risk classification (≥ threshold) "
            "between WHO Non-Lab and alternative models. Cohen’s κ quantifies agreement beyond chance."
        )

        disc_thresh = st.select_slider(
            "High-risk threshold (%)",
            options=[5, 7.5, 10, 15, 20], value=20,
            key="_mr_disc_thr",
        )

        cmp_pairs = {
            "FRS vs WHO": "risk_frs_nonlab",
            "SCORE2-AP vs WHO": "risk_score2_ap",
            "Globorisk vs WHO": "risk_globorisk",
        }

        d_rows = []
        for pair, ca in cmp_pairs.items():
            if ca not in df_mr.columns or "risk_nonlab" not in df_mr.columns:
                continue
            res = compute_discordance_matrix(df_mr, ca, "risk_nonlab", disc_thresh)
            if res["n_valid"] == 0:
                continue
            mname = pair.split(" vs ")[0]
            d_rows.append({
                "Comparison": pair,
                "N (valid)": f"{res['n_valid']:,}",
                "Both High": res["agree_high"],
                "Both Low": res["agree_low"],
                f"Only {mname} ≥{disc_thresh}%": res["a_high_b_low"],
                f"Only WHO ≥{disc_thresh}%": res["a_low_b_high"],
                "Concordance": f"{res['concordance_rate']}%",
                "Discordance": f"{res['discordance_rate']}%",
                "Cohen’s κ": f"{res['kappa']:.3f}",
            })
        if d_rows:
            st.dataframe(pd.DataFrame(d_rows).set_index("Comparison"), use_container_width=True)

        # ── mini 2×2 confusion matrices ──
        st.markdown("#### Confusion Matrices (2 × 2)")
        cm_cols = st.columns(len(cmp_pairs))
        for i, (pair, ca) in enumerate(cmp_pairs.items()):
            with cm_cols[i]:
                if ca in df_mr.columns and "risk_nonlab" in df_mr.columns:
                    res = compute_discordance_matrix(df_mr, ca, "risk_nonlab", disc_thresh)
                    if res["n_valid"] > 0:
                        mname = pair.split(" vs ")[0]
                        cm = pd.DataFrame(
                            [[res["agree_high"], res["a_low_b_high"]],
                             [res["a_high_b_low"], res["agree_low"]]],
                            index=[f"{mname} ≥{disc_thresh}%", f"{mname} <{disc_thresh}%"],
                            columns=[f"WHO ≥{disc_thresh}%", f"WHO <{disc_thresh}%"],
                        )
                        st.markdown(f"**{mname}**")
                        st.dataframe(cm, use_container_width=True)

        # -------------------------------------------------------------------
        # TABLE F – Reclassification Summary (≥20 %)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table F: High-Risk Reclassification Summary (≥20 % threshold)")
        st.caption(
            "Net reclassification when switching from WHO Non-Lab to each alternative. "
            "'Only Model Flags' = high-risk by the model but not by WHO."
        )

        rc_rows = []
        for lbl, col in [("FRS (Non-Lab)","risk_frs_nonlab"),
                          ("SCORE2-AP","risk_score2_ap"),
                          ("Globorisk","risk_globorisk")]:
            if col not in df_mr.columns or "risk_nonlab" not in df_mr.columns:
                continue
            v = df_mr[["risk_nonlab", col]].dropna()
            w_hi = (v["risk_nonlab"] >= 20).sum()
            m_hi = (v[col] >= 20).sum()
            both = ((v["risk_nonlab"] >= 20) & (v[col] >= 20)).sum()
            only_m = ((v["risk_nonlab"] < 20) & (v[col] >= 20)).sum()
            only_w = ((v["risk_nonlab"] >= 20) & (v[col] < 20)).sum()
            rc_rows.append({
                "Model": lbl,
                "N": f"{len(v):,}",
                "WHO ≥20 %": w_hi,
                "Model ≥20 %": m_hi,
                "Both ≥20 %": both,
                "Only Model Flags": only_m,
                "Only WHO Flags": only_w,
                "Net Δ": m_hi - w_hi,
            })
        if rc_rows:
            st.dataframe(pd.DataFrame(rc_rows).set_index("Model"), use_container_width=True)

        # -------------------------------------------------------------------
        # TABLE G – Globorisk: Postmenopausal Women Focus
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Table G: Globorisk — Postmenopausal Women Focus (Women ≥ 50)")
        st.caption(
            "Globorisk applies a Bangladesh-specific recalibration (SEAR-D γ-factor) "
            "and a postmenopausal adjustment for women ≥ 50. Prior PHC studies show "
            "it identifies additional high-risk women missed by WHO non-lab charts."
        )

        _sx = "gender_key" if "gender_key" in df_mr.columns else "gender"
        women_df = df_mr[df_mr[_sx] == "women"].copy()

        if len(women_df) > 0:
            women_df["meno_group"] = women_df["age"].apply(
                lambda x: "Postmenopausal (≥50)" if x >= 50 else "Premenopausal (<50)"
            )

            meno_rows = []
            for grp in ["Premenopausal (<50)", "Postmenopausal (≥50)"]:
                sub = women_df[women_df["meno_group"] == grp]
                for lbl, col in [("WHO Non-Lab","risk_nonlab"),
                                  ("FRS (Non-Lab)","risk_frs_nonlab"),
                                  ("SCORE2-AP","risk_score2_ap"),
                                  ("Globorisk","risk_globorisk")]:
                    if col not in sub.columns:
                        continue
                    s = pd.to_numeric(sub[col], errors='coerce').dropna()
                    if len(s) == 0:
                        continue
                    meno_rows.append({
                        "Group": grp, "Model": lbl,
                        "N": f"{len(s):,}",
                        "Mean (SD)": f"{s.mean():.2f} ({s.std():.2f})",
                        "≥10 %": f"{(s>=10).mean()*100:.1f}%",
                        "≥20 %": f"{(s>=20).mean()*100:.1f}%",
                    })
            if meno_rows:
                st.dataframe(
                    pd.DataFrame(meno_rows).set_index(["Group","Model"]),
                    use_container_width=True,
                )

            # discordance metrics for postmenopausal women
            postm = women_df[women_df["age"] >= 50]
            if "risk_globorisk" in postm.columns and "risk_nonlab" in postm.columns:
                st.markdown("**Discordance: Globorisk vs WHO — Postmenopausal Women**")
                for thr in [10, 20]:
                    rp = compute_discordance_matrix(postm, "risk_globorisk", "risk_nonlab", thr)
                    if rp["n_valid"] > 0:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(f"Discordance (≥{thr}%)", f"{rp['discordance_rate']}%")
                        c2.metric("Cohen’s κ", f"{rp['kappa']:.3f}")
                        c3.metric("Only Globorisk flags", rp["a_high_b_low"])
                        c4.metric("Only WHO flags", rp["a_low_b_high"])
        else:
            st.info("No female participants in the current cohort.")

        # ── interpretation ──
        st.markdown("---")
        with st.expander("📝 Key Findings & Interpretation", expanded=False):
            st.markdown("""
| Pattern | Explanation |
|---------|------------|
| **FRS > WHO** | FRS was derived from a US (Framingham) cohort and consistently **overestimates** 10-year CVD risk in South-Asian populations. |
| **SCORE2-AP ≈ WHO (with variance)** | SCORE2 Asia-Pacific is regionally recalibrated (C-index ≈ 0.71). Moderate discordance suggests regional calibration matters. |
| **Globorisk > WHO in women ≥ 50** | Country-specific recalibration (Bangladesh γ ≈ 1.18–1.22) plus postmenopausal adjustment flags additional high-risk women missed by WHO charts. |
| **High discordance (>15 %)** | Clinical decisions (e.g. statin initiation) would differ depending on model choice—argues for local validation studies. |
            """)
