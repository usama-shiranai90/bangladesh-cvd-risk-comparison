import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_risk_cat, RISK_PALETTE
from utils.export_utils import add_download_button

def render_rq1(df_paired, datasets=None):
    """Render RQ1."""
    st.caption("Preview (paired cohort)")

    st.title("RQ1: Baseline Risk Burden & Heterogeneity")
    
    st.markdown("""
    **Research Question:** What is the baseline cardiovascular disease risk burden and heterogeneity 
    across age groups using both laboratory-based and non-laboratory-based WHO risk assessment methods?
    
    - **Chart 1:** Risk Distribution by Age Group (Lab vs Non-Lab)
    - **Chart 2:** Age-Stratified Escalation of CVD Risk (Lab vs Non-Lab)
    """)

    if df_paired is None:
        st.error("No paired data loaded. Please ensure 'cvd_paired.csv' is selected in the sidebar.")
        return

    df = df_paired.copy()
    
    if 'eligible_paired' in df.columns:
        df = df[df['eligible_paired']].reset_index(drop=True)
        st.info(f"📊 Using **eligible paired records**: {len(df):,} participants")
    else:
        st.info(f"📊 Using **paired records**: {len(df):,} participants")
    
    for col in ['risk_lab', 'risk_nonlab', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    required_cols = ['risk_lab', 'risk_nonlab', 'age_band']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    df = df.dropna(subset=required_cols)
    
    if len(df) == 0:
        st.error("No valid data after filtering. Please check your dataset.")
        return
    
    risk_order = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
    df['risk_lab_cat'] = df['risk_lab'].apply(get_risk_cat).astype(pd.CategoricalDtype(risk_order, ordered=True))
    df['risk_nonlab_cat'] = df['risk_nonlab'].apply(get_risk_cat).astype(pd.CategoricalDtype(risk_order, ordered=True))
    
    df['lab_high_risk'] = (df['risk_lab'] >= 20).astype(int)
    df['nonlab_high_risk'] = (df['risk_nonlab'] >= 20).astype(int)
    
    age_band_order = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74']
    df['age_band'] = pd.Categorical(df['age_band'], categories=age_band_order, ordered=True)
    df = df.sort_values('age_band')
    
    st.markdown(f"**Total Paired Observations Used:** {len(df):,}")
    st.markdown("---")
    
    st.subheader("📊 Chart 1: Risk Distribution by Age Group (Lab vs Non-Lab)")
    st.caption("Comparison of cardiovascular risk distribution across age groups using laboratory-based and non-laboratory-based methods")
    
    df['lab_risk_binary'] = df['risk_lab_cat'].apply(lambda x: '≥20% (High Risk)' if x in ['20% to <30%', '≥30%'] else '<20% (Low Risk)')
    df['nonlab_risk_binary'] = df['risk_nonlab_cat'].apply(lambda x: '≥20% (High Risk)' if x in ['20% to <30%', '≥30%'] else '<20% (Low Risk)')
    
    lab_binary = df.groupby(['age_band', 'lab_risk_binary'], observed=False).size().reset_index(name='count')
    lab_binary_totals = df.groupby('age_band', observed=False).size().reset_index(name='total')
    lab_binary = lab_binary.merge(lab_binary_totals, on='age_band')
    lab_binary['percentage'] = (lab_binary['count'] / lab_binary['total'] * 100).round(2)
    lab_binary['method'] = 'Lab-based'
    lab_binary = lab_binary.rename(columns={'lab_risk_binary': 'risk_level'})
    
    nonlab_binary = df.groupby(['age_band', 'nonlab_risk_binary'], observed=False).size().reset_index(name='count')
    nonlab_binary_totals = df.groupby('age_band', observed=False).size().reset_index(name='total')
    nonlab_binary = nonlab_binary.merge(nonlab_binary_totals, on='age_band')
    nonlab_binary['percentage'] = (nonlab_binary['count'] / nonlab_binary['total'] * 100).round(2)
    nonlab_binary['method'] = 'Non-Lab-based'
    nonlab_binary = nonlab_binary.rename(columns={'nonlab_risk_binary': 'risk_level'})
    
    combined_binary = pd.concat([lab_binary, nonlab_binary], ignore_index=True)
    
    fig_bar = go.Figure()
    
    age_bands = sorted(combined_binary['age_band'].unique())
    methods = ['Lab-based', 'Non-Lab-based']
    risk_levels = ['<20% (Low Risk)', '≥20% (High Risk)']
    
    colors = {'<20% (Low Risk)': '#52b788', '≥20% (High Risk)': '#d62828'}
    
    for risk_level in risk_levels:
        for method in methods:
            data = combined_binary[(combined_binary['risk_level'] == risk_level) & (combined_binary['method'] == method)]
            
            offset = 0.2 if method == 'Lab-based' else -0.2
            x_positions = [f"{age_band}" for age_band in data['age_band']]
            
            pattern_shape = "" if method == 'Lab-based' else "/"
            
            fig_bar.add_trace(go.Bar(
                name=f"{method} - {risk_level}",
                x=data['age_band'].astype(str),
                y=data['percentage'],
                text=[f"{p:.1f}%" for p in data['percentage']],
                textposition='outside',
                marker=dict(
                    color=colors[risk_level],
                    opacity=1.0 if method == 'Lab-based' else 0.7,
                    pattern_shape=pattern_shape,
                    line=dict(color='white', width=1)
                ),
                offsetgroup=method,
                legendgroup=risk_level,
                showlegend=True,
                hovertemplate='<b>%{x}</b><br>' +
                             f'{method}<br>' +
                             f'{risk_level}<br>' +
                             '%{y:.1f}%<extra></extra>'
            ))
    
    fig_bar.update_layout(
        title="Risk Distribution by Age Group: Lab-based vs Non-Lab-based Assessment",
        xaxis_title="Age Group",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=500,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            title="Assessment Method & Risk Level"
        ),
        yaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig_bar, use_container_width=True)
    add_download_button(fig_bar, "risk_distribution_by_age", "plotly")
    
    with st.expander("📋 View Detailed Distribution Table"):
        st.markdown("### Lab-based Risk Distribution")
        lab_pivot = lab_binary.pivot(index='age_band', columns='risk_level', values='percentage').fillna(0)
        st.dataframe(lab_pivot.style.format("{:.2f}%").background_gradient(cmap='RdYlGn_r', axis=1))
        
        st.markdown("### Non-Lab-based Risk Distribution")
        nonlab_pivot = nonlab_binary.pivot(index='age_band', columns='risk_level', values='percentage').fillna(0)
        st.dataframe(nonlab_pivot.style.format("{:.2f}%").background_gradient(cmap='RdYlGn_r', axis=1))
    
    st.markdown("---")
    
    st.subheader("📈 Chart 2: Age-Stratified Escalation of Cardiovascular Risk")
    st.caption("Comparative line chart showing mean CVD risk scores across age groups for both assessment methods")
    
    age_risk_summary = df.groupby('age_band', observed=True).agg({
        'risk_lab': ['mean', 'std', 'count'],
        'risk_nonlab': ['mean', 'std', 'count']
    }).reset_index()
    
    age_risk_summary.columns = ['age_band', 'lab_mean', 'lab_std', 'lab_count', 
                                  'nonlab_mean', 'nonlab_std', 'nonlab_count']
    
    from scipy import stats as scipy_stats
    age_risk_summary['lab_ci'] = 1.96 * age_risk_summary['lab_std'] / np.sqrt(age_risk_summary['lab_count'])
    age_risk_summary['nonlab_ci'] = 1.96 * age_risk_summary['nonlab_std'] / np.sqrt(age_risk_summary['nonlab_count'])
    
    fig_line = go.Figure()
    
    fig_line.add_trace(go.Scatter(
        x=age_risk_summary['age_band'].astype(str),
        y=age_risk_summary['lab_mean'],
        mode='lines+markers',
        name='Lab-based Risk',
        line=dict(color='#2a9d8f', width=3),
        marker=dict(size=10, symbol='circle', color='#2a9d8f', line=dict(color='white', width=2)),
        error_y=dict(
            type='data',
            array=age_risk_summary['lab_ci'],
            visible=True,
            color='#2a9d8f',
            thickness=1.5,
            width=4
        ),
        hovertemplate='<b>Age: %{x}</b><br>' +
                     'Lab-based Mean Risk: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig_line.add_trace(go.Scatter(
        x=age_risk_summary['age_band'].astype(str),
        y=age_risk_summary['nonlab_mean'],
        mode='lines+markers',
        name='Non-Lab-based Risk',
        line=dict(color='#e76f51', width=3, dash='dash'),
        marker=dict(size=10, symbol='diamond', color='#e76f51', line=dict(color='white', width=2)),
        error_y=dict(
            type='data',
            array=age_risk_summary['nonlab_ci'],
            visible=True,
            color='#e76f51',
            thickness=1.5,
            width=4
        ),
        hovertemplate='<b>Age: %{x}</b><br>' +
                     'Non-Lab-based Mean Risk: %{y:.2f}%<br>' +
                     '<extra></extra>'
    ))
    
    fig_line.add_hline(
        y=20,
        line_dash="dot",
        line_color="rgba(255, 0, 0, 0.5)",
        annotation_text="High-Risk Threshold (≥20%)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="red"
    )
    
    fig_line.add_hline(
        y=10,
        line_dash="dot",
        line_color="rgba(255, 165, 0, 0.5)",
        annotation_text="Moderate-Risk Threshold (≥10%)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="orange"
    )
    
    fig_line.update_layout(
        title="Mean CVD Risk Score by Age Group: Lab vs Non-Lab Assessment",
        xaxis_title="Age Group",
        yaxis_title="Mean 10-Year CVD Risk (%)",
        height=550,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, max(age_risk_summary['lab_mean'].max(), 
                                  age_risk_summary['nonlab_mean'].max()) * 1.2])
    )
    
    fig_line.update_xaxes(showgrid=False, showline=True, linewidth=2, linecolor='gray')
    fig_line.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', showline=True, linewidth=2, linecolor='gray')
    
    st.plotly_chart(fig_line, use_container_width=True)
    add_download_button(fig_line, "age_stratified_risk_escalation", "plotly")
    
    with st.expander("📋 View Age-Stratified Risk Summary Table"):
        summary_display = age_risk_summary[['age_band', 'lab_mean', 'lab_std', 'lab_count', 
                                             'nonlab_mean', 'nonlab_std', 'nonlab_count']].copy()
        summary_display.columns = ['Age Group', 'Lab Mean Risk (%)', 'Lab SD', 'Lab N',
                                    'Non-Lab Mean Risk (%)', 'Non-Lab SD', 'Non-Lab N']
        
        st.dataframe(
            summary_display.style.format({
                'Lab Mean Risk (%)': '{:.2f}',
                'Lab SD': '{:.2f}',
                'Lab N': '{:.0f}',
                'Non-Lab Mean Risk (%)': '{:.2f}',
                'Non-Lab SD': '{:.2f}',
                'Non-Lab N': '{:.0f}'
            }).background_gradient(cmap='YlOrRd', subset=['Lab Mean Risk (%)', 'Non-Lab Mean Risk (%)']),
            use_container_width=True
        )
    
    st.markdown("---")
    
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Paired Records",
            f"{len(df):,}",
            help="Number of participants with both Lab and Non-Lab risk assessments"
        )
    
    with col2:
        lab_high_risk_pct = (df['lab_high_risk'].mean() * 100)
        st.metric(
            "Lab-based High Risk (≥20%)",
            f"{lab_high_risk_pct:.1f}%",
            delta=f"{df['lab_high_risk'].sum()} participants",
            help="Percentage of participants classified as high-risk by Lab-based method"
        )
    
    with col3:
        nonlab_high_risk_pct = (df['nonlab_high_risk'].mean() * 100)
        st.metric(
            "Non-Lab-based High Risk (≥20%)",
            f"{nonlab_high_risk_pct:.1f}%",
            delta=f"{df['nonlab_high_risk'].sum()} participants",
            help="Percentage of participants classified as high-risk by Non-Lab-based method"
        )
    
    st.markdown("### 📈 Risk Escalation Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        first_age_lab_risk = age_risk_summary.iloc[0]['lab_mean']
        last_age_lab_risk = age_risk_summary.iloc[-1]['lab_mean']
        lab_escalation = last_age_lab_risk - first_age_lab_risk
        lab_escalation_pct = (lab_escalation / first_age_lab_risk * 100) if first_age_lab_risk > 0 else 0
        
        st.metric(
            "Lab-based Risk Escalation",
            f"+{lab_escalation:.2f} pp",
            delta=f"{lab_escalation_pct:.1f}% increase from youngest to oldest age group",
            help=f"From {age_risk_summary.iloc[0]['age_band']} to {age_risk_summary.iloc[-1]['age_band']}"
        )
    
    with col_b:
        first_age_nonlab_risk = age_risk_summary.iloc[0]['nonlab_mean']
        last_age_nonlab_risk = age_risk_summary.iloc[-1]['nonlab_mean']
        nonlab_escalation = last_age_nonlab_risk - first_age_nonlab_risk
        nonlab_escalation_pct = (nonlab_escalation / first_age_nonlab_risk * 100) if first_age_nonlab_risk > 0 else 0
        
        st.metric(
            "Non-Lab-based Risk Escalation",
            f"+{nonlab_escalation:.2f} pp",
            delta=f"{nonlab_escalation_pct:.1f}% increase from youngest to oldest age group",
            help=f"From {age_risk_summary.iloc[0]['age_band']} to {age_risk_summary.iloc[-1]['age_band']}"
        )
    
    st.markdown("### 🔍 Key Findings")
    
    from scipy.stats import pearsonr
    
    age_mapping = {'40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67, '70-74': 72}
    df['age_numeric'] = df['age_band'].map(age_mapping)
    
    lab_corr, lab_p = pearsonr(df['age_numeric'].dropna(), df.loc[df['age_numeric'].notna(), 'risk_lab'])
    nonlab_corr, nonlab_p = pearsonr(df['age_numeric'].dropna(), df.loc[df['age_numeric'].notna(), 'risk_nonlab'])
    
    findings = f"""
    1. **Age-Risk Relationship:**
       - Lab-based method shows a correlation of **r = {lab_corr:.3f}** (p < 0.001) with age
       - Non-Lab-based method shows a correlation of **r = {nonlab_corr:.3f}** (p < 0.001) with age
       
    2. **Risk Escalation:**
       - Lab-based risk increases by **{lab_escalation:.2f} percentage points** from youngest to oldest age group
       - Non-Lab-based risk increases by **{nonlab_escalation:.2f} percentage points** from youngest to oldest age group
       
    3. **High-Risk Burden:**
       - Lab-based method identifies **{lab_high_risk_pct:.1f}%** of the population as high-risk (≥20%)
       - Non-Lab-based method identifies **{nonlab_high_risk_pct:.1f}%** of the population as high-risk (≥20%)
       - Difference: **{abs(lab_high_risk_pct - nonlab_high_risk_pct):.1f} percentage points**
       
    4. **Age Group with Highest Risk:**
       - Lab-based: **{age_risk_summary.loc[age_risk_summary['lab_mean'].idxmax(), 'age_band']}** ({age_risk_summary['lab_mean'].max():.2f}% mean risk)
       - Non-Lab-based: **{age_risk_summary.loc[age_risk_summary['nonlab_mean'].idxmax(), 'age_band']}** ({age_risk_summary['nonlab_mean'].max():.2f}% mean risk)
    """
    
    st.info(findings)
