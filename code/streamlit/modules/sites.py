import streamlit as st
import numpy as np
import plotly.graph_objects as go

def render_sites(df_merged):
    st.title("🏥 Site Heterogeneity Analysis")
    st.markdown("Analyzing variability in high-risk prevalence (≥20%) across different service points.")

    st.dataframe(df_merged)

    if df_merged is None: st.stop()


    # Calculate stats per site
    site_stats = df_merged.groupby("site_id").agg({
        "risk_nonlab": lambda x: (x >= 20).sum(),
        "pid": "count",
        "site_title": "first"
    }).reset_index()
    site_stats.columns = ["site_id", "high_risk_n", "total_n", "site_title"]
    site_stats["prevalence"] = site_stats["high_risk_n"] / site_stats["total_n"] * 100

    # Filter small sites
    min_n = st.slider("Min Sample Size per Site", 10, 200, 30)
    filtered_sites = site_stats[site_stats["total_n"] >= min_n].copy()

    # Funnel Plot
    st.subheader("Funnel Plot: Prevalence vs Sample Size")

    # Global mean
    global_p = filtered_sites["high_risk_n"].sum() / filtered_sites["total_n"].sum()

    # Confidence Limits (Approx)
    # 95% CI = p +/- 1.96 * sqrt(p(1-p)/n)
    n_range = np.linspace(filtered_sites["total_n"].min(), filtered_sites["total_n"].max(), 100)

    def get_ci(p, n, z):
        se = np.sqrt(p * (1 - p) / n)
        return (p + z * se) * 100, (p - z * se) * 100

    upper_95, lower_95 = get_ci(global_p, n_range, 1.96)
    upper_99, lower_99 = get_ci(global_p, n_range, 3.0)

    fig_funnel = go.Figure()

    # Control limits
    fig_funnel.add_trace(go.Scatter(x=n_range, y=upper_99, mode='lines', line=dict(dash='dot', color='gray'), name='99.7% Limit'))
    fig_funnel.add_trace(go.Scatter(x=n_range, y=lower_99, mode='lines', line=dict(dash='dot', color='gray'), showlegend=False))
    fig_funnel.add_trace(go.Scatter(x=n_range, y=upper_95, mode='lines', line=dict(color='gray'), name='95% Limit'))
    fig_funnel.add_trace(go.Scatter(x=n_range, y=lower_95, mode='lines', line=dict(color='gray'), showlegend=False))

    # Mean line
    fig_funnel.add_trace(go.Scatter(x=n_range, y=[global_p*100]*len(n_range), mode='lines', line=dict(color='black'), name='Mean Prevalence'))

    # Points
    fig_funnel.add_trace(go.Scatter(
        x=filtered_sites["total_n"],
        y=filtered_sites["prevalence"],
        mode='markers',
        text=filtered_sites["site_title"],
        marker=dict(color='#006a4e', size=8, opacity=0.7),
        name='Sites'
    ))

    fig_funnel.update_layout(
        xaxis_title="Site Sample Size (N)",
        yaxis_title="High Risk Prevalence (%)",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.caption(f"Global Weighted Prevalence: {global_p*100:.2f}% | Analyzed {len(filtered_sites)} sites with N>={min_n}")
