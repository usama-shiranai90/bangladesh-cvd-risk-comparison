import streamlit as st
import plotly.express as px

def render_overview(df_merged, totals):
    st.title("🇧🇩 National Overview (Bangladesh)")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_n = totals.get('total_n', 0)
    
    with kpi1:
        st.metric("Total Screening", f"{total_n:,}", delta="WHO Domain (40-74y)")
    
    with kpi2:
        high_risk = 0
        if df_merged is not None and "risk_nonlab" in df_merged.columns:
            high_risk = (df_merged["risk_nonlab"] >= 20).sum()
        pct = (high_risk / total_n * 100) if total_n else 0
        st.metric("Non-Lab High Risk (≥20%)", f"{high_risk:,}", f"{pct:.1f}% Prevalence", delta_color="inverse")
        
    with kpi3:
        high_risk = 0
        if df_merged is not None and "risk_lab" in df_merged.columns:
            high_risk = (df_merged["risk_lab"] >= 20).sum()
        pct = (high_risk / total_n * 100) if total_n else 0
        st.metric("Lab High Risk (≥20%)", f"{high_risk:,}", f"{pct:.1f}% Prevalence", delta_color="inverse")
        
    with kpi4:
        # Lab coverage
        paired_n = totals.get('paired_n', 0)
        cov = (paired_n / total_n * 100) if total_n else 0
        st.metric("Lab Data Coverage", f"{paired_n:,}", f"{cov:.1f}% of Cohort")

    # Map
    st.subheader("📍 Site Coverage Map")
    if df_merged is not None and "site_latitude" in df_merged.columns:
        # Aggregation by site
        title = "project_title"
       
        site_stats = df_merged.groupby("site_id").agg({
            "site_latitude": "first",
            "site_longitude": "first",
            "risk_nonlab": lambda x: (x >= 20).mean() * 100,
            "pid": "count",
            title: "first"
        }).reset_index()
        site_stats.rename(columns={"risk_nonlab": "High Risk %", "pid": "Sample Size"}, inplace=True)
        
   

        # Helper to fill missing titles

        site_stats[title] = site_stats[title].fillna("Site " + site_stats['site_id'].astype(str))
        site_stats.loc[site_stats[title].astype(str).str.strip() == "", title] = "Site " + site_stats['site_id'].astype(str)
        
        site_stats = site_stats.dropna(subset=["site_latitude", "site_longitude"])
        
        fig_map = px.scatter_mapbox(
            site_stats,
            lat="site_latitude",
            lon="site_longitude",
            size="Sample Size",
            color="High Risk %",
            color_continuous_scale="RdYlGn_r", # Red is high risk
            hover_name=title,
            hover_data=["High Risk %", "Sample Size"],
            zoom=6,
            center={"lat": 23.6850, "lon": 90.3563}, # Bangladesh Center
            mapbox_style="carto-positron",
            height=600,
            size_max=30
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Geographical data missing.")
