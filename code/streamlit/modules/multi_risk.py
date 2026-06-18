"""
Multi-Model CVD Risk Analysis Module
=====================================

Streamlit page that implements:
  Tab 1 — Interactive Individual Calculator (FRS / SCORE2-AP / Globorisk)
  Tab 2 — Batch Computation on PHC Dataset
  Tab 3 — Model Comparison & Discordance vs WHO
  Tab 4 — Subgroup Deep-Dive (sex, age, smoking, diabetes)
  Tab 5 — Globorisk Focus (postmenopausal, country-specific)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.helpers import RISK_PALETTE, get_risk_cat
from utils.risk_engines import (
    compute_frs_lab, compute_frs_nonlab,
    compute_score2_ap, compute_globorisk,
    add_all_risk_scores, compute_discordance_matrix,
)

MODEL_COLORS = {
    "WHO Non-Lab": "#2a9d8f",
    "WHO Lab": "#264653",
    "FRS Lab": "#e76f51",
    "FRS Non-Lab": "#f4a261",
    "SCORE2-AP": "#6a4c93",
    "Globorisk": "#1982c4",
}

RISK_BINS = [-np.inf, 5, 10, 20, 30, np.inf]
RISK_LABELS = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
RISK_CAT_COLORS = {
    "<5%": "#2a9d8f",
    "5% to <10%": "#e9c46a",
    "10% to <20%": "#f4a261",
    "20% to <30%": "#e76f51",
    "≥30%": "#d62828",
}


def _card(label: str, value: str, color: str, subtitle: str = ""):
    """Render a styled metric card."""
    return f"""
    <div style='background:linear-gradient(135deg, {color}dd, {color}99);
                padding:22px 18px;border-radius:14px;color:white;
                text-align:center;box-shadow:0 6px 20px {color}44;
                transition:transform 0.2s;min-height:130px;
                display:flex;flex-direction:column;justify-content:center;'>
        <p style='margin:0;font-size:0.82em;opacity:0.88;letter-spacing:0.5px;
                   text-transform:uppercase;font-weight:600;'>{label}</p>
        <h2 style='margin:6px 0 2px;font-size:2.2em;color:white;font-weight:700;'>{value}</h2>
        <p style='margin:0;font-size:0.78em;opacity:0.75;'>{subtitle}</p>
    </div>"""


def _section_header(icon: str, title: str, desc: str = ""):
    """Section header."""
    st.markdown(f"""
    <div style='margin:28px 0 12px;'>
        <h3 style='margin:0;'>{icon} {title}</h3>
        {"<p style='color:#6c757d;margin:4px 0 0;font-size:0.92em;'>" + desc + "</p>" if desc else ""}
    </div>""", unsafe_allow_html=True)


def render_multi_risk(datasets: dict):
    """Main entry point for the Multi-Model Risk Analysis page."""
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                padding:32px 28px;border-radius:16px;margin-bottom:24px;
                box-shadow:0 8px 32px rgba(0,0,0,0.25);'>
        <h1 style='color:white;margin:0;font-size:2em;'>
            🫀 Multi-Model CVD Risk Analysis
        </h1>
        <p style='color:#a8b2d1;margin:8px 0 0;font-size:1.05em;'>
            Framingham Risk Score · SCORE2 Asia-Pacific · Globorisk — compared against WHO/ISH charts
        </p>
    </div>""", unsafe_allow_html=True)

    tab_calc, tab_batch, tab_compare, tab_subgroup, tab_globo, tab_tables = st.tabs([
        "🧮 Individual Calculator",
        "📊 Batch Computation",
        "⚖️ Model Comparison",
        "👥 Subgroup Analysis",
        "🌍 Globorisk Focus",
        "📋 Publication Tables",
    ])

    with tab_calc:
        _render_individual_calculator()

    with tab_batch:
        _render_batch_computation(datasets)

    with tab_compare:
        _render_model_comparison(datasets)

    with tab_subgroup:
        _render_subgroup_analysis(datasets)

    with tab_globo:
        _render_globorisk_focus(datasets)

    with tab_tables:
        _render_publication_tables(datasets)


def _render_individual_calculator():
    """Render individual calculator."""
    _section_header("🧮", "Individual Multi-Model Risk Calculator",
                    "Compute 10-year CVD risk side-by-side across all four models.")

    c1, c2 = st.columns(2)
    with c1:
        age_in = st.number_input("Age (years)", 30, 80, 55, key="mr_age")
        sex_in = st.selectbox("Sex", ["Male", "Female"], key="mr_sex")
        sbp_in = st.number_input("Systolic BP (mmHg)", 70, 260, 135, key="mr_sbp")
        smoker_in = st.selectbox(
            "Current Smoker?", ["No", "Yes"],
            key="mr_smoke")
    with c2:
        bmi_in = st.number_input("BMI (kg/m²)", 10.0, 60.0, 26.0, key="mr_bmi")
        diab_in = st.selectbox("Diabetes?", ["No", "Yes"], key="mr_diab")
        has_lab = st.toggle("Include lab data (TC / HDL)?", key="mr_has_lab")
        tc_in, hdl_in = None, None
        if has_lab:
            tc_in = st.number_input("Total Cholesterol (mmol/L)", 2.0, 15.0, 5.5, key="mr_tc")
            hdl_in = st.number_input("HDL Cholesterol (mmol/L)", 0.3, 5.0, 1.2, key="mr_hdl")

    if st.button("⚡ Compute All Models", type="primary", key="mr_calc"):
        sex_key = "men" if sex_in == "Male" else "women"
        smoke_bool = smoker_in == "Yes"
        diab_bool = diab_in == "Yes"

        results = {}
        if has_lab and tc_in and hdl_in:
            results["FRS Lab"] = compute_frs_lab(
                age_in, sex_key, sbp_in, tc_in, hdl_in, smoke_bool, diab_bool)
        results["FRS Non-Lab"] = compute_frs_nonlab(
            age_in, sex_key, sbp_in, bmi_in, smoke_bool, diab_bool)
        results["SCORE2-AP"] = compute_score2_ap(
            age_in, sex_key, sbp_in,
            tc_in if has_lab else None,
            hdl_in if has_lab else None,
            bmi_in, smoke_bool, diab_bool)
        results["Globorisk"] = compute_globorisk(
            age_in, sex_key, sbp_in, bmi_in,
            tc_in if has_lab else None,
            hdl_in if has_lab else None,
            smoke_bool, diab_bool)

        st.markdown("### Results")
        cols = st.columns(len(results))
        for i, (name, val) in enumerate(results.items()):
            with cols[i]:
                if val is not None:
                    cat = get_risk_cat(val)
                    color = RISK_CAT_COLORS.get(cat, "#333")
                    st.markdown(_card(name, f"{val}%", color, f"Category: {cat}"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(_card(name, "N/A", "#6c757d", "Out of range"),
                                unsafe_allow_html=True)

        model_names = list(results.keys())
        model_vals = [v if v is not None else 0 for v in results.values()]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=model_vals + [model_vals[0]],
            theta=model_names + [model_names[0]],
            fill='toself',
            fillcolor='rgba(106, 76, 147, 0.15)',
            line=dict(color='#6a4c93', width=2),
            marker=dict(size=8, color=[MODEL_COLORS.get(n, "#333") for n in model_names] + [MODEL_COLORS.get(model_names[0], "#333")]),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(model_vals) * 1.3 + 5]),
            ),
            showlegend=False,
            title="Model Risk Comparison (Radar)",
            height=380,
            margin=dict(t=50, b=30, l=60, r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Interpretation Notes:**
        - **FRS** was developed on US populations (Framingham cohort) and tends to **overestimate** risk
          in South Asian populations compared to WHO/ISH.
        - **SCORE2 Asia-Pacific** (2024/2025) is recalibrated for the Asia-Pacific region (C-index ≈ 0.71).
        - **Globorisk** applies country-specific recalibration for Bangladesh, and adds a postmenopausal
          risk adjustment for women ≥ 50.
        """)


@st.cache_data(show_spinner="Computing risk scores on full dataset…")
def _compute_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Cache the expensive row-wise computation."""
    return add_all_risk_scores(df)


def _render_batch_computation(datasets: dict):
    """Render batch computation."""
    _section_header("📊", "Batch Risk Computation on PHC Data",
                    "Apply FRS, SCORE2-AP, and Globorisk to the entire dataset and compare with existing WHO scores.")

    ds_opts = {k: v for k, v in datasets.items() if v is not None}
    if not ds_opts:
        st.warning("No datasets available. Load data from the sidebar first.")
        return

    chosen = st.selectbox("Select dataset", list(ds_opts.keys()), key="mr_batch_ds")
    df_raw = ds_opts[chosen]

    if df_raw is None or len(df_raw) == 0:
        st.warning("Selected dataset is empty.")
        return

    with st.spinner("Computing FRS, SCORE2-AP, Globorisk on dataset…"):
        df = _compute_batch(df_raw)

    _section_header("📈", "Summary Statistics")

    risk_cols = {
        "WHO Non-Lab": "risk_nonlab",
        "FRS Non-Lab": "risk_frs_nonlab",
        "SCORE2-AP": "risk_score2_ap",
        "Globorisk": "risk_globorisk",
    }

    if "risk_lab" in df.columns and df["risk_lab"].notna().sum() > 0:
        risk_cols["WHO Lab"] = "risk_lab"
    if "risk_frs_lab" in df.columns and df["risk_frs_lab"].notna().sum() > 0:
        risk_cols["FRS Lab"] = "risk_frs_lab"

    kpi_cols = st.columns(len(risk_cols))
    for i, (label, col) in enumerate(risk_cols.items()):
        with kpi_cols[i]:
            if col in df.columns:
                valid = df[col].dropna()
                mean_val = valid.mean()
                n_valid = len(valid)
                color = MODEL_COLORS.get(label, "#333")
                st.markdown(
                    _card(label, f"{mean_val:.1f}%", color,
                          f"n={n_valid:,} · median={valid.median():.1f}%"),
                    unsafe_allow_html=True)
            else:
                st.markdown(_card(label, "N/A", "#6c757d", "Column missing"),
                            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _section_header("📊", "Risk Distribution Comparison")

    plot_data = []
    for label, col in risk_cols.items():
        if col in df.columns:
            series = df[col].dropna()
            for v in series:
                plot_data.append({"Model": label, "10-Year Risk (%)": v})

    if plot_data:
        df_plot = pd.DataFrame(plot_data)

        chart_type = st.radio("Chart type", ["Violin", "Box", "Histogram"],
                              horizontal=True, key="mr_dist_type")

        if chart_type == "Violin":
            fig = px.violin(df_plot, x="Model", y="10-Year Risk (%)",
                            color="Model", color_discrete_map=MODEL_COLORS,
                            box=True, points="outliers")
        elif chart_type == "Box":
            fig = px.box(df_plot, x="Model", y="10-Year Risk (%)",
                         color="Model", color_discrete_map=MODEL_COLORS,
                         notched=True)
        else:
            fig = px.histogram(df_plot, x="10-Year Risk (%)", color="Model",
                               color_discrete_map=MODEL_COLORS,
                               barmode="overlay", opacity=0.6, nbins=30)

        fig.update_layout(height=420, template="plotly_white",
                          font=dict(family="Inter, sans-serif"),
                          legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Descriptive Statistics Table", expanded=False):
        desc_rows = []
        for label, col in risk_cols.items():
            if col in df.columns:
                s = df[col].dropna()
                desc_rows.append({
                    "Model": label,
                    "N": len(s),
                    "Mean": round(s.mean(), 2),
                    "SD": round(s.std(), 2),
                    "Median": round(s.median(), 2),
                    "IQR": f"{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}",
                    "Min": round(s.min(), 1),
                    "Max": round(s.max(), 1),
                    "≥10%": f"{(s >= 10).sum()} ({(s >= 10).mean()*100:.1f}%)",
                    "≥20%": f"{(s >= 20).sum()} ({(s >= 20).mean()*100:.1f}%)",
                })
        if desc_rows:
            st.dataframe(pd.DataFrame(desc_rows).set_index("Model"), use_container_width=True)

    _section_header("🏷️", "Risk Category Distribution")
    cat_cols = {
        "WHO Non-Lab": "risk_nonlab_cat",
        "FRS Non-Lab": "risk_frs_cat",
        "SCORE2-AP": "risk_score2_cat",
        "Globorisk": "risk_globorisk_cat",
    }

    cat_data = []
    for label, col in cat_cols.items():
        if col in df.columns:
            counts = df[col].value_counts(dropna=True)
            total = counts.sum()
            for cat, cnt in counts.items():
                cat_data.append({
                    "Model": label,
                    "Risk Category": str(cat),
                    "Count": cnt,
                    "Percentage": round(cnt / total * 100, 1),
                })

    if cat_data:
        df_cat = pd.DataFrame(cat_data)
        fig = px.bar(df_cat, x="Model", y="Percentage", color="Risk Category",
                     color_discrete_map=RISK_CAT_COLORS,
                     barmode="stack",
                     text="Percentage",
                     category_orders={"Risk Category": RISK_LABELS})
        fig.update_traces(texttemplate='%{text:.0f}%', textposition='inside')
        fig.update_layout(height=450, template="plotly_white",
                          yaxis_title="Proportion (%)",
                          font=dict(family="Inter, sans-serif"),
                          legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)


def _render_model_comparison(datasets: dict):
    """Render model comparison."""
    _section_header("⚖️", "Model Comparison & Discordance Analysis",
                    "Head-to-head comparison of multi-model risk estimates against WHO/ISH reference.")

    ds_opts = {k: v for k, v in datasets.items() if v is not None}
    if not ds_opts:
        st.warning("No datasets available.")
        return

    chosen = st.selectbox("Dataset", list(ds_opts.keys()), key="mr_cmp_ds")
    df_raw = ds_opts[chosen]
    if df_raw is None or len(df_raw) == 0:
        st.warning("Empty dataset.")
        return

    df = _compute_batch(df_raw)

    _section_header("🔬", "Pairwise Risk Scatter (vs. WHO Non-Lab)")

    compare_models = {
        "FRS Non-Lab": "risk_frs_nonlab",
        "SCORE2-AP": "risk_score2_ap",
        "Globorisk": "risk_globorisk",
    }

    cols_scatter = st.columns(3)
    for i, (label, col) in enumerate(compare_models.items()):
        with cols_scatter[i]:
            if col in df.columns and "risk_nonlab" in df.columns:
                valid = df[["risk_nonlab", col]].dropna()
                if len(valid) > 5000:
                    valid = valid.sample(5000, random_state=42)

                fig = px.scatter(valid, x="risk_nonlab", y=col,
                                 opacity=0.25,
                                 color_discrete_sequence=[MODEL_COLORS.get(label, "#333")])
                max_val = max(valid["risk_nonlab"].max(), valid[col].max())
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(color="red", dash="dash", width=1.5))
                corr = valid["risk_nonlab"].corr(valid[col])
                fig.update_layout(
                    title=f"{label} (ρ={corr:.3f})",
                    xaxis_title="WHO Non-Lab (%)",
                    yaxis_title=f"{label} (%)",
                    height=350,
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif", size=11),
                    margin=dict(t=50, b=40, l=50, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

    _section_header("📐", "Bland-Altman Analysis (Difference vs. Mean)")

    ba_cols = st.columns(3)
    for i, (label, col) in enumerate(compare_models.items()):
        with ba_cols[i]:
            if col in df.columns and "risk_nonlab" in df.columns:
                valid = df[["risk_nonlab", col]].dropna()
                if len(valid) > 5000:
                    valid = valid.sample(5000, random_state=42)

                mean_val = (valid["risk_nonlab"] + valid[col]) / 2
                diff_val = valid[col] - valid["risk_nonlab"]

                mean_diff = diff_val.mean()
                sd_diff = diff_val.std()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=mean_val, y=diff_val, mode='markers',
                    marker=dict(size=3, color=MODEL_COLORS.get(label, "#333"), opacity=0.3),
                    name="Difference"))
                fig.add_hline(y=mean_diff, line_dash="solid", line_color="black",
                              annotation_text=f"Mean: {mean_diff:.2f}")
                fig.add_hline(y=mean_diff + 1.96 * sd_diff, line_dash="dash",
                              line_color="red",
                              annotation_text=f"+1.96SD: {mean_diff + 1.96*sd_diff:.1f}")
                fig.add_hline(y=mean_diff - 1.96 * sd_diff, line_dash="dash",
                              line_color="red",
                              annotation_text=f"-1.96SD: {mean_diff - 1.96*sd_diff:.1f}")

                fig.update_layout(
                    title=f"{label} − WHO",
                    xaxis_title="Mean Risk (%)",
                    yaxis_title="Difference (%)",
                    height=350,
                    template="plotly_white",
                    font=dict(family="Inter, sans-serif", size=11),
                    margin=dict(t=50, b=40, l=50, r=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    _section_header("🔀", "Discordance at Clinical Thresholds")

    threshold = st.select_slider("High-risk threshold (%)",
                                 options=[5, 7.5, 10, 15, 20],
                                 value=20, key="mr_thresh")

    disc_models = {
        "FRS Non-Lab vs WHO": ("risk_frs_nonlab", "risk_nonlab"),
        "SCORE2-AP vs WHO": ("risk_score2_ap", "risk_nonlab"),
        "Globorisk vs WHO": ("risk_globorisk", "risk_nonlab"),
    }

    disc_cols = st.columns(3)
    for i, (pair_label, (col_a, col_b)) in enumerate(disc_models.items()):
        with disc_cols[i]:
            if col_a in df.columns and col_b in df.columns:
                result = compute_discordance_matrix(df, col_a, col_b, threshold)
                if result["n_valid"] > 0:
                    st.markdown(f"**{pair_label}**")

                    model_name = pair_label.split(" vs ")[0]
                    cm = pd.DataFrame(
                        [[result["agree_high"], result["a_low_b_high"]],
                         [result["a_high_b_low"], result["agree_low"]]],
                        index=[f"{model_name} ≥{threshold}%", f"{model_name} <{threshold}%"],
                        columns=[f"WHO ≥{threshold}%", f"WHO <{threshold}%"],
                    )
                    st.dataframe(cm, use_container_width=True)

                    col_k1, col_k2 = st.columns(2)
                    with col_k1:
                        st.metric("Discordance", f"{result['discordance_rate']}%")
                    with col_k2:
                        st.metric("Cohen's κ", f"{result['kappa']}")
                else:
                    st.info(f"No valid pairs for {pair_label}")

    with st.expander("📝 Interpretation Guide", expanded=False):
        st.markdown("""
        **Key takeaways from multi-model comparison:**

        | Finding | Explanation |
        |---------|------------|
        | **FRS > WHO** | Expected: Framingham was calibrated on a US cohort with higher baseline CVD rates. Prior studies consistently show FRS overestimates in South Asian PHC populations. |
        | **SCORE2-AP ≈ WHO** | SCORE2 Asia-Pacific (2024/2025) is recalibrated for the region (C-index 0.71). Agreement suggests regional calibration matters. |
        | **Globorisk > WHO in women ≥50** | Globorisk's postmenopausal adjustment flags additional high-risk women that WHO non-lab charts miss. |
        | **High discordance (>15%)** | Indicates models classify a meaningful proportion of patients differently—clinical decisions would change based on model choice. |
        | **κ > 0.6** | Substantial agreement; κ < 0.4 indicates poor agreement between models. |
        """)


def _render_subgroup_analysis(datasets: dict):
    """Render subgroup analysis."""
    _section_header("👥", "Subgroup-Stratified Risk Comparison",
                    "Examine how different models perform across sex, age, smoking, and diabetes strata.")

    ds_opts = {k: v for k, v in datasets.items() if v is not None}
    if not ds_opts:
        st.warning("No datasets available.")
        return

    chosen = st.selectbox("Dataset", list(ds_opts.keys()), key="mr_sub_ds")
    df_raw = ds_opts[chosen]
    if df_raw is None or len(df_raw) == 0:
        return

    df = _compute_batch(df_raw)

    risk_cols = {
        "WHO Non-Lab": "risk_nonlab",
        "FRS Non-Lab": "risk_frs_nonlab",
        "SCORE2-AP": "risk_score2_ap",
        "Globorisk": "risk_globorisk",
    }

    _section_header("⚤", "Mean Risk by Sex")
    sex_col = "gender_key" if "gender_key" in df.columns else "gender"
    if sex_col in df.columns:
        sex_data = []
        for sex_val in ["men", "women"]:
            subset = df[df[sex_col] == sex_val]
            for label, col in risk_cols.items():
                if col in subset.columns:
                    mean_r = subset[col].dropna().mean()
                    if not pd.isna(mean_r):
                        sex_data.append({
                            "Sex": sex_val.title(),
                            "Model": label,
                            "Mean Risk (%)": round(mean_r, 2),
                        })

        if sex_data:
            df_sex = pd.DataFrame(sex_data)
            fig = px.bar(df_sex, x="Model", y="Mean Risk (%)", color="Sex",
                         barmode="group",
                         color_discrete_map={"Men": "#457b9d", "Women": "#e63946"},
                         text_auto=".1f")
            fig.update_layout(height=400, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

    _section_header("📅", "Mean Risk by Age Band")
    if "age_band" in df.columns:
        age_data = []
        for ab in df["age_band"].cat.categories if hasattr(df["age_band"], "cat") else df["age_band"].dropna().unique():
            subset = df[df["age_band"] == ab]
            for label, col in risk_cols.items():
                if col in subset.columns:
                    mean_r = subset[col].dropna().mean()
                    if not pd.isna(mean_r):
                        age_data.append({
                            "Age Band": str(ab),
                            "Model": label,
                            "Mean Risk (%)": round(mean_r, 2),
                        })

        if age_data:
            df_age = pd.DataFrame(age_data)
            fig = px.line(df_age, x="Age Band", y="Mean Risk (%)", color="Model",
                          color_discrete_map=MODEL_COLORS,
                          markers=True)
            fig.update_layout(height=400, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

    _section_header("🚬", "Mean Risk by Smoking Status")
    smoke_col = "smoker_key" if "smoker_key" in df.columns else "smoker"
    if smoke_col in df.columns:
        smoke_data = []
        for sm_val in ["yes", "no"]:
            subset = df[df[smoke_col] == sm_val]
            for label, col in risk_cols.items():
                if col in subset.columns:
                    mean_r = subset[col].dropna().mean()
                    if not pd.isna(mean_r):
                        smoke_data.append({
                            "Smoker": sm_val.title(),
                            "Model": label,
                            "Mean Risk (%)": round(mean_r, 2),
                        })

        if smoke_data:
            df_smoke = pd.DataFrame(smoke_data)
            fig = px.bar(df_smoke, x="Model", y="Mean Risk (%)", color="Smoker",
                         barmode="group",
                         color_discrete_map={"Yes": "#e76f51", "No": "#2a9d8f"},
                         text_auto=".1f")
            fig.update_layout(height=400, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

    _section_header("🩸", "Mean Risk by Diabetes Status")
    diab_col = "has_diabetes"
    if diab_col in df.columns:
        diab_data = []
        for dv in [True, False]:
            subset = df[df[diab_col] == dv]
            label_d = "Diabetic" if dv else "Non-Diabetic"
            for label, col in risk_cols.items():
                if col in subset.columns:
                    mean_r = subset[col].dropna().mean()
                    if not pd.isna(mean_r):
                        diab_data.append({
                            "Diabetes": label_d,
                            "Model": label,
                            "Mean Risk (%)": round(mean_r, 2),
                        })

        if diab_data:
            df_diab = pd.DataFrame(diab_data)
            fig = px.bar(df_diab, x="Model", y="Mean Risk (%)", color="Diabetes",
                         barmode="group",
                         color_discrete_map={"Diabetic": "#d62828", "Non-Diabetic": "#2a9d8f"},
                         text_auto=".1f")
            fig.update_layout(height=400, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

    _section_header("📊", "High-Risk Reclassification Across Models (≥20% threshold)")

    if all(c in df.columns for c in ["risk_nonlab", "risk_frs_nonlab", "risk_score2_ap", "risk_globorisk"]):
        reclass_data = []
        for label, col in [("FRS Non-Lab", "risk_frs_nonlab"),
                           ("SCORE2-AP", "risk_score2_ap"),
                           ("Globorisk", "risk_globorisk")]:
            valid = df[["risk_nonlab", col]].dropna()
            who_high = (valid["risk_nonlab"] >= 20).sum()
            model_high = (valid[col] >= 20).sum()
            both_high = ((valid["risk_nonlab"] >= 20) & (valid[col] >= 20)).sum()
            only_model = ((valid["risk_nonlab"] < 20) & (valid[col] >= 20)).sum()
            only_who = ((valid["risk_nonlab"] >= 20) & (valid[col] < 20)).sum()

            reclass_data.append({
                "Model": label,
                "N valid": len(valid),
                "WHO High-Risk": who_high,
                "Model High-Risk": model_high,
                "Both High": both_high,
                "Only Model Flags": only_model,
                "Only WHO Flags": only_who,
                "Net Reclassification": model_high - who_high,
            })

        st.dataframe(pd.DataFrame(reclass_data).set_index("Model"), use_container_width=True)


def _render_globorisk_focus(datasets: dict):
    """Render globorisk focus."""
    _section_header("🌍", "Globorisk — Country-Specific Analysis",
                    "Bangladesh-calibrated model with postmenopausal adjustment. "
                    "Prior PHC-linked studies show Globorisk flags more high-risk among postmenopausal women.")

    ds_opts = {k: v for k, v in datasets.items() if v is not None}
    if not ds_opts:
        st.warning("No datasets available.")
        return

    chosen = st.selectbox("Dataset", list(ds_opts.keys()), key="mr_glob_ds")
    df_raw = ds_opts[chosen]
    if df_raw is None or len(df_raw) == 0:
        return

    df = _compute_batch(df_raw)

    sex_col = "gender_key" if "gender_key" in df.columns else "gender"

    _section_header("👩‍🦳", "Postmenopausal Women (age ≥ 50) vs. Younger Women")

    women = df[df[sex_col] == "women"].copy()
    if len(women) == 0:
        st.info("No female participants in selected dataset.")
        return

    women["menopause_group"] = women["age"].apply(
        lambda x: "Postmenopausal (≥50)" if x >= 50 else "Premenopausal (<50)")

    risk_cols_cmp = {
        "WHO Non-Lab": "risk_nonlab",
        "FRS Non-Lab": "risk_frs_nonlab",
        "SCORE2-AP": "risk_score2_ap",
        "Globorisk": "risk_globorisk",
    }

    meno_data = []
    for group in ["Premenopausal (<50)", "Postmenopausal (≥50)"]:
        subset = women[women["menopause_group"] == group]
        for label, col in risk_cols_cmp.items():
            if col in subset.columns:
                mean_r = subset[col].dropna().mean()
                high_pct = (subset[col].dropna() >= 20).mean() * 100
                if not pd.isna(mean_r):
                    meno_data.append({
                        "Group": group,
                        "Model": label,
                        "Mean Risk (%)": round(mean_r, 2),
                        "% High Risk (≥20%)": round(high_pct, 1),
                        "N": len(subset[col].dropna()),
                    })

    if meno_data:
        df_meno = pd.DataFrame(meno_data)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df_meno, x="Model", y="Mean Risk (%)", color="Group",
                         barmode="group",
                         color_discrete_map={
                             "Premenopausal (<50)": "#2a9d8f",
                             "Postmenopausal (≥50)": "#e63946",
                         },
                         text_auto=".1f",
                         title="Mean Risk by Menopausal Status")
            fig.update_layout(height=420, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(df_meno, x="Model", y="% High Risk (≥20%)", color="Group",
                         barmode="group",
                         color_discrete_map={
                             "Premenopausal (<50)": "#2a9d8f",
                             "Postmenopausal (≥50)": "#e63946",
                         },
                         text_auto=".1f",
                         title="% Classified High-Risk (≥20%)")
            fig.update_layout(height=420, template="plotly_white",
                              font=dict(family="Inter, sans-serif"),
                              legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Detailed Table", expanded=False):
            st.dataframe(df_meno.set_index(["Group", "Model"]), use_container_width=True)

    _section_header("🔍", "Globorisk vs. WHO (Women Only)")

    if "risk_globorisk" in women.columns and "risk_nonlab" in women.columns:
        valid_w = women[["risk_nonlab", "risk_globorisk", "age", "menopause_group"]].dropna()
        if len(valid_w) > 3000:
            valid_w = valid_w.sample(3000, random_state=42)

        fig = px.scatter(valid_w, x="risk_nonlab", y="risk_globorisk",
                         color="menopause_group",
                         color_discrete_map={
                             "Premenopausal (<50)": "#2a9d8f",
                             "Postmenopausal (≥50)": "#e63946",
                         },
                         opacity=0.4, hover_data=["age"])
        max_val = max(valid_w["risk_nonlab"].max(), valid_w["risk_globorisk"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                      line=dict(color="gray", dash="dash", width=1.5))
        fig.add_hline(y=20, line_dash="dot", line_color="#d62828", opacity=0.5,
                      annotation_text="Globorisk 20%")
        fig.add_vline(x=20, line_dash="dot", line_color="#d62828", opacity=0.5,
                      annotation_text="WHO 20%")

        fig.update_layout(
            title="Globorisk vs. WHO Non-Lab (Women)",
            xaxis_title="WHO Non-Lab Risk (%)",
            yaxis_title="Globorisk Risk (%)",
            height=500,
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

    _section_header("📊", "Discordance: Globorisk vs. WHO — Postmenopausal Women")

    postmeno = women[women["age"] >= 50]
    if "risk_globorisk" in postmeno.columns and "risk_nonlab" in postmeno.columns:
        for thresh in [10, 20]:
            result = compute_discordance_matrix(postmeno, "risk_globorisk", "risk_nonlab", thresh)
            if result["n_valid"] > 0:
                st.markdown(f"**At ≥{thresh}% threshold (n={result['n_valid']})**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Discordance", f"{result['discordance_rate']}%")
                with c2:
                    st.metric("κ", f"{result['kappa']}")
                with c3:
                    st.metric("Only Globorisk Flags", result['a_high_b_low'])
                with c4:
                    st.metric("Only WHO Flags", result['a_low_b_high'])

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f3460,#1a1a2e);
                padding:24px;border-radius:14px;margin-top:20px;
                border-left:4px solid #1982c4;'>
        <h4 style='color:#a8b2d1;margin:0 0 12px;'>🔑 Key Findings — Globorisk for Bangladesh PHC</h4>
        <ul style='color:#cbd5e1;margin:0;'>
            <li><strong>Country recalibration matters:</strong> Bangladesh hazard rates are ~18-22% higher
                than US-Framingham baseline, captured by Globorisk's γ-factor.</li>
            <li><strong>Postmenopausal uplift:</strong> Women ≥50 receive an additional risk increment,
                which prior PHC studies show identifies high-risk women missed by WHO charts.</li>
            <li><strong>Clinical implication:</strong> If using only WHO non-lab charts, some postmenopausal
                women who need statins/antihypertensives may be under-treated.</li>
        </ul>
    </div>""", unsafe_allow_html=True)


def _render_publication_tables(datasets: dict):
    """Comprehensive publication-ready tables:"""
    _section_header("📋", "Publication-Ready Tables",
                    "Comprehensive structured tables suitable for a manuscript or report.")

    ds_opts = {k: v for k, v in datasets.items() if v is not None}
    if not ds_opts:
        st.warning("No datasets available. Load data from the sidebar first.")
        return

    chosen = st.selectbox("Dataset", list(ds_opts.keys()), key="mr_pub_ds")
    df_raw = ds_opts[chosen]
    if df_raw is None or len(df_raw) == 0:
        st.warning("Selected dataset is empty.")
        return

    with st.spinner("Computing all risk scores…"):
        df = _compute_batch(df_raw)

    sex_col = "gender_key" if "gender_key" in df.columns else "gender"

    def _pct(n, total):
        """Pct."""
        return f"{n:,} ({n/total*100:.1f}%)"
    def _mean_sd(s):
        """Mean sd."""
        return f"{s.mean():.1f} ± {s.std():.1f}"
    def _median_iqr(s):
        """Median iqr."""
        return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"
    def _n_miss(s, total):
        """N miss."""
        return f"{s.isna().sum():,} ({s.isna().mean()*100:.1f}%)"

    def _kappa_pair(df_k, col_a, col_b):
        """Cohen's κ on categorised 5-band labels."""
        try:
            from sklearn.metrics import cohen_kappa_score
            valid = df_k[[col_a, col_b]].dropna()
            if len(valid) < 10:
                return np.nan, len(valid)
            return round(cohen_kappa_score(valid[col_a].astype(str),
                                           valid[col_b].astype(str),
                                           weights="linear"), 3), len(valid)
        except Exception:
            return np.nan, 0

    ALL_RISK = {
        "WHO Non-Lab":  ("risk_nonlab",   "risk_nonlab_cat"),
        "FRS Non-Lab":  ("risk_frs_nonlab","risk_frs_cat"),
        "SCORE2-AP":    ("risk_score2_ap", "risk_score2_cat"),
        "Globorisk":    ("risk_globorisk", "risk_globorisk_cat"),
    }
    if "risk_lab" in df.columns and df["risk_lab"].notna().any():
        ALL_RISK["WHO Lab"] = ("risk_lab", "risk_lab_cat")

    for label, (rcol, catcol) in ALL_RISK.items():
        if rcol in df.columns and catcol not in df.columns:
            df[catcol] = pd.cut(
                df[rcol], bins=RISK_BINS, labels=RISK_LABELS, right=False
            ).astype(pd.CategoricalDtype(RISK_LABELS, ordered=True))

    st.markdown("---")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 1 — Participant Characteristics</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Continuous variables: mean ± SD (median, IQR where skewed).
            Categorical variables: N (%). Stratified by sex.
            p-values: Mann-Whitney U (continuous) or &#967;² (categorical).
        </p>
    </div>""", unsafe_allow_html=True)

    men_df   = df[df[sex_col] == "men"]
    women_df = df[df[sex_col] == "women"]
    N        = len(df)
    N_m      = len(men_df)
    N_f      = len(women_df)

    def _row(label, overall_val, men_val, women_val, p_val=""):
        """Row."""
        return {"Characteristic": label,
                f"Overall (N={N:,})": overall_val,
                f"Men (n={N_m:,})": men_val,
                f"Women (n={N_f:,})": women_val,
                "p-value": p_val}

    from scipy import stats as _stats

    def _chi2_p(col, val_a, val_b):
        """Chi2 p."""
        try:
            ct = pd.crosstab(df[col].map({val_a: val_a, val_b: val_b}),
                             df[sex_col].map({"men": "men", "women": "women"})).values
            _, p, _, _ = _stats.chi2_contingency(ct)
            return "<0.001" if p < 0.001 else f"{p:.3f}"
        except Exception:
            return ""

    def _mw_p(col):
        """Mw p."""
        try:
            s_m = men_df[col].dropna()
            s_f = women_df[col].dropna()
            _, p = _stats.mannwhitneyu(s_m, s_f, alternative="two-sided")
            return "<0.001" if p < 0.001 else f"{p:.3f}"
        except Exception:
            return ""

    t1_rows = []

    t1_rows.append(_row("Age (years), mean ± SD",
                        _mean_sd(df["age"]), _mean_sd(men_df["age"]), _mean_sd(women_df["age"]),
                        _mw_p("age")))
    t1_rows.append(_row("Age (years), median (IQR)",
                        _median_iqr(df["age"]), _median_iqr(men_df["age"]), _median_iqr(women_df["age"])))

    if "age_band" in df.columns:
        for ab in ["40-44","45-49","50-54","55-59","60-64","65-69","70-74"]:
            ab_n = (df["age_band"] == ab).sum()
            ab_m = (men_df["age_band"] == ab).sum()
            ab_f = (women_df["age_band"] == ab).sum()
            t1_rows.append(_row(f"  Age band {ab}",
                                _pct(ab_n, N), _pct(ab_m, N_m), _pct(ab_f, N_f)))

    for col, label in [("height","Height (cm)"),("weight","Weight (kg)"),("bmi","BMI (kg/m²)")]:
        if col in df.columns:
            t1_rows.append(_row(f"{label}, mean ± SD",
                                _mean_sd(df[col]), _mean_sd(men_df[col]), _mean_sd(women_df[col]),
                                _mw_p(col)))

    if "bmi" in df.columns:
        bmi_bins   = [0, 18.5, 23, 25, 30, 200]
        bmi_labels = ["Underweight (<18.5)", "Normal (18.5–22.9)",
                      "Overweight-Asian (23–24.9)", "Overweight (25–29.9)", "Obese (≥30)"]
        df["_bmi_t1"]    = pd.cut(df["bmi"],      bins=bmi_bins, labels=bmi_labels, right=False)
        men_df["_bmi_t1"]   = pd.cut(men_df["bmi"],   bins=bmi_bins, labels=bmi_labels, right=False)
        women_df["_bmi_t1"] = pd.cut(women_df["bmi"],  bins=bmi_bins, labels=bmi_labels, right=False)
        for bl in bmi_labels:
            bn = (df["_bmi_t1"] == bl).sum()
            bm = (men_df["_bmi_t1"] == bl).sum()
            bf = (women_df["_bmi_t1"] == bl).sum()
            t1_rows.append(_row(f"  {bl}", _pct(bn, N), _pct(bm, N_m), _pct(bf, N_f)))

    if "whr" in df.columns:
        t1_rows.append(_row("Waist-Hip Ratio, mean ± SD",
                            _mean_sd(df["whr"].dropna()),
                            _mean_sd(men_df["whr"].dropna()),
                            _mean_sd(women_df["whr"].dropna()),
                            _mw_p("whr")))

    for col, label in [("sbp","Systolic BP (mmHg)"),("dbp","Diastolic BP (mmHg)")]:
        if col in df.columns:
            t1_rows.append(_row(f"{label}, mean ± SD",
                                _mean_sd(df[col].dropna()),
                                _mean_sd(men_df[col].dropna()),
                                _mean_sd(women_df[col].dropna()),
                                _mw_p(col)))

    if "bp_category" in df.columns:
        for bpc in df["bp_category"].dropna().unique():
            bn = (df["bp_category"] == bpc).sum()
            bm = (men_df["bp_category"] == bpc).sum()
            bf = (women_df["bp_category"] == bpc).sum()
            t1_rows.append(_row(f"  BP: {bpc}", _pct(bn, N), _pct(bm, N_m), _pct(bf, N_f)))

    if "bg_mgdl" in df.columns:
        t1_rows.append(_row("Fasting blood glucose (mg/dL), mean ± SD",
                            _mean_sd(df["bg_mgdl"]),
                            _mean_sd(men_df["bg_mgdl"]),
                            _mean_sd(women_df["bg_mgdl"]),
                            _mw_p("bg_mgdl")))

    if "cholesterol_mmolL" in df.columns:
        chol_n = df["cholesterol_mmolL"].notna().sum()
        t1_rows.append(_row(
            f"Total cholesterol (mmol/L), mean ± SD [n={chol_n:,}]",
            _mean_sd(df["cholesterol_mmolL"].dropna()),
            _mean_sd(men_df["cholesterol_mmolL"].dropna()),
            _mean_sd(women_df["cholesterol_mmolL"].dropna()),
            _mw_p("cholesterol_mmolL")))

    if "has_diabetes" in df.columns:
        dm_n = df["has_diabetes"].sum(); dm_m = men_df["has_diabetes"].sum(); dm_f = women_df["has_diabetes"].sum()
        t1_rows.append(_row("Diabetes mellitus", _pct(int(dm_n), N), _pct(int(dm_m), N_m), _pct(int(dm_f), N_f), _mw_p("has_diabetes")))

    smoke_col = "smoker_key" if "smoker_key" in df.columns else None
    if smoke_col and smoke_col in df.columns:
        sm_n = (df[smoke_col] == "yes").sum()
        sm_m = (men_df[smoke_col] == "yes").sum()
        sm_f = (women_df[smoke_col] == "yes").sum()
        t1_rows.append(_row("Current smoker", _pct(int(sm_n), N), _pct(int(sm_m), N_m), _pct(int(sm_f), N_f)))

    if "arrhythmia" in df.columns:
        ar_n = (df["arrhythmia"] == "Abnormal").sum()
        ar_m = (men_df["arrhythmia"] == "Abnormal").sum()
        ar_f = (women_df["arrhythmia"] == "Abnormal").sum()
        t1_rows.append(_row("Arrhythmia (Abnormal)", _pct(int(ar_n), N), _pct(int(ar_m), N_m), _pct(int(ar_f), N_f)))

    t1_rows.append(_row("─── CVD Risk Scores ───", "", "", ""))
    for label, (rcol, _) in ALL_RISK.items():
        if rcol in df.columns:
            valid = df[rcol].dropna()
            vm    = men_df[rcol].dropna()   if rcol in men_df.columns   else pd.Series(dtype=float)
            vf    = women_df[rcol].dropna() if rcol in women_df.columns else pd.Series(dtype=float)
            t1_rows.append(_row(
                f"{label} (%), mean ± SD",
                _mean_sd(valid), _mean_sd(vm), _mean_sd(vf),
                _mw_p(rcol) if len(vm) > 5 and len(vf) > 5 else ""
            ))
            t1_rows.append(_row(
                f"  ≥20% high risk",
                _pct((valid >= 20).sum(), len(valid)),
                _pct((vm >= 20).sum(), len(vm)) if len(vm) > 0 else "—",
                _pct((vf >= 20).sum(), len(vf)) if len(vf) > 0 else "—"
            ))

    t1_rows.append(_row("─── Missing Data ───", "", "", ""))
    for col, label in [("sbp","SBP"),("bmi","BMI"),("whr","WHR"),("bg_mgdl","Blood Glucose"),("cholesterol_mmolL","Cholesterol")]:
        if col in df.columns:
            t1_rows.append(_row(f"Missing: {label}",
                                _n_miss(df[col], N),
                                _n_miss(men_df[col] if col in men_df.columns else pd.Series(), N_m),
                                _n_miss(women_df[col] if col in women_df.columns else pd.Series(), N_f)))

    def _render_t1_html(rows, N, N_m, N_f):
        """Render t1 html."""
        SECTION_STYLE = ("background:#0f3460;color:white;font-weight:700;"
                         "font-size:.82em;letter-spacing:.5px;text-transform:uppercase;"
                         "padding:7px 12px;")
        INDENT_STYLE  = "color:#555;font-size:.82em;padding-left:28px;"

        def _p_badge(p):
            """P badge."""
            if not p: return ""
            if p == "<0.001":
                return "<span style='background:#c0392b;color:white;padding:1px 6px;border-radius:8px;font-size:.75em;font-weight:700;'>&lt;0.001</span>"
            try:
                v = float(p)
                if v < 0.05:
                    return f"<span style='background:#e74c3c;color:white;padding:1px 6px;border-radius:8px;font-size:.75em;font-weight:700;'>{p}</span>"
                return f"<span style='color:#888;font-size:.82em;'>{p}</span>"
            except:
                return f"<span style='font-size:.82em;'>{p}</span>"

        html = [
            "<style>",
            ".t1-wrap{overflow-x:auto;margin:12px 0 24px;}",
            ".t1{border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.83em;width:100%;}",
            ".t1 th,.t1 td{padding:7px 12px;border:1px solid #e0e0e0;vertical-align:middle;}",
            ".t1 .char-col{text-align:left;min-width:260px;background:#fafafa;}",
            ".t1 .val-col{text-align:center;}",
            ".t1 .p-col{text-align:center;min-width:80px;}",
            ".t1 thead th{background:#1a1a2e;color:white;font-weight:600;font-size:.85em;text-align:center;}",
            ".t1 thead th.char-head{text-align:left;}",
            ".t1 .alt-row td,.t1 .alt-row th{background:#f8f9fd;}",
            "</style>",
            "<div class='t1-wrap'><table class='t1'>",
            "<thead><tr>",
            "  <th class='char-head'>Characteristic</th>",
            f" <th>Overall<br/><span style='font-weight:400;opacity:.8;font-size:.85em;'>N={N:,}</span></th>",
            f" <th style='background:#457b9d;'>Males<br/><span style='font-weight:400;opacity:.8;font-size:.85em;'>n={N_m:,}</span></th>",
            f" <th style='background:#e63946;'>Females<br/><span style='font-weight:400;opacity:.8;font-size:.85em;'>n={N_f:,}</span></th>",
            "  <th class='p-col'>p-value</th>",
            "</tr></thead><tbody>",
        ]

        alt = False
        for i, row in enumerate(rows):
            char = row["Characteristic"]
            ov   = row.get(f"Overall (N={N:,})", row.get(list(row.keys())[1], ""))
            mn   = row.get(f"Men (n={N_m:,})",   row.get(list(row.keys())[2], ""))
            wm   = row.get(f"Women (n={N_f:,})", row.get(list(row.keys())[3], ""))
            pv   = row.get("p-value", "")

            if "───" in char:
                lbl = char.replace("─", "").strip()
                html.append(f"<tr><td colspan='5' style='{SECTION_STYLE}'>{lbl}</td></tr>")
                alt = False
                continue

            is_indent = char.startswith("  ")
            alt = not alt
            row_cls = "alt-row" if alt else ""

            char_html = (f"<span style='{INDENT_STYLE}'>{char.strip()}</span>"
                         if is_indent else f"<b>{char}</b>")

            html.append(f"<tr class='{row_cls}'>")
            html.append(f"<td class='char-col'>{char_html}</td>")
            html.append(f"<td class='val-col'>{ov}</td>")
            html.append(f"<td class='val-col'>{mn}</td>")
            html.append(f"<td class='val-col'>{wm}</td>")
            html.append(f"<td class='p-col'>{_p_badge(pv)}</td>")
            html.append("</tr>")

        html.append("</tbody></table></div>")
        return "".join(html)

    t1_df = pd.DataFrame(t1_rows)
    html1 = _render_t1_html(t1_rows, N, N_m, N_f)
    st.markdown(html1, unsafe_allow_html=True)
    st.download_button("📥 Download Table 1 (CSV)", t1_df.to_csv(index=False),
                       file_name="table1_participant_characteristics.csv", mime="text/csv")

    st.markdown("---")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 2 — Lab vs Non-Lab Risk Agreement by Sex</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Agreement between WHO Lab and WHO Non-Lab risk scores across 5 risk bands.
            <b style='color:#7fc8f8;'>Diagonal (green)</b> = exact agreement &nbsp;|
            <b style='color:#f4857a;'>Off-diagonal (red)</b> = reclassification. N (row %).
        </p>
    </div>""", unsafe_allow_html=True)

    def _lab_nonlab_agreement_table(sub_df, sex_label):
        """Build agreement cross-tab between WHO Lab and WHO Non-Lab categories."""
        if "risk_lab_cat" not in sub_df.columns or "risk_nonlab_cat" not in sub_df.columns:
            return None
        valid = sub_df[["risk_lab_cat","risk_nonlab_cat"]].dropna()
        if len(valid) < 5:
            return None
        ct = pd.crosstab(
            valid["risk_lab_cat"].astype(str),
            valid["risk_nonlab_cat"].astype(str),
            margins=True, margins_name="Total"
        )
        cats_p = [c for c in RISK_LABELS if c in ct.index]
        ct = ct.reindex(index=cats_p + ["Total"], columns=cats_p + ["Total"], fill_value=0)
        return ct

    def _render_t2_html(ct, sex_label, n_valid, agree_pct, kappa_val, disc_val):
        """Render a styled agreement matrix with summary stats below."""
        band_colors = {
            "<5%":         ("#d4edda", "#155724"),
            "5% to <10%":  ("#fff3cd", "#856404"),
            "10% to <20%": ("#fdebd0", "#7e5109"),
            "20% to <30%": ("#fadbd8", "#922b21"),
            "≥30%":        ("#f9ebea", "#641e16"),
            "Total":       ("#eaecee", "#212529"),
        }
        sex_color = "#457b9d" if sex_label == "Males" else "#e63946"
        cats = [c for c in ct.index if c != "Total"]

        def _cell_bg(r, c, n, rt):
            """Cell bg."""
            if r == "Total" or c == "Total":
                return "background:#f0f0f0;font-weight:700;"
            pct = n / rt * 100 if rt > 0 else 0
            if r == c:
                alpha = min(0.95, max(0.08, pct / 100))
                return f"background:rgba(39,174,96,{alpha:.2f});color:{'#0d4a26' if pct>30 else '#333'};"
            if pct == 0:
                return "background:#fafafa;color:#ccc;"
            alpha = min(0.85, 0.08 + pct / 60)
            return f"background:rgba(231,76,60,{alpha:.2f});color:{'white' if alpha>0.5 else '#333'};"

        def _kappa_badge(k):
            """Kappa badge."""
            if pd.isna(k): return "—"
            if k > 0.80: bg, lbl = "#1a5276", "Almost perfect"
            elif k > 0.60: bg, lbl = "#1e8449", "Substantial"
            elif k > 0.40: bg, lbl = "#b7950b", "Moderate"
            elif k > 0.20: bg, lbl = "#7d6608", "Fair"
            else:          bg, lbl = "#922b21", "Slight"
            return (f"<b style='font-size:1.1em;'>{k:.3f}</b> "
                    f"<span style='background:{bg};color:white;padding:1px 7px;"
                    f"border-radius:10px;font-size:.78em;font-weight:700;'>{lbl}</span>")

        html = [
            "<style>.t2{{border-collapse:collapse;font-family:'Inter',sans-serif;"
            "font-size:.83em;width:100%;}}",
            ".t2 th,.t2 td{{padding:7px 11px;border:1px solid #dee2e6;text-align:center;"
            "vertical-align:middle;}}",
            ".t2 .r-hdr{{text-align:left;font-weight:600;background:#f8f9fa;white-space:nowrap;}}",
            ".t2 .total-row td,.t2 .total-row th{{background:#eaecee!important;font-weight:700;}}",
            ".t2-stat{{display:flex;gap:16px;margin:12px 0 0;flex-wrap:wrap;}}",
            ".t2-stat-box{{background:#f8f9fa;border:1px solid #dee2e6;border-radius:10px;"
            "padding:10px 18px;min-width:160px;text-align:center;}}",
            ".t2-stat-label{{font-size:.73em;color:#888;text-transform:uppercase;"
            "letter-spacing:.4px;margin-bottom:4px;}}",
            ".t2-stat-val{{font-size:1.15em;font-weight:700;color:#1a1a2e;}}",
            "</style>",
            f"<div style='margin-bottom:8px;'>"
            f"<span style='background:{sex_color};color:white;padding:3px 12px;"
            f"border-radius:12px;font-weight:700;font-size:.9em;'>{sex_label}</span></div>",
            "<table class='t2'>",
        ]

        html.append("<thead><tr>")
        html.append(f"<th class='r-hdr' style='background:#1a1a2e;color:white;'>"
                    f"WHO Lab ↓ / WHO Non-Lab →</th>")
        for band in cats + ["Total"]:
            bg, fg = band_colors.get(band, ("#eee","#333"))
            html.append(f"<th style='background:{bg};color:{fg};font-size:.78em;'>{band}</th>")
        html.append("</tr></thead><tbody>")

        for row_label in cats + ["Total"]:
            is_total = row_label == "Total"
            bg, fg = band_colors.get(row_label, ("#f8f9fa","#333"))
            cls = "total-row" if is_total else ""
            html.append(f"<tr class='{cls}'>")
            html.append(f"<td class='r-hdr' style='background:{bg};color:{fg};'>{row_label}</td>")
            rt = ct.at[row_label, "Total"] if row_label in ct.index else 0
            for col_label in cats + ["Total"]:
                n_val = ct.at[row_label, col_label] if (row_label in ct.index and col_label in ct.columns) else 0
                pct   = n_val / rt * 100 if rt > 0 and not is_total else None
                style = _cell_bg(row_label, col_label, n_val, rt) if not is_total else "background:#eaecee;font-weight:700;"
                cell  = (f"{n_val:,}<br/><span style='font-size:.72em;opacity:.75;'>({pct:.1f}%)</span>"
                         if pct is not None else f"<b>{n_val:,}</b>")
                html.append(f"<td style='{style}'>{cell}</td>")
            html.append("</tr>")

        html.append("</tbody></table>")

        disc_str = f"{disc_val}%" if disc_val not in ("—", None, "") else "—"
        html.append(
            "<div class='t2-stat'>"
            f"<div class='t2-stat-box'><div class='t2-stat-label'>Valid pairs</div>"
            f"<div class='t2-stat-val'>{n_valid:,}</div></div>"
            f"<div class='t2-stat-box'><div class='t2-stat-label'>Exact agreement</div>"
            f"<div class='t2-stat-val'>{agree_pct:.1f}%</div></div>"
            f"<div class='t2-stat-box'><div class='t2-stat-label'>Weighted \u03ba</div>"
            f"<div class='t2-stat-val'>{_kappa_badge(kappa_val)}</div></div>"
            f"<div class='t2-stat-box'><div class='t2-stat-label'>Discordance ≥20%</div>"
            f"<div class='t2-stat-val'>{disc_str}</div></div>"
            "</div>"
        )
        return "".join(html)

    tab2a, tab2b = st.tabs(["2A · Males", "2B · Females"])

    with tab2a:
        men_sub = df[df[sex_col] == "men"]
        ct2a = _lab_nonlab_agreement_table(men_sub, "Males")
        if ct2a is not None:
            valid_m   = men_sub[["risk_lab_cat","risk_nonlab_cat"]].dropna()
            agree_m   = (valid_m["risk_lab_cat"].astype(str) == valid_m["risk_nonlab_cat"].astype(str)).mean()*100
            kappa_m, n_m = _kappa_pair(men_sub, "risk_lab_cat", "risk_nonlab_cat")
            disc_m    = compute_discordance_matrix(men_sub, "risk_lab", "risk_nonlab", 20) if "risk_lab" in men_sub.columns else {}
            html2a = _render_t2_html(ct2a, "Males", n_m, agree_m, kappa_m,
                                     disc_m.get("discordance_rate", "—") if disc_m else "—")
            st.markdown(html2a, unsafe_allow_html=True)
            st.download_button("📥 Download Table 2A (CSV)",
                               ct2a.to_csv(), file_name="table2a_lab_nonlab_males.csv", mime="text/csv")
        else:
            st.info("WHO Lab data not available — load 'who_lab' or 'paired' dataset.")

    with tab2b:
        women_sub = df[df[sex_col] == "women"]
        ct2b = _lab_nonlab_agreement_table(women_sub, "Females")
        if ct2b is not None:
            valid_f   = women_sub[["risk_lab_cat","risk_nonlab_cat"]].dropna()
            agree_f   = (valid_f["risk_lab_cat"].astype(str) == valid_f["risk_nonlab_cat"].astype(str)).mean()*100
            kappa_f, n_f = _kappa_pair(women_sub, "risk_lab_cat", "risk_nonlab_cat")
            disc_f    = compute_discordance_matrix(women_sub, "risk_lab", "risk_nonlab", 20) if "risk_lab" in women_sub.columns else {}
            html2b = _render_t2_html(ct2b, "Females", n_f, agree_f, kappa_f,
                                     disc_f.get("discordance_rate", "—") if disc_f else "—")
            st.markdown(html2b, unsafe_allow_html=True)
            st.download_button("📥 Download Table 2B (CSV)",
                               ct2b.to_csv(), file_name="table2b_lab_nonlab_females.csv", mime="text/csv")
        else:
            st.info("WHO Lab data not available in selected dataset.")

    st.markdown("---")


    def _render_agreement_html(ct_dict, mod_a, mod_b, n_tots, agree_pcts, kappas):
        """Renders a side-by-side (Overall / Men / Women) styled agreement matrix."""
        band_colors = {
            "<5%":         ("#d4edda", "#155724"),
            "5% to <10%":  ("#fff3cd", "#856404"),
            "10% to <20%": ("#fdebd0", "#7e5109"),
            "20% to <30%": ("#fadbd8", "#922b21"),
            "≥30%":        ("#f9ebea", "#641e16"),
            "Total":       ("#eaecee", "#212529"),
        }

        def _cell_bg(row_label, col_label, n, row_total):
            """Colour diagonal green, off-diagonal warm proportional to N."""
            if row_label == "Total" or col_label == "Total":
                return "background:#f0f0f0;font-weight:600;"
            pct = n / row_total * 100 if row_total > 0 else 0
            if row_label == col_label:
                intensity = min(255, int(180 + pct * 0.75))
                return f"background:rgba(39,174,96,{pct/100:.2f});color:{'#0d4a26' if pct>30 else '#333'};"
            else:
                if pct == 0:
                    return "background:#fafafa;color:#bbb;"
                alpha = min(0.85, 0.08 + pct / 60)
                return f"background:rgba(231,76,60,{alpha:.2f});color:{'white' if alpha>0.5 else '#333'};"

        groups = list(ct_dict.keys())
        n_groups = len(groups)

        html = [
            "<style>",
            ".t3-wrap{overflow-x:auto;margin:12px 0 24px;}",
            ".t3{border-collapse:collapse;font-size:0.82em;font-family:'Inter',sans-serif;width:100%;}",
            ".t3 th,.t3 td{padding:7px 10px;border:1px solid #dee2e6;text-align:center;vertical-align:middle;}",
            ".t3 .row-hdr{text-align:left;font-weight:600;background:#f8f9fa;white-space:nowrap;min-width:130px;}",
            ".t3 .grp-hdr{font-weight:700;font-size:1em;background:#1a1a2e;color:white;letter-spacing:.5px;}",
            ".t3 .col-hdr{font-weight:600;font-size:0.8em;}",
            ".t3 .stat-row td{background:#f8f9fa!important;font-style:italic;font-size:0.78em;color:#555;}",
            ".t3 .total-row td,.t3 .total-row th{background:#eaecee!important;font-weight:700;}",
            ".kbadge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.78em;font-weight:700;}",
            "</style>",
            "<div class='t3-wrap'><table class='t3'>",
        ]

        sample_ct = list(ct_dict.values())[0]
        cats = [c for c in sample_ct.index if c != "Total"]
        n_cats = len(cats)

        html.append("<thead>")
        html.append(f"<tr><th rowspan='3' class='row-hdr' style='background:#1a1a2e;color:white;'>{mod_a}<br/><span style='opacity:.6;font-size:.78em;font-weight:400;'>\u2193 rows</span></th>")
        for g_label in groups:
            bg = "#2a9d8f" if g_label=="Overall" else ("#457b9d" if g_label=="Men" else "#e63946")
            html.append(f"<th colspan='{n_cats+1}' class='grp-hdr' style='background:{bg};'>{g_label}</th>")
        html.append("</tr>")

        html.append("<tr>")
        for _ in groups:
            html.append(f"<th colspan='{n_cats+1}' style='background:#f1f3f5;font-size:.78em;color:#555;font-style:italic;'>{mod_b} →</th>")
        html.append("</tr>")

        html.append("<tr>")
        for _ in groups:
            for band in cats + ["Total"]:
                bg, fg = band_colors.get(band, ("#eee", "#333"))
                html.append(f"<th class='col-hdr' style='background:{bg};color:{fg};font-size:.75em;'>{band}</th>")
        html.append("</tr></thead><tbody>")

        for i, row_label in enumerate(cats + ["Total"]):
            is_total = row_label == "Total"
            row_class = "total-row" if is_total else ""
            bg, fg = band_colors.get(row_label, ("#f8f9fa", "#333"))
            html.append(f"<tr class='{row_class}'>")
            html.append(f"<td class='row-hdr' style='background:{bg};color:{fg};'>{row_label}</td>")
            for g_label, sub_ct in ct_dict.items():
                row_total = sub_ct.at[row_label, "Total"] if row_label in sub_ct.index else 0
                for col_label in cats + ["Total"]:
                    n_val = sub_ct.at[row_label, col_label] if (row_label in sub_ct.index and col_label in sub_ct.columns) else 0
                    pct   = n_val / row_total * 100 if row_total > 0 and not is_total else None
                    style = _cell_bg(row_label, col_label, n_val, row_total) if not is_total else "background:#eaecee;font-weight:700;"
                    cell_txt = f"{n_val:,}<br/><span style='font-size:.72em;opacity:.75;'>({pct:.1f}%)</span>" if pct is not None else f"<b>{n_val:,}</b>"
                    html.append(f"<td style='{style}'>{cell_txt}</td>")
            html.append("</tr>")

        def _kappa_badge(k):
            """Kappa badge."""
            if pd.isna(k):
                return "<span>—</span>"
            if k > 0.80:   col, lbl = "#155724", "Almost perfect"
            elif k > 0.60: col, lbl = "#1a5276", "Substantial"
            elif k > 0.40: col, lbl = "#784212", "Moderate"
            elif k > 0.20: col, lbl = "#6c3483", "Fair"
            else:          col, lbl = "#922b21", "Slight"
            return f"<span class='kbadge' style='background:{col};color:white;'>κ={k:.3f} — {lbl}</span>"

        stat_labels = [
            ("n (valid)",     [f"{n_tots[g]:,}" for g in groups]),
            ("Exact agree.",  [f"{agree_pcts[g]:.1f}%"  for g in groups]),
            ("Weighted κ",    [_kappa_badge(kappas[g])   for g in groups]),
        ]
        for lbl, vals in stat_labels:
            html.append("<tr class='stat-row'>")
            html.append(f"<td class='row-hdr' style='font-style:italic;'>{lbl}</td>")
            for g_idx, g_label in enumerate(groups):
                html.append(f"<td colspan='{n_cats+1}' style='text-align:center;'>{vals[g_idx]}</td>")
            html.append("</tr>")

        html.append("</tbody></table></div>")
        return "".join(html)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 3 — Multi-Model Category Agreement Matrices</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            5 × 5 cross-tabulation of 10-year CVD risk categories for each model pair.
            <b style='color:#7fc8f8;'>Diagonal (green)</b> = exact agreement.
            <b style='color:#f4857a;'>Off-diagonal (red)</b> = reclassification intensity.
            Row % within each cell. Stratified by sex.
        </p>
    </div>""", unsafe_allow_html=True)

    model_names_avail = [m for m, (rc, cc) in ALL_RISK.items() if cc in df.columns]
    pairs = [(a, b) for i, a in enumerate(model_names_avail)
             for b in model_names_avail[i+1:]]
    pair_labels = [f"{a} vs {b}" for a, b in pairs]

    if pair_labels:
        col_sel, col_info = st.columns([3, 1])
        with col_sel:
            sel_pair = st.selectbox("Select model pair", pair_labels, key="pub_pair_sel")
        with col_info:
            st.markdown("<div style='margin-top:28px;font-size:.82em;color:#6c757d;'>"
                        "🟢 Diagonal = agreement &nbsp;|&nbsp; 🔴 Off-diagonal = reclassification</div>",
                        unsafe_allow_html=True)

        idx      = pair_labels.index(sel_pair)
        mod_a, mod_b = pairs[idx]
        col_a_cat = ALL_RISK[mod_a][1]
        col_b_cat = ALL_RISK[mod_b][1]

        ct_dict     = {}
        n_tots      = {}
        agree_pcts  = {}
        kappas      = {}

        for group_label, sub_df in [("Overall", df),
                                     ("Men",     df[df[sex_col]=="men"]),
                                     ("Women",   df[df[sex_col]=="women"])]:
            valid = sub_df[[col_a_cat, col_b_cat]].dropna()
            if len(valid) < 5:
                continue
            ct = pd.crosstab(
                valid[col_a_cat].astype(str),
                valid[col_b_cat].astype(str),
                margins=True, margins_name="Total"
            )
            cats_p = [c for c in RISK_LABELS if c in ct.index]
            ct = ct.reindex(index=cats_p + ["Total"],
                            columns=cats_p + ["Total"], fill_value=0)
            n_tot     = len(valid)
            agree_n   = sum(ct.at[rl, rl] for rl in cats_p if rl in ct.index and rl in ct.columns)
            kappa, _  = _kappa_pair(valid.rename(columns={col_a_cat: mod_a, col_b_cat: mod_b}),
                                    mod_a, mod_b)
            ct_dict[group_label]    = ct
            n_tots[group_label]     = n_tot
            agree_pcts[group_label] = agree_n / n_tot * 100
            kappas[group_label]     = kappa

        if ct_dict:
            html3 = _render_agreement_html(ct_dict, mod_a, mod_b, n_tots, agree_pcts, kappas)
            st.markdown(html3, unsafe_allow_html=True)

            t3_rows = []
            for g, ct in ct_dict.items():
                for ri in ct.index:
                    for ci in ct.columns:
                        t3_rows.append({"Group": g, mod_a: ri, mod_b: ci, "N": ct.at[ri, ci]})
            st.download_button("📥 Download Table 3 (CSV)",
                               pd.DataFrame(t3_rows).to_csv(index=False),
                               file_name="table3_agreement_matrix.csv", mime="text/csv")

    st.markdown("---")


    def _render_kappa_html(kappa_data, groups):
        """kappa_data: list of dicts with keys:"""
        def _kappa_color(k):
            """Kappa color."""
            if pd.isna(k):   return ("#f0f0f0", "#999",  "—")
            if k > 0.80:     return ("#1a5276", "white", "Almost perfect")
            if k > 0.60:     return ("#1e8449", "white", "Substantial")
            if k > 0.40:     return ("#b7950b", "white", "Moderate")
            if k > 0.20:     return ("#7d6608", "white", "Fair")
            return             ("#922b21", "white", "Slight")

        def _disc_color(d):
            """Disc color."""
            if d == "—" or d is None: return "#f0f0f0"
            try:
                v = float(d)
                if v < 5:   return "#d5f5e3"
                if v < 10:  return "#fef9e7"
                if v < 20:  return "#fdebd0"
                return              "#fadbd8"
            except: return "#f0f0f0"

        pairs_uniq = list(dict.fromkeys(d["pair"] for d in kappa_data))
        grp_cols = groups
        metrics = ["N", "κ", "Strength", "Conc. ≥20%", "Disc. ≥20%"]

        lookup = {(d["pair"], d["group"]): d for d in kappa_data}

        MODEL_COL = {
            "WHO Non-Lab": "#2a9d8f", "WHO Lab": "#264653",
            "FRS Non-Lab": "#f4a261", "FRS Lab": "#e76f51",
            "SCORE2-AP":   "#6a4c93", "Globorisk": "#1982c4",
        }

        html = [
            "<style>",
            ".t4-wrap{overflow-x:auto;margin:12px 0 24px;}",
            ".t4{border-collapse:collapse;font-family:'Inter',sans-serif;font-size:0.82em;width:100%;}",
            ".t4 th,.t4 td{padding:8px 12px;border:1px solid #dee2e6;vertical-align:middle;}",
            ".t4 .pair-hdr{text-align:left;font-weight:700;min-width:220px;background:#f8f9fa;}",
            ".t4 .grp-span{text-align:center;font-weight:700;font-size:.85em;letter-spacing:.4px;color:white;}",
            ".t4 .met-hdr{text-align:center;font-size:.75em;font-weight:600;color:#555;background:#f1f3f5;}",
            ".t4 .n-cell{text-align:center;color:#555;font-size:.82em;}",
            ".t4 .k-cell{text-align:center;font-weight:700;font-size:.95em;}",
            ".t4 .s-cell{text-align:center;font-size:.75em;}",
            ".t4 .p-cell{text-align:center;font-size:.82em;}",
            ".pair-tag{display:inline-block;padding:2px 7px;border-radius:10px;",
            "  font-size:.72em;color:white;margin-right:4px;vertical-align:middle;}",
            "</style>",
            "<div class='t4-wrap'><table class='t4'>",
        ]

        html.append("<thead><tr>")
        html.append(f"<th rowspan='2' class='pair-hdr' style='background:#1a1a2e;color:white;'"
                    f">Model Pair</th>")
        grp_colors = {"Overall": "#2a9d8f", "Men": "#457b9d", "Women": "#e63946"}
        for g in grp_cols:
            html.append(f"<th colspan='{len(metrics)}' class='grp-span' "
                        f"style='background:{grp_colors.get(g,'#555')};'>{g}</th>")
        html.append("</tr>")

        html.append("<tr>")
        for _ in grp_cols:
            for m in metrics:
                html.append(f"<th class='met-hdr'>{m}</th>")
        html.append("</tr></thead><tbody>")

        for pair in pairs_uniq:
            a_lbl, b_lbl = pair.split(" vs ", 1)
            a_color = MODEL_COL.get(a_lbl, "#555")
            b_color = MODEL_COL.get(b_lbl, "#555")
            pair_cell = (f"<span class='pair-tag' style='background:{a_color};'>{a_lbl}</span>"
                         f"<span style='color:#999;font-size:.8em;'>vs</span> "
                         f"<span class='pair-tag' style='background:{b_color};'>{b_lbl}</span>")
            html.append(f"<tr><td class='pair-hdr'>{pair_cell}</td>")

            for g in grp_cols:
                row = lookup.get((pair, g))
                if row is None:
                    html.append(f"<td colspan='{len(metrics)}' class='n-cell' style='color:#bbb;'>—</td>")
                    continue
                k    = row.get("kappa")
                bg, fg, strength = _kappa_color(k)
                conc = row.get("concordance_20", "—")
                disc = row.get("discordance_20", "—")
                d_bg = _disc_color(disc)
                n_v  = row.get("n_valid", "—")

                html.append(f"<td class='n-cell'>{n_v:,}</td>")
                html.append(f"<td class='k-cell' style='background:{bg};color:{fg};'>"
                             f"{'—' if pd.isna(k) else f'{k:.3f}'}</td>")
                html.append(f"<td class='s-cell'><span style='padding:2px 6px;border-radius:8px;"
                             f"background:{bg};color:{fg};font-size:.75em;'>{strength}</span></td>")
                html.append(f"<td class='p-cell'>{conc}%</td>")
                html.append(f"<td class='p-cell' style='background:{d_bg};'>{disc}%</td>")
            html.append("</tr>")

        html.append("<tr><td colspan='100%' style='background:#f8f9fa;font-size:.75em;color:#555;"
                    "padding:8px 12px;text-align:left;border-top:2px solid #dee2e6;'>")
        for bg, lbl in [("#1a5276","Almost perfect κ>0.80"),("#1e8449","Substantial 0.61–0.80"),
                        ("#b7950b","Moderate 0.41–0.60"),("#7d6608","Fair 0.21–0.40"),("#922b21","Slight ≤0.20")]:
            html.append(f"<span style='display:inline-block;margin-right:14px;'>"
                        f"<span style='background:{bg};color:white;padding:1px 6px;border-radius:6px;"
                        f"font-size:.85em;font-weight:700;'>{lbl[:2]}</span> {lbl[3:]}</span>")
        html.append("Conc./Disc. = Concordance / Discordance at ≥20%.")
        html.append("</td></tr>")
        html.append("</tbody></table></div>")
        return "".join(html)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 4 — Pairwise Cohen&#39;s &#954; (Weighted, Linear) Summary</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Weighted (linear) &#954; for all model pairs × stratification groups.
            Cell colour encodes agreement strength. Concordance/discordance at
            <b style='color:#7fc8f8;'>&#8805;20% clinical threshold</b> also shown.
        </p>
    </div>""", unsafe_allow_html=True)

    kappa_data  = []
    groups_seen = ["Overall", "Men", "Women"]
    for group_label, sub_df in [("Overall", df),
                                  ("Men",     df[df[sex_col]=="men"]),
                                  ("Women",   df[df[sex_col]=="women"])]:
        for a_label, b_label in pairs:
            col_a_cat = ALL_RISK[a_label][1]
            col_b_cat = ALL_RISK[b_label][1]
            if col_a_cat not in sub_df.columns or col_b_cat not in sub_df.columns:
                continue
            kappa, n_valid = _kappa_pair(sub_df, col_a_cat, col_b_cat)
            col_a_r = ALL_RISK[a_label][0]
            col_b_r = ALL_RISK[b_label][0]
            disc_res = {}
            if col_a_r in sub_df.columns and col_b_r in sub_df.columns:
                disc_res = compute_discordance_matrix(sub_df, col_a_r, col_b_r, 20)

            if pd.isna(kappa):
                strength = "—"
            elif kappa > 0.80: strength = "Almost perfect"
            elif kappa > 0.60: strength = "Substantial"
            elif kappa > 0.40: strength = "Moderate"
            elif kappa > 0.20: strength = "Fair"
            else:              strength = "Slight"

            kappa_data.append({
                "pair":            f"{a_label} vs {b_label}",
                "group":           group_label,
                "n_valid":         n_valid,
                "kappa":           kappa,
                "strength":        strength,
                "concordance_20": disc_res.get("concordance_rate", "—"),
                "discordance_20": disc_res.get("discordance_rate", "—"),
            })

    if kappa_data:
        html4 = _render_kappa_html(kappa_data, groups_seen)
        st.markdown(html4, unsafe_allow_html=True)

        t4_csv = pd.DataFrame([{
            "Group": d["group"], "Model A": d["pair"].split(" vs ")[0],
            "Model B": d["pair"].split(" vs ")[1], "N": d["n_valid"],
            "Weighted κ": d["kappa"], "Strength": d["strength"],
            "Concordance ≥20% (%)": d["concordance_20"],
            "Discordance ≥20% (%)": d["discordance_20"],
        } for d in kappa_data])
        st.download_button("📥 Download Table 4 (CSV)", t4_csv.to_csv(index=False),
                           file_name="table4_pairwise_kappa.csv", mime="text/csv")

    st.markdown("---")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 5 — Reclassification Table vs WHO Non-Lab (All 5 Risk Bands)</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Rows = WHO Non-Lab band (reference). Columns = comparison model band.
            <b style='color:#7fc8f8;'>Diagonal (green)</b> = same category &nbsp;|
            <b style='color:#93c5fd;'>Upper triangle (blue)</b> = up-classified ↑ &nbsp;|
            <b style='color:#fca5a5;'>Lower triangle (red)</b> = down-classified ↓.
        </p>
    </div>""", unsafe_allow_html=True)

    def _render_t5_html(ct, cats_p, n_tot, agree, up_cls, dn_cls, group_label, cmp_model):
        """Render t5 html."""
        band_colors = {
            "<5%":         ("#d4edda", "#155724"),
            "5% to <10%":  ("#fff3cd", "#856404"),
            "10% to <20%": ("#fdebd0", "#7e5109"),
            "20% to <30%": ("#fadbd8", "#922b21"),
            "≥30%":        ("#f9ebea", "#641e16"),
            "Total":       ("#eaecee", "#212529"),
        }
        grp_color = {"Overall": "#2a9d8f", "Men": "#457b9d", "Women": "#e63946"}.get(group_label, "#555")
        cat_idx = {c: i for i, c in enumerate(cats_p)}

        def _cls_style(ri, ci, n, rt):
            """Cls style."""
            if ri == "Total" or ci == "Total":
                return "background:#eaecee;font-weight:700;"
            pct = n / rt * 100 if rt > 0 else 0
            r_i, c_i = cat_idx.get(ri, -1), cat_idx.get(ci, -1)
            if r_i == c_i:
                alpha = min(0.9, max(0.08, pct / 100))
                return f"background:rgba(39,174,96,{alpha:.2f});color:{'#0d4a26' if pct>30 else '#333'};"
            if pct == 0:
                return "background:#fafafa;color:#ccc;"
            alpha = min(0.8, 0.06 + pct / 70)
            if c_i > r_i:
                return f"background:rgba(37,99,235,{alpha:.2f});color:{'white' if alpha>0.45 else '#1e3a8a'};"
            else:
                return f"background:rgba(220,38,38,{alpha:.2f});color:{'white' if alpha>0.45 else '#7f1d1d'};"

        html = [
            f"""<div style='margin:12px 0 4px;display:flex;align-items:center;gap:10px;'>
            <span style='background:{grp_color};color:white;padding:3px 14px;border-radius:12px;
                         font-weight:700;font-size:.9em;'>{group_label}</span>
            <span style='font-size:.82em;color:#555;'>
                n={n_tot:,} &nbsp;|
                Same: <b>{agree}</b> ({agree/n_tot*100:.1f}%) &nbsp;|
                <span style='color:#1d4ed8;'>↑ Up: <b>{up_cls}</b> ({up_cls/n_tot*100:.1f}%)</span> &nbsp;|
                <span style='color:#dc2626;'>↓ Down: <b>{dn_cls}</b> ({dn_cls/n_tot*100:.1f}%)</span>
            </span></div>""",
            "<table style='border-collapse:collapse;font-family:Inter,sans-serif;"
            "font-size:.82em;width:100%;margin-bottom:16px;'>",
            "<thead><tr>",
            f"<th style='background:#1a1a2e;color:white;padding:7px 10px;border:1px solid #dee2e6;"
            f"text-align:left;white-space:nowrap;'>WHO Non-Lab ↓ / {cmp_model} →</th>",
        ]
        for band in cats_p + ["Total"]:
            bg, fg = band_colors.get(band, ("#eee","#333"))
            html.append(f"<th style='background:{bg};color:{fg};padding:7px 10px;border:1px solid #dee2e6;"
                        f"text-align:center;font-size:.78em;'>{band}</th>")
        html.append("</tr></thead><tbody>")

        for row_label in cats_p + ["Total"]:
            bg, fg = band_colors.get(row_label, ("#f8f9fa","#333"))
            html.append("<tr>")
            html.append(f"<td style='background:{bg};color:{fg};font-weight:600;padding:7px 10px;"
                        f"border:1px solid #dee2e6;white-space:nowrap;'>{row_label}</td>")
            rt = ct.at[row_label, "Total"] if row_label in ct.index else 0
            is_total = row_label == "Total"
            for col_label in cats_p + ["Total"]:
                n_val = ct.at[row_label, col_label] if (row_label in ct.index and col_label in ct.columns) else 0
                pct   = n_val / rt * 100 if rt > 0 and not is_total else None
                style = _cls_style(row_label, col_label, n_val, rt) if not is_total else "background:#eaecee;font-weight:700;"
                cell  = (f"{n_val:,}<br/><span style='font-size:.7em;opacity:.75;'>({pct:.1f}%)</span>"
                         if pct is not None else f"<b>{n_val:,}</b>")
                html.append(f"<td style='{style}padding:7px 10px;border:1px solid #dee2e6;"
                            f"text-align:center;'>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    ref_cat_col = ALL_RISK["WHO Non-Lab"][1]
    compare_models_t5 = {k: v for k, v in ALL_RISK.items() if k != "WHO Non-Lab" and v[1] in df.columns}

    col_sel5, col_leg5 = st.columns([3, 1])
    with col_sel5:
        sel_model_t5 = st.selectbox("Compare vs WHO Non-Lab:", list(compare_models_t5.keys()), key="pub_t5_model")
    with col_leg5:
        st.markdown("<div style='margin-top:28px;font-size:.81em;color:#6c757d;'>"
                    "🟢 Same &nbsp;🟦 Up-classified ↑ &nbsp;🔴 Down-classified ↓</div>",
                    unsafe_allow_html=True)
    cmp_cat_col = compare_models_t5[sel_model_t5][1]

    t5_all_rows = []
    for group_label, sub_df in [("Overall", df),
                                  ("Men",     df[df[sex_col]=="men"]),
                                  ("Women",   df[df[sex_col]=="women"])]:
        valid = sub_df[[ref_cat_col, cmp_cat_col]].dropna()
        if len(valid) < 5:
            continue
        ct = pd.crosstab(
            valid[ref_cat_col].astype(str),
            valid[cmp_cat_col].astype(str),
            margins=True, margins_name="Total"
        )
        cats_p = [c for c in RISK_LABELS if c in ct.columns]
        ct = ct.reindex(index=cats_p + ["Total"], columns=cats_p + ["Total"], fill_value=0)
        n_tot  = len(valid)
        agree  = sum(ct.at[rl, rl] for rl in cats_p if rl in ct.index and rl in ct.columns)
        up_cls = sum(ct.at[r, c]
                     for ri, r in enumerate(cats_p) for ci, c in enumerate(cats_p)
                     if ci > ri and r in ct.index and c in ct.columns)
        dn_cls = sum(ct.at[r, c]
                     for ri, r in enumerate(cats_p) for ci, c in enumerate(cats_p)
                     if ci < ri and r in ct.index and c in ct.columns)
        st.markdown(
            _render_t5_html(ct, cats_p, n_tot, agree, up_cls, dn_cls, group_label, sel_model_t5),
            unsafe_allow_html=True
        )
        for ri in ct.index:
            for ci in ct.columns:
                t5_all_rows.append({"Group": group_label, "WHO Non-Lab": ri,
                                    sel_model_t5: ci, "N": ct.at[ri, ci]})

    if t5_all_rows:
        st.download_button("📥 Download Table 5 (CSV)",
                           pd.DataFrame(t5_all_rows).to_csv(index=False),
                           file_name="table5_reclassification.csv", mime="text/csv")

    st.markdown("---")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 6 — Mean CVD Risk (%) by Sex and Age Band — All Models</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Mean ± SD 10-year CVD risk (%) with prevalence at ≥10% and ≥20% risk
            per sex &times; 5-year age band. Cell colour scales with mean risk level.
        </p>
    </div>""", unsafe_allow_html=True)

    MODEL_COLORS_6 = {
        "WHO Non-Lab": "#2a9d8f", "WHO Lab": "#264653",
        "FRS Non-Lab": "#f4a261", "FRS Lab": "#e76f51",
        "SCORE2-AP":   "#6a4c93", "Globorisk": "#1982c4",
    }

    if "age_band" in df.columns:
        age_bands_ordered = ["40-44","45-49","50-54","55-59","60-64","65-69","70-74"]
        risk_models = [(lbl, rc) for lbl, (rc, _) in ALL_RISK.items() if rc in df.columns]
        t6_flat = []

        def _risk_cell_style(mean_val):
            """Green → yellow → red gradient for mean risk."""
            try:
                v = float(str(mean_val).split("±")[0])
            except:
                return "background:#f8f9fa;color:#bbb;"
            if v < 5:    return "background:#d4edda;color:#155724;"
            if v < 10:   return "background:#fff3cd;color:#856404;"
            if v < 20:   return "background:#fdebd0;color:#7e5109;"
            if v < 30:   return "background:#fadbd8;color:#922b21;"
            return              "background:#f9ebea;color:#641e16;"

        html6 = [
            "<div style='overflow-x:auto;margin:12px 0 24px;'>",
            "<table style='border-collapse:collapse;font-family:Inter,sans-serif;"
            "font-size:.80em;width:100%;'>",
            "<thead>",
            "<tr>",
            "<th rowspan='3' style='background:#1a1a2e;color:white;padding:8px 12px;"
            "border:1px solid #dee2e6;'>Age Band</th>",
            "<th rowspan='3' style='background:#1a1a2e;color:white;padding:8px 12px;"
            "border:1px solid #dee2e6;'>N</th>",
        ]
        for sx_lbl, sx_clr in [("Males","#457b9d"),("Females","#e63946")]:
            html6.append(f"<th colspan='{len(risk_models)*3}' "
                         f"style='background:{sx_clr};color:white;font-weight:700;"
                         f"padding:7px;border:1px solid #dee2e6;text-align:center;'>{sx_lbl}</th>")
        html6.append("</tr>")

        html6.append("<tr>")
        for _ in range(2):
            for lbl, _ in risk_models:
                c = MODEL_COLORS_6.get(lbl, "#555")
                html6.append(f"<th colspan='3' style='background:{c};color:white;"
                             f"font-size:.78em;font-weight:700;padding:6px 8px;"
                             f"border:1px solid #dee2e6;text-align:center;'>{lbl}</th>")
        html6.append("</tr>")

        html6.append("<tr>")
        for _ in range(2 * len(risk_models)):
            for metric in ["Mean±SD", "≥10%", "≥20%"]:
                html6.append(f"<th style='background:#f1f3f5;color:#555;font-size:.72em;"
                             f"font-weight:600;padding:5px 8px;border:1px solid #dee2e6;"
                             f"text-align:center;white-space:nowrap;'>{metric}</th>")
        html6.append("</tr></thead><tbody>")

        alt = False
        for sex_val, sex_lbl in [("men","Males"),("women","Females")]:
            sub = df[df[sex_col] == sex_val]
            first_in_sex = True
            for ab in age_bands_ordered:
                cell_df = sub[sub["age_band"] == ab]
                n_cell  = len(cell_df)
                alt = not alt
                row_bg  = "#f8f9fd" if alt else "white"
                html6.append(f"<tr style='background:{row_bg};'>")
                html6.append(f"<td style='font-weight:600;padding:7px 10px;"
                             f"border:1px solid #dee2e6;white-space:nowrap;'>{ab}</td>")
                html6.append(f"<td style='text-align:center;padding:7px 10px;"
                             f"border:1px solid #dee2e6;color:#555;'>{n_cell:,}</td>")


                for s_val2, _ in [("men","Males"),("women","Females")]:
                    if s_val2 != sex_val:
                        for lbl, _ in risk_models:
                            html6.append(f"<td colspan='3' style='background:#f0f0f0;"
                                        f"border:1px solid #dee2e6;'></td>")
                        continue
                    for lbl, rcol in risk_models:
                        s = cell_df[rcol].dropna() if rcol in cell_df.columns else pd.Series(dtype=float)
                        mean_sd = f"{s.mean():.1f}±{s.std():.1f}" if len(s) > 0 else "—"
                        pct10   = f"{(s>=10).mean()*100:.0f}%"     if len(s) > 0 else "—"
                        pct20   = f"{(s>=20).mean()*100:.0f}%"     if len(s) > 0 else "—"
                        rs      = _risk_cell_style(mean_sd)
                        c = MODEL_COLORS_6.get(lbl, "#555")
                        border  = f"border-left:3px solid {c};"
                        html6.append(f"<td style='{rs}{border}padding:7px 8px;"
                                    f"border-top:1px solid #dee2e6;border-bottom:1px solid #dee2e6;"
                                    f"text-align:center;font-weight:600;'>{mean_sd}</td>")
                        html6.append(f"<td style='background:{row_bg};padding:7px 8px;"
                                    f"border:1px solid #dee2e6;text-align:center;"
                                    f"font-size:.82em;color:#555;'>{pct10}</td>")
                        html6.append(f"<td style='background:{row_bg};padding:7px 8px;"
                                    f"border:1px solid #dee2e6;text-align:center;"
                                    f"font-size:.82em;color:#555;'>{pct20}</td>")

                html6.append("</tr>")

                for lbl, rcol in risk_models:
                    if rcol in cell_df.columns:
                        s = cell_df[rcol].dropna()
                        t6_flat.append({"Sex": sex_lbl, "Age Band": ab,
                                        "Model": lbl, "N": n_cell,
                                        "Mean±SD": f"{s.mean():.1f}±{s.std():.1f}" if len(s)>0 else "—",
                                        "≥10%": f"{(s>=10).mean()*100:.1f}%" if len(s)>0 else "—",
                                        "≥20%": f"{(s>=20).mean()*100:.1f}%" if len(s)>0 else "—"})

        html6.append("</tbody></table></div>")
        st.markdown("".join(html6), unsafe_allow_html=True)
        if t6_flat:
            st.download_button("📥 Download Table 6 (CSV)",
                               pd.DataFrame(t6_flat).to_csv(index=False),
                               file_name="table6_risk_by_sex_age.csv", mime="text/csv")

    st.markdown("---")

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 24px 14px;
                border-radius:12px;margin-bottom:16px;'>
        <h3 style='color:white;margin:0;font-size:1.3em;'>Table 7 — Net Reclassification Index (NRI) vs WHO Non-Lab</h3>
        <p style='color:#a8b2d1;margin:6px 0 0;font-size:.88em;'>
            Category-based NRI (%) = (proportion up-classified − proportion down-classified) × 100.
            <b style='color:#93c5fd;'>Positive NRI</b> = model flags more high-risk than WHO.
            <b style='color:#fca5a5;'>Negative NRI</b> = model is more conservative.
        </p>
    </div>""", unsafe_allow_html=True)

    nri_rows = []
    ref_cat = ALL_RISK["WHO Non-Lab"][1]
    for group_label, sub_df in [("Overall", df),
                                  ("Men",     df[df[sex_col]=="men"]),
                                  ("Women",   df[df[sex_col]=="women"])]:
        for label, (rcol, cat_col) in ALL_RISK.items():
            if label == "WHO Non-Lab" or cat_col not in sub_df.columns or ref_cat not in sub_df.columns:
                continue
            valid = sub_df[[ref_cat, cat_col]].dropna()
            if len(valid) < 5:
                continue
            cats = RISK_LABELS
            cat_idx_a = {c: i for i, c in enumerate(cats)}
            ref_vals  = valid[ref_cat].astype(str).map(cat_idx_a)
            cmp_vals  = valid[cat_col].astype(str).map(cat_idx_a)
            up   = (cmp_vals > ref_vals).sum()
            down = (cmp_vals < ref_vals).sum()
            same = (cmp_vals == ref_vals).sum()
            nri  = (up - down) / len(valid) * 100
            rcol_r = ALL_RISK["WHO Non-Lab"][0]
            if rcol_r in sub_df.columns and rcol in sub_df.columns:
                bv = sub_df[[rcol_r, rcol]].dropna()
                wh = bv[rcol_r] >= 20;  ch = bv[rcol] >= 20
                bin_nri = ((~wh & ch).sum() - (wh & ~ch).sum()) / len(bv) * 100
            else:
                bin_nri = np.nan
            nri_rows.append({
                "group": group_label, "model": label, "n": len(valid),
                "up": int(up), "down": int(down), "same": int(same),
                "nri": round(nri, 1),
                "bin_nri": round(bin_nri, 1) if not np.isnan(bin_nri) else None,
            })

    if nri_rows:
        models_uniq = list(dict.fromkeys(r["model"] for r in nri_rows))
        groups_nri  = ["Overall", "Men", "Women"]
        lookup_nri  = {(r["model"], r["group"]): r for r in nri_rows}
        metrics_nri = ["N", "Up ↑", "Down ↓", "Same", "Cat. NRI (%)", "Bin. NRI (%)", "Direction"]
        grp_colors  = {"Overall": "#2a9d8f", "Men": "#457b9d", "Women": "#e63946"}

        def _nri_cell(v):
            """Nri cell."""
            if v is None: return "background:#f0f0f0;color:#bbb;", "—"
            if v > 2:   return "background:rgba(37,99,235,.18);color:#1d4ed8;font-weight:700;", f"+{v:.1f}%"
            if v < -2:  return "background:rgba(220,38,38,.18);color:#dc2626;font-weight:700;", f"{v:.1f}%"
            return "background:#f8f9fa;color:#555;", f"{v:.1f}%"

        def _dir_badge(v):
            """Dir badge."""
            if v is None: return "—"
            if v > 1:   return "<span style='background:#1d4ed8;color:white;padding:2px 8px;border-radius:10px;font-size:.75em;font-weight:700;'>↑ Higher risk</span>"
            if v < -1:  return "<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:10px;font-size:.75em;font-weight:700;'>↓ Lower risk</span>"
            return "<span style='background:#6b7280;color:white;padding:2px 8px;border-radius:10px;font-size:.75em;font-weight:700;'>≈ Neutral</span>"

        html7 = [
            "<div style='overflow-x:auto;margin:12px 0 24px;'>",
            "<table style='border-collapse:collapse;font-family:Inter,sans-serif;font-size:.82em;width:100%;'>",
            "<thead><tr>",
            "<th rowspan='2' style='background:#1a1a2e;color:white;padding:8px 14px;"
            "border:1px solid #dee2e6;text-align:left;min-width:140px;'>Model vs WHO</th>",
        ]
        for g in groups_nri:
            html7.append(f"<th colspan='{len(metrics_nri)}' style='background:{grp_colors[g]};"
                         f"color:white;font-weight:700;padding:7px;border:1px solid #dee2e6;"
                         f"text-align:center;'>{g}</th>")
        html7.append("</tr><tr>")
        for _ in groups_nri:
            for m in metrics_nri:
                html7.append(f"<th style='background:#f1f3f5;color:#555;font-size:.72em;"
                             f"font-weight:600;padding:5px 8px;border:1px solid #dee2e6;"
                             f"text-align:center;white-space:nowrap;'>{m}</th>")
        html7.append("</tr></thead><tbody>")

        alt = False
        for model in models_uniq:
            alt = not alt
            row_bg = "#f8f9fd" if alt else "white"
            mc = MODEL_COLORS_6.get(model, "#555")
            html7.append(f"<tr style='background:{row_bg};'>")
            html7.append(f"<td style='font-weight:700;padding:8px 14px;border:1px solid #dee2e6;"
                         f"border-left:4px solid {mc};white-space:nowrap;'>{model}</td>")
            for g in groups_nri:
                r = lookup_nri.get((model, g))
                if r is None:
                    html7.append(f"<td colspan='{len(metrics_nri)}' style='background:#f0f0f0;"
                                 f"border:1px solid #dee2e6;text-align:center;color:#bbb;'>—</td>")
                    continue
                nri_s, nri_v  = _nri_cell(r["nri"])
                bnri_s, bnri_v = _nri_cell(r["bin_nri"])
                html7.append(f"<td style='text-align:center;padding:7px 8px;border:1px solid #dee2e6;"
                             f"color:#555;'>{r['n']:,}</td>")
                html7.append(f"<td style='text-align:center;padding:7px 8px;border:1px solid #dee2e6;"
                             f"color:#1d4ed8;'>{r['up']:,}</td>")
                html7.append(f"<td style='text-align:center;padding:7px 8px;border:1px solid #dee2e6;"
                             f"color:#dc2626;'>{r['down']:,}</td>")
                html7.append(f"<td style='text-align:center;padding:7px 8px;border:1px solid #dee2e6;"
                             f"color:#555;'>{r['same']:,}</td>")
                html7.append(f"<td style='{nri_s}padding:7px 8px;border:1px solid #dee2e6;"
                             f"text-align:center;'>{nri_v}</td>")
                html7.append(f"<td style='{bnri_s}padding:7px 8px;border:1px solid #dee2e6;"
                             f"text-align:center;'>{bnri_v}</td>")
                html7.append(f"<td style='padding:7px 8px;border:1px solid #dee2e6;"
                             f"text-align:center;'>{_dir_badge(r['nri'])}</td>")
            html7.append("</tr>")

        html7.append("<tr><td colspan='100%' style='background:#f8f9fa;font-size:.73em;"
                     "color:#666;padding:8px 14px;border-top:2px solid #dee2e6;text-align:left;'>")
        html7.append("<b>Cat. NRI</b> = category-based NRI (all 5 bands). "
                     "<b>Bin. NRI</b> = binary NRI at ≥20% threshold. "
                     "<span style='color:#1d4ed8;font-weight:700;'>Blue positive</span> = flags more high-risk than WHO. "
                     "<span style='color:#dc2626;font-weight:700;'>Red negative</span> = more conservative.")
        html7.append("</td></tr></tbody></table></div>")

        st.markdown("".join(html7), unsafe_allow_html=True)
        t7_csv = pd.DataFrame([{
            "Group": r["group"], "Model": r["model"], "N": r["n"],
            "Up": r["up"], "Down": r["down"], "Same": r["same"],
            "Cat NRI (%)": r["nri"], "Binary NRI (%)": r["bin_nri"],
        } for r in nri_rows])
        st.download_button("📥 Download Table 7 (CSV)", t7_csv.to_csv(index=False),
                           file_name="table7_nri.csv", mime="text/csv")

    st.markdown("---")
    with st.expander("📝 Table Interpretation Guide", expanded=False):
        st.markdown("""
| Table | Purpose | Key Metric |
|-------|---------|------------|
| **1** | Participant characteristics | Demonstrates comparability of subgroups; p-values from Mann-Whitney U (continuous) |
| **2A/2B** | Lab vs Non-Lab agreement — Male / Female | Weighted κ; exact 5-band cross-tab reveals where discordance clusters |
| **3** | All model-pair category cross-tabs | Identifies systematic over/under-classification between any two models |
| **4** | Pairwise κ summary | Single-row comparison of all model pairs; strength interpretation provided |
| **5** | Reclassification table vs WHO | Quantifies up/downclassification; diagonal = agreement |
| **6** | Sex × age-band mean risk (all models) | Reveals age gradient and sex divergence across models |
| **7** | NRI vs WHO Non-Lab | Positive NRI = model flags more high-risk than WHO; negative = more conservative |

**Cohen's κ strength interpretation (Landis & Koch 1977):**
- ≤0.20: Slight | 0.21–0.40: Fair | 0.41–0.60: Moderate | 0.61–0.80: Substantial | >0.80: Almost perfect

**NRI:** Category-based NRI = proportion reclassified up − proportion reclassified down.
Positive values indicate the model assigns patients to higher risk bands than WHO; 
negative values indicate a more conservative model.
        """)
