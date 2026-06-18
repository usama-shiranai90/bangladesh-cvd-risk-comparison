import streamlit as st
import os
from utils.data_loader import find_available_datasets, load_csv_safe


def render_sidebar(parent_dir):
    """Render sidebar."""
    with st.sidebar:
        st.title("PHC Cardiovascular")

        st.subheader("Navigation")

        nav_section = st.radio("Section", ["Analysis Modules", "Tools & Data"], horizontal=True, label_visibility="collapsed")
        if nav_section == "Analysis Modules":
            st.caption("Research & Insights")
            selected_page = st.radio("Module",
                ["Overview", "RQ0: Baseline Burden", "RQ1: Safety & Discordance",
                "RQ1.1: Safety & Discordance",
                "Site Heterogeneity",
                "Journal Figures",
                "📄 Journal Figures Anti"],
                label_visibility="collapsed"
            )
        else:
            st.caption("Exploration Tools")
            selected_page = st.radio("Tool",
                ["Risk Calculator", "Multi-Model Risk", "Data Browser", "Deep EDA"],
                label_visibility="collapsed"
            )

        st.markdown("---")
        st.subheader("Dataset Configuration")

        analyzed_dir = os.path.join(parent_dir, "cvd", "resource", "analyzed", "v2.30")
        available = find_available_datasets(analyzed_dir)


        site_path = os.path.join(parent_dir, "cvd", "resource", "service_site_with_geographical_points.csv")
        available["Sites"] = site_path


        options = list(available.keys())
        default_idx = 0
        if "cvd_who_nonlab_domain.csv" in options:
            default_idx = options.index("cvd_who_nonlab_domain.csv")

        selected_main = st.selectbox("Select Non-Lab Data", options, index=default_idx)

        if selected_main:
            st.session_state['path_main'] = available[selected_main]

        lab_default_idx = 0
        if "cvd_who_lab_domain.csv" in options:
            lab_default_idx = options.index("cvd_who_lab_domain.csv")
        elif "cvd_lab.csv" in options:
            lab_default_idx = options.index("cvd_lab.csv")

        selected_lab = st.selectbox("Select Lab Data", options, index=lab_default_idx)
        if selected_lab:
            st.session_state['path_lab'] = available[selected_lab]

        if "cvd_paired.csv" in available:
            st.session_state['path_paired'] = available["cvd_paired.csv"]
        else:
            st.session_state['path_paired'] = None


        st.info(f"**Loaded:** {selected_main}")

        return selected_page
