import streamlit as st
import os
from utils.data_loader import find_available_datasets, load_csv_safe


def render_sidebar(parent_dir):
    with st.sidebar:
        # logo_path = os.path.join(parent_dir, "streamlit", "assets", "logo.png")
        # if os.path.exists(logo_path):
        #      st.image(logo_path, width=120)
        # else:
        #      st.image("https://cdn-icons-png.flaticon.com/512/323/323295.png", width=50) # Fallback
        st.title("PHC Cardiovascular")

        # Navigation
        st.subheader("Navigation")

        # Section Selector
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

        # Dataset Selector
        # Default analyzed path
        analyzed_dir = os.path.join(parent_dir, "cvd", "resource", "analyzed", "v2.30")
        available = find_available_datasets(analyzed_dir)

        #print("Available datasets found:", available)

        # Also add the site file
        site_path = os.path.join(parent_dir, "cvd", "resource", "service_site_with_geographical_points.csv")
        available["Sites"] = site_path

        # # Old confirmation files
        # archieve_dir = os.path.join(parent_dir, "cvd", "resource", "analyzed", "archieve")
        # archieve_available = find_available_datasets(archieve_dir)
        # available.update(archieve_available)

        # st.text(available.keys())
        # st.text(archieve_available.keys())

        # --- Selector for Main Cohort ---
        # st.caption("Primary Non-Lab Dataset")
        # Default options
        options = list(available.keys())
        default_idx = 0
        if "cvd_who_nonlab_domain.csv" in options:
            default_idx = options.index("cvd_who_nonlab_domain.csv")

        selected_main = st.selectbox("Select Non-Lab Data", options, index=default_idx)

        if selected_main:
            st.session_state['path_main'] = available[selected_main]

        # --- Selector for Lab Cohort ---
        # st.caption("Primary Lab Dataset")
        lab_default_idx = 0
        if "cvd_who_lab_domain.csv" in options:
            lab_default_idx = options.index("cvd_who_lab_domain.csv")
        elif "cvd_lab.csv" in options:
            lab_default_idx = options.index("cvd_lab.csv")

        selected_lab = st.selectbox("Select Lab Data", options, index=lab_default_idx)
        if selected_lab:
            st.session_state['path_lab'] = available[selected_lab]

        # --- Selector for Paired ---
        if "cvd_paired.csv" in available:
            st.session_state['path_paired'] = available["cvd_paired.csv"]
        else:
            st.session_state['path_paired'] = None

        # Always set site path
        # st.session_state['path_sites'] = site_path

        st.info(f"**Loaded:** {selected_main}")

        return selected_page
