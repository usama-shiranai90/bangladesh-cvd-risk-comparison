import streamlit as st
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
 
if not hasattr(st, '_patched_for_export'):
    st._patched_for_export = True
    
    st._orig_dataframe = st.dataframe
    st._orig_markdown = st.markdown
    st._orig_pyplot = st.pyplot
    st._orig_plotly_chart = getattr(st, 'plotly_chart', None)

    def patched_dataframe(*args, **kwargs):
        """Patched dataframe."""
        import pandas as pd
        target = args[0] if len(args) > 0 else kwargs.get('data')
        if target is not None and isinstance(target, pd.DataFrame):
            if 'export_tables' in st.session_state:
                count = len(st.session_state['export_tables']) + 1
                st.session_state['export_tables'][f"table_{count}"] = target
        return st._orig_dataframe(*args, **kwargs)

    def patched_markdown(body, *args, **kwargs):
        """Patched markdown."""
        if kwargs.get('unsafe_allow_html', False) and isinstance(body, str) and "<table" in body.lower():
            if 'export_html' in st.session_state:
                count = len(st.session_state['export_html']) + 1
                st.session_state['export_html'][f"html_table_{count}"] = body
        return st._orig_markdown(body, *args, **kwargs)

    def patched_pyplot(*args, **kwargs):
        """Patched pyplot."""
        import matplotlib.pyplot as plt
        target = args[0] if len(args) > 0 else kwargs.get('fig')
        if target is None:
            target = plt.gcf()
        if target is not None and target.get_axes():
            if 'export_figures' in st.session_state:
                count = len(st.session_state['export_figures']) + 1
                st.session_state['export_figures'][f"figure_{count}"] = target
        return st._orig_pyplot(*args, **kwargs)

    def patched_plotly(*args, **kwargs):
        """Patched plotly."""
        target = args[0] if len(args) > 0 else kwargs.get('figure_or_data')
        if target is not None:
            if 'export_figures' in st.session_state:
                count = len(st.session_state['export_figures']) + 1
                st.session_state['export_figures'][f"plotly_{count}"] = target
        if st._orig_plotly_chart:
            return st._orig_plotly_chart(*args, **kwargs)

    st.dataframe = patched_dataframe
    st.markdown = patched_markdown
    st.pyplot = patched_pyplot
    if hasattr(st, 'plotly_chart'):
        st.plotly_chart = patched_plotly

for k in ['export_tables', 'export_figures', 'export_html']:
    if k not in st.session_state:
        st.session_state[k] = {}

st.session_state['export_tables'].clear()
st.session_state['export_figures'].clear()
st.session_state['export_html'].clear()
from components.styling import apply_custom_css
from components.sidebar import render_sidebar
from utils.data_loader import load_csv_safe, get_merged_data

from modules.overview import render_overview
from modules.sites import render_sites
from modules.calculator import render_calculator
from modules.browser import render_browser
from modules.rq0 import render_rq0
from modules.rq1 import render_rq1
from modules.rq1_backup import render_backup_rq1

from modules.journal_figures import render_journal_figures
from modules.multi_risk import render_multi_risk
from modules.deep_eda import render_deep_eda

def get_cvd_paths():
    """Get CVD paths."""
    base_res = os.path.join(parent_dir, "cvd", "resource")
    base_an = os.path.join(base_res, "analyzed", "v2.30")
    return base_res, base_an

st.set_page_config(
    page_title="Bangladesh CVD Risk Tracker",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

if 'path_main' not in st.session_state:
    st.session_state['path_main'] = None
if 'path_lab' not in st.session_state:
    st.session_state['path_lab'] = None
if 'path_paired' not in st.session_state:
    st.session_state['path_paired'] = None
if 'path_sites' not in st.session_state:
    st.session_state['path_sites'] = None

selected_page = render_sidebar(parent_dir)
if 'last_page' not in st.session_state or st.session_state['last_page'] != selected_page:
    st.session_state['ready_zip'] = None
    st.session_state['last_page'] = selected_page

df_main = None
df_sites = None
df_paired = None

if st.session_state['path_sites']:
    df_sites = load_csv_safe(st.session_state['path_sites'])

if st.session_state['path_main']:
    raw_main = load_csv_safe(st.session_state['path_main'])
    df_main = get_merged_data(raw_main, df_sites)

if st.session_state['path_paired']:
    df_paired = load_csv_safe(st.session_state['path_paired'])

totals = {
    'total_n': len(df_main) if df_main is not None else 0,
    'paired_n': len(df_paired) if df_paired is not None else 0
}

if selected_page == "Overview":
    render_overview(df_main, totals)

elif selected_page == "RQ0: Baseline Burden":
    base_res, base_an = get_cvd_paths()
    
    rq_datasets = {
        "sites": load_csv_safe(os.path.join(base_res, "service_site_with_geographical_points.csv")),
        "nonlab": load_csv_safe(os.path.join(base_an, "cvd_nonlab.csv")),
        "lab": load_csv_safe(os.path.join(base_an, "cvd_lab.csv")),
        "paired": load_csv_safe(os.path.join(base_an, "cvd_paired.csv")),
        "who_nonlab": load_csv_safe(os.path.join(base_an, "cvd_who_nonlab_domain.csv")),
        "who_lab": load_csv_safe(os.path.join(base_an, "cvd_who_lab_domain.csv"))
    }
    
    primary_rq_df = df_main if df_main is not None else rq_datasets['who_nonlab']
    
    selected_dataset_name = None
    if primary_rq_df is not None and rq_datasets['sites'] is not None:
        primary_rq_df = get_merged_data(primary_rq_df, rq_datasets['sites'])

    if st.session_state.get('path_main'):
        selected_dataset_name = os.path.basename(st.session_state['path_main'])
    
    render_rq0(primary_rq_df, datasets=rq_datasets, selected_dataset_name=selected_dataset_name)

elif selected_page == "RQ1: Safety & Discordance":
    target_paired = df_paired
    if target_paired is None:
         base_res, base_an = get_cvd_paths()
         target_paired = load_csv_safe(os.path.join(base_an, "cvd_paired.csv"))
         
    render_rq1(target_paired)

elif selected_page == "RQ1.1: Safety & Discordance":
    target_paired = df_paired
    if target_paired is None:
         base_res, base_an = get_cvd_paths()
         target_paired = load_csv_safe(os.path.join(base_an, "cvd_paired.csv"))
         
    render_backup_rq1(target_paired)


elif selected_page == "Site Heterogeneity":

    base_res, base_an = get_cvd_paths()

    rq_datasets = {
        "sites": load_csv_safe(os.path.join(base_res, "service_site_with_geographical_points.csv")),
        "nonlab": load_csv_safe(os.path.join(base_an, "cvd_nonlab.csv")),
        "lab": load_csv_safe(os.path.join(base_an, "cvd_lab.csv")),
        "paired": load_csv_safe(os.path.join(base_an, "cvd_paired.csv")),
        "who_nonlab": load_csv_safe(os.path.join(base_an, "cvd_who_nonlab_domain.csv")),
        "who_lab": load_csv_safe(os.path.join(base_an, "cvd_who_lab_domain.csv"))
    }

    primary_rq_df = df_main if df_main is not None else rq_datasets['who_nonlab']

    selected_dataset_name = None
    if primary_rq_df is not None and rq_datasets['sites'] is not None:
        primary_rq_df = get_merged_data(primary_rq_df, rq_datasets['sites'])

    if st.session_state.get('path_main'):
        selected_dataset_name = os.path.basename(st.session_state['path_main'])
    render_sites(primary_rq_df)

elif selected_page == "Risk Calculator":
    render_calculator()

elif selected_page == "Multi-Model Risk":
    base_res, base_an = get_cvd_paths()
    multi_datasets = {
        "who_nonlab": load_csv_safe(os.path.join(base_an, "cvd_who_nonlab_domain.csv")),
        "who_lab": load_csv_safe(os.path.join(base_an, "cvd_who_lab_domain.csv")),
        "paired": load_csv_safe(os.path.join(base_an, "cvd_paired.csv")),
    }
    render_multi_risk(multi_datasets)

elif selected_page == "Data Browser":
    render_browser(df_main)

elif selected_page == "Journal Figures":
    base_res, base_an = get_cvd_paths()
    jf_sites = load_csv_safe(os.path.join(base_res, "service_site_with_geographical_points.csv"))
    jf_who_nl = load_csv_safe(os.path.join(base_an, "cvd_who_nonlab_domain.csv"))
    jf_who_l  = load_csv_safe(os.path.join(base_an, "cvd_who_lab_domain.csv"))
    if jf_sites is not None:
        jf_who_nl = get_merged_data(jf_who_nl, jf_sites)
        jf_who_l  = get_merged_data(jf_who_l, jf_sites)
    jf_datasets = {
        "nonlab": load_csv_safe(os.path.join(base_an, "cvd_nonlab.csv")),
        "lab": load_csv_safe(os.path.join(base_an, "cvd_lab.csv")),
        "who_nonlab": jf_who_nl,
        "who_lab": jf_who_l,
    }
    
    st.title("Journal Figures")
    tab1, tab2 = st.tabs(["Journal Figures", "Data Explorer"])
    with tab1:
        render_journal_figures(jf_datasets)

elif selected_page == "Deep EDA":
    base_res, base_an = get_cvd_paths()
    deep_datasets = {
        "who_nonlab": load_csv_safe(os.path.join(base_an, "cvd_who_nonlab_domain.csv")),
        "who_lab":    load_csv_safe(os.path.join(base_an, "cvd_who_lab_domain.csv")),
        "sites":      load_csv_safe(os.path.join(base_res, "service_site_with_geographical_points.csv")),
    }
    render_deep_eda(deep_datasets.get("who_nonlab"), datasets=deep_datasets)

def generate_export_zip():
    """Generate export zip."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in st.session_state['export_tables'].items():
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr(f"{name}.csv", csv_buf.getvalue())
            
        for name, html in st.session_state['export_html'].items():
            full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html}</body></html>"
            zf.writestr(f"{name}.html", full_html)
            
        for name, fig in st.session_state['export_figures'].items():
            img_buf = io.BytesIO()
            try:
                if hasattr(fig, "savefig"):
                    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=300)
                    zf.writestr(f"{name}.png", img_buf.getvalue())
                elif hasattr(fig, "write_image"):
                    fig.write_image(img_buf, format="png")
                    zf.writestr(f"{name}.png", img_buf.getvalue())
            except Exception as e:
                with io.StringIO() as err_buf:
                    err_buf.write(f"Error saving image: {e}")
                    zf.writestr(f"{name}_error.txt", err_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()

with st.sidebar:
    st.markdown("---")
    st.subheader("📥 Export Page Content")
    st.caption("Export all tables and diagrams shown on the current page to a ZIP file.")
    
    if st.button("Prepare Download (ZIP)", key="prepare_zip_btn"):
        st.session_state['ready_zip'] = generate_export_zip()
        
    if st.session_state.get('ready_zip'):
        st.download_button(
            label="📦 Download Ready (ZIP)",
            data=st.session_state['ready_zip'],
            file_name=f"{selected_page.replace(' ', '_')}_export.zip",
            mime="application/zip",
            key="download_ready_zip_btn"
        )
