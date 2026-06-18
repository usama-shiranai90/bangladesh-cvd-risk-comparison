import streamlit as st

def render_browser(df_merged):
    st.title("📂 Data Browser")
    st.info("Explore the underlying dataset for Bangladesh")
    
    if df_merged is not None:
        grid_opts = df_merged.columns.tolist()
        cols = st.multiselect("Columns to view", grid_opts, default=["pid", "age", "gender", "smoker", "cholesterol", "sbp", "height", "weight" ,"bmi" , "risk_nonlab", "risk_lab"])
        
        search = st.text_input("Search PID")
        show_df = df_merged[cols].copy()
        
        if search:
            show_df = show_df[show_df["pid"].astype(str).str.contains(search)]
            
        st.dataframe(show_df.head(200), use_container_width=True)
        st.caption(f"Showing {len(show_df)} records")
    else:
        st.warning("No data available.")
