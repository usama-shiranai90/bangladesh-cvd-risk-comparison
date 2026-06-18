import streamlit as st
import pandas as pd
from utils.helpers import get_risk_cat, RISK_PALETTE

# Try importing CVD library
try:
    from cvd.library import feature_engineering as fe
    from cvd.library import risk_data as rd
except ImportError as e:
    st.error("CVD Library not found. Calculator disabled.")
    st.warning(e)
    fe = None
    rd = None

def render_calculator():
    st.title("❤️ Individual Risk Calculator")
    st.markdown("Estimate 10-year cardiovascular disease risk using WHO charts for South Asia (SEAR D).")
    
    if fe is None or rd is None:
        st.warning("Backend library missing.")
        return

    c1, c2 = st.columns(2)
    with c1:
        age_in = st.number_input("Age", 20, 100, 50)
        gender_in = st.selectbox("Gender", [1,0], format_func=lambda x: "Male" if x==1 else "Female")
        sbp_in = st.number_input("Systolic BP (mmHg)", 70, 280, 120)
        smoker_in = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Smoker" if x==1 else "Non-Smoker")
    with c2:
        bmi_in = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
        has_lab = st.toggle("Include Lab Data?")
        chol_in = 4.0
        diab_in = "no_diabetes"
        if has_lab:
            chol_in = st.number_input("Cholesterol (mmol/L)", 2.0, 15.0, 5.0)
            is_diab = st.selectbox("Diabetes", ["No", "Yes"])
            diab_in = "with_diabetes" if is_diab == "Yes" else "no_diabetes"

    if st.button("Calculate Risk", type="primary"):
        # Fix SBP labels
        fe.SBP_LABELS = ["<120", "120-139", "140-159", "160-179", ">="]
        
        demo_df = pd.DataFrame([{
            "age": age_in, "gender": gender_in, "sbp": sbp_in, "bmi": bmi_in,
            "cholesterol_mmolL": chol_in, "smoker": smoker_in,
            "has_diabetes": True if diab_in == "with_diabetes" else False,
            "smoker_who": smoker_in
        }])
        
     

        # st.info(rd.risk_data['non_lab']['men'])
        demo_df = fe.prepare_who_analysis_df(demo_df)
        demo_df = fe.add_who_risks(demo_df, rd.risk_data)
        res = demo_df.iloc[0]
        # st.dataframe(res)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            r = res['risk_nonlab']
            if pd.notna(r):
                # Map color
                cat = get_risk_cat(r)
                c = RISK_PALETTE.get(cat, "#333")
                st.markdown(f"""
                <div style='background:{c};padding:25px;border-radius:12px;color:white;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>
                    <h3 style='color:white;margin:0;opacity:0.9'>Non-Lab Risk</h3>
                    <h1 style='color:white;font-size:3.5em;margin:10px 0'>{r}%</h1>
                    <span style='background:rgba(255,255,255,0.2);padding:4px 12px;border-radius:20px'>{cat}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Non-Lab Risk: Out of range (40-74y)")
        
        if has_lab:
            with col_res2:
                r_l = res['risk_lab']
                if pd.notna(r_l):
                    cat = get_risk_cat(r_l)
                    c = RISK_PALETTE.get(cat, "#333")
                    st.markdown(f"""
                    <div style='background:{c};padding:25px;border-radius:12px;color:white;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>
                        <h3 style='color:white;margin:0;opacity:0.9'>Lab Risk</h3>
                        <h1 style='color:white;font-size:3.5em;margin:10px 0'>{r_l}%</h1>
                        <span style='background:rgba(255,255,255,0.2);padding:4px 12px;border-radius:20px'>{cat}</span>
                    </div>
                    """, unsafe_allow_html=True)
