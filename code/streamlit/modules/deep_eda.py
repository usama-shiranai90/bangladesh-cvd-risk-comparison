"""
Deep EDA Module — PHC CVD Dataset
Focused deep-dive analyses revealing clinically meaningful patterns:
  1. Diabetes × CVD Risk Interaction
  2. Hypertension Profile & Cascade
  3. Abdominal Obesity Paradox (BMI vs WHR)
  4. Blood Glucose Distribution & Pre-diabetes Burden
  5. Site Heterogeneity Deep-Dive
  6. Co-morbidity Clustering
  7. Smoking & Gender Gap
  8. Arrhythmia & Elevated Pulse
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

try:
    import scienceplots
    plt.style.use(['science', 'nature', 'no-latex'])
    SCIPLOT = True
except Exception:
    SCIPLOT = False

# ── colour palette ─────────────────────────────────────────────────────────
_C = {
    "male":   "#2980b9",
    "female": "#e74c3c",
    "diab":   "#e67e22",
    "nodiab": "#27ae60",
    "htn":    "#c0392b",
    "normal": "#2ecc71",
    "warn":   "#f39c12",
    "accent": "#8e44ad",
    "grey":   "#95a5a6",
}
RISK_COLORS = {
    "<5%":         "#27ae60",
    "5% to <10%":  "#f1c40f",
    "10% to <20%": "#e67e22",
    "20% to <30%": "#e74c3c",
    "≥30%":        "#922b21",
}
RISK_BINS   = [-np.inf, 5, 10, 20, 30, np.inf]
RISK_LABELS = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]


# ── helpers ────────────────────────────────────────────────────────────────
def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def _savefig(fig, key):
    from utils.export_utils import add_download_button
    st.pyplot(fig)
    try:
        add_download_button(fig, key, "matplotlib")
    except Exception:
        pass
    plt.close(fig)

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["risk_nonlab"]  = pd.to_numeric(d.get("risk_nonlab",  pd.Series()), errors="coerce")
    d["bmi"]          = pd.to_numeric(d.get("bmi",          pd.Series()), errors="coerce")
    d["sbp"]          = pd.to_numeric(d.get("sbp",          pd.Series()), errors="coerce")
    d["dbp"]          = pd.to_numeric(d.get("dbp",          pd.Series()), errors="coerce")
    d["bg_mgdl"]      = pd.to_numeric(d.get("bg_mgdl",      pd.Series()), errors="coerce")
    d["whr"]          = pd.to_numeric(d.get("whr",          pd.Series()), errors="coerce")
    d["age"]          = pd.to_numeric(d.get("age",          pd.Series()), errors="coerce")
    d["pulse"]        = pd.to_numeric(d.get("pulse",        pd.Series()), errors="coerce")

    # gender_key normalised → 'men'/'women'
    if "gender_key" in d.columns:
        sex = d["gender_key"]
    elif "gender" in d.columns:
        sex = d["gender"].map({"M": "men", "F": "women", "Male": "men", "Female": "women"})
    else:
        sex = pd.Series("unknown", index=d.index)
    d["_sex"] = sex

    # binary diabetes
    if "has_diabetes" in d.columns:
        d["_diab"] = d["has_diabetes"].astype(bool)
    elif "diab_group" in d.columns:
        d["_diab"] = d["diab_group"] == "with_diabetes"
    else:
        d["_diab"] = False

    # binary smoker
    if "smoker_key" in d.columns:
        d["_smoke"] = d["smoker_key"] == "yes"
    elif "smoker" in d.columns:
        d["_smoke"] = d["smoker"].isin(["Smoker", "yes", "1", 1])
    else:
        d["_smoke"] = False

    # age_band
    if "age_band" not in d.columns:
        bins   = [40, 45, 50, 55, 60, 65, 70, 75]
        labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]
        d["age_band"] = pd.cut(d["age"], bins=bins, labels=labels, right=False)

    # risk category
    d["_risk_cat"] = pd.cut(d["risk_nonlab"], bins=RISK_BINS, labels=RISK_LABELS, right=False)

    # hr flags
    d["_hr10"] = (d["risk_nonlab"] >= 10).astype(int)
    d["_hr20"] = (d["risk_nonlab"] >= 20).astype(int)

    # HTN
    if "sbp" in d.columns:
        d["_htn"] = d["sbp"] >= 140

    # abdominal obesity (WHO thresholds for South Asia)
    if "whr" in d.columns:
        d["_abdo_obes"] = np.where(
            d["_sex"] == "men",   d["whr"] >= 0.90,
            np.where(d["_sex"] == "women", d["whr"] >= 0.85, False)
        ).astype(bool)

    # BMI category (Asian thresholds)
    if "bmi" in d.columns:
        d["_bmi_cat"] = pd.cut(
            d["bmi"],
            bins=[0, 18.5, 23, 25, 30, 200],
            labels=["Underweight", "Normal", "Overweight\n(Asian)", "Overweight", "Obese"],
            right=False,
        )

    return d


# ══════════════════════════════════════════════════════════════════════════════
def render_deep_eda(df_merged, datasets=None):
# ══════════════════════════════════════════════════════════════════════════════
    st.title("🔬 Deep EDA — PHC CVD Cohort")
    st.info(
        "Eight targeted analytical deep-dives revealing clinically actionable patterns "
        "in the PHC dataset (N=14,085 eligible non-lab cohort; age 40-74)."
    )

    if df_merged is None or len(df_merged) == 0:
        st.error("No data available. Please load a dataset.")
        return

    # use who_nonlab from datasets if available, else fall back
    src = df_merged
    if datasets and "who_nonlab" in datasets and datasets["who_nonlab"] is not None:
        src = datasets["who_nonlab"]

    df = _prep(src)

    tabs = st.tabs([
        "1️⃣ Diabetes × CVD",
        "2️⃣ Hypertension Cascade",
        "3️⃣ Abdominal Obesity Paradox",
        "4️⃣ Blood Glucose Burden",
        "5️⃣ Site Heterogeneity",
        "6️⃣ Co-morbidity Clustering",
        "7️⃣ Smoking & Gender Gap",
        "8️⃣ Arrhythmia & Pulse",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: Diabetes × CVD Risk Interaction
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Diabetes × CVD Risk Interaction")
        st.caption(
            "Diabetes diagnosis is present in 41.6% of men and 31.9% of women — "
            "far above the national average. This section quantifies how diabetes "
            "modifies CVD risk across age and sex strata."
        )

        d = df.dropna(subset=["risk_nonlab", "_diab", "_sex", "age_band"])

        # ── summary metrics ──
        c1, c2, c3, c4 = st.columns(4)
        for sex, label, col in [("men","Male",c1),("women","Female",c2)]:
            sub = d[d["_sex"] == sex]
            rate = sub["_diab"].mean() * 100
            col.metric(f"Diabetes prevalence ({label})", f"{rate:.1f}%")
        m_diab   = d[d["_sex"]=="men"]["_diab"].mean()*100
        f_diab   = d[d["_sex"]=="women"]["_diab"].mean()*100
        diff     = m_diab - f_diab
        c3.metric("M−F Diabetes Gap", f"{diff:+.1f} pp")
        overall  = d["_diab"].mean()*100
        c4.metric("Overall Prevalence", f"{overall:.1f}%")

        st.markdown("---")

        # ── Mean CVD risk: diabetic vs non-diabetic × age × sex ──
        g = d.groupby(["_sex","age_band","_diab"])["risk_nonlab"].mean().reset_index()
        g.columns = ["sex","age_band","diabetic","mean_risk"]
        g["label"] = g["diabetic"].map({True:"Diabetic", False:"Non-diabetic"})

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        age_bands = ["40-44","45-49","50-54","55-59","60-64","65-69","70-74"]
        x = np.arange(len(age_bands))
        w = 0.35

        for ax, sex, title in zip(axes, ["men","women"], ["Men","Women"]):
            sub = g[g["sex"] == sex]
            d_yes = sub[sub["diabetic"]==True].set_index("age_band")["mean_risk"].reindex(age_bands).fillna(0)
            d_no  = sub[sub["diabetic"]==False].set_index("age_band")["mean_risk"].reindex(age_bands).fillna(0)
            ax.bar(x - w/2, d_yes, w, label="Diabetic",     color=_C["diab"],  edgecolor="black", linewidth=0.5)
            ax.bar(x + w/2, d_no,  w, label="Non-diabetic", color=_C["nodiab"],edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontweight="bold")
            ax.set_xticks(x); ax.set_xticklabels(age_bands, rotation=30, ha="right")
            ax.set_ylabel("Mean 10-year CVD Risk (%)"); ax.legend()
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            # delta annotation
            for i, (yn, nn) in enumerate(zip(d_yes, d_no)):
                if yn > 0:
                    ax.annotate(f"+{yn-nn:.1f}",
                                xy=(x[i]-w/2, yn), xytext=(0,4), textcoords="offset points",
                                ha="center", fontsize=7, color=_C["diab"])
        plt.suptitle("Mean CVD Risk (Non-Lab): Diabetic vs Non-Diabetic by Age & Sex",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        _savefig(fig, "deep_diab_cvd")

        # ── high-risk prevalence table ──
        st.markdown("**High-Risk (≥10%) Prevalence by Diabetes Status & Gender**")
        tbl = d.groupby(["_sex","_diab"])[["_hr10","_hr20"]].mean().mul(100).round(1)
        tbl.index = tbl.index.map(lambda x: f"{x[0].capitalize()} – {'Diabetic' if x[1] else 'Non-diabetic'}")
        tbl.columns = ["≥10% HR (%)", "≥20% HR (%)"]
        st.dataframe(tbl, use_container_width=True)

        # ── Mann-Whitney test ──
        st.markdown("**Statistical Test (Mann-Whitney U): Diabetic vs Non-diabetic CVD risk**")
        rows = []
        for sex in ["men","women"]:
            s1 = d[(d["_sex"]==sex) & (d["_diab"]==True)]["risk_nonlab"]
            s2 = d[(d["_sex"]==sex) & (d["_diab"]==False)]["risk_nonlab"]
            if len(s1) > 5 and len(s2) > 5:
                u, p = stats.mannwhitneyu(s1, s2, alternative="greater")
                rows.append({
                    "Group": sex.capitalize(),
                    "n (Diabetic)": len(s1), "Median Risk (Diab)": round(s1.median(),1),
                    "n (Non-diab)": len(s2), "Median Risk (Non-diab)": round(s2.median(),1),
                    "U": round(u,0), "p-value": f"{'<0.001' if p<0.001 else round(p,4)}"
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: Hypertension Cascade
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Hypertension Profile & CVD Risk Cascade")
        st.caption(
            "SBP ranges from 70–233 mmHg. ~46% of the cohort is hypertensive (SBP ≥ 140). "
            "This section maps the SBP→CVD risk relationship and quantifies co-morbidity with diabetes."
        )

        d = df.dropna(subset=["sbp", "risk_nonlab", "_sex"])

        # SBP categories
        sbp_bins   = [0, 120, 130, 140, 160, 300]
        sbp_labels = ["Normal\n(<120)", "Elevated\n(120-129)", "HTN Stage 1\n(130-139)", "HTN Stage 2\n(140-159)", "Severe\n(≥160)"]
        d["_sbp_cat"] = pd.cut(d["sbp"], bins=sbp_bins, labels=sbp_labels, right=False)

        # ── summary metrics ──
        c1, c2, c3, c4 = st.columns(4)
        htn_rate = (d["sbp"] >= 140).mean() * 100
        c1.metric("Hypertension prevalence (≥140)", f"{htn_rate:.1f}%")
        severe = (d["sbp"] >= 160).mean() * 100
        c2.metric("Severe HTN (≥160 mmHg)", f"{severe:.1f}%")
        c3.metric("Mean SBP", f"{d['sbp'].mean():.1f} mmHg")
        dbt_htn = (d["_diab"] & (d["sbp"] >= 140)).mean() * 100 if "_diab" in d else 0
        c4.metric("Diabetes + HTN co-morbidity", f"{dbt_htn:.1f}%")

        st.markdown("---")

        # ── Bar: Mean CVD risk per SBP category ──
        bp_risk = d.groupby(["_sbp_cat","_sex"])["risk_nonlab"].mean().reset_index()
        bp_risk.columns = ["sbp_cat","sex","mean_risk"]

        fig, ax = _fig(12, 5)
        cats = sbp_labels
        x = np.arange(len(cats))
        w = 0.35
        m = bp_risk[bp_risk["sex"]=="men"].set_index("sbp_cat")["mean_risk"].reindex(cats).fillna(0)
        f = bp_risk[bp_risk["sex"]=="women"].set_index("sbp_cat")["mean_risk"].reindex(cats).fillna(0)
        ax.bar(x-w/2, m, w, label="Men",   color=_C["male"],   edgecolor="black", linewidth=0.5)
        ax.bar(x+w/2, f, w, label="Women", color=_C["female"], edgecolor="black", linewidth=0.5)
        for i, v in enumerate(m):
            ax.text(x[i]-w/2, v+0.1, f"{v:.1f}", ha="center", fontsize=8)
        for i, v in enumerate(f):
            ax.text(x[i]+w/2, v+0.1, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel("Mean 10-year CVD Risk (%)"); ax.legend()
        ax.set_title("Mean CVD Risk by SBP Category & Sex", fontweight="bold")
        plt.tight_layout()
        _savefig(fig, "deep_htn_cvd")

        # ── table ──
        st.markdown("**SBP Category Distribution & High-Risk Prevalence**")
        tbl_bp = d.groupby("_sbp_cat").agg(
            n=("sbp","count"),
            pct_total=("sbp", lambda s: len(s)/len(d)*100),
            mean_risk=("risk_nonlab","mean"),
            pct_hr10=("_hr10","mean"),
            pct_hr20=("_hr20","mean"),
        ).round(1)
        tbl_bp.columns = ["N", "% of Cohort", "Mean Risk", "≥10% HR", "≥20% HR"]
        tbl_bp["% of Cohort"] = tbl_bp["% of Cohort"].map("{:.1f}%".format)
        tbl_bp["≥10% HR"] = tbl_bp["≥10% HR"].mul(100).map("{:.1f}%".format)
        tbl_bp["≥20% HR"] = tbl_bp["≥20% HR"].mul(100).map("{:.1f}%".format)
        st.dataframe(tbl_bp, use_container_width=True)

        # ── Scatter: SBP vs CVD risk coloured by diabetes ──
        st.markdown("**SBP vs CVD Risk (coloured by diabetes status)**")
        fig2, ax2 = _fig(10, 5)
        for diab, lbl, col in [(True,"Diabetic",_C["diab"]),(False,"Non-diabetic",_C["nodiab"])]:
            sub = d[d["_diab"]==diab].sample(min(600, len(d[d["_diab"]==diab])), random_state=42)
            ax2.scatter(sub["sbp"], sub["risk_nonlab"], alpha=0.4, s=15, c=col, label=lbl)
        ax2.set_xlabel("Systolic BP (mmHg)"); ax2.set_ylabel("10-year CVD Risk (%)")
        ax2.set_title("SBP vs CVD Risk by Diabetes Status", fontweight="bold")
        ax2.axvline(140, color="red", linestyle="--", linewidth=1, alpha=0.7, label="HTN threshold")
        ax2.legend(); plt.tight_layout()
        _savefig(fig2, "deep_sbp_scatter")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: Abdominal Obesity Paradox
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Abdominal Obesity Paradox — BMI vs WHR")
        st.caption(
            "Mean BMI ≈ 23.6 appears 'normal' by WHO thresholds, but 46% are overweight/obese "
            "by South-Asian thresholds (BMI ≥ 23). Crucially, WHR reveals large abdominal fat "
            "burdens independent of BMI — the 'South Asian phenotype' of central obesity with "
            "normal-range BMI. Women show WHR ≈ 0.91 at median (clinically high)."
        )

        d = df.dropna(subset=["bmi","whr","risk_nonlab","_sex"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean BMI", f"{d['bmi'].mean():.1f} kg/m²")
        c2.metric("Mean WHR (Men)",   f"{d[d['_sex']=='men']['whr'].mean():.2f}")
        c3.metric("Mean WHR (Women)", f"{d[d['_sex']=='women']['whr'].mean():.2f}")
        abdob = d["_abdo_obes"].mean()*100 if "_abdo_obes" in d else 0
        c4.metric("Abdominal Obesity Rate", f"{abdob:.1f}%")

        st.markdown("---")

        # ── BMI cat distribution ──
        st.markdown("**BMI Category Distribution (Asian Thresholds)**")
        bmi_vc = d["_bmi_cat"].value_counts().reindex(
            ["Underweight","Normal","Overweight\n(Asian)","Overweight","Obese"]
        ).dropna()
        fig, ax = _fig(10, 4)
        colors_bmi = [_C["grey"],_C["nodiab"],_C["warn"],_C["diab"],_C["htn"]]
        bars = ax.bar(bmi_vc.index, bmi_vc.values, color=colors_bmi[:len(bmi_vc)], edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, bmi_vc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f"{v}\n({v/len(d)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Count"); ax.set_title("BMI Distribution (Asian cutoffs)", fontweight="bold")
        plt.tight_layout()
        _savefig(fig, "deep_bmi_dist")

        # ── Normal BMI but abdominal obese ──
        if "_abdo_obes" in d.columns and "_bmi_cat" in d.columns:
            paradox = d[d["_bmi_cat"].isin(["Normal","Underweight"]) & d["_abdo_obes"]]
            st.markdown(f"**'Thin-Fat' Paradox: Normal/Underweight BMI + Abdominal Obesity**")
            st.metric(
                "Individuals with BMI < 23 but high WHR",
                f"{len(paradox):,}  ({len(paradox)/len(d)*100:.1f}% of cohort)"
            )
            if len(paradox) > 0:
                par_vs_norm = pd.DataFrame({
                    "Group": ["Normal BMI, Normal WHR", "Normal BMI, High WHR (Thin-Fat)"],
                    "N": [
                        len(d[d["_bmi_cat"].isin(["Normal","Underweight"]) & ~d["_abdo_obes"]]),
                        len(paradox)
                    ],
                    "Mean CVD Risk": [
                        d[d["_bmi_cat"].isin(["Normal","Underweight"]) & ~d["_abdo_obes"]]["risk_nonlab"].mean(),
                        paradox["risk_nonlab"].mean()
                    ],
                    "≥10% HR rate": [
                        d[d["_bmi_cat"].isin(["Normal","Underweight"]) & ~d["_abdo_obes"]]["_hr10"].mean()*100,
                        paradox["_hr10"].mean()*100
                    ],
                }).round(2)
                st.dataframe(par_vs_norm, use_container_width=True, hide_index=True)

        # ── WHR vs CVD Risk scatter ──
        st.markdown("**WHR vs CVD Risk (sampled)**")
        fig2, ax2 = _fig(10, 5)
        for sex, col_s in [("men",_C["male"]),("women",_C["female"])]:
            sub = d[d["_sex"]==sex].sample(min(500, len(d[d["_sex"]==sex])), random_state=1)
            ax2.scatter(sub["whr"], sub["risk_nonlab"], alpha=0.4, s=14, c=col_s, label=sex.capitalize())
        ax2.axvline(0.90, color=_C["male"],   linestyle="--", linewidth=1, alpha=0.7, label="Men WHR threshold (0.90)")
        ax2.axvline(0.85, color=_C["female"], linestyle="--", linewidth=1, alpha=0.7, label="Women WHR threshold (0.85)")
        ax2.set_xlabel("Waist-Hip Ratio"); ax2.set_ylabel("CVD Risk (%)")
        ax2.set_title("WHR vs 10-year CVD Risk by Sex", fontweight="bold")
        ax2.legend(); plt.tight_layout()
        _savefig(fig2, "deep_whr_scatter")

        # ── BMI vs WHR risk table ──
        st.markdown("**Mean CVD Risk by BMI × WHR Category**")
        if "_abdo_obes" in d.columns and "_bmi_cat" in d.columns:
            d["_bmi_simple"] = d["_bmi_cat"].astype(str).map({
                "Normal": "BMI<23", "Underweight": "BMI<23",
                "Overweight\n(Asian)": "BMI 23-25", "Overweight": "BMI>25", "Obese": "BMI>25"
            })
            cross = d.groupby(["_bmi_simple","_abdo_obes"])["risk_nonlab"].agg(["mean","count"]).round(2)
            cross.index = cross.index.map(lambda x: f"{x[0]} / {'Abdominal Obese' if x[1] else 'Normal WHR'}")
            cross.columns = ["Mean CVD Risk", "N"]
            st.dataframe(cross, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: Blood Glucose Burden
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Blood Glucose Distribution & Pre-diabetes Burden")
        st.caption(
            "Median blood glucose is 113 mg/dL (men) and 108 mg/dL (women) — both in the "
            "pre-diabetic / impaired fasting range. A substantial proportion exceed diagnostic "
            "thresholds without formal diabetes diagnoses."
        )

        d = df.dropna(subset=["bg_mgdl","_sex"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median BG (Men)",   f"{d[d['_sex']=='men']['bg_mgdl'].median():.0f} mg/dL")
        c2.metric("Median BG (Women)", f"{d[d['_sex']=='women']['bg_mgdl'].median():.0f} mg/dL")
        prediab_m = (d[d["_sex"]=="men"]["bg_mgdl"].between(100,125)).mean()*100
        prediab_f = (d[d["_sex"]=="women"]["bg_mgdl"].between(100,125)).mean()*100
        c3.metric("Pre-diabetic range (Men)",   f"{prediab_m:.1f}%")
        c4.metric("Pre-diabetic range (Women)", f"{prediab_f:.1f}%")

        st.markdown("---")

        # ── BG category distribution ──
        st.markdown("**Blood Glucose Classification (Fasting-equivalent thresholds)**")
        bg_bins   = [0, 100, 126, 200, 9999]
        bg_labels = ["Normoglycaemia\n(<100)", "Pre-diabetes\n(100-125)", "Diabetes\n(126-199)", "Severe hyperglycaemia\n(≥200)"]
        d["_bg_cat"] = pd.cut(d["bg_mgdl"], bins=bg_bins, labels=bg_labels, right=False)

        fig, ax = _fig(11, 5)
        bg_m = d[d["_sex"]=="men"]["_bg_cat"].value_counts().reindex(bg_labels)
        bg_f = d[d["_sex"]=="women"]["_bg_cat"].value_counts().reindex(bg_labels)
        x = np.arange(len(bg_labels)); w = 0.35
        ax.bar(x-w/2, bg_m, w, label="Men",   color=_C["male"],   edgecolor="black", linewidth=0.5)
        ax.bar(x+w/2, bg_f, w, label="Women", color=_C["female"], edgecolor="black", linewidth=0.5)
        for i, (m_v, f_v) in enumerate(zip(bg_m.fillna(0), bg_f.fillna(0))):
            ax.text(x[i]-w/2, m_v+10, f"{m_v:.0f}", ha="center", fontsize=8, color=_C["male"])
            ax.text(x[i]+w/2, f_v+10, f"{f_v:.0f}", ha="center", fontsize=8, color=_C["female"])
        ax.set_xticks(x); ax.set_xticklabels(bg_labels, fontsize=9)
        ax.set_ylabel("Count"); ax.legend()
        ax.set_title("Blood Glucose Category Distribution by Sex", fontweight="bold")
        plt.tight_layout()
        _savefig(fig, "deep_bg_dist")

        # ── Mean CVD risk by BG category ──
        dd = d.dropna(subset=["risk_nonlab"])
        bg_risk = dd.groupby("_bg_cat")["risk_nonlab"].agg(["mean","median","count"]).round(2)
        bg_risk.columns = ["Mean CVD Risk", "Median CVD Risk", "N"]
        st.markdown("**Mean CVD Risk by Blood Glucose Category**")
        st.dataframe(bg_risk, use_container_width=True)

        # ── Hidden hyperglycaemia (glucose ≥126 but no diabetes diagnosis) ──
        if "_diab" in d.columns:
            hidden = d[(d["bg_mgdl"] >= 126) & (d["_diab"]==False)]
            st.markdown(f"**Undiagnosed Hyperglycaemia (BG ≥ 126 but no diabetes diagnosis): {len(hidden):,} "
                        f"({len(hidden)/len(d)*100:.1f}% of cohort)**")
            if len(hidden) > 0:
                st.dataframe(pd.DataFrame({
                    "Metric": ["N", "Mean CVD Risk", "≥10% HR Rate", "Median Age"],
                    "Value": [
                        len(hidden),
                        f"{hidden['risk_nonlab'].mean():.2f}%",
                        f"{hidden['_hr10'].mean()*100:.1f}%",
                        f"{hidden['age'].median():.0f} yrs",
                    ]
                }), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: Site Heterogeneity Deep-Dive
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Site-Level CVD Risk Heterogeneity")
        st.caption(
            "78 PHC sites range from mean CVD risk of 2.4% to 11.0% — a 4.6× spread. "
            "Site-level high-risk (≥10%) prevalences range 0%–50%. This massive heterogeneity "
            "suggests strong geographic, demographic, or structural confounders."
        )

        d = df.dropna(subset=["site_id","risk_nonlab"])

        site = d.groupby("site_id").agg(
            n        =("risk_nonlab","count"),
            mean_risk=("risk_nonlab","mean"),
            std_risk =("risk_nonlab","std"),
            pct_hr10 =("_hr10","mean"),
            pct_hr20 =("_hr20","mean"),
        ).reset_index()
        site["pct_hr10"] = site["pct_hr10"] * 100
        site["pct_hr20"] = site["pct_hr20"] * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Number of Sites", len(site))
        c2.metric("Site Risk Range", f"{site['mean_risk'].min():.1f}% – {site['mean_risk'].max():.1f}%")
        c3.metric("Coefficient of Variation", f"{site['mean_risk'].std()/site['mean_risk'].mean()*100:.1f}%")
        c4.metric("Max Site ≥10% HR Rate", f"{site['pct_hr10'].max():.1f}%")

        st.markdown("---")

        # ── Sorted site bar chart ──
        site_sorted = site.sort_values("mean_risk")
        fig, ax = _fig(14, 6)
        colors_s = ["#27ae60" if r < 5 else "#e67e22" if r < 8 else "#c0392b"
                    for r in site_sorted["mean_risk"]]
        bars = ax.bar(range(len(site_sorted)), site_sorted["mean_risk"], color=colors_s, edgecolor="none")
        ax.axhline(site_sorted["mean_risk"].mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {site_sorted['mean_risk'].mean():.2f}%")
        ax.set_xlabel("Sites (ranked by mean risk)"); ax.set_ylabel("Mean 10-year CVD Risk (%)")
        ax.set_title("Site-Level CVD Risk Ranked (Green <5%, Orange 5-8%, Red >8%)", fontweight="bold")
        ax.legend()
        patches = [mpatches.Patch(color="#27ae60", label="<5%"),
                   mpatches.Patch(color="#e67e22", label="5–8%"),
                   mpatches.Patch(color="#c0392b", label=">8%")]
        ax.legend(handles=patches + [plt.Line2D([0],[0],color="black",linestyle="--",label="Mean")])
        plt.tight_layout()
        _savefig(fig, "deep_site_ranked")

        # ── Top / bottom sites table ──
        st.markdown("**Top-10 Highest Risk Sites**")
        st.dataframe(
            site.nlargest(10,"mean_risk")[["site_id","n","mean_risk","pct_hr10","pct_hr20"]]
            .rename(columns={"site_id":"Site","n":"N","mean_risk":"Mean Risk (%)","pct_hr10":"≥10% HR (%)","pct_hr20":"≥20% HR (%)"})
            .round(1), use_container_width=True, hide_index=True
        )
        st.markdown("**Bottom-10 Lowest Risk Sites**")
        st.dataframe(
            site.nsmallest(10,"mean_risk")[["site_id","n","mean_risk","pct_hr10","pct_hr20"]]
            .rename(columns={"site_id":"Site","n":"N","mean_risk":"Mean Risk (%)","pct_hr10":"≥10% HR (%)","pct_hr20":"≥20% HR (%)"})
            .round(1), use_container_width=True, hide_index=True
        )

        # ── Scatter: site size vs mean risk ──
        st.markdown("**Site Size vs Mean Risk (bubble = ≥10% HR rate)**")
        fig2, ax2 = _fig(9, 5)
        sc = ax2.scatter(site["n"], site["mean_risk"],
                         s=site["pct_hr10"]*10 + 20,
                         c=site["pct_hr10"], cmap="RdYlGn_r",
                         alpha=0.75, edgecolors="black", linewidth=0.4)
        plt.colorbar(sc, ax=ax2, label="≥10% HR Rate (%)")
        ax2.set_xlabel("Site Sample Size (n)"); ax2.set_ylabel("Mean CVD Risk (%)")
        ax2.set_title("Site Size vs Mean CVD Risk (bubble size = ≥10% HR%)", fontweight="bold")
        plt.tight_layout()
        _savefig(fig2, "deep_site_scatter")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6: Co-morbidity Clustering
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Co-morbidity Clustering")
        st.caption(
            "Diabetes, hypertension, abdominal obesity, and hyperglycaemia cluster together. "
            "This section enumerates all 2- and 3-way combinations and their CVD risk burden."
        )

        d = df.dropna(subset=["risk_nonlab","sbp"])
        d["_htn"] = (d["sbp"] >= 140).astype(bool)

        flags = {"Diabetes": "_diab", "HTN": "_htn", "Abdominal Obesity": "_abdo_obes"}
        avail = {k: v for k, v in flags.items() if v in d.columns}

        # ── Single-flag rates ──
        st.markdown("**Single Co-morbidity Prevalence**")
        single_rows = []
        for lbl, col in avail.items():
            s = d[col]
            single_rows.append({
                "Condition": lbl,
                "N": s.sum(),
                "Prevalence (%)": f"{s.mean()*100:.1f}%",
                "Mean CVD Risk (with)": f"{d[s==True]['risk_nonlab'].mean():.2f}%",
                "Mean CVD Risk (without)": f"{d[s==False]['risk_nonlab'].mean():.2f}%",
                "Risk Ratio": f"{d[s==True]['risk_nonlab'].mean() / max(d[s==False]['risk_nonlab'].mean(), 0.01):.2f}×",
            })
        st.dataframe(pd.DataFrame(single_rows), use_container_width=True, hide_index=True)

        # ── 2-way combinations ──
        st.markdown("**Two-way Co-morbidity Combinations**")
        combs2 = []
        keys = list(avail.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a, b = keys[i], keys[j]
                mask = d[avail[a]] & d[avail[b]]
                if mask.sum() == 0:
                    continue
                combs2.append({
                    "Combination": f"{a} + {b}",
                    "N": mask.sum(),
                    "% of cohort": f"{mask.mean()*100:.1f}%",
                    "Mean CVD Risk": f"{d[mask]['risk_nonlab'].mean():.2f}%",
                    "≥10% HR (%)": f"{d[mask]['_hr10'].mean()*100:.1f}%",
                })
        if combs2:
            st.dataframe(pd.DataFrame(combs2), use_container_width=True, hide_index=True)

        # ── Triple ──
        if len(avail) >= 3:
            st.markdown("**Triple Co-morbidity**")
            cols_v = list(avail.values())
            triple = d[cols_v[0]] & d[cols_v[1]] & d[cols_v[2]]
            st.metric(
                "All-three co-morbid (Diabetes + HTN + Abdominal Obesity)",
                f"{triple.sum():,}  ({triple.mean()*100:.1f}%)"
            )
            if triple.sum() > 0:
                st.metric("Mean CVD Risk (triple co-morbidity)", f"{d[triple]['risk_nonlab'].mean():.2f}%")

        # ── Heatmap: risk by diabetes × HTN ──
        st.markdown("**Mean CVD Risk Heatmap: Diabetes × HTN**")
        pivot = d.groupby(["_diab","_htn"])["risk_nonlab"].mean().unstack()
        pivot.index = ["No Diabetes","Diabetes"]
        pivot.columns = ["No HTN","HTN"]
        fig, ax = _fig(6, 3.5)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label":"Mean CVD Risk (%)"}, annot_kws={"size":13})
        ax.set_title("Mean CVD Risk: Diabetes × HTN", fontweight="bold")
        plt.tight_layout()
        _savefig(fig, "deep_comorbidity_heatmap")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7: Smoking & Gender Gap
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Smoking Prevalence & Gender Gap")
        st.caption(
            "Smoking is remarkably absent in women — only 2/6,505 (0.03%) are recorded as smokers, "
            "vs 83/7,580 (1.1%) of men. This extreme gender gap reflects cultural norms in Bangladesh "
            "but may also reflect under-reporting. Notably, even the low male rate is well below "
            "national estimates (~40% of adult men smoke), which may reflect PHC selection bias."
        )

        d = df.dropna(subset=["_sex","_smoke"])

        # ── metrics ──
        c1, c2, c3, c4 = st.columns(4)
        m_smoke = d[d["_sex"]=="men"]["_smoke"].mean()*100
        f_smoke = d[d["_sex"]=="women"]["_smoke"].mean()*100
        c1.metric("Smoking rate — Men",   f"{m_smoke:.2f}%")
        c2.metric("Smoking rate — Women", f"{f_smoke:.2f}%")
        c3.metric("Male:Female ratio", f"{m_smoke/max(f_smoke,0.001):.0f}:1")
        c4.metric("Expected national (~40%)", "Potential under-reporting ⚠️")

        st.markdown("---")
        st.warning(
            "⚠️ **Under-reporting likely**: National Bangladesh survey data (GATS 2009) report "
            "~43.3% of adult men currently use tobacco. PHC cohort shows 1.1% — a 39pp gap. "
            "This likely reflects: (1) PHC intake survey framing, (2) social desirability bias, "
            "or (3) selection of healthier patients at PHCs."
        )

        # ── smoking × CVD risk ──
        st.markdown("**CVD Risk: Smokers vs Non-Smokers**")
        d_r = d.dropna(subset=["risk_nonlab"])
        smoke_risk = d_r.groupby(["_sex","_smoke"])["risk_nonlab"].agg(["mean","median","count"]).round(2)
        smoke_risk.index = smoke_risk.index.map(
            lambda x: f"{x[0].capitalize()} – {'Smoker' if x[1] else 'Non-smoker'}"
        )
        smoke_risk.columns = ["Mean CVD Risk","Median CVD Risk","N"]
        st.dataframe(smoke_risk, use_container_width=True)

        # ── Age distribution of smokers ──
        smokers = d[(d["_smoke"]==True) & d["age"].notna()]
        if len(smokers) > 5:
            st.markdown("**Age Distribution of Smokers**")
            fig, ax = _fig(8, 4)
            for sex, col_s in [("men",_C["male"])]:   # women too few
                sub = smokers[smokers["_sex"]==sex]
                if len(sub) > 2:
                    ax.hist(sub["age"], bins=10, color=col_s, alpha=0.7, label=sex.capitalize(), edgecolor="black")
            ax.set_xlabel("Age"); ax.set_ylabel("Count")
            ax.set_title("Age Distribution of Smokers (Men)", fontweight="bold")
            ax.legend(); plt.tight_layout()
            _savefig(fig, "deep_smoker_age")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 8: Arrhythmia & Pulse
    # ──────────────────────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Arrhythmia & Elevated Pulse Analysis")
        st.caption(
            "625 individuals (5.1% of those with ECG data) have abnormal arrhythmia findings. "
            "Women have a slightly higher arrhythmia rate than men (5.9% vs 4.4%). "
            "Elevated resting pulse rate (>100 bpm) is a known CVD risk modifier."
        )

        d = df.dropna(subset=["_sex"])

        c1, c2, c3, c4 = st.columns(4)
        if "arrhythmia" in d.columns:
            arrh = d["arrhythmia"].dropna()
            rate = (arrh == "Abnormal").mean() * 100
            c1.metric("Overall Arrhythmia Rate", f"{rate:.1f}%")
            m_r = (d[d["_sex"]=="men"]["arrhythmia"] == "Abnormal").mean()*100
            f_r = (d[d["_sex"]=="women"]["arrhythmia"] == "Abnormal").mean()*100
            c2.metric("Arrhythmia — Men",   f"{m_r:.1f}%")
            c3.metric("Arrhythmia — Women", f"{f_r:.1f}%")
            c4.metric("F > M by",           f"{f_r - m_r:+.1f} pp")

        st.markdown("---")

        # ── Arrhythmia × CVD risk ──
        if "arrhythmia" in d.columns:
            d_r = d.dropna(subset=["risk_nonlab","arrhythmia"])
            arrh_risk = d_r.groupby(["_sex","arrhythmia"])["risk_nonlab"].agg(["mean","median","count"]).round(2)
            arrh_risk.index = arrh_risk.index.map(lambda x: f"{x[0].capitalize()} – {x[1]}")
            arrh_risk.columns = ["Mean CVD Risk","Median CVD Risk","N"]
            st.markdown("**CVD Risk by Arrhythmia Status**")
            st.dataframe(arrh_risk, use_container_width=True)

        # ── Pulse distribution ──
        if "pulse" in d.columns:
            d_p = d.dropna(subset=["pulse"])
            st.markdown("**Resting Pulse Distribution**")
            fig, ax = _fig(10, 4)
            for sex, col_s in [("men",_C["male"]),("women",_C["female"])]:
                sub = d_p[d_p["_sex"]==sex]["pulse"]
                ax.hist(sub, bins=30, alpha=0.6, color=col_s, label=sex.capitalize(), edgecolor="black", linewidth=0.3)
            ax.axvline(100, color="red", linestyle="--", linewidth=1.2, label="Tachycardia threshold (100 bpm)")
            ax.set_xlabel("Resting Pulse (bpm)"); ax.set_ylabel("Count")
            ax.set_title("Resting Pulse Distribution by Sex", fontweight="bold")
            ax.legend(); plt.tight_layout()
            _savefig(fig, "deep_pulse_dist")

            # tachycardia
            tachy_m = (d_p[d_p["_sex"]=="men"]["pulse"] > 100).mean()*100
            tachy_f = (d_p[d_p["_sex"]=="women"]["pulse"] > 100).mean()*100
            cc1, cc2 = st.columns(2)
            cc1.metric("Tachycardia (>100 bpm) — Men",   f"{tachy_m:.1f}%")
            cc2.metric("Tachycardia (>100 bpm) — Women", f"{tachy_f:.1f}%")

            # Pulse × CVD risk
            d_pr = d_p.dropna(subset=["risk_nonlab"])
            d_pr["_pulse_cat"] = pd.cut(d_pr["pulse"],
                                        bins=[0,60,80,100,200],
                                        labels=["Bradycardia\n(<60)","Normal\n(60-79)","High-Normal\n(80-99)","Tachycardia\n(≥100)"],
                                        right=False)
            pulse_risk = d_pr.groupby("_pulse_cat")["risk_nonlab"].agg(["mean","count"]).round(2)
            pulse_risk.columns = ["Mean CVD Risk","N"]
            st.markdown("**Mean CVD Risk by Pulse Category**")
            st.dataframe(pulse_risk, use_container_width=True)

        # ── Interpretation ──
        st.markdown("---")
        with st.expander("📝 Clinical Interpretation", expanded=False):
            st.markdown("""
| Finding | Clinical Significance |
|---------|----------------------|
| **5.1% Arrhythmia** | Likely AF/flutter or PVCs detected on spot ECG; warrants follow-up 12-lead ECG and Holter monitoring |
| **Higher in women** | Female-excess arrhythmia may reflect SVT predominance or anxiety-related tachycardia |
| **Pulse >100 bpm** | Tachycardia is independently associated with higher CVD mortality (HR ~1.3× per 10 bpm increment) |
| **Arrhythmia + High Risk** | Combination warrants urgent cardiology referral under PHC protocol |
            """)
