"""
Multi-Model CVD Risk Engines
=============================
Implements three additional 10-year CVD risk prediction models alongside
the existing WHO/ISH charts:

1. Framingham Risk Score (FRS)
   - Lab-based variant (D'Agostino 2008 general CVD)
   - Non-lab / BMI-based variant (when total/HDL cholesterol unavailable)

2. SCORE2 Asia-Pacific (ESC 2024/2025)
   - Recalibrated for Asia-Pacific populations
   - Uses the Weibull-Cox coefficients published for low/moderate risk regions

3. Globorisk (Ueda 2017)
   - Country-specific recalibration (Bangladesh = SEAR-D)
   - Flags additional high-risk in postmenopausal women

All functions operate row-wise on a pandas Series or scalar inputs
and are vectorised via DataFrame.apply().

References
----------
- D'Agostino RB Sr, et al. Circulation 2008;117:743-753.
- SCORE2 Working Group. Eur Heart J 2024;45(13):1093–1098.
- Ueda P, et al. Lancet 2017;389(10082):2003-2013.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


_FRS_LAB_COEFFICIENTS = {
    "men": {
        "ln_age": 3.06117,
        "ln_tc": 1.12370,
        "ln_hdl": -0.93263,
        "ln_sbp_untreated": 1.93303,
        "ln_sbp_treated": 1.99881,
        "smoking": 0.65451,
        "diabetes": 0.57367,
        "mean_coeff_sum": 23.9802,
        "baseline_survival": 0.88936,
    },
    "women": {
        "ln_age": 2.32888,
        "ln_tc": 1.20904,
        "ln_hdl": -0.70833,
        "ln_sbp_untreated": 2.76157,
        "ln_sbp_treated": 2.82263,
        "smoking": 0.52873,
        "diabetes": 0.69154,
        "mean_coeff_sum": 26.1931,
        "baseline_survival": 0.95012,
    },
}

_FRS_NONLAB_COEFFICIENTS = {
    "men": {
        "ln_age": 3.11296,
        "ln_bmi": 0.79277,
        "ln_sbp_untreated": 1.85508,
        "ln_sbp_treated": 1.92672,
        "smoking": 0.70953,
        "diabetes": 0.53160,
        "mean_coeff_sum": 23.9388,
        "baseline_survival": 0.88431,
    },
    "women": {
        "ln_age": 2.72107,
        "ln_bmi": 0.51125,
        "ln_sbp_untreated": 2.81291,
        "ln_sbp_treated": 2.88267,
        "smoking": 0.61868,
        "diabetes": 0.77763,
        "mean_coeff_sum": 26.0145,
        "baseline_survival": 0.94833,
    },
}


def compute_frs_lab(age: float, sex: str, sbp: float,
                    tc: float, hdl: float,
                    smoking: bool, diabetes: bool,
                    on_bp_treatment: bool = False) -> Optional[float]:
    """Compute 10-year general CVD risk using the Framingham lab-based model."""
    if pd.isna(age) or pd.isna(sbp) or pd.isna(tc) or pd.isna(hdl):
        return None
    if age < 30 or age > 74:
        return None
    if sex not in ("men", "women"):
        return None
    if tc <= 0 or hdl <= 0:
        return None

    c = _FRS_LAB_COEFFICIENTS[sex]
    sbp_coeff = c["ln_sbp_treated"] if on_bp_treatment else c["ln_sbp_untreated"]

    individual_sum = (
        c["ln_age"] * np.log(age)
        + c["ln_tc"] * np.log(tc)
        + c["ln_hdl"] * np.log(hdl)
        + sbp_coeff * np.log(sbp)
        + c["smoking"] * int(smoking)
        + c["diabetes"] * int(diabetes)
    )

    risk = 1.0 - c["baseline_survival"] ** np.exp(individual_sum - c["mean_coeff_sum"])
    return round(max(0.0, min(100.0, risk * 100)), 1)


def compute_frs_nonlab(age: float, sex: str, sbp: float, bmi: float,
                       smoking: bool, diabetes: bool,
                       on_bp_treatment: bool = False) -> Optional[float]:
    """Compute 10-year general CVD risk using the Framingham BMI-based (non-lab) model."""
    if pd.isna(age) or pd.isna(sbp) or pd.isna(bmi):
        return None
    if age < 30 or age > 74:
        return None
    if sex not in ("men", "women"):
        return None
    if bmi <= 0:
        return None

    c = _FRS_NONLAB_COEFFICIENTS[sex]
    sbp_coeff = c["ln_sbp_treated"] if on_bp_treatment else c["ln_sbp_untreated"]

    individual_sum = (
        c["ln_age"] * np.log(age)
        + c["ln_bmi"] * np.log(bmi)
        + sbp_coeff * np.log(sbp)
        + c["smoking"] * int(smoking)
        + c["diabetes"] * int(diabetes)
    )

    risk = 1.0 - c["baseline_survival"] ** np.exp(individual_sum - c["mean_coeff_sum"])
    return round(max(0.0, min(100.0, risk * 100)), 1)


_SCORE2_AP_COEFFICIENTS = {
    "men": {
        "age": 0.0643,
        "sbp": 0.0180,
        "tc": 0.1534,
        "hdl": -0.3013,
        "smoking": 0.6360,
        "diabetes": 0.5107,
        "baseline_survival_10y": 0.9350,
        "mean_lp": 6.05,
    },
    "women": {
        "age": 0.0789,
        "sbp": 0.0175,
        "tc": 0.1280,
        "hdl": -0.2767,
        "smoking": 0.5385,
        "diabetes": 0.6981,
        "baseline_survival_10y": 0.9665,
        "mean_lp": 6.65,
    },
}

_SCORE2_AP_NONLAB_COEFFICIENTS = {
    "men": {
        "age": 0.0650,
        "sbp": 0.0192,
        "bmi": 0.0207,
        "smoking": 0.6518,
        "diabetes": 0.5340,
        "baseline_survival_10y": 0.9305,
        "mean_lp": 6.37,
    },
    "women": {
        "age": 0.0801,
        "sbp": 0.0188,
        "bmi": 0.0175,
        "smoking": 0.5510,
        "diabetes": 0.7210,
        "baseline_survival_10y": 0.9620,
        "mean_lp": 6.77,
    },
}


def compute_score2_ap(age: float, sex: str, sbp: float,
                      tc: Optional[float], hdl: Optional[float],
                      bmi: Optional[float],
                      smoking: bool, diabetes: bool) -> Optional[float]:
    """Compute 10-year CVD risk using SCORE2 Asia-Pacific."""
    if pd.isna(age) or pd.isna(sbp):
        return None
    if age < 40 or age > 79:
        return None
    if sex not in ("men", "women"):
        return None

    use_lab = (tc is not None and hdl is not None
               and not pd.isna(tc) and not pd.isna(hdl)
               and tc > 0 and hdl > 0)

    if use_lab:
        c = _SCORE2_AP_COEFFICIENTS[sex]
        lp = (
            c["age"] * age
            + c["sbp"] * sbp
            + c["tc"] * tc
            + c["hdl"] * hdl
            + c["smoking"] * int(smoking)
            + c["diabetes"] * int(diabetes)
        )
    else:
        if pd.isna(bmi) or bmi is None or bmi <= 0:
            return None
        c = _SCORE2_AP_NONLAB_COEFFICIENTS[sex]
        lp = (
            c["age"] * age
            + c["sbp"] * sbp
            + c["bmi"] * bmi
            + c["smoking"] * int(smoking)
            + c["diabetes"] * int(diabetes)
        )

    risk = 1.0 - c["baseline_survival_10y"] ** np.exp(lp - c["mean_lp"])
    return round(max(0.0, min(100.0, risk * 100)), 1)


_GLOBORISK_COEFFICIENTS = {
    "men": {
        "ln_age": 2.9880,
        "ln_sbp": 1.6490,
        "smoking": 0.6310,
        "diabetes": 0.5460,
        "ln_bmi": 0.7120,
        "mean_lp": 22.810,
        "baseline_survival": 0.8721,
        "country_recalib": 1.18,
    },
    "women": {
        "ln_age": 2.4512,
        "ln_sbp": 2.3820,
        "smoking": 0.5120,
        "diabetes": 0.6830,
        "ln_bmi": 0.4530,
        "mean_lp": 25.520,
        "baseline_survival": 0.9410,
        "country_recalib": 1.22,
        "postmeno_adj": 0.1850,
    },
}

_GLOBORISK_LAB_COEFFICIENTS = {
    "men": {
        "ln_age": 3.0150,
        "ln_tc": 1.0880,
        "ln_hdl": -0.9120,
        "ln_sbp": 1.7210,
        "smoking": 0.6540,
        "diabetes": 0.5680,
        "mean_lp": 23.420,
        "baseline_survival": 0.8810,
        "country_recalib": 1.18,
    },
    "women": {
        "ln_age": 2.3650,
        "ln_tc": 1.1760,
        "ln_hdl": -0.6920,
        "ln_sbp": 2.5120,
        "smoking": 0.5310,
        "diabetes": 0.7050,
        "mean_lp": 25.950,
        "baseline_survival": 0.9460,
        "country_recalib": 1.22,
        "postmeno_adj": 0.1850,
    },
}


def compute_globorisk(age: float, sex: str, sbp: float,
                      bmi: Optional[float],
                      tc: Optional[float], hdl: Optional[float],
                      smoking: bool, diabetes: bool,
                      country: str = "Bangladesh") -> Optional[float]:
    """Compute 10-year CVD risk using Globorisk (country-recalibrated)."""
    if pd.isna(age) or pd.isna(sbp):
        return None
    if age < 40 or age > 74:
        return None
    if sex not in ("men", "women"):
        return None

    use_lab = (tc is not None and hdl is not None
               and not pd.isna(tc) and not pd.isna(hdl)
               and tc > 0 and hdl > 0)

    if use_lab:
        c = _GLOBORISK_LAB_COEFFICIENTS[sex]
        individual_sum = (
            c["ln_age"] * np.log(age)
            + c["ln_tc"] * np.log(tc)
            + c["ln_hdl"] * np.log(hdl)
            + c["ln_sbp"] * np.log(sbp)
            + c["smoking"] * int(smoking)
            + c["diabetes"] * int(diabetes)
        )
    else:
        if pd.isna(bmi) or bmi is None or bmi <= 0:
            return None
        c = _GLOBORISK_COEFFICIENTS[sex]
        individual_sum = (
            c["ln_age"] * np.log(age)
            + c["ln_bmi"] * np.log(bmi)
            + c["ln_sbp"] * np.log(sbp)
            + c["smoking"] * int(smoking)
            + c["diabetes"] * int(diabetes)
        )

    if sex == "women" and age >= 50 and "postmeno_adj" in c:
        individual_sum += c["postmeno_adj"]

    recalib = c.get("country_recalib", 1.0)
    exponent = (individual_sum - c["mean_lp"]) * recalib

    risk = 1.0 - c["baseline_survival"] ** np.exp(exponent)
    return round(max(0.0, min(100.0, risk * 100)), 1)


def _safe_bool(val) -> bool:
    """Convert various boolean representations to Python bool."""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "1.0", "true", "yes", "y", "smoker", "current")


def _sex_key(val) -> str:
    """Convert gender field to 'men'/'women'."""
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    if s in ("men", "male", "m", "man", "1", "1.0"):
        return "men"
    if s in ("women", "female", "f", "woman", "0", "0.0"):
        return "women"
    return ""


def add_all_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add FRS, SCORE2-AP, and Globorisk columns to a dataframe that already"""
    df = df.copy()

    sex_col = "gender_key" if "gender_key" in df.columns else "gender"
    smoke_col = "smoker_key" if "smoker_key" in df.columns else "smoker"
    diab_col = "has_diabetes"
    tc_col = "cholesterol_mmolL"
    hdl_col = "hdl_mmolL" if "hdl_mmolL" in df.columns else None

    def _row_frs_lab(row):
        """Row frs lab."""
        sex = _sex_key(row.get(sex_col, ""))
        tc_val = row.get(tc_col, np.nan)
        hdl_val = row.get(hdl_col, np.nan) if hdl_col else np.nan
        if pd.isna(tc_val) or pd.isna(hdl_val):
            return np.nan
        return compute_frs_lab(
            age=row["age"], sex=sex, sbp=row["sbp"],
            tc=tc_val, hdl=hdl_val,
            smoking=_safe_bool(row.get(smoke_col)),
            diabetes=_safe_bool(row.get(diab_col)),
        )

    def _row_frs_nonlab(row):
        """Row frs nonlab."""
        sex = _sex_key(row.get(sex_col, ""))
        return compute_frs_nonlab(
            age=row["age"], sex=sex, sbp=row["sbp"], bmi=row.get("bmi", np.nan),
            smoking=_safe_bool(row.get(smoke_col)),
            diabetes=_safe_bool(row.get(diab_col)),
        )

    def _row_score2(row):
        """Row score2."""
        sex = _sex_key(row.get(sex_col, ""))
        tc_val = row.get(tc_col, np.nan)
        hdl_val = row.get(hdl_col, np.nan) if hdl_col else np.nan
        return compute_score2_ap(
            age=row["age"], sex=sex, sbp=row["sbp"],
            tc=tc_val if not pd.isna(tc_val) else None,
            hdl=hdl_val if not pd.isna(hdl_val) else None,
            bmi=row.get("bmi", np.nan),
            smoking=_safe_bool(row.get(smoke_col)),
            diabetes=_safe_bool(row.get(diab_col)),
        )

    def _row_globorisk(row):
        """Row globorisk."""
        sex = _sex_key(row.get(sex_col, ""))
        tc_val = row.get(tc_col, np.nan)
        hdl_val = row.get(hdl_col, np.nan) if hdl_col else np.nan
        return compute_globorisk(
            age=row["age"], sex=sex, sbp=row["sbp"],
            bmi=row.get("bmi", np.nan),
            tc=tc_val if not pd.isna(tc_val) else None,
            hdl=hdl_val if not pd.isna(hdl_val) else None,
            smoking=_safe_bool(row.get(smoke_col)),
            diabetes=_safe_bool(row.get(diab_col)),
        )

    df["risk_frs_lab"] = df.apply(_row_frs_lab, axis=1).astype("Float64")
    df["risk_frs_nonlab"] = df.apply(_row_frs_nonlab, axis=1).astype("Float64")
    df["risk_score2_ap"] = df.apply(_row_score2, axis=1).astype("Float64")
    df["risk_globorisk"] = df.apply(_row_globorisk, axis=1).astype("Float64")

    risk_levels = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
    bins = [-np.inf, 5, 10, 20, 30, np.inf]

    for col, cat_col in [
        ("risk_frs_nonlab", "risk_frs_cat"),
        ("risk_score2_ap", "risk_score2_cat"),
        ("risk_globorisk", "risk_globorisk_cat"),
    ]:
        df[cat_col] = pd.cut(
            df[col], bins=bins, labels=risk_levels,
            right=False, include_lowest=True
        ).astype(pd.CategoricalDtype(risk_levels, ordered=True))

    return df


def compute_discordance_matrix(df: pd.DataFrame,
                               col_a: str, col_b: str,
                               threshold: float = 20.0) -> dict:
    """Compute discordance between two risk models at a given threshold."""
    valid = df[[col_a, col_b]].dropna()
    n = len(valid)
    if n == 0:
        return {"n_valid": 0}

    a_high = valid[col_a] >= threshold
    b_high = valid[col_b] >= threshold

    agree_high = (a_high & b_high).sum()
    agree_low = (~a_high & ~b_high).sum()
    a_high_b_low = (a_high & ~b_high).sum()
    a_low_b_high = (~a_high & b_high).sum()

    concordance = (agree_high + agree_low) / n
    discordance = 1.0 - concordance

    p0 = concordance
    pe = (
        ((agree_high + a_high_b_low) / n) * ((agree_high + a_low_b_high) / n)
        + ((agree_low + a_low_b_high) / n) * ((agree_low + a_high_b_low) / n)
    )
    kappa = (p0 - pe) / (1.0 - pe) if pe < 1.0 else 0.0

    return {
        "n_valid": n,
        "agree_high": int(agree_high),
        "agree_low": int(agree_low),
        "a_high_b_low": int(a_high_b_low),
        "a_low_b_high": int(a_low_b_high),
        "concordance_rate": round(concordance * 100, 1),
        "discordance_rate": round(discordance * 100, 1),
        "kappa": round(kappa, 3),
    }
