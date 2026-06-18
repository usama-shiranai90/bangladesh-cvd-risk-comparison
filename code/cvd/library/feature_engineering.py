"""
Feature engineering utilities for the CVD project.

This module re-exports feature engineering related helpers from data_utils to
provide a cleaner, responsibility-separated API without breaking backward compatibility.

It also contains the WHO CVD Risk calculation logic.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

from .data_utils import (
    normalize_bg_type,
    classify_pbs_mgdl,
    normcols,
    parse_dt,
    to_numeric,
    first_existing,
    to_numeric_safe,
    gender_std,
    blood_group_std,
    marital_std,
    age_bucket,
    bmi_class,
    district_from_address,
    yn,
    gtype,
    bp_category,
    detect_temp_unit,
    detect_glu_unit,
    norm_sex,
    norm_smoker_from_status,
    who_age_band,
    who_sbp_band,
    bmi_idx,
    chol_idx,
    diabetes_key,
    fix_id_dtype,
)

# Import huge risk tables separately if needed, or rely on passing them in
# from .risk_data import risk_data  # Uncomment if we want to bind it strictly here

__all__ = [
    'normalize_bg_type',
    'classify_pbs_mgdl',
    'normcols',
    'parse_dt',
    'to_numeric',
    'first_existing',
    'to_numeric_safe',
    'gender_std',
    'blood_group_std',
    'marital_std',
    'age_bucket',
    'bmi_class',
    'district_from_address',
    'yn',
    'gtype',
    'bp_category',
    'detect_temp_unit',
    'detect_glu_unit',
    'norm_sex',
    'norm_smoker_from_status',
    'who_age_band',
    'who_sbp_band',
    'bmi_idx',
    'chol_idx',
    'diabetes_key',
    'fix_id_dtype',
    'encode_categoricals',
    'prepare_who_analysis_df',
    'add_who_risks',
    'standardize_core_fields',
    'add_cholesterol_mmolL',
    'build_cohorts',
    'add_who_bins_and_domain_flags',
]


def encode_categoricals(df, categorical_columns):
    df = df.copy()
    encoders = {}

    # --- PRE-PROCESSING FIX ---
    # Convert "Unknown" text to real NaNs so they don't get encoded
    cleanup_vals = ["Unknown", "unknown", "UNK", "None"]  # Add your specific label
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].replace(cleanup_vals, np.nan)

    for col in categorical_columns:
        if col in df.columns:
            # Only encode non-NaN values to preserve NaNs
            valid_mask = df[col].notna()
            le = LabelEncoder()
            # Fit only on valid text (Yes/No)
            df.loc[valid_mask, col] = le.fit_transform(df.loc[valid_mask, col].astype(str))
            # Ensure the column is numeric (object -> float to hold NaNs)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            encoders[col] = le

    return df, encoders


# ============================================================
# WHO CVD Risk (Non-lab + Lab) — LOGIC
# ============================================================

STR = "string"

AGE_BINS = [40, 45, 50, 55, 60, 65, 70, 75]
AGE_LABELS = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]

SBP_BINS = [-np.inf, 120, 140, 160, 180, np.inf]
SBP_LABELS = ["<120", "120-139", "140-159", "160-179", ">="]

BMI_BINS = [-np.inf, 20, 25, 30, 35, np.inf]  # -> 0..4
BMI_LABELS = ["<20", "20-24", "25-29", "30-34", ">=35"]

CHOL_BINS = [-np.inf, 4, 5, 6, 7, np.inf]  # mmol/L -> 0..4
CHOL_LABELS = ["<4", "4-4.9", "5-5.9", "6-6.9", ">=7"]

RISK_LEVELS = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]


# ----------------------------
# B) Small utils (DEFINE ONCE)
# ----------------------------
def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=np.nan) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns created as NA: {missing}")
        for c in missing:
            df[c] = fill
    return df


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_bool_na(s: pd.Series) -> pd.Series:
    """Coerce to pandas BooleanDtype, keeping NA."""
    if s is None:
        return pd.Series(pd.NA, dtype="boolean")
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    x = s.astype(STR).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n"}
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[x.isin(true_set)] = True
    out[x.isin(false_set)] = False
    return out


def normalize_gender_key(s: pd.Series) -> pd.Series:
    """Return 'men'/'women'/NA."""
    x = s.astype(STR).str.strip().str.lower()
    out = x.map({
        "m": "men", "male": "men", "man": "men", "men": "men" , "1": "men", "1.0": "men",
        "f": "women", "female": "women", "woman": "women", "women": "women", "0": "men", "0.0": "men",
    })
    return out.astype(STR)


def _normalize_yes_no_na(s: pd.Series) -> pd.Series:
    """Return 'yes'/'no'/NA. (WHO lookup needs exact yes/no)"""
    x = s.astype(STR).str.strip().str.lower()
    yes = x.isin(["yes", "y", "1", "true", "smoker", "current"])
    no = x.isin(["no", "n", "0", "false", "non-smoker", "nonsmoker", "non smoker", "never", "ex-smoker", "exsmoker"])
    out = pd.Series(pd.NA, index=s.index, dtype=STR)
    out[yes] = "yes"
    out[no] = "no"
    return out


def normalize_smoker_key(smoker: pd.Series, smoker_who: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Row-wise preference:
      - use smoker_who if present for that row
      - else fallback to smoker
    Returns: (smoker_key, smoker_key_source)
    """
    s1 = _normalize_yes_no_na(smoker) if smoker is not None else pd.Series(pd.NA, dtype=STR)
    if smoker_who is None:
        src = pd.Series(np.where(s1.notna(), "smoker", "missing"), index=s1.index, dtype=STR)
        return s1, src

    s2 = _normalize_yes_no_na(smoker_who)
    key = s2.where(s2.notna(), s1)

    src = pd.Series("missing", index=key.index, dtype=STR)
    src[s1.notna()] = "smoker"
    src[s2.notna()] = "smoker_who"
    return key, src


def infer_cholesterol_mmolL(chol: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Heuristic:
      - if median >= 25 -> assume mg/dL, convert: mmol/L = mg/dL / 38.67
      - else assume mmol/L
    """
    x = _to_num(chol)
    med = x.dropna().median()
    unit = pd.Series(pd.NA, index=x.index, dtype=STR)
    if pd.isna(med):
        return x, unit
    if med >= 25:
        return x / 38.67, pd.Series("mg/dL→mmol/L (inferred)", index=x.index, dtype=STR)
    return x, pd.Series("mmol/L (inferred)", index=x.index, dtype=STR)


def to_index(series: pd.Series, bins: list[float]) -> pd.Series:
    """Bin -> index 0..4 (Int64)."""
    x = _to_num(series)
    return pd.cut(x, bins=bins, labels=[0, 1, 2, 3, 4], right=False).astype("Int64")


def risk_bucket(series: pd.Series) -> pd.Series:
    x = _to_num(series)
    return pd.cut(
        x,
        bins=[-np.inf, 5, 10, 20, 30, np.inf],
        labels=RISK_LEVELS,
        right=False,
        include_lowest=True
    ).astype(pd.CategoricalDtype(RISK_LEVELS, ordered=True))


# ----------------------------
# C) WHO risk lookup map builders (DEFINE ONCE)
# ----------------------------
def build_nonlab_map(risk_data: dict) -> Dict[tuple, float]:
    """(sex, smoker, age_band, sbp_band, bmi_i) -> risk%"""
    out: Dict[tuple, float] = {}
    for sx, d1 in risk_data["non_lab"].items():
        for sm, d2 in d1.items():
            for ab, d3 in d2.items():
                for sb, arr in d3.items():
                    for bmi_i, val in enumerate(arr):
                        out[(sx, sm, ab, sb, bmi_i)] = float(val)
    return out


def build_lab_map(risk_data: dict) -> Dict[tuple, float]:
    """(diab_group, sex, smoker, age_band, sbp_band, chol_i) -> risk%"""
    out: Dict[tuple, float] = {}
    for diab_group, d0 in risk_data["lab"].items():
        for sx, d1 in d0.items():
            for sm, d2 in d1.items():
                for ab, d3 in d2.items():
                    for sb, arr in d3.items():
                        for chol_i, val in enumerate(arr):
                            out[(diab_group, sx, sm, ab, sb, chol_i)] = float(val)
    return out


# ----------------------------
# 0) Core standardization (ONE VERSION)
# ----------------------------
def standardize_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_cols(df, ["gender", "age", "sbp", "dbp", "bmi", "cholesterol", "smoker", "smoker_who", "has_diabetes"])

    # numeric coercions
    for c in ["age", "sbp", "dbp", "bmi", "cholesterol"]:
        df[c] = _to_num(df[c])

    # keys
    df["gender_key"] = normalize_gender_key(df["gender"])
    df["smoker_key"], df["smoker_key_source"] = normalize_smoker_key(df["smoker"], df["smoker_who"])
    df["has_diabetes"] = _to_bool_na(df["has_diabetes"])

    # plausibility flags (no auto-drop)
    df["flag_age_implausible"] = ((df["age"] < 0) | (df["age"] > 120)).fillna(False)
    df["flag_bmi_implausible"] = ((df["bmi"] < 10) | (df["bmi"] > 70)).fillna(False)
    df["flag_sbp_implausible"] = ((df["sbp"] < 70) | (df["sbp"] > 260)).fillna(False)
    df["flag_dbp_implausible"] = ((df["dbp"] < 30) | (df["dbp"] > 160)).fillna(False)

    return df


# ----------------------------
# 1) Cholesterol canonicalization (ONE VERSION)
# ----------------------------
def add_cholesterol_mmolL(df: pd.DataFrame, chol_col: str = "cholesterol") -> pd.DataFrame:
    df = df.copy()
    if "cholesterol_mmolL" in df.columns and df["cholesterol_mmolL"].notna().any():
        df["cholesterol_mmolL"] = _to_num(df["cholesterol_mmolL"])
        df["cholesterol_unit_inferred"] = pd.Series("mmol/L (existing)", index=df.index, dtype=STR)
        return df

    df = _ensure_cols(df, [chol_col])
    df["cholesterol_mmolL"], df["cholesterol_unit_inferred"] = infer_cholesterol_mmolL(df[chol_col])
    return df


# ----------------------------
# 2) Eligibility cohorts (ONE VERSION)
# ----------------------------
def build_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_cols(df, ["gender_key", "age", "smoker_key", "sbp", "bmi", "has_diabetes", "cholesterol_mmolL"])

    nonlab_req = ["gender_key", "age", "smoker_key", "sbp", "bmi"]
    lab_req = ["gender_key", "age", "smoker_key", "sbp", "has_diabetes", "cholesterol_mmolL"]

    df["eligible_nonlab"] = df[nonlab_req].notna().all(axis=1)
    df["eligible_lab"] = df[lab_req].notna().all(axis=1)
    df["eligible_paired"] = df["eligible_nonlab"] & df["eligible_lab"]
    # df["who_chol_col_used"] = "cholesterol_mmolL"
    return df


# ----------------------------
# 3) WHO domain flags + WHO bins (ONE VERSION)
# ----------------------------
def add_who_bins_and_domain_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # WHO domain flags (do not drop)
    df["in_who_age_domain"] = df["age"].between(40, 74, inclusive="both")
    df["in_who_sbp_domain"] = df["sbp"].between(70, 260, inclusive="both")

    df["who_domain_ok_nonlab"] = df["eligible_nonlab"] & df["in_who_age_domain"] & df["in_who_sbp_domain"]
    df["who_domain_ok_lab"] = df["eligible_lab"] & df["in_who_age_domain"] & df["in_who_sbp_domain"]

    # bands
    df["age_band"] = pd.cut(df["age"], bins=AGE_BINS, right=False, labels=AGE_LABELS)
    df["sbp_band"] = pd.cut(df["sbp"], bins=SBP_BINS, right=False, labels=SBP_LABELS)
    df["bmi_band"] = pd.cut(df["bmi"], bins=BMI_BINS, right=False, labels=BMI_LABELS)

    if "cholesterol_mmolL" in df.columns:
        df["chol_band"] = pd.cut(df["cholesterol_mmolL"], bins=CHOL_BINS, right=False, labels=CHOL_LABELS)
    else:
        df["chol_band"] = pd.NA

    # ordered categoricals (stable plots / crosstabs)
    df["age_band"] = df["age_band"].astype(pd.CategoricalDtype(AGE_LABELS, ordered=True))
    df["sbp_band"] = df["sbp_band"].astype(pd.CategoricalDtype(SBP_LABELS, ordered=True))
    df["bmi_band"] = df["bmi_band"].astype(pd.CategoricalDtype(BMI_LABELS, ordered=True))
    df["chol_band"] = df["chol_band"].astype(pd.CategoricalDtype(CHOL_LABELS, ordered=True))
    return df


# ----------------------------
# 4) WHO risk computation (ONE VERSION)
# ----------------------------
def add_who_risks(df: pd.DataFrame, risk_data: dict) -> pd.DataFrame:
    df = df.copy()

    # build lookup maps (OK to do here; for speed you can prebuild once and reuse)
    print("processing........")
    nonlab_map = build_nonlab_map(risk_data)
    print("nonlab_map", nonlab_map)
    lab_map = build_lab_map(risk_data)

    # indices required by WHO tables
    df["bmi_i"] = to_index(df["bmi"], BMI_BINS)
    df["chol_i"] = to_index(df["cholesterol_mmolL"], CHOL_BINS)

    df["diab_group"] = df["has_diabetes"].map({True: "with_diabetes", False: "no_diabetes"}).astype(STR)

    # --- non-lab lookup ---
    nonlab_keys = list(zip(
        df["gender_key"],
        df["smoker_key"],
        df["age_band"].astype(str),
        df["sbp_band"].astype(str),
        df["bmi_i"].astype("Int64"),
    ))
    df["risk_nonlab"] = pd.Series(nonlab_keys, index=df.index).map(nonlab_map)

    # --- lab lookup ---
    lab_keys = list(zip(
        df["diab_group"],
        df["gender_key"],
        df["smoker_key"],
        df["age_band"].astype(str),
        df["sbp_band"].astype(str),
        df["chol_i"].astype("Int64"),
    ))
    df["risk_lab"] = pd.Series(lab_keys, index=df.index).map(lab_map)

    # --- comparisons (safe) ---
    df["risk_diff_nonlab_minus_lab"] = df["risk_nonlab"] - df["risk_lab"]
    df["underestimates"] = df["eligible_paired"] & (df["risk_nonlab"] < df["risk_lab"])

    df["highrisk_lab_20"] = df["risk_lab"].ge(20) & df["risk_lab"].notna()
    df["highrisk_nonlab_20"] = df["risk_nonlab"].ge(20) & df["risk_nonlab"].notna()
    df["missed_highrisk_20"] = df["highrisk_lab_20"] & (~df["highrisk_nonlab_20"])

    # buckets (optional but useful)
    df["risk_nonlab_cat"] = risk_bucket(df["risk_nonlab"])
    df["risk_lab_cat"] = risk_bucket(df["risk_lab"])

    return df


# ----------------------------
# 5) One-call prep (YOUR df2 step)
# ----------------------------
def prepare_who_analysis_df(df_who: pd.DataFrame) -> pd.DataFrame:
    df = standardize_core_fields(df_who)
    df = add_cholesterol_mmolL(df, chol_col="cholesterol")
    df = build_cohorts(df)
    df = add_who_bins_and_domain_flags(df)
    return df