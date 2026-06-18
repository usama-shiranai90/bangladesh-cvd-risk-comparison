"""
data_utils.py — Data Cleaning and Analysis Utilities

This module provides helper functions for:
- Column normalization and parsing
- Standardizing categorical and text fields
- Quick exploratory summaries

Author: OneEyeOwl
"""

import math
import re
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from IPython.display import display

from .constants import BD_DISTRICT_CANON, ALIAS
# Re-export selected functions from split modules to make data_utils a facade
from .eda import quick_analysis, deep_analysis
from .visualization import annotate_bars, check_Outliers


# Blood glucose
def normalize_bg_type(x):
    """Fallback normalizer. If you already have normalize_bg_type imported, remove this."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    BG_TYPE_MAP = {
        "pbs": "pbs",
        "post": "pbs",
        "postprandial": "pbs",
        "pp": "pbs",
        "2hr": "pbs",
        "2h": "pbs",
        "rbs": "pbs",
        "random": "pbs",
        "fbs": "fbs",
        "fasting": "fbs",
    }
    mapping = {
        "fbs": "fbs", "fasting": "fbs", "fast": "fbs",
        "rbs": "rbs", "random": "rbs",
        "pbs": "pbs", "pp": "pbs", "ppbs": "pbs", "post": "pbs", "postprandial": "pbs", "2hr": "pbs", "2h": "pbs",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return np.nan


def classify_pbs_mgdl(x):
    if pd.isna(x): return np.nan
    x = float(x)
    if x < 54:  return "Lower Warning (<54)"
    if x < 140: return "Normal (<140)"
    if x < 200: return "Prediabetes (140–199)"
    if x < 300: return "Diabetes (200–299)"
    if x < 540: return "Severe High (300–539)"
    return "Critical High (>=540)"



# ---------------------------------------------------------------------
# 🧹 Data Cleaning Utilities
# ---------------------------------------------------------------------

def normcols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (lowercase, snake_case, alphanumeric only)."""
    out = df.copy()
    out.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c.strip()).lower() for c in out.columns]
    return out


def parse_dt(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to datetime, safely removing timezones."""
    d = pd.to_datetime(series, errors="coerce")
    try:
        d = d.dt.tz_localize(None)
    except Exception:
        pass
    return d


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric, coercing invalid values."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def first_existing(df: pd.DataFrame, aliases: Tuple[str, ...]) -> Optional[str]:
    """Return the first existing column name from a list of aliases."""
    for a in aliases:
        if a in df.columns:
            return a
    return None


def to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc": return s
    cleaned = s.astype(str).str.replace(r"[^0-9\.\-]+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


# ---------------------------------------------------------------------
# 🔤 Standardization Functions
# ---------------------------------------------------------------------

def gender_std(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"", "none", "null", "nan", "na", "n/a"}:
        return np.nan

    if s in {"f", "female", "woman", "girl", "Female", "Woman"}:
        return "F"
    if s in {"m", "male", "man", "boy", "Male", "Boy"}:
        return "M"
    return np.nan


def blood_group_std(x):
    """Standardize blood group entries (A+, O-, etc.)."""
    if pd.isna(x):
        return np.nan
    s = str(x).upper().replace(" ", "")
    s = (s.replace("POSITIVE", "+").replace("NEGATIVE", "-")
         .replace("POS", "+").replace("NEG", "-")
         .replace("PLUS", "+").replace("MINUS", "-"))
    valid = {"A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
    return s if s in valid else np.nan


def marital_std(x):
    """Normalize marital status values to 'Married' or 'Unmarried'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "married", "y", "yes", "1"}:
        return "Married"
    if s in {"u", "unmarried", "single", "n", "no", "0"}:
        return "Unmarried"
    return np.nan


def age_bucket(age):
    """Convert age (int) into labeled age buckets."""
    if pd.isna(age):
        return "Unknown"
    a = int(age)
    if a < 18: return "<18"
    if a < 30: return "18-29"
    if a < 45: return "30-44"
    if a < 60: return "45-59"
    return "60+"


def bmi_class(b):
    """Categorize BMI into WHO standard classes."""
    if pd.isna(b): return np.nan
    if b < 18.5: return "Underweight"
    if b < 25: return "Normal"
    if b < 30: return "Overweight"
    return "Obese"


def district_from_address(addr: str) -> Any:
    """Extract the most likely district name from a text address."""
    if pd.isna(addr):
        return np.nan
    s = re.sub(r'[^A-Za-z ]+', ' ', str(addr)).strip().title()
    for k, v in ALIAS.items():  # apply aliases first
        if k in s:
            s = s.replace(k, v)
    tokens = set(s.split())
    for d in BD_DISTRICT_CANON:  # exact token match
        if set(d.split()).issubset(tokens):
            return d
    for d in BD_DISTRICT_CANON:  # substring fallback
        if d in s:
            return d
    return np.nan


def yn(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "y", "1", "true"}:  return "Yes"
    if s in {"no", "n", "0", "false"}:  return "No"
    return np.nan


def gtype(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"fbs", "fb", "fasting"}: return "FBS"
    if s in {"rbs", "random"}:       return "RBS"
    if s in {"2hr", "2h", "pp"}:      return "2H-PP"
    return np.nan


def bp_category(sys, dia):
    if pd.isna(sys) or pd.isna(dia): return np.nan
    if sys < 120 and dia < 80: return "Normal"
    if 120 <= sys <= 129 and dia < 80: return "Elevated"
    if (130 <= sys <= 139) or (80 <= dia <= 89): return "HTN Stage 1"
    if (140 <= sys <= 179) or (90 <= dia <= 119): return "HTN Stage 2"
    if sys >= 180 or dia >= 120: return "Hypertensive Crisis"
    return "Other"


# ---------------------------------------------------------------------
# 🌡️ UNIT DETECTORS & LABELING HELPERS
# ---------------------------------------------------------------------

def detect_temp_unit(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty: return "unknown"
    m = x.median()
    return "F" if 90 < m < 110 else ("C" if 30 <= m <= 45 else "unknown")


def detect_glu_unit(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty: return "unknown"
    m = x.median()
    return "mmol/L" if 2 <= m <= 25 else ("mg/dL" if m > 40 else "unknown")


def _bands(lo, hi):
    lo = float(lo) if lo is not None else -np.inf
    hi = float(hi) if hi is not None else np.inf
    return lo, hi


def label_by_ranges(x: pd.Series, bands: list[tuple[str, float, float]]) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype="object")
    x = pd.to_numeric(x, errors="coerce")
    for color, lo, hi in bands:
        out[x.ge(lo) & x.lt(hi)] = color
    return out


def label_discrete(x: pd.Series, mapping: dict[str, list[str]]) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype="object")
    xc = x.astype(str).str.strip().str.lower()
    for color, values in mapping.items():
        if not values:
            continue
        keys = [str(v).strip().lower() for v in values]
        out[xc.isin(keys)] = color
    return out


def label_sex_split(df: pd.DataFrame, value_col: str, sex_col: str, male_bands, female_bands) -> pd.Series:
    res = pd.Series(np.nan, index=df.index, dtype="object")
    if sex_col not in df.columns or value_col not in df.columns:
        return res
    sx = df[sex_col].astype(str).str.strip().str.lower()
    m_mask = sx.str.startswith("m")
    f_mask = sx.str.startswith("f")
    if m_mask.any():
        res.loc[m_mask] = label_by_ranges(pd.to_numeric(df.loc[m_mask, value_col], errors="coerce"), male_bands)
    if f_mask.any():
        res.loc[f_mask] = label_by_ranges(pd.to_numeric(df.loc[f_mask, value_col], errors="coerce"), female_bands)
    return res


def build_bands_from_tv(TV, prefix):
    colors = ["GREEN", "YELLOW", "ORANGE", "RED"]
    out = []
    for color in colors:
        kmin, kmax = f"{prefix}_{color}_MIN", f"{prefix}_{color}_MAX"
        if kmin in TV and kmax in TV:
            out.append((color, *_bands(TV[kmin], TV[kmax])))
    return out


def apply_vectorized_triage(df: pd.DataFrame, TV: dict, sex_col: str = "gender") -> pd.DataFrame:
    d = df.copy()

    rename_for_triage = {
        "oxygen_of_blood": "blood_oxygenation",
        "bp_sys": "blood_pressure_systolic",
        "bp_dia": "blood_pressure_diastolic",
        "blood_glucose": "blood_sugar",
        "cholesterol": "blood_cholesterol",
        "uric_acid": "blood_uric_acid",
        "urinary_glucose": "u_sugar",
        "urinary_protein": "u_protein",
        "urinary_urobilinogen": "u_urobilinogen",
        "waist_hip_ratio": "waistHipRatio",
        "bmi": "bmi",
        "temperature": "temperature",
    }
    for src, dst in rename_for_triage.items():
        if src in d.columns and dst not in d.columns:
            d[dst] = d[src]

    # Temperature °F → °C
    if "temperature" in d.columns:
        f_mask = pd.to_numeric(d["temperature"], errors="coerce") > 45
        d.loc[f_mask, "temperature"] = (pd.to_numeric(d.loc[f_mask, "temperature"], errors="coerce") - 32) * (5 / 9)

    # Glucose mg/dL → mmol/L heuristic
    if "blood_sugar" in d.columns:
        bs = pd.to_numeric(d["blood_sugar"], errors="coerce")
        mgdl_like = bs.between(30, 800)
        d["blood_sugar_mmol"] = np.where(mgdl_like, bs / 18.0, bs)
        d["blood_sugar_for_rules"] = d["blood_sugar_mmol"]

    # Bands and labeling
    if "bmi" in d.columns:
        bands_bmi = build_bands_from_tv(TV, "BMI")
        if bands_bmi: d["BMI_COLOR".lower()] = label_by_ranges(d["bmi"], bands_bmi)

    if "temperature" in d.columns:
        bands_t = build_bands_from_tv(TV, "TEMPERATURE".lower())
        if bands_t: d["TEMPERATURE_COLOR".lower()] = label_by_ranges(d["temperature"], bands_t)

    if "blood_oxygenation" in d.columns:
        bands_spo2 = build_bands_from_tv(TV, "SPO2".lower())
        if bands_spo2: d["SPO2_COLOR".lower()] = label_by_ranges(d["blood_oxygenation"], bands_spo2)

    if "blood_pressure_systolic" in d.columns:
        bands_sys = build_bands_from_tv(TV, "BP_SYS")
        if bands_sys: d["BP_SYS_COLOR"] = label_by_ranges(d["blood_pressure_systolic"], bands_sys)
    if "blood_pressure_diastolic" in d.columns:
        bands_dia = build_bands_from_tv(TV, "BP_DIA")
        if bands_dia: d["BP_DIA_COLOR"] = label_by_ranges(d["blood_pressure_diastolic"], bands_dia)

    if "blood_sugar_for_rules" in d.columns:
        g = pd.to_numeric(d["blood_sugar_for_rules"], errors="coerce")
        bands_bs = [
            ("GREEN", *_bands(TV.get('BLOOD_SUGAR_GREEN_MIN', -np.inf), TV.get('BLOOD_SUGAR_GREEN_MAX', np.inf))),
            ("YELLOW", *_bands(TV.get('BLOOD_SUGAR_YELLOW_MIN', -np.inf), TV.get('BLOOD_SUGAR_YELLOW_MAX', np.inf))),
            ("ORANGE", *_bands(TV.get('BLOOD_SUGAR_ORANGE_MIN', -np.inf), TV.get('BLOOD_SUGAR_ORANGE_MAX', np.inf))),
            ("RED", *_bands(TV.get('BLOOD_SUGAR_RED_MIN', -np.inf), TV.get('BLOOD_SUGAR_RED_MAX', np.inf))),
        ]
        d["BLOOD_SUGAR_COLOR"] = label_by_ranges(g, bands_bs)

    if "blood_hemoglobin" in d.columns:
        hb = pd.to_numeric(d["blood_hemoglobin"], errors="coerce")
        bands_hb = [
            ("GREEN",
             *_bands(TV.get('BLOOD_HEMOGLOBIN_GREEN_MIN', -np.inf), TV.get('BLOOD_HEMOGLOBIN_GREEN_MAX', np.inf))),
            ("YELLOW",
             *_bands(TV.get('BLOOD_HEMOGLOBIN_YELLOW_MIN', -np.inf), TV.get('BLOOD_HEMOGLOBIN_YELLOW_MAX', np.inf))),
            ("ORANGE",
             *_bands(TV.get('BLOOD_HEMOGLOBIN_ORANGE_MIN', -np.inf), TV.get('BLOOD_HEMOGLOBIN_ORANGE_MAX', np.inf))),
            ("RED", *_bands(TV.get('BLOOD_HEMOGLOBIN_RED_MIN', -np.inf), TV.get('BLOOD_HEMOGLOBIN_RED_MAX', np.inf))),
        ]
        d["BLOOD_HEMOGLOBIN_COLOR"] = label_by_ranges(hb, bands_hb)

    if "blood_cholesterol" in d.columns:
        bands_chol = [
            ("GREEN",
             *_bands(TV.get('BLOOD_CHOLESTEROL_GREEN_MIN', -np.inf), TV.get('BLOOD_CHOLESTEROL_GREEN_MAX', np.inf))),
            ("YELLOW",
             *_bands(TV.get('BLOOD_CHOLESTEROL_YELLOW_MIN', -np.inf), TV.get('BLOOD_CHOLESTEROL_YELLOW_MAX', np.inf))),
            ("ORANGE",
             *_bands(TV.get('BLOOD_CHOLESTEROL_ORANGE_MIN', -np.inf), TV.get('BLOOD_CHOLESTEROL_ORANGE_MAX', np.inf))),
            ("RED", *_bands(TV.get('BLOOD_CHOLESTEROL_RED_MIN', -np.inf), TV.get('BLOOD_CHOLESTEROL_RED_MAX', np.inf))),
        ]
        d["BLOOD_CHOLESTEROL_COLOR"] = label_by_ranges(pd.to_numeric(d["blood_cholesterol"], errors="coerce"),
                                                       bands_chol)

    if "blood_uric_acid" in d.columns:
        if sex_col in d.columns:
            male_bands = [("GREEN", 3.5, 7.0), ("ORANGE", 7.1, 7.9), ("RED", 8.0, 12.0)]
            female_bands = [("GREEN", 3.5, 6.0), ("ORANGE", 6.1, 6.9), ("RED", 7.0, 12.0)]
            d["BLOOD_URIC_ACID_COLOR"] = label_sex_split(d, "blood_uric_acid", sex_col, male_bands, female_bands)
        else:
            d["BLOOD_URIC_ACID_COLOR"] = label_by_ranges(pd.to_numeric(d["blood_uric_acid"], errors="coerce"),
                                                         [("GREEN", 3.5, 7.0), ("ORANGE", 7.1, 7.9),
                                                          ("RED", 8.0, 12.0)])

    # Discrete urine
    if "u_sugar" in d.columns:
        us_map = {"GREEN": ["-"], "YELLOW": ["+-", "±"], "ORANGE": ["+", "++", "+++", "++++", "others"], "RED": []}
        d["URINARY_SUGAR_COLOR"] = label_discrete(d["u_sugar"], us_map)
    if "u_protein" in d.columns:
        up_map = {"GREEN": ["-"], "YELLOW": ["+-", "±"], "ORANGE": ["+", "++", "+++", "++++", "others"], "RED": []}
        d["URINARY_PROTEIN_COLOR"] = label_discrete(d["u_protein"], up_map)
    if "u_urobilinogen" in d.columns:
        uu_map = {"GREEN": ["+-", "±"], "YELLOW": ["+-", "±"], "ORANGE": ["-", "+", "++", "+++", "++++", "others"],
                  "RED": []}
        d["UROBILINOGEN_COLOR"] = label_discrete(d["u_urobilinogen"], uu_map)

    # Overall health status by max severity
    COLOR_RANK = {"RED": 3, "ORANGE": 2, "YELLOW": 1, "GREEN": 0}
    color_cols = [c for c in d.columns if c.endswith("_COLOR")]
    if color_cols:
        def max_color(row):
            best = None
            best_rank = -1
            for c in color_cols:
                val = row.get(c, np.nan)
                if pd.isna(val):
                    continue
                r = COLOR_RANK.get(val, -1)
                if r > best_rank:
                    best_rank = r
                    best = val
            return best

        d["HEALTH_STATUS_COLOR".lower()] = d[color_cols].apply(max_color, axis=1)

    return d


# ---------------------------------------------------------------------
# 🧾 Utility Functions
# ---------------------------------------------------------------------

def fix_id_dtype(df, key, is_index=False):
    """Convert ID column or index to pandas nullable integer (Int64)."""
    if is_index:
        df.index = pd.to_numeric(df.index, errors="coerce").astype("Int64")
    else:
        if key not in df.columns:
            raise KeyError(f"Column '{key}' not found in DataFrame.")
        df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
    return df


# -------------------------
# Helpers: WHO normalization + binning
# -------------------------
def norm_sex(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male", "man", "men"}:
        return "men"
    if s in {"f", "female", "woman", "women"}:
        return "women"
    return np.nan


def norm_smoker_from_status(x):
    # input can be smoker_status ("Yes"/"No"/"Ex-smoker") or raw smoker
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"yes", "y", "1", "true", "smoker", "current"}:
        return "yes"
    if s in {"no", "n", "0", "false", "non-smoker", "nonsmoker", "never"}:
        return "no"
    if s in {"ex", "ex-smoker", "former", "quit"}:
        return "no"  # WHO charts are yes/no
    # if coming from smoker_status:
    if s in {"ex-smoker"}:
        return "no"
    return np.nan


def who_age_band(age):
    if pd.isna(age):
        return np.nan
    try:
        age = int(float(age))
    except Exception:
        return np.nan
    if age < 40 or age > 74:
        return np.nan
    low = 40 + 5 * ((age - 40) // 5)
    high = low + 4
    if low > 70:
        low, high = 70, 74
    return f"{low}-{high}"


def who_sbp_band(sbp):
    # ✅ FIX: must match your dict keys: "<120", "120-139", "140-159", "160-179", ">="
    if pd.isna(sbp):
        return np.nan
    try:
        sbp = float(sbp)
    except Exception:
        return np.nan
    if sbp < 120:
        return "<120"
    if sbp < 140:
        return "120-139"
    if sbp < 160:
        return "140-159"
    if sbp < 180:
        return "160-179"
    return ">="  # 180+


def bmi_idx(bmi):
    if pd.isna(bmi):
        return np.nan
    try:
        b = float(bmi)
    except Exception:
        return np.nan
    if b < 20:
        return 0
    if b < 25:
        return 1
    if b < 30:
        return 2
    if b < 35:
        return 3
    return 4


def chol_idx(chol_mmol):
    if pd.isna(chol_mmol):
        return np.nan
    try:
        c = float(chol_mmol)
    except Exception:
        return np.nan
    if c < 4:
        return 0
    if c < 5:
        return 1
    if c < 6:
        return 2
    if c < 7:
        return 3
    return 4


def diabetes_key(v):
    if pd.isna(v):
        return np.nan
    return "with_diabetes" if bool(v) else "no_diabetes"


# Rebind facade exports to ensure split-module implementations take precedence
# (importing again at the end to override any local legacy definitions)
from .eda import quick_analysis as _qa, deep_analysis as _da
from .visualization import annotate_bars as _ab, check_Outliers as _co

# Expose under original names
quick_analysis = _qa
deep_analysis = _da
annotate_bars = _ab
check_Outliers = _co
