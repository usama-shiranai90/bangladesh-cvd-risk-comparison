import pandas as pd
import numpy as np
import streamlit as st

def n_pct(n, total, decimals=1):
    """N pct."""
    if total == 0:
        return "0 (0.0%)"
    return f"{int(n):,} ({(n / total) * 100:.{decimals}f}%)"

def summarize_continuous(df, col, decimals=1):
    """Return tuple (mean_sd_str, n_count)."""
    if col not in df.columns:
        return "N/A", 0
    v = pd.to_numeric(df[col], errors="coerce")
    n = int(v.notna().sum())
    if n == 0:
        return "N/A", 0
    return f"{v.mean():.{decimals}f} ± {v.std():.{decimals}f}", n

def count_age_bin(df, age_col="age", lo=None, hi=None, lo_inclusive=True, hi_inclusive=True):
    """Count age bin."""
    if age_col not in df.columns:
        return 0
    a = pd.to_numeric(df[age_col], errors="coerce")
    m = a.notna()
    if lo is not None:
        m &= (a >= lo) if lo_inclusive else (a > lo)
    if hi is not None:
        m &= (a <= hi) if hi_inclusive else (a < hi)
    return int(m.sum())

def sex_counts(df, col="gender"):
    """Sex counts."""
    if col not in df.columns:
        return (0, 0, len(df))
    s = df[col].astype("string").str.strip().str.lower()
    men = int(s.isin(["m", "male", "men", "man"]).sum())
    women = int(s.isin(["f", "female", "women", "woman"]).sum())
    unk = int(len(df) - (men + women))
    return men, women, unk

def smoker_count(df, col="smoker"):
    """Smoker count."""
    if col not in df.columns:
        return 0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).eq(1).sum())
    t = s.astype("string").str.strip().str.lower()
    return int(t.isin(["yes", "y", "true", "1", "smoker", "current", "current smoker"]).sum())

def non_smoker_count(df, col="smoker"):
    """Count non-smokers in the dataframe."""
    if col not in df.columns:
        return 0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).eq(0).sum())
    t = s.astype("string").str.strip().str.lower()
    return int(t.isin(["no", "n", "false", "0", "non-smoker", "non smoker", "never", "never smoker"]).sum())

def smoker_missing_count(df, col="smoker"):
    """Count missing smoker values."""
    if col not in df.columns:
        return len(df)
    return int(df[col].isna().sum())

def diabetes_count(df, col="has_diabetes"):
    """Diabetes count."""
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())

def bp_category_counts(df, col="bp_category"):
    """Get counts for each BP category."""
    if col not in df.columns:
        return {}
    counts = df[col].value_counts(dropna=False)
    return counts.to_dict()

def bp_category_count(df, category, col="bp_category"):
    """Count specific BP category."""
    if col not in df.columns:
        return 0
    s = df[col].astype("string").str.strip().str.lower()
    category_lower = category.lower()
    return int(s.str.contains(category_lower, na=False).sum())

def bmi_band_counts(df, col="bmi_band"):
    """Get counts for each BMI band."""
    if col not in df.columns:
        return {}
    counts = df[col].value_counts(dropna=False)
    return counts.to_dict()

def bmi_band_count(df, band, col="bmi_band"):
    """Count specific BMI band."""
    if col not in df.columns:
        return 0
    return int((df[col] == band).sum())

def attach_location_type(df, df_sites=None, site_col="site_id", out_col="location_type"):
    """Attach location type."""
    d = df.copy()
    if out_col in d.columns:
        x = d[out_col].astype("string").str.strip().str.title()
        x = x.replace({"Semi-urban": "Semi-Urban", "Semi Urban": "Semi-Urban"})
        d[out_col] = x.fillna("Unknown").replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
        return d
    
    if "urban_rural" in d.columns:
        x = d["urban_rural"].astype("string").str.strip().str.title()
        x = x.replace({"Semi-Urban": "Semi-Urban", "Semi Urban": "Semi-Urban"})
        d[out_col] = x.fillna("Unknown").replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
        return d

    if df_sites is None or site_col not in d.columns or site_col not in df_sites.columns:
        d[out_col] = "Unknown"
        return d

    tmp = df_sites[[site_col, "location_type"]].drop_duplicates(site_col)
    out = d.merge(tmp, on=site_col, how="left")
    x = out["location_type"].astype("string").str.strip().str.title()
    x = x.replace({"Semi-urban": "Semi-Urban", "Semi Urban": "Semi-Urban"})
    out[out_col] = x.fillna("Unknown").replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
    return out


def _clean_site_text(s: pd.Series, unknown="Unknown") -> pd.Series:
    """Clean site text."""
    x = s.astype("string").str.strip()
    x = x.fillna(unknown).replace({"Nan": unknown, "None": unknown, "": unknown})
    return x

def _clean_location_type(s: pd.Series, unknown="Unknown") -> pd.Series:
    """Clean location type."""
    x = s.astype("string").str.strip().str.title()
    x = x.replace({"Semi-urban": "Semi-Urban", "Semi Urban": "Semi-Urban"})
    x = x.fillna(unknown).replace({"Nan": unknown, "None": unknown, "": unknown})
    return x


def attach_site_info(
    df,
    df_sites=None,
    site_col="site_id",
    location_out="location_type",
    title_out="site_title",
    name_out="project_title",
):
    """Attach location_type, site_title, site_name to df."""
    d = df.copy()

    if location_out in d.columns:
        d[location_out] = _clean_location_type(d[location_out])
    elif "urban_rural" in d.columns:
        d[location_out] = _clean_location_type(d["urban_rural"])

    if title_out in d.columns:
        d[title_out] = _clean_site_text(d[title_out])
    if name_out in d.columns:
        d[name_out] = _clean_site_text(d[name_out])

    have_all = all(c in d.columns for c in [location_out, title_out, name_out])
    if have_all:
        return d

    can_merge = (
        df_sites is not None
        and isinstance(df_sites, pd.DataFrame)
        and site_col in d.columns
        and site_col in df_sites.columns
    )

    if can_merge:
        want_cols = [site_col]
        for c in [location_out, title_out, name_out]:
            if c in df_sites.columns:
                want_cols.append(c)

        tmp = df_sites[want_cols].drop_duplicates(site_col)
        out = d.merge(tmp, on=site_col, how="left", suffixes=("", "_site"))

        if location_out not in out.columns:
            out[location_out] = "Unknown"
        else:
            out[location_out] = _clean_location_type(out[location_out])

        if title_out not in out.columns:
            out[title_out] = "Unknown"
        else:
            out[title_out] = _clean_site_text(out[title_out])

        if name_out not in out.columns:
            out[name_out] = "Unknown"
        else:
            out[name_out] = _clean_site_text(out[name_out])

        for c in [location_out, title_out, name_out]:
            if c not in out.columns:
                out[c] = "Unknown"
            out[c] = out[c].fillna("Unknown").replace({"": "Unknown"})

        return out

def bmi_category_series(df, col="bmi"):
    """Bmi category series."""
    if col not in df.columns:
        return pd.Series(["N/A"] * len(df), index=df.index)
    b = pd.to_numeric(df[col], errors="coerce")
    cat = pd.cut(
        b,
        bins=[-np.inf, 18.5, 25, 30, np.inf],
        right=False,
        labels=["Underweight (<18.5)", "Normal (18.5–24.9)", "Overweight (25–29.9)", "Obese (≥30)"]
    )
    return cat.astype("object").fillna("Missing")

TABLE_CSS = """
<style>
    table.report-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 14px;
        color: #333;
    }
    table.report-table th {
        background-color: #f8f9fa;
        color: #444;
        font-weight: 600;
        text-align: center;
        border: 1px solid #dee2e6;
        padding: 8px;
    }
    table.report-table td {
        border: 1px solid #dee2e6;
        padding: 6px 10px;
        vertical-align: middle;
    }
    table.report-table tr:nth-child(even) {
        background-color: #fcfcfc;
    }
    .section-row td {
        background-color: #e9ecef !important;
        font-weight: bold;
        text-align: left;
        color: #495057;
    }
    .var-row td:first-child {
        text-align: left;
        padding-left: 20px;
    }
    .data-cell {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .header-top {
        border-bottom: 2px solid #555 !important;
    }
    .sub-header {
        border-bottom: 1px solid #555 !important;
    }
</style>
"""

def generate_html_table1(
    df_nonlab, df_lab, df_who_nonlab_domain, df_who_lab_domain,
    df_sites=None,
    site_col="site_id"
):
    """Generate html table1."""
    cols_keys = ["NL_Gen", "L_Gen", "NL_WHO", "L_WHO"]
    inputs = [df_nonlab, df_lab, df_who_nonlab_domain, df_who_lab_domain]
    dfs = {}

    for i, k in enumerate(cols_keys):
        d = inputs[i].copy() if inputs[i] is not None else pd.DataFrame()
        if not d.empty:
            d = attach_site_info(d, df_sites=df_sites, site_col=site_col)
        dfs[k] = d

    html = [TABLE_CSS]
    html.append('<table class="report-table">')

    html.append('<thead>')
    html.append('<tr>')
    html.append('<th rowspan="2" style="text-align:left; border-right:2px solid #ddd;">Variable</th>')
    html.append('<th colspan="2" class="header-top">General Cohort</th>')
    html.append('<th colspan="2" class="header-top">WHO Domain Cohort</th>')
    html.append('</tr>')
    html.append('<tr>')
    html.append('<th class="sub-header">Non-Lab</th><th class="sub-header">Lab</th>')
    html.append('<th class="sub-header">Non-Lab</th><th class="sub-header">Lab</th>')
    html.append('</tr>')
    html.append('</thead>')

    html.append('<tbody>')

    def add_section(title):
        """Add section."""
        html.append(f'<tr class="section-row"><td colspan="5">{title}</td></tr>')

    def add_row(label, val_func, is_n_row=False):
        """Add row."""
        row_html = '<tr class="var-row">'
        style = 'font-style:italic; color:#666;' if is_n_row else ''
        row_html += f'<td style="{style}">{label}</td>'
        for k in cols_keys:
            val = val_func(dfs[k])
            row_html += f'<td class="data-cell">{val}</td>'
        row_html += '</tr>'
        html.append(row_html)


    add_section("Study Sample")
    add_row("Total N", lambda d: f"{len(d):,}")

    add_section("Age")
    def agg_age(d):
        """Agg age."""
        s, n = summarize_continuous(d, "age", 1)
        return s
    add_row("Mean ± SD (years)", agg_age)

    def agg_age_n(d):
        """Agg age n."""
        _, n = summarize_continuous(d, "age", 1)
        return f"(n={n:,})"
    add_row("<i>(n for Mean)</i>", agg_age_n, is_n_row=True)

    add_row("Age ≥ 60 yr, n (%)", lambda d: n_pct(count_age_bin(d, "age", lo=60), len(d)))

    add_section("Age distribution")
    add_row("&lt; 40, n (%)", lambda d: n_pct(count_age_bin(d, "age", hi=39), len(d)))

    bands = [(40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74)]
    for lo, hi in bands:
        add_row(f"{lo}–{hi}, n (%)", lambda d, lo=lo, hi=hi: n_pct(count_age_bin(d, "age", lo=lo, hi=hi), len(d)))
    add_row("≥ 75, n (%)", lambda d: n_pct(count_age_bin(d, "age", lo=75), len(d)))

    add_section("Sex, n (%)")
    add_row("Male", lambda d: n_pct(sex_counts(d)[0], len(d)))
    add_row("Female", lambda d: n_pct(sex_counts(d)[1], len(d)))

    add_section("Vitals (Mean ± SD)")
    add_row("SBP (mmHg)", lambda d: summarize_continuous(d, "sbp")[0])
    add_row("BMI (kg/m²)", lambda d: summarize_continuous(d, "bmi")[0])
    add_row("<i>Note: N matches Total usually</i>", lambda d: "", is_n_row=True)

    add_section("Risk factors")
    add_row("Current smoker, n (%)", lambda d: n_pct(smoker_count(d), len(d)))
    add_row("Diabetes, n (%)", lambda d: n_pct(diabetes_count(d), len(d)))
    add_row("Total Chol. (mmol/L)", lambda d: summarize_continuous(d, "cholesterol_mmolL", 2)[0])
    add_row("<i>(Available N for Chol)</i>", lambda d: f"(n={summarize_continuous(d, 'cholesterol_mmolL')[1]:,})", is_n_row=True)

    add_section("Smoking Status, n (%)")
    add_row("Smoker", lambda d: n_pct(smoker_count(d), len(d)))
    add_row("Non-Smoker", lambda d: n_pct(non_smoker_count(d), len(d)))
    add_row("Missing", lambda d: n_pct(smoker_missing_count(d), len(d)), is_n_row=True)

    add_section("Blood Pressure Category, n (%)")
    bp_categories = ["Normal", "Elevated", "HTN Stage 1", "HTN Stage 2", "Hypertensive Crisis"]
    for bp_cat in bp_categories:
        def make_bp_counter(category):
            """Make bp counter."""
            def counter(d):
                """Counter."""
                if "bp_category" not in d.columns:
                    return "N/A"
                count = int((d["bp_category"] == category).sum())
                return n_pct(count, len(d))
            return counter
        add_row(bp_cat, make_bp_counter(bp_cat))

    def bp_missing(d):
        """Bp missing."""
        if "bp_category" not in d.columns:
            return "N/A"
        return n_pct(int(d["bp_category"].isna().sum()), len(d))
    add_row("Missing", bp_missing, is_n_row=True)

    add_section("BMI Category, n (%)")
    bmi_bands = ["<20", "20-24", "25-29", "30-34", ">=35"]
    bmi_labels = ["Underweight (<20)", "Normal (20-24)", "Overweight (25-29)", "Obese I (30-34)", "Obese II/III (≥35)"]
    for band, label in zip(bmi_bands, bmi_labels):
        def make_bmi_counter(bmi_band):
            """Make bmi counter."""
            def counter(d):
                """Counter."""
                if "bmi_band" not in d.columns:
                    return "N/A"
                count = int((d["bmi_band"] == bmi_band).sum())
                return n_pct(count, len(d))
            return counter
        add_row(label, make_bmi_counter(band))

    def bmi_missing(d):
        """Bmi missing."""
        if "bmi_band" not in d.columns:
            return "N/A"
        return n_pct(int(d["bmi_band"].isna().sum()), len(d))
    add_row("Missing", bmi_missing, is_n_row=True)

    add_section("Location Type, n (%)")
    location_types = ["Urban", "Rural", "Semi-Urban"]
    for loc_type in location_types:
        def make_loc_counter(location):
            """Make loc counter."""
            def counter(d):
                """Counter."""
                if "location_type" not in d.columns:
                    return "N/A"
                count = int((d["location_type"] == location).sum())
                return n_pct(count, len(d))
            return counter
        add_row(loc_type, make_loc_counter(loc_type))

    add_section("Top sites, n (%)")

    ref_df = dfs["NL_Gen"]
    if ref_df.empty or site_col not in ref_df.columns:
        ref_df = dfs["NL_WHO"]

    top_sites = []
    if not ref_df.empty and site_col in ref_df.columns:
        top_sites = ref_df[site_col].value_counts().head(5).index.tolist()

    def site_label_for(site_id):
        """Prefer site_title, then site_name, else fall back to site_id as string."""
        label = str(site_id)
        if ref_df.empty or site_col not in ref_df.columns:
            return label

        m = ref_df.loc[ref_df[site_col] == site_id]
        if m.empty:
            return label
        
        for c in ["project_title", "site_title", "facility_name"]:
            if c in m.columns:
                v = m.iloc[0][c]
                if pd.notna(v) and str(v).strip():
                    return str(v).strip()
        return label

    def site_percentage(d, site, site_col):
        """Site percentage."""
        print("---- Debug site_percentage ----")
        print("Rows:", len(d))
        print("Empty:", d.empty)
        print("Columns:", list(d.columns))
        print("Looking for column:", site_col)
        print("Target site:", site)

        if d.empty:
            count = 0
        elif site_col not in d.columns:
            count = 0
        else:
            matches = d[site_col] == site
            print("Match sample:", matches.head())
            count = int(matches.sum())

        print("Count:", count)
        print("------------------------------")

        return n_pct(count, len(d))

    for site in top_sites:
        label = site_label_for(site)
        add_row(
            label,
            lambda d, site=site: n_pct(
                int((d[site_col] == site).sum()) if (not d.empty and site_col in d.columns) else 0,
                len(d)
            )
        )
        add_row(label, lambda d, site=site: site_percentage(d, site, site_col))
    
    html.append('</tbody></table>')
    return "\n".join(html)
