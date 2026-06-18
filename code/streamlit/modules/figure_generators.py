"""
figure_generators.py
====================
Programmatically creates all six -compliant journal figures for the
CVD risk manuscript.  Every figure is generated fresh from the real data —
no pre-existing image files are used.

All figures conform strictly to  Reviews Artwork Guidelines via
the shared `utils.nature_style` module.

Public API
----------
    make_fig1(nl_df, site_df)   -> (fig, axes)   Non-lab risk pyramid + stacked bars
    make_fig2(nl_df)            -> (fig, axes)   Lab risk profile (diabetes split)
    make_fig3(paired_df)        -> (fig, axes)   Concordance heatmap + Sankey
    make_fig4(nl_df)            -> (fig, axes)   Age escalation + threshold
    make_fig5(paired_df)        -> (fig, axes)   Bland-Altman by gender
    make_fig6(nl_df, site_df)   -> (fig, ax)     Site heterogeneity dot plot
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from utils.nature_style import (
    apply_nature_style, add_panel_labels, clean_axes, save_figure,
    OKABE_ITO, PAIR_LAB_NONLAB, PAIR_MALE_FEMALE,
    FONT_BASE, FONT_PANEL, LW_DEFAULT, LW_THIN, LW_THICK, LW_MIN,
)

warnings.filterwarnings("ignore", category=UserWarning)


RISK_CATS   = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]
RISK_COLORS = {
    "<5%":         "#CCCCCC",
    "5% to <10%":  "#F0E442",
    "10% to <20%": "#E69F00",
    "20% to <30%": "#D55E00",
    "≥30%":        "#000000",
}
AGE_BANDS = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74"]

C_LAB    = PAIR_LAB_NONLAB[0]
C_NONLAB = PAIR_LAB_NONLAB[1]
C_MALE   = PAIR_MALE_FEMALE[0]
C_FEMALE = PAIR_MALE_FEMALE[1]


def _ensure_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure age band."""
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 40) & (df["age"] < 75)].copy()
    if "age_band" not in df.columns or df["age_band"].nunique() < 2:
        bins   = [40, 45, 50, 55, 60, 65, 70, 75]
        labels = AGE_BANDS
        df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    df["age_band"] = pd.Categorical(df["age_band"], categories=AGE_BANDS, ordered=True)
    return df


def _cat_pct(df: pd.DataFrame, group_col: str, cat_col: str) -> pd.DataFrame:
    """Return a wide DataFrame: index=group_col, columns=risk categories (% of row)."""
    ct = df.groupby([group_col, cat_col], observed=True).size().unstack(fill_value=0)
    for c in RISK_CATS:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[RISK_CATS]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    return pct


def _wilson_ci(n_pos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = n_pos / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, (centre - margin) * 100), min(100, (centre + margin) * 100)


def make_fig1(nl_df: pd.DataFrame, site_df: pd.DataFrame | None = None) -> tuple:
    """Fig. 1 | Non-lab WHO CVD risk profile."""
    df = _ensure_age_band(nl_df).copy()
    df["risk_nonlab_cat"] = pd.Categorical(
        df["risk_nonlab_cat"], categories=RISK_CATS, ordered=True
    )
    df["gender"] = df["gender"].str.upper().str.strip()

    fig, axes = apply_nature_style(nrows=1, ncols=2, layout="double",
                                   height_in=3.4, constrained_layout=True)
    ax_a, ax_b = axes.flat

    bands = AGE_BANDS
    n_bands = len(bands)
    y_pos = np.arange(n_bands)

    for gender, color, sign in [("M", C_MALE, -1), ("F", C_FEMALE, 1)]:
        gdf = df[df["gender"] == gender]
        left = np.zeros(n_bands)
        for cat in RISK_CATS:
            vals = []
            for b in bands:
                sub = gdf[gdf["age_band"] == b]
                n_total = len(sub)
                n_cat   = (sub["risk_nonlab_cat"] == cat).sum()
                vals.append(n_cat / n_total * 100 if n_total > 0 else 0)
            vals = np.array(vals)
            ax_a.barh(
                y_pos, vals * sign, left=left * sign,
                color=RISK_COLORS[cat], edgecolor="white",
                linewidth=0.2, height=0.72,
            )
            left = left + vals

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(bands, fontsize=FONT_BASE)
    ax_a.set_xlabel("Population share (%)", fontsize=FONT_BASE)
    ax_a.axvline(0, color="black", linewidth=LW_THIN)
    ax_a.set_title("Risk by age and sex", fontsize=FONT_BASE, pad=3)

    xmax = ax_a.get_xlim()[1]
    ax_a.set_xlim(-xmax * 1.05, xmax * 1.05)
    ticks = np.arange(0, int(xmax) + 20, 20)
    ax_a.set_xticks(sorted(set([*-ticks[ticks > 0], *ticks])))
    ax_a.set_xticklabels([str(abs(int(t))) for t in ax_a.get_xticks()], fontsize=FONT_BASE - 1)

    ax_a.text(-xmax * 0.5, n_bands - 0.3, "Male", color=C_MALE,
              fontsize=FONT_BASE, fontweight="bold", ha="center")
    ax_a.text( xmax * 0.5, n_bands - 0.3, "Female", color=C_FEMALE,
              fontsize=FONT_BASE, fontweight="bold", ha="center")
    clean_axes(ax_a, left=True, bottom=True)

    for i, band in enumerate(bands):
        sub  = df[df["age_band"] == band]
        n_t  = len(sub)
        left = 0
        for cat in RISK_CATS:
            pct = (sub["risk_nonlab_cat"] == cat).sum() / n_t * 100 if n_t > 0 else 0
            ax_b.bar(i, pct, bottom=left, color=RISK_COLORS[cat],
                     edgecolor="white", linewidth=0.2, width=0.72)
            left += pct

    ax_b.set_xticks(range(len(bands)))
    ax_b.set_xticklabels([b.replace("-", "–") for b in bands],
                         rotation=30, ha="right", fontsize=FONT_BASE - 1)
    ax_b.set_ylabel("Population share (%)", fontsize=FONT_BASE)
    ax_b.set_ylim(0, 100)
    ax_b.set_title("Risk by age group (all)", fontsize=FONT_BASE, pad=3)
    clean_axes(ax_b, left=True, bottom=True)

    patches = [mpatches.Patch(facecolor=RISK_COLORS[c], label=c, linewidth=LW_MIN)
               for c in RISK_CATS]
    fig.legend(handles=patches, title="10-yr CVD Risk",
               fontsize=FONT_BASE - 1, title_fontsize=FONT_BASE - 1,
               loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    add_panel_labels(list(axes.flat))
    return fig, axes


def make_fig2(nl_df: pd.DataFrame) -> tuple:
    """Fig. 2 | Lab-based WHO CVD risk profile."""
    df_all = _ensure_age_band(nl_df).copy()
    df_all["risk_lab_cat"] = pd.Categorical(
        df_all["risk_lab_cat"], categories=RISK_CATS, ordered=True
    )
    df_all["gender"] = df_all["gender"].str.upper().str.strip()

    df_lab = df_all[df_all["eligible_lab"] == True].copy() if "eligible_lab" in df_all.columns else df_all.dropna(subset=["risk_lab"])

    fig, axes = apply_nature_style(nrows=1, ncols=2, layout="double",
                                   height_in=3.6, constrained_layout=True)
    ax_a, ax_b = axes.flat

    bands   = AGE_BANDS
    n_bands = len(bands)
    y_pos   = np.arange(n_bands)

    for gender, color, sign in [("M", C_MALE, -1), ("F", C_FEMALE, 1)]:
        gdf  = df_lab[df_lab["gender"] == gender]
        left = np.zeros(n_bands)
        for cat in RISK_CATS:
            vals = []
            for b in bands:
                sub   = gdf[gdf["age_band"] == b]
                n_tot = len(sub)
                n_c   = (sub["risk_lab_cat"] == cat).sum()
                vals.append(n_c / n_tot * 100 if n_tot > 0 else 0)
            vals = np.array(vals)
            ax_a.barh(y_pos, vals * sign, left=left * sign,
                      color=RISK_COLORS[cat], edgecolor="white",
                      linewidth=0.2, height=0.72)
            left = left + vals

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(bands, fontsize=FONT_BASE)
    ax_a.set_xlabel("Population share (%)", fontsize=FONT_BASE)
    ax_a.axvline(0, color="black", linewidth=LW_THIN)
    ax_a.set_title("Lab risk by age and sex", fontsize=FONT_BASE, pad=3)
    xmax = max(abs(ax_a.get_xlim()[0]), ax_a.get_xlim()[1])
    ax_a.set_xlim(-xmax * 1.05, xmax * 1.05)
    ax_a.set_xticklabels([str(abs(int(t))) for t in ax_a.get_xticks()], fontsize=FONT_BASE - 1)
    ax_a.text(-xmax * 0.5, n_bands - 0.3, "Male",   color=C_MALE,   fontsize=FONT_BASE, fontweight="bold", ha="center")
    ax_a.text( xmax * 0.5, n_bands - 0.3, "Female", color=C_FEMALE, fontsize=FONT_BASE, fontweight="bold", ha="center")
    clean_axes(ax_a, left=True, bottom=True)

    if "has_diabetes" in df_lab.columns and "sbp_band" in df_lab.columns:
        sbp_order = sorted(df_lab["sbp_band"].dropna().unique())
        diab_labels = {True: "Diabetic", False: "Non-Diabetic"}
        bar_labels, bar_pcts, bar_colors = [], [], []

        for diab in [True, False]:
            sub_d = df_lab[df_lab["has_diabetes"] == diab]
            for sbp in sbp_order:
                sub = sub_d[sub_d["sbp_band"] == sbp]
                if len(sub) < 5:
                    continue
                row = []
                for cat in RISK_CATS:
                    row.append((sub["risk_lab_cat"] == cat).sum() / len(sub) * 100)
                bar_pcts.append(row)
                bar_labels.append(f"{'D' if diab else 'ND'}\n{sbp}")
                bar_colors.append("#56B4E9" if diab else "#E69F00")

        y_pts = np.arange(len(bar_labels))
        for j, cat in enumerate(RISK_CATS):
            left_arr = np.sum([np.array([r[k] for r in bar_pcts]) for k in range(j)], axis=0) if j > 0 else np.zeros(len(bar_pcts))
            ax_b.barh(y_pts,
                      [r[j] for r in bar_pcts],
                      left=left_arr,
                      color=RISK_COLORS[cat], edgecolor="white",
                      linewidth=0.2, height=0.65)

        ax_b.set_yticks(y_pts)
        ax_b.set_yticklabels(bar_labels, fontsize=max(FONT_BASE - 2, 5))
        ax_b.set_xlabel("Population share (%)", fontsize=FONT_BASE)
        ax_b.set_xlim(0, 100)
        ax_b.set_title("Lab risk by diabetes × SBP", fontsize=FONT_BASE, pad=3)

        n_diab = sum(1 for lbl in bar_labels if lbl.startswith("D\n"))
        if 0 < n_diab < len(bar_labels):
            ax_b.axhline(n_diab - 0.5, color="black", linewidth=LW_THIN, linestyle="--", alpha=0.5)
            ax_b.text(50, n_diab - 0.55, "Diabetic ↑  |  Non-Diabetic ↓",
                      ha="center", fontsize=max(FONT_BASE - 2, 5), style="italic", color="#555")
    else:
        ax_b.text(0.5, 0.5, "Diabetes / SBP data not available",
                  ha="center", va="center", transform=ax_b.transAxes, fontsize=FONT_BASE)

    clean_axes(ax_b, left=True, bottom=True)

    patches = [mpatches.Patch(facecolor=RISK_COLORS[c], label=c, linewidth=LW_MIN) for c in RISK_CATS]
    fig.legend(handles=patches, title="10-yr CVD Risk", fontsize=FONT_BASE - 1,
               title_fontsize=FONT_BASE - 1, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    add_panel_labels(list(axes.flat))
    return fig, axes


def make_fig3(paired_df: pd.DataFrame) -> tuple:
    """Fig. 3 | Concordance: heatmap (a) + Sankey alluvial (b)."""
    df = paired_df.copy()
    df["risk_nonlab_cat"] = pd.Categorical(df["risk_nonlab_cat"], categories=RISK_CATS, ordered=True)
    df["risk_lab_cat"]    = pd.Categorical(df["risk_lab_cat"],    categories=RISK_CATS, ordered=True)
    df = df.dropna(subset=["risk_nonlab_cat", "risk_lab_cat"])

    ct = pd.crosstab(df["risk_nonlab_cat"], df["risk_lab_cat"])
    for c in RISK_CATS:
        if c not in ct.columns: ct[c] = 0
        if c not in ct.index:   ct.loc[c] = 0
    ct = ct.loc[RISK_CATS, RISK_CATS]

    fig, axes = apply_nature_style(nrows=1, ncols=2, layout="double",
                                   height_in=3.5, constrained_layout=True)
    ax_a, ax_b = axes.flat

    pct_row = ct.div(ct.sum(axis=1), axis=0) * 100
    n_cats  = len(RISK_CATS)

    cmap_diag  = mpl.colors.to_rgba("#0072B2", alpha=0.9)
    cmap_above = mpl.colors.to_rgba("#E69F00", alpha=0.85)
    cmap_below = mpl.colors.to_rgba("#CCCCCC", alpha=0.70)

    for r_i, rcat in enumerate(RISK_CATS):
        for c_i, ccat in enumerate(RISK_CATS):
            val = pct_row.iloc[r_i, c_i]
            raw = ct.iloc[r_i, c_i]
            if r_i == c_i:
                fc = plt.cm.Blues(0.4 + val / 200)
            elif c_i > r_i:
                fc = plt.cm.Oranges(0.3 + val / 150)
            else:
                fc = (0.9, 0.9, 0.9, 0.6)
            rect = mpatches.FancyBboxPatch(
                (c_i - 0.45, r_i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.03",
                facecolor=fc, edgecolor="white", linewidth=0.4,
            )
            ax_a.add_patch(rect)
            if val > 1:
                fc_txt = "white" if val > 40 else "black"
                ax_a.text(c_i, r_i, f"{val:.0f}%\n(n={raw:,})",
                          ha="center", va="center", fontsize=max(FONT_BASE - 2, 5),
                          color=fc_txt, linespacing=1.3)

    ax_a.set_xlim(-0.5, n_cats - 0.5)
    ax_a.set_ylim(-0.5, n_cats - 0.5)
    ax_a.invert_yaxis()
    ax_a.set_xticks(range(n_cats))
    ax_a.set_yticks(range(n_cats))
    short = ["<5%", "5–10%", "10–20%", "20–30%", "≥30%"]
    ax_a.set_xticklabels(short, rotation=30, ha="right", fontsize=max(FONT_BASE - 1, 5))
    ax_a.set_yticklabels(short, fontsize=max(FONT_BASE - 1, 5))
    ax_a.set_xlabel("Laboratory model category", fontsize=FONT_BASE)
    ax_a.set_ylabel("Non-laboratory model category", fontsize=FONT_BASE)
    ax_a.set_title("Agreement heatmap", fontsize=FONT_BASE, pad=3)
    ax_a.set_aspect("equal")
    for sp in ax_a.spines.values(): sp.set_visible(False)
    ax_a.tick_params(length=0)

    ax_a.plot([-0.5, n_cats - 0.5], [-0.5, n_cats - 0.5],
              color="#D55E00", linewidth=LW_THIN, linestyle="--", zorder=5)

    nl_counts = df["risk_nonlab_cat"].value_counts().reindex(RISK_CATS).fillna(0)
    lb_counts = df["risk_lab_cat"].value_counts().reindex(RISK_CATS).fillna(0)
    total     = len(df)

    ax_b.set_xlim(-0.2, 1.2)
    ax_b.set_ylim(0, total)

    bar_w = 0.08
    nl_tops = {}
    lb_tops = {}
    nl_cum, lb_cum = 0, 0

    for cat in RISK_CATS:
        n_nl = nl_counts[cat]; n_lb = lb_counts[cat]
        r, g, b, a = mpl.colors.to_rgba(RISK_COLORS[cat]) if cat != "<5%" else (0.75, 0.75, 0.75, 1.0)
        ax_b.barh(nl_cum + n_nl / 2, bar_w, left=-bar_w, height=n_nl, color=RISK_COLORS[cat] if cat != "<5%" else "#AAAAAA",
                  edgecolor="white", linewidth=0.3, align="center")
        ax_b.barh(lb_cum + n_lb / 2, bar_w, left=1.0,    height=n_lb, color=RISK_COLORS[cat] if cat != "<5%" else "#AAAAAA",
                  edgecolor="white", linewidth=0.3, align="center")
        nl_tops[cat] = (nl_cum, nl_cum + n_nl)
        lb_tops[cat] = (lb_cum, lb_cum + n_lb)
        nl_cum += n_nl; lb_cum += n_lb

    flow_locs = {c: nl_tops[c][0] for c in RISK_CATS}
    flow_locs_lb = {c: lb_tops[c][0] for c in RISK_CATS}

    for r_cat in RISK_CATS:
        for c_cat in RISK_CATS:
            n = ct.loc[r_cat, c_cat]
            if n < 5:
                flow_locs[r_cat] += n; flow_locs_lb[c_cat] += n
                continue
            y0s = flow_locs[r_cat]
            y0e = y0s + n
            y1s = flow_locs_lb[c_cat]
            y1e = y1s + n

            from matplotlib.patches import PathPatch
            from matplotlib.path import Path as MPath
            xs = [0, 0.35, 0.65, 1.0, 1.0, 0.65, 0.35, 0, 0]
            ys = [y0s, y0s, y1s, y1s, y1e, y1e, y0e, y0e, y0s]
            verts = list(zip(xs, ys))
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            path = MPath(verts, codes)
            is_concordant = (r_cat == c_cat)
            color = RISK_COLORS[r_cat] if cat != "<5%" else "#BBBBBB"
            alpha = 0.55 if is_concordant else 0.30
            pp = PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
            ax_b.add_patch(pp)

            flow_locs[r_cat]    += n
            flow_locs_lb[c_cat] += n

    nl_cum = 0
    for cat in RISK_CATS:
        n_nl = nl_counts[cat]
        ax_b.text(-bar_w - 0.01, nl_cum + n_nl / 2,
                  f"{cat}\n({n_nl/total*100:.0f}%)",
                  ha="right", va="center", fontsize=max(FONT_BASE - 2, 5))
        nl_cum += n_nl
    lb_cum = 0
    for cat in RISK_CATS:
        n_lb = lb_counts[cat]
        ax_b.text(1.0 + bar_w + 0.01, lb_cum + n_lb / 2,
                  f"{cat}\n({n_lb/total*100:.0f}%)",
                  ha="left", va="center", fontsize=max(FONT_BASE - 2, 5))
        lb_cum += n_lb

    ax_b.text(0,   total * 1.03, "Non-Lab", ha="center", fontsize=FONT_BASE, fontweight="bold", color=C_NONLAB)
    ax_b.text(1.0, total * 1.03, "Lab",     ha="center", fontsize=FONT_BASE, fontweight="bold", color=C_LAB)

    ax_b.set_title("Reclassification flow", fontsize=FONT_BASE, pad=3)
    ax_b.axis("off")

    add_panel_labels(list(axes.flat))
    return fig, axes


def make_fig4(nl_df: pd.DataFrame) -> tuple:
    """Fig. 4 | Age-stratified escalation + prevalence-by-threshold."""
    df = _ensure_age_band(nl_df).copy()
    df["risk_nonlab"] = pd.to_numeric(df["risk_nonlab"], errors="coerce")
    df["risk_lab"]    = pd.to_numeric(df["risk_lab"],    errors="coerce")

    fig, axes = apply_nature_style(nrows=1, ncols=2, layout="double",
                                   height_in=3.2, constrained_layout=True)
    ax_a, ax_b = axes.flat

    bands   = AGE_BANDS
    y_10, lo_10, hi_10 = [], [], []
    y_20, lo_20, hi_20 = [], [], []
    ns = []
    for b in bands:
        sub = df[df["age_band"] == b].dropna(subset=["risk_nonlab"])
        n   = len(sub)
        ns.append(n)
        n10  = (sub["risk_nonlab"] >= 10).sum()
        n20  = (sub["risk_nonlab"] >= 20).sum()
        lo, hi = _wilson_ci(n10, n)
        y_10.append(n10 / n * 100 if n else 0); lo_10.append(lo); hi_10.append(hi)
        lo, hi = _wilson_ci(n20, n)
        y_20.append(n20 / n * 100 if n else 0); lo_20.append(lo); hi_20.append(hi)

    x = np.arange(len(bands))
    y_10 = np.array(y_10); lo_10 = np.array(lo_10); hi_10 = np.array(hi_10)
    y_20 = np.array(y_20); lo_20 = np.array(lo_20); hi_20 = np.array(hi_20)

    ax_a.fill_between(x, lo_10, hi_10, alpha=0.15, color=C_LAB)
    ax_a.plot(x, y_10, "o-", color=C_LAB, linewidth=LW_DEFAULT, markersize=3.5,
              markerfacecolor=C_LAB, label="≥10% (Non-Lab)")
    ax_a.fill_between(x, lo_20, hi_20, alpha=0.15, color=C_NONLAB)
    ax_a.plot(x, y_20, "s--", color=C_NONLAB, linewidth=LW_DEFAULT, markersize=3.0,
              markerfacecolor=C_NONLAB, label="≥20% (Non-Lab)")

    for i, (xi, n) in enumerate(zip(x, ns)):
        ax_a.text(xi, -4, f"n={n:,}", ha="center", fontsize=max(FONT_BASE - 2, 5),
                  color="#777", rotation=45)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([b.replace("-", "–") for b in bands], rotation=30, ha="right", fontsize=FONT_BASE - 1)
    ax_a.set_ylabel("Prevalence of high risk (%)", fontsize=FONT_BASE)
    ax_a.set_ylim(-8, 100)
    ax_a.legend(frameon=False, fontsize=FONT_BASE - 1, loc="upper left")
    ax_a.set_title("Age-stratified risk escalation", fontsize=FONT_BASE, pad=3)
    clean_axes(ax_a, left=True, bottom=True)

    thresholds = np.arange(1, 31)
    df_nl  = df.dropna(subset=["risk_nonlab"])
    df_lab = df.dropna(subset=["risk_lab"])

    prev_nl  = [(df_nl["risk_nonlab"]  >= t).mean() * 100 for t in thresholds]
    prev_lab = [(df_lab["risk_lab"]    >= t).mean() * 100 for t in thresholds]

    ax_b.plot(thresholds, prev_nl,  color=C_NONLAB, linewidth=LW_DEFAULT, label="Non-Laboratory", linestyle="-")
    ax_b.plot(thresholds, prev_lab, color=C_LAB,    linewidth=LW_DEFAULT, label="Laboratory",     linestyle="--")
    ax_b.fill_between(thresholds, prev_nl, prev_lab, alpha=0.10, color="grey")

    ax_b.axvline(10, color="black", linewidth=LW_MIN, linestyle=":", alpha=0.7)
    ax_b.axvline(20, color="black", linewidth=LW_MIN, linestyle=":", alpha=0.7)
    ax_b.text(10.3, 95, "10%", fontsize=max(FONT_BASE - 2, 5), color="#444")
    ax_b.text(20.3, 95, "20%", fontsize=max(FONT_BASE - 2, 5), color="#444")
    ax_b.text(14.5, 60, "Gap", fontsize=max(FONT_BASE - 2, 5), color="#999", style="italic")

    ax_b.set_xlabel("Risk threshold (%)", fontsize=FONT_BASE)
    ax_b.set_ylabel("Prevalence (%)", fontsize=FONT_BASE)
    ax_b.set_title("Prevalence by risk threshold", fontsize=FONT_BASE, pad=3)
    ax_b.legend(frameon=False, fontsize=FONT_BASE - 1, loc="upper right")
    ax_b.set_xlim(1, 30)
    ax_b.set_ylim(0, 102)
    clean_axes(ax_b, left=True, bottom=True)

    add_panel_labels(list(axes.flat))
    return fig, axes


def make_fig5(paired_df: pd.DataFrame) -> tuple:
    """Fig. 5 | Bland–Altman plots stratified by gender."""
    df = paired_df.copy()
    df["risk_nonlab"] = pd.to_numeric(df["risk_nonlab"], errors="coerce")
    df["risk_lab"]    = pd.to_numeric(df["risk_lab"],    errors="coerce")
    df["gender"]      = df["gender"].str.upper().str.strip()
    df = df.dropna(subset=["risk_nonlab", "risk_lab"])

    df["mean_risk"] = (df["risk_nonlab"] + df["risk_lab"]) / 2.0
    df["diff_risk"] = df["risk_lab"] - df["risk_nonlab"]

    fig, axes = apply_nature_style(nrows=1, ncols=2, layout="double",
                                   height_in=3.0, constrained_layout=True)

    for ax, (gname, color, label) in zip(axes.flat, [
        ("M", C_MALE,   "Males (n={n})"),
        ("F", C_FEMALE, "Females (n={n})"),
    ]):
        sub = df[df["gender"] == gname]
        n   = len(sub)
        if n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        mean_diff = sub["diff_risk"].mean()
        sd_diff   = sub["diff_risk"].std()
        loa_hi    = mean_diff + 1.96 * sd_diff
        loa_lo    = mean_diff - 1.96 * sd_diff

        from matplotlib.colors import Normalize
        xy  = np.vstack([sub["mean_risk"], sub["diff_risk"]])
        try:
            kde = stats.gaussian_kde(xy)
            zz  = kde(xy)
        except Exception:
            zz  = np.ones(n)
        norm = Normalize(vmin=zz.min(), vmax=zz.max())

        sc = ax.scatter(sub["mean_risk"], sub["diff_risk"],
                        c=zz, cmap="Blues", norm=norm,
                        s=4, alpha=0.55, linewidths=0,
                        rasterized=True)

        xmin, xmax = sub["mean_risk"].min(), sub["mean_risk"].max()
        for y_val, ls, lw, lbl in [
            (mean_diff, "-",  LW_DEFAULT, f"Mean bias: {mean_diff:+.1f} pp"),
            (loa_hi,    "--", LW_THIN,    f"+1.96 SD: {loa_hi:+.1f} pp"),
            (loa_lo,    "--", LW_THIN,    f"−1.96 SD: {loa_lo:+.1f} pp"),
        ]:
            ax.axhline(y_val, color=color, linewidth=lw, linestyle=ls, alpha=0.85)
            ax.text(xmax * 1.01, y_val, lbl, fontsize=max(FONT_BASE - 2, 5),
                    va="center", color=color, clip_on=False)

        ax.axhline(0, color="black", linewidth=LW_MIN, linestyle=":", alpha=0.5)

        ax.set_xlabel("Mean risk (Lab + Non-Lab) / 2  (%)", fontsize=FONT_BASE)
        ax.set_ylabel("Difference (Lab − Non-Lab)  (pp)", fontsize=FONT_BASE)
        ax.set_title(label.format(n=n), fontsize=FONT_BASE, pad=3)
        clean_axes(ax, left=True, bottom=True)

    add_panel_labels(list(axes.flat))
    return fig, axes


def make_fig6(nl_df: pd.DataFrame, site_df: pd.DataFrame | None = None) -> tuple:
    """Fig. 6 | Site-level heterogeneity: Cleveland dot plot with Wilson CIs."""
    df = _ensure_age_band(nl_df).copy()
    df["risk_nonlab"] = pd.to_numeric(df["risk_nonlab"], errors="coerce")

    if site_df is not None:
        loc_col = next((c for c in site_df.columns
                        if "urban" in c.lower() or "type" in c.lower()), None)
        if loc_col:
            df = df.merge(site_df[["site_id", loc_col]], on="site_id", how="left")
            df.rename(columns={loc_col: "_loc_type"}, inplace=True)
        else:
            df["_loc_type"] = "Unknown"
    else:
        df["_loc_type"] = "Unknown"

    sites_agg = []
    for sid, sg in df.groupby("site_id"):
        sg = sg.dropna(subset=["risk_nonlab"])
        n  = len(sg)
        if n < 10:
            continue
        n_hi = (sg["risk_nonlab"] >= 10).sum()
        prev = n_hi / n * 100
        lo, hi = _wilson_ci(n_hi, n)
        loc = sg["_loc_type"].mode()[0] if "_loc_type" in sg else "Unknown"
        sites_agg.append({"site_id": sid, "n": n, "prev": prev,
                          "lo": lo, "hi": hi, "loc": loc})

    if not sites_agg:
        fig, axes = apply_nature_style(nrows=1, ncols=1, layout="single")
        axes.flat[0].text(0.5, 0.5, "No site data", ha="center", va="center", transform=axes.flat[0].transAxes)
        return fig, axes

    site_agg_df = pd.DataFrame(sites_agg).sort_values("prev").reset_index(drop=True)
    n_sites = len(site_agg_df)

    loc_color_map = {"Urban": "#0072B2", "Semi-Urban": "#E69F00", "Rural": "#009E73", "Unknown": "#CCCCCC"}
    loc_map_lower = {k.lower(): v for k, v in loc_color_map.items()}

    def _get_loc_color(loc: str) -> str:
        """Get loc color."""
        for k, v in loc_map_lower.items():
            if k in str(loc).lower():
                return v
        return "#CCCCCC"

    h = min(max(n_sites * 0.22, 3.0), 9.72)
    fig, axes = apply_nature_style(nrows=1, ncols=1, layout="double",
                                   height_in=h, constrained_layout=True)
    ax = axes.flat[0]

    y_pos = np.arange(n_sites)
    pooled = site_agg_df["prev"].mean()

    for i, row in site_agg_df.iterrows():
        color = _get_loc_color(row["loc"])
        ax.plot([row["lo"], row["hi"]], [i, i], color=color, linewidth=LW_THIN, alpha=0.7)
        ax.scatter(row["prev"], i, color=color, s=18, zorder=4, edgecolors="white", linewidths=0.3)

    ax.axvline(pooled, color="black", linewidth=LW_DEFAULT, linestyle="--", zorder=2, alpha=0.8)
    ax.text(pooled + 0.4, n_sites - 1, f"Pooled\n{pooled:.1f}%",
            fontsize=max(FONT_BASE - 1, 5), va="top", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(site_agg_df["site_id"].astype(str), fontsize=max(FONT_BASE - 2, 5))
    ax.set_xlabel("Prevalence of ≥10% ten-year CVD risk (%)", fontsize=FONT_BASE)
    ax.set_title("Site-level CVD risk prevalence", fontsize=FONT_BASE, pad=4)
    ax.set_xlim(0, min(site_agg_df["hi"].max() + 5, 100))

    seen_locs = site_agg_df["loc"].unique()
    leg_handles = [
        mpatches.Patch(facecolor=_get_loc_color(loc), label=loc, linewidth=LW_MIN)
        for loc in ["Urban", "Semi-Urban", "Rural", "Unknown"]
        if any(loc.lower() in str(x).lower() for x in seen_locs)
    ]
    ax.legend(handles=leg_handles, title="Location type", frameon=False,
              fontsize=FONT_BASE - 1, title_fontsize=FONT_BASE - 1, loc="lower right")
    clean_axes(ax, left=True, bottom=True)

    return fig, axes.reshape(1)
