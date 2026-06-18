"""
nature_style.py
===============
Reusable Matplotlib configurator that strictly enforces the
 Reviews / Scientific Reports Artwork Guidelines.

Usage
-----
    from utils.nature_style import apply_nature_style, OKABE_ITO, save_figure

    fig, axes = apply_nature_style(ncols=2, layout="double")
    # … plot code …
    save_figure(fig, "output_dir/my_figure")  # saves .pdf and .svg
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Literal, Sequence

# ---------------------------------------------------------------------------
# 1. Dimensional constants ( Reviews Artwork Guidelines)
# ---------------------------------------------------------------------------
# Single-column : 89 mm  → 3.504 inches
# Double-column : 183 mm → 7.205 inches
# Maximum height: 247 mm → 9.724 inches
SINGLE_COL_IN: float = 89  / 25.4   # 3.504 in
DOUBLE_COL_IN: float = 183 / 25.4   # 7.205 in
MAX_HEIGHT_IN: float = 247 / 25.4   # 9.724 in

# ---------------------------------------------------------------------------
# 2. Typographic constants
# ---------------------------------------------------------------------------
FONT_BASE   = 7    # pt  – axis labels, legend, tick labels
FONT_TITLE  = 8    # pt  – panel titles / sub-headings
FONT_PANEL  = 8    # pt  – panel labels (a, b, c)
FONT_MIN    = 5    # pt  – absolute minimum for any text
FONT_FAMILY = "sans-serif"
FONT_SANS   = ["Arial", "Helvetica", "DejaVu Sans"]  # fallback chain

# ---------------------------------------------------------------------------
# 3. Line-weight constants (pt → points used directly by Matplotlib)
# ---------------------------------------------------------------------------
LW_DEFAULT  = 0.75   # pt – plot lines, axes
LW_THIN     = 0.5    # pt – ticks, minor lines
LW_MIN      = 0.25   # pt – absolute minimum
LW_THICK    = 1.0    # pt – emphasis lines, error bar caps

# ---------------------------------------------------------------------------
# 4. Okabe-Ito colorblind-safe palette (8 colours)
# Red-green combinations **never** used together to distinguish data.
# ---------------------------------------------------------------------------
OKABE_ITO: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion (avoid pairing with #009E73 for key distinctions)
    "#CC79A7",  # reddish-purple
    "#000000",  # black
]

# Curated 2-colour pairs guaranteed to be distinguishable by all CVD types
PAIR_MALE_FEMALE: tuple[str, str] = ("#0072B2", "#CC79A7")   # blue / pink-purple
PAIR_LAB_NONLAB:  tuple[str, str] = ("#0072B2", "#E69F00")   # blue / orange
PAIR_HIGH_LOW:    tuple[str, str] = ("#D55E00", "#56B4E9")   # vermillion / sky-blue

# Contextual / Main Backgrounds
NATURE_STONE = ['#F3F2E9', '#E6E2D1', '#CFCBA9', '#B2AD81', '#8C8861', '#666345']
NATURE_GREY  = ['#EBECEF', '#D1D4DB', '#A8AEBD', '#7E869B', '#5A6175', '#393E4D']

# Main Accents (Primary Data)
NATURE_RED   = ['#F6CDCD', '#EFA0A0', '#E26D6D', '#CE3737', '#A12626', '#711A1A']
NATURE_BLUE  = ['#CDE3F6', '#A0CBEF', '#6DABDE', '#3783CE', '#2661A1', '#1A4271']
NATURE_YELLOW= ['#F6EECD', '#EFDCA0', '#E2C66D', '#CEAD37', '#A18626', '#715D1A']

# Extended Palette (For complex categorical data)
NATURE_OLIVE = ['#EEF4B8', '#DCE87C', '#C2D148', '#99A82B', '#6D7A1A', '#454F0D']
NATURE_GREEN = ['#D1E8CC', '#A5D49B', '#72BB62', '#3D9B2B', '#27701A', '#154A0F']
NATURE_TEAL  = ['#CAEAEB', '#92D7D9', '#54BDC1', '#219DA1', '#127073', '#0A494B']
NATURE_PURPLE= ['#E4CAEA', '#CD92D9', '#B154C1', '#8F21A1', '#651273', '#400A4B']
NATURE_ORANGE= ['#F6DECC', '#EFBE9B', '#E29762', '#CE6B2B', '#A14E1A', '#71320F']



# ---------------------------------------------------------------------------
# 5. Core Matplotlib rc configuration
# ---------------------------------------------------------------------------

def configure_rc() -> None:
    """
    Apply all  Reviews Artwork Guidelines to Matplotlib's global rc
    parameters.  Call once at module load — side-effect free otherwise.
    """
    rc: dict = {
        # --- Font ---
        "font.family":          FONT_FAMILY,
        "font.sans-serif":      FONT_SANS,
        "font.size":            FONT_BASE,
        "axes.titlesize":       FONT_TITLE,
        "axes.labelsize":       FONT_BASE,
        "xtick.labelsize":      FONT_BASE,
        "ytick.labelsize":      FONT_BASE,
        "legend.fontsize":      FONT_BASE,
        "legend.title_fontsize": FONT_BASE,

        # --- Line weights ---
        "axes.linewidth":       LW_THIN,
        "xtick.major.width":    LW_THIN,
        "ytick.major.width":    LW_THIN,
        "xtick.minor.width":    LW_MIN,
        "ytick.minor.width":    LW_MIN,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "lines.linewidth":      LW_DEFAULT,
        "patch.linewidth":      LW_THIN,
        "errorbar.capsize":     2.5,
        "hatch.linewidth":      LW_THIN,

        # --- Spines: remove top + right ---
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # --- Color cycle ---
        "axes.prop_cycle":      mpl.cycler(color=OKABE_ITO),

        # --- Grid ---
        "axes.grid":            False,
        "grid.linewidth":       LW_MIN,
        "grid.alpha":           0.4,

        # --- Legend ---
        "legend.frameon":       False,
        "legend.borderpad":     0.2,
        "legend.labelspacing":  0.3,

        # --- Background ---
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",

        # --- PDF / SVG output ---
        "pdf.fonttype":         42,   # embed TrueType fonts in PDF
        "svg.fonttype":         "none",
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "figure.dpi":           150,  # screen preview
    }
    mpl.rcParams.update(rc)


# Apply immediately on import
configure_rc()


# ---------------------------------------------------------------------------
# 6. Figure factory
# ---------------------------------------------------------------------------

def apply_nature_style(
    nrows: int = 1,
    ncols: int = 1,
    layout: Literal["single", "double"] = "double",
    height_in: float | None = None,
    aspect: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a -compliant Figure + Axes array.

    Parameters
    ----------
    nrows, ncols : grid dimensions
    layout       : "single" (89 mm) or "double" (183 mm) column width
    height_in    : explicit height in inches (optional)
    aspect       : height-to-width ratio when height_in is None
    **subplot_kw : forwarded to plt.subplots()

    Returns
    -------
    fig, axes  – (axes is always an ndarray, even if nrows==ncols==1)
    """
    configure_rc()   # always re-apply in case global state was reset

    width_in = SINGLE_COL_IN if layout == "single" else DOUBLE_COL_IN

    if height_in is None:
        if aspect is not None:
            height_in = width_in * aspect
        else:
            # Default: golden-ratio-ish per row
            height_in = min(width_in * 0.65 * nrows / max(ncols, 1), MAX_HEIGHT_IN)

    height_in = min(height_in, MAX_HEIGHT_IN)

    default_kw = {
        "constrained_layout": True,
        "figsize": (width_in, height_in),
    }
    default_kw.update(subplot_kw)

    fig, axes = plt.subplots(nrows, ncols, **default_kw)

    # Normalise axes to always be an ndarray
    axes = np.atleast_1d(axes)

    return fig, axes


# ---------------------------------------------------------------------------
# 7. Panel-label helper
# ---------------------------------------------------------------------------

def add_panel_labels(
    axes: Sequence[plt.Axes],
    labels: Sequence[str] | None = None,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = FONT_PANEL,
) -> None:
    """
    Add lowercase bold panel labels (a, b, c…) to each axis.

    Parameters
    ----------
    axes   : sequence of Axes objects
    labels : explicit labels; defaults to ['a','b','c',…]
    x, y   : label position in axis-fraction coordinates
    """
    if labels is None:
        labels = [chr(ord("a") + i) for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(
            x, y, label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="bottom",
            ha="right",
            fontfamily=FONT_FAMILY,
        )


# ---------------------------------------------------------------------------
# 8. Spine / tick cleanup helpers
# ---------------------------------------------------------------------------

def clean_axes(
    ax: plt.Axes,
    left: bool = True,
    bottom: bool = True,
) -> None:
    """
    Remove unwanted spines and ensure ticks point outward.
    Top + right spines are already disabled via rc; call this for
    finer control (e.g. removing left spine for horizontal bar charts).
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)

    if left:
        ax.spines["left"].set_linewidth(LW_THIN)
    if bottom:
        ax.spines["bottom"].set_linewidth(LW_THIN)

    ax.tick_params(direction="out", width=LW_THIN)


# ---------------------------------------------------------------------------
# 9. Save utility — always exports PDF + SVG
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    stem: str | Path,
    dpi: int = 600,
    formats: tuple[str, ...] = ("pdf", "svg"),
    tight: bool = True,
) -> list[str]:
    """
    Save *fig* to PDF and SVG (or any combination via *formats*).

    Parameters
    ----------
    fig     : the Matplotlib Figure to save
    stem    : path without extension (directory will be created)
    dpi     : resolution for raster elements embedded in vector formats
    formats : output formats to write
    tight   : whether to use bbox_inches='tight'

    Returns
    -------
    List of absolute paths to saved files.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(
            str(path),
            format=fmt,
            dpi=dpi,
            bbox_inches="tight" if tight else None,
            facecolor=fig.get_facecolor(),
        )
        saved.append(str(path))

    return saved


# ---------------------------------------------------------------------------
# 10. Convenience: risk-band colour map (publication-safe, no red-green pair)
# ---------------------------------------------------------------------------
# Maps WHO risk band label → face colour
RISK_BAND_COLORS: dict[str, str] = {
    "<5%":          "#CCCCCC",   # light grey
    "5% to <10%":   "#F0E442",   # yellow (Okabe-Ito)
    "10% to <20%":  "#E69F00",   # orange
    "20% to <30%":  "#D55E00",   # vermillion
    "≥30%":         "#000000",   # black
}

RISK_BAND_ORDER = ["<5%", "5% to <10%", "10% to <20%", "20% to <30%", "≥30%"]


def risk_legend_patches(ax: plt.Axes | None = None) -> list[mpatches.Patch]:
    """Return legend handles for the five WHO risk bands."""
    patches = [
        mpatches.Patch(facecolor=RISK_BAND_COLORS[b], label=b, linewidth=LW_MIN)
        for b in RISK_BAND_ORDER
    ]
    if ax is not None:
        ax.legend(handles=patches, title="10-yr CVD Risk",
                  loc="upper right", frameon=False,
                  fontsize=FONT_BASE, title_fontsize=FONT_BASE)
    return patches
