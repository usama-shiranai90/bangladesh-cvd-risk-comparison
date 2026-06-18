"""
nature_style.py
===============
Reusable Matplotlib configurator that strictly enforces the
standard publication artwork guidelines.

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

SINGLE_COL_IN: float = 89  / 25.4
DOUBLE_COL_IN: float = 183 / 25.4
MAX_HEIGHT_IN: float = 247 / 25.4

FONT_BASE   = 7
FONT_TITLE  = 8
FONT_PANEL  = 8
FONT_MIN    = 5
FONT_FAMILY = "sans-serif"
FONT_SANS   = ["Arial", "Helvetica", "DejaVu Sans"]

LW_DEFAULT  = 0.75
LW_THIN     = 0.5
LW_MIN      = 0.25
LW_THICK    = 1.0

OKABE_ITO: list[str] = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

PAIR_MALE_FEMALE: tuple[str, str] = ("#0072B2", "#CC79A7")
PAIR_LAB_NONLAB:  tuple[str, str] = ("#0072B2", "#E69F00")
PAIR_HIGH_LOW:    tuple[str, str] = ("#D55E00", "#56B4E9")

NATURE_STONE = ['#F3F2E9', '#E6E2D1', '#CFCBA9', '#B2AD81', '#8C8861', '#666345']
NATURE_GREY  = ['#EBECEF', '#D1D4DB', '#A8AEBD', '#7E869B', '#5A6175', '#393E4D']

NATURE_RED   = ['#F6CDCD', '#EFA0A0', '#E26D6D', '#CE3737', '#A12626', '#711A1A']
NATURE_BLUE  = ['#CDE3F6', '#A0CBEF', '#6DABDE', '#3783CE', '#2661A1', '#1A4271']
NATURE_YELLOW= ['#F6EECD', '#EFDCA0', '#E2C66D', '#CEAD37', '#A18626', '#715D1A']

NATURE_OLIVE = ['#EEF4B8', '#DCE87C', '#C2D148', '#99A82B', '#6D7A1A', '#454F0D']
NATURE_GREEN = ['#D1E8CC', '#A5D49B', '#72BB62', '#3D9B2B', '#27701A', '#154A0F']
NATURE_TEAL  = ['#CAEAEB', '#92D7D9', '#54BDC1', '#219DA1', '#127073', '#0A494B']
NATURE_PURPLE= ['#E4CAEA', '#CD92D9', '#B154C1', '#8F21A1', '#651273', '#400A4B']
NATURE_ORANGE= ['#F6DECC', '#EFBE9B', '#E29762', '#CE6B2B', '#A14E1A', '#71320F']


def configure_rc() -> None:
    """Apply all  Reviews Artwork Guidelines to Matplotlib's global rc"""
    rc: dict = {
        "font.family":          FONT_FAMILY,
        "font.sans-serif":      FONT_SANS,
        "font.size":            FONT_BASE,
        "axes.titlesize":       FONT_TITLE,
        "axes.labelsize":       FONT_BASE,
        "xtick.labelsize":      FONT_BASE,
        "ytick.labelsize":      FONT_BASE,
        "legend.fontsize":      FONT_BASE,
        "legend.title_fontsize": FONT_BASE,

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

        "axes.spines.top":      False,
        "axes.spines.right":    False,

        "axes.prop_cycle":      mpl.cycler(color=OKABE_ITO),

        "axes.grid":            False,
        "grid.linewidth":       LW_MIN,
        "grid.alpha":           0.4,

        "legend.frameon":       False,
        "legend.borderpad":     0.2,
        "legend.labelspacing":  0.3,

        "figure.facecolor":     "white",
        "axes.facecolor":       "white",

        "pdf.fonttype":         42,
        "svg.fonttype":         "none",
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "figure.dpi":           150,
    }
    mpl.rcParams.update(rc)


configure_rc()


def apply_nature_style(
    nrows: int = 1,
    ncols: int = 1,
    layout: Literal["single", "double"] = "double",
    height_in: float | None = None,
    aspect: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a -compliant Figure + Axes array."""
    configure_rc()

    width_in = SINGLE_COL_IN if layout == "single" else DOUBLE_COL_IN

    if height_in is None:
        if aspect is not None:
            height_in = width_in * aspect
        else:
            height_in = min(width_in * 0.65 * nrows / max(ncols, 1), MAX_HEIGHT_IN)

    height_in = min(height_in, MAX_HEIGHT_IN)

    default_kw = {
        "constrained_layout": True,
        "figsize": (width_in, height_in),
    }
    default_kw.update(subplot_kw)

    fig, axes = plt.subplots(nrows, ncols, **default_kw)

    axes = np.atleast_1d(axes)

    return fig, axes


def add_panel_labels(
    axes: Sequence[plt.Axes],
    labels: Sequence[str] | None = None,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: int = FONT_PANEL,
) -> None:
    """Add lowercase bold panel labels (a, b, c…) to each axis."""
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


def clean_axes(
    ax: plt.Axes,
    left: bool = True,
    bottom: bool = True,
) -> None:
    """Remove unwanted spines and ensure ticks point outward."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)

    if left:
        ax.spines["left"].set_linewidth(LW_THIN)
    if bottom:
        ax.spines["bottom"].set_linewidth(LW_THIN)

    ax.tick_params(direction="out", width=LW_THIN)


def save_figure(
    fig: plt.Figure,
    stem: str | Path,
    dpi: int = 600,
    formats: tuple[str, ...] = ("pdf", "svg"),
    tight: bool = True,
) -> list[str]:
    """Save *fig* to PDF and SVG (or any combination via *formats*)."""
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


RISK_BAND_COLORS: dict[str, str] = {
    "<5%":          "#CCCCCC",
    "5% to <10%":   "#F0E442",
    "10% to <20%":  "#E69F00",
    "20% to <30%":  "#D55E00",
    "≥30%":         "#000000",
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
