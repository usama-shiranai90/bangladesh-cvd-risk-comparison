"""
SciencePlots helpers for consistent visualization styling across the application.
Provides standardized colors, fonts, and plotting functions for publication-ready figures.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path
import seaborn as sns
import numpy as np
import pandas as pd
import io
import streamlit as st


def setup_scienceplots_style():
    """Initialize SciencePlots styling with fallback for systems without it."""
    try:
        import scienceplots
        plt.style.use(['science', 'nature', 'no-latex'])
        sns.set_theme(style="whitegrid", context="talk")
        return True
    except Exception:
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        return False

SCIENCEPLOTS_AVAILABLE = setup_scienceplots_style()


NATURE_COLORS = {
    'blue': '#0C5DA5',
    'orange': '#FF9500',
    'red': '#FF2C00',
    'purple': '#845B97',
    'green': '#2E7D32',
    'yellow': '#FDD835',
    'teal': '#00B8D4',
    'gray': '#6E6E6E',
}

RISK_COLORS = {
    '<5%': '#2E7D32',
    '5% to <10%': '#FDD835',
    '10% to <20%': '#FB8C00',
    '20% to <30%': '#E53935',
    '≥30%': '#8E24AA'
}

RISK_LABELS = ['<5%', '5% to <10%', '10% to <20%', '20% to <30%', '≥30%']

AGREEMENT_COLORS = {
    'Concordance': '#BDBDBD',
    'Underestimation': '#D32F2F',
    'Overestimation': '#FBC02D',
}

COMPARISON_COLORS = {
    'lab': '#0C5DA5',
    'nonlab': '#845B97',
    'Lab-based': '#0C5DA5',
    'Non-lab based': '#845B97',
}


def get_figure_and_ax(figsize=(10, 6), nrows=1, ncols=1, **kwargs):
    """Create a figure with standardized styling."""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    return fig, ax


def apply_nature_style(ax, title=None, xlabel=None, ylabel=None, legend=True, grid=True):
    """Apply Journal-like styling to an axis."""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='serif')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontfamily='serif')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontfamily='serif')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    if legend and ax.get_legend():
        ax.legend(frameon=True, framealpha=0.9, edgecolor='gray')
    
    ax.tick_params(axis='both', which='major', labelsize=10, direction='out', length=4)
    
    plt.tight_layout()


def create_grouped_bar_chart(data, x_col, y_cols, labels, colors, title, xlabel, ylabel,
                             figsize=(10, 6), error_cols=None, show_values=True):
    """Create a grouped bar chart with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    x = np.arange(len(data))
    width = 0.8 / len(y_cols)
    
    for i, (y_col, label, color) in enumerate(zip(y_cols, labels, colors)):
        offset = (i - len(y_cols)/2 + 0.5) * width
        yerr = data[error_cols[i]] if error_cols else None
        
        bars = ax.bar(x + offset, data[y_col], width, label=label, color=color,
                      yerr=yerr, capsize=3, edgecolor='black', linewidth=0.5)
        
        if show_values:
            for bar, val in zip(bars, data[y_col]):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           ha='center', va='bottom',
                           fontsize=8, fontfamily='serif')
    
    ax.set_xticks(x)
    ax.set_xticklabels(data[x_col], rotation=45, ha='right')
    
    apply_nature_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    
    return fig


def create_line_chart(data, x_col, y_cols, labels, colors, title, xlabel, ylabel,
                      figsize=(10, 6), markers=None, linestyles=None, error_cols=None,
                      reference_lines=None, show_ci=False):
    """Create a multi-series line chart with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    markers = markers or ['o'] * len(y_cols)
    linestyles = linestyles or ['-'] * len(y_cols)
    
    for i, (y_col, label, color) in enumerate(zip(y_cols, labels, colors)):
        ax.plot(data[x_col], data[y_col], 
                marker=markers[i], 
                linestyle=linestyles[i],
                color=color, 
                label=label, 
                linewidth=2, 
                markersize=8)
        
        if show_ci and error_cols and len(error_cols) > i:
            lower_col, upper_col = error_cols[i]
            ax.fill_between(data[x_col], data[lower_col], data[upper_col],
                           alpha=0.2, color=color)
    
    if reference_lines:
        for ref in reference_lines:
            ax.axhline(y=ref['y'], color=ref.get('color', 'gray'), 
                      linestyle='--', linewidth=1, label=ref.get('label', ''))
    
    apply_nature_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    
    return fig


def create_stacked_bar_chart(data, x_col, stack_cols, colors, labels, title, xlabel, ylabel,
                              figsize=(10, 6), show_percentages=True, horizontal=False):
    """Create a stacked bar chart with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    
    for col, color, label in zip(stack_cols, colors, labels):
        values = data[col].fillna(0).values
        
        if horizontal:
            bars = ax.barh(x, values, left=bottom, label=label, color=color,
                          edgecolor='black', linewidth=0.5)
        else:
            bars = ax.bar(x, values, bottom=bottom, label=label, color=color,
                         edgecolor='black', linewidth=0.5)
        
        if show_percentages:
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 3:
                    if horizontal:
                        ax.text(bottom[i] + val/2, i, f'{val:.0f}%',
                               ha='center', va='center', fontsize=8, color='white',
                               fontweight='bold', fontfamily='serif')
                    else:
                        ax.text(i, bottom[i] + val/2, f'{val:.0f}%',
                               ha='center', va='center', fontsize=8, color='white',
                               fontweight='bold', fontfamily='serif')
        
        bottom += values
    
    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(data[x_col])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(data[x_col], rotation=45, ha='right')
    
    apply_nature_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc='upper right', frameon=True)
    
    return fig


def create_heatmap(data, title, xlabel, ylabel, cmap='RdYlGn_r', center=None,
                   annot=True, fmt='.1f', figsize=(10, 8), cbar_label=''):
    """Create a heatmap with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    sns.heatmap(data, ax=ax, cmap=cmap, center=center, annot=annot, fmt=fmt,
                linewidths=0.5, linecolor='white', cbar_kws={'label': cbar_label})
    
    apply_nature_style(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=False, legend=False)
    
    return fig


def create_confusion_matrix(data, row_labels, col_labels, title,
                            xlabel='Predicted', ylabel='Actual',
                            cmap='Blues', figsize=(8, 6)):
    """Create a confusion matrix heatmap with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    im = ax.imshow(data, cmap=cmap)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontfamily='serif')
    
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            color = "white" if val > data.max() / 2 else "black"
            ax.text(j, i, f'{val:.0f}', ha="center", va="center", color=color,
                   fontsize=10, fontfamily='serif')
    
    apply_nature_style(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=False, legend=False)
    
    return fig


def create_pie_donut(values, labels, colors, title, figsize=(8, 8), hole=0.3):
    """Create a donut chart with consistent styling."""
    fig, ax = get_figure_and_ax(figsize=figsize)
    
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', pctdistance=0.75,
        startangle=90, wedgeprops=dict(width=1-hole, edgecolor='white')
    )
    
    for text in texts:
        text.set_fontfamily('serif')
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontfamily('serif')
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='serif')
    
    return fig


def create_sankey_alluvial(left_values, right_values, flow_matrix,
                           left_labels, right_labels, left_title, right_title,
                           title, figsize=(12, 8), left_color='#0C5DA5', right_color='#845B97'):
    """Create a Sankey-style alluvial diagram using Matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)
    
    CATS = left_labels
    COLOR_CONCORDANT = "#C8C8C8"
    COLOR_UNDER = "#FF2C00"
    COLOR_OVER = "#FF9500"
    ALPHA_RIBBON = 0.6
    
    if isinstance(flow_matrix, dict):
        flows = pd.DataFrame(flow_matrix).reindex(index=CATS, columns=CATS, fill_value=0)
    else:
        flows = flow_matrix
    
    left_counts = {cat: left_values.get(cat, 0) for cat in CATS}
    right_counts = {cat: right_values.get(cat, 0) for cat in CATS}
    
    total_n = sum(left_counts.values())
    
    GAP_PROPORTION = 0.05
    gap_height = total_n * GAP_PROPORTION
    total_drawing_height = total_n + (len(CATS) - 1) * gap_height
    
    X_LEFT_BAR = 0.05
    X_RIGHT_BAR = 0.95
    BAR_WIDTH = 0.03
    
    y_left_start = {}
    current_y = total_drawing_height
    for cat in CATS:
        h = left_counts[cat]
        y_left_start[cat] = current_y - h
        current_y -= (h + gap_height)
    
    y_right_start = {}
    current_y = total_drawing_height
    for cat in CATS:
        h = right_counts[cat]
        y_right_start[cat] = current_y - h
        current_y -= (h + gap_height)
    
    left_offsets = {c: 0 for c in CATS}
    right_offsets = {c: 0 for c in CATS}
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_drawing_height)
    ax.axis('off')
    
    for src in CATS:
        for tgt in CATS:
            val = flows.loc[src, tgt] if src in flows.index and tgt in flows.columns else 0
            if val > 0:
                y1_top = y_left_start[src] + left_counts[src] - left_offsets[src]
                y1_bot = y1_top - val
                left_offsets[src] += val
                
                y2_top = y_right_start[tgt] + right_counts[tgt] - right_offsets[tgt]
                y2_bot = y2_top - val
                right_offsets[tgt] += val
                
                if src == tgt:
                    color = COLOR_CONCORDANT
                    zorder = 1
                elif CATS.index(src) > CATS.index(tgt):
                    color = COLOR_UNDER
                    zorder = 2
                else:
                    color = COLOR_OVER
                    zorder = 2
                
                midpoint = (X_LEFT_BAR + BAR_WIDTH + X_RIGHT_BAR - BAR_WIDTH) / 2
                
                verts = [
                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),
                    (X_LEFT_BAR + BAR_WIDTH, y1_top),
                    (midpoint, y1_top),
                    (midpoint, y2_top),
                    (X_RIGHT_BAR - BAR_WIDTH, y2_top),
                    (X_RIGHT_BAR - BAR_WIDTH, y2_bot),
                    (midpoint, y2_bot),
                    (midpoint, y1_bot),
                    (X_LEFT_BAR + BAR_WIDTH, y1_bot),
                ]
                
                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.LINETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4
                ]
                
                path = Path(verts, codes)
                patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none',
                                           alpha=ALPHA_RIBBON, zorder=zorder)
                ax.add_patch(patch)
    
    for cat in CATS:
        if left_counts[cat] > 0:
            rect = mpatches.Rectangle(
                (X_LEFT_BAR, y_left_start[cat]),
                BAR_WIDTH,
                left_counts[cat],
                facecolor=left_color,
                edgecolor='black',
                linewidth=0.5,
                zorder=3
            )
            ax.add_patch(rect)
            
            y_center = y_left_start[cat] + left_counts[cat]/2
            pct = (left_counts[cat] / total_n) * 100
            label_str = f"{cat}\n({pct:.1f}%)" if pct > 3 else cat
            
            ax.text(X_LEFT_BAR - 0.01, y_center, label_str,
                   ha='right', va='center', fontsize=10, fontfamily='serif')
    
    for cat in CATS:
        if right_counts[cat] > 0:
            rect = mpatches.Rectangle(
                (X_RIGHT_BAR - BAR_WIDTH, y_right_start[cat]),
                BAR_WIDTH,
                right_counts[cat],
                facecolor=right_color,
                edgecolor='black',
                linewidth=0.5,
                zorder=3
            )
            ax.add_patch(rect)
            
            y_center = y_right_start[cat] + right_counts[cat]/2
            pct = (right_counts[cat] / total_n) * 100
            label_str = f"{cat}\n({pct:.1f}%)" if pct > 3 else cat
            
            ax.text(X_RIGHT_BAR + 0.01, y_center, label_str,
                   ha='left', va='center', fontsize=10, fontfamily='serif')
    
    ax.text(X_LEFT_BAR + BAR_WIDTH/2, total_drawing_height + gap_height,
           left_title, ha='center', va='bottom', fontsize=14,
           fontweight='bold', color=left_color, fontfamily='serif')
    
    ax.text(X_RIGHT_BAR - BAR_WIDTH/2, total_drawing_height + gap_height,
           right_title, ha='center', va='bottom', fontsize=14,
           fontweight='bold', color=right_color, fontfamily='serif')
    
    ax.set_title(f"{title} (N={total_n})", fontsize=16, pad=30, fontfamily='serif')
    
    plt.tight_layout()
    return fig


def add_svg_download_button(fig, filename_prefix, label=None, key=None):
    """Add an SVG download button for a Matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    
    button_label = label or f"📥 Download {filename_prefix}.svg"
    
    st.download_button(
        label=button_label,
        data=buf.getvalue(),
        file_name=f"{filename_prefix}.svg",
        mime="image/svg+xml",
        key=key
    )


def add_png_download_button(fig, filename_prefix, dpi=300, label=None, key=None):
    """Add a PNG download button for a Matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    
    button_label = label or f"📥 Download {filename_prefix}.png"
    
    st.download_button(
        label=button_label,
        data=buf.getvalue(),
        file_name=f"{filename_prefix}.png",
        mime="image/png",
        key=key
    )


def add_dual_download_buttons(fig, filename_prefix, cols=None, key_prefix=None):
    """Add both SVG and PNG download buttons side by side."""
    if cols is None:
        col1, col2, _ = st.columns([1, 1, 3])
    else:
        col1, col2 = cols
    
    key_svg = f"{key_prefix}_svg" if key_prefix else None
    key_png = f"{key_prefix}_png" if key_prefix else None
    
    with col1:
        add_svg_download_button(fig, filename_prefix, key=key_svg)
    with col2:
        add_png_download_button(fig, filename_prefix, key=key_png)
