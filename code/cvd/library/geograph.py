import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import branca.colormap as cm
import numpy as np

ADM_SHP_FILES = {
    "Division": {
        "shp_file": "bgd_admbnda_adm1_bbs_20201113.shp",
        "name_col": "ADM1_EN"
    },
    "District": {
        "shp_file": "bgd_admbnda_adm2_bbs_20201113.shp",
        "name_col": "ADM2_EN"
    },
    "Upazila": {
        "shp_file": "bgd_admbnda_adm3_bbs_20201113.shp",
        "name_col": "ADM3_EN"
    }
}

SITE_NAME_COL = "site_title"
PATIENT_ID_COL = "cid"
LAT_COL = "site_latitude"
LON_COL = "site_longitude"
UPAZILA_COL_CSV = "upazila_name"


def clean_coordinates(df, lat_col=LAT_COL, lon_col=LON_COL):
    """Clean coordinates."""
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    return df.dropna(subset=[lat_col, lon_col])


def aggregate_sites(df):
    """Aggregate sites."""
    return (
        df.groupby([SITE_NAME_COL, LAT_COL, LON_COL])[PATIENT_ID_COL]
        .nunique()
        .reset_index(name="patient_count")
    )


def build_sites_geodataframe(df_sites_agg):
    """Build sites geodataframe."""
    gdf = gpd.GeoDataFrame(
        df_sites_agg,
        geometry=gpd.points_from_xy(df_sites_agg[LON_COL], df_sites_agg[LAT_COL]),
        crs="EPSG:4326"
    )

    if gdf["patient_count"].nunique() >= 3:
        gdf["volume_cat"] = pd.qcut(
            gdf["patient_count"], q=3, labels=["Low", "Medium", "High"]
        )
    else:
        gdf["volume_cat"] = "Medium"

    return gdf


def plot_admin_level_spatial(
        level_name,
        cfg,
        gdf_sites,
        base_map_dir
):
    """Plot admin level spatial."""
    print(f"Generating map for {level_name}...")

    shp_path = base_map_dir + cfg["shp_file"]
    name_col = cfg["name_col"]

    boundaries = gpd.read_file(shp_path).to_crs("EPSG:4326")

    sites_with_region = gpd.sjoin(
        gdf_sites,
        boundaries[[name_col, "geometry"]],
        how="inner",
        predicate="within"
    )

    region_totals = (
        sites_with_region
        .groupby(name_col)["patient_count"]
        .sum()
        .reset_index(name="total_patients")
    )

    level_map = boundaries.merge(region_totals, on=name_col, how="left")
    level_map = level_map.rename(columns={name_col: "admin_name"})
    level_map["total_patients"] = level_map["total_patients"].fillna(0)
    level_map["has_patients"] = level_map["total_patients"] > 0

    fig, ax = plt.subplots(figsize=(16, 16))

    level_map.plot(
        column="total_patients",
        cmap="Blues",
        legend=True,
        ax=ax,
        edgecolor="none",
        missing_kwds={"color": "#f9f9f9", "label": "No Service Sites"},
        legend_kwds={
            "label": "Total Patients",
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.05
        }
    )

    level_map[level_map["has_patients"]].boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    level_map[~level_map["has_patients"]].boundary.plot(ax=ax, edgecolor="#ccc", linewidth=0.5)

    label_limit = 50 if level_name == "Upazila" else 30
    labels = (
        level_map[level_map["has_patients"]]
        .sort_values("total_patients", ascending=False)
        .head(label_limit)
    )

    labels["rep_point"] = labels.geometry.representative_point()

    for _, row in labels.iterrows():
        ax.annotate(
            f"{row['admin_name']}\n{int(row['total_patients'])}",
            xy=(row.rep_point.x, row.rep_point.y),
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.6)
        )

    styles = {
        "Low": ("v", "#4CAF50"),
        "Medium": ("s", "#FFC107"),
        "High": ("^", "#F44336"),
    }

    for cat, (marker, color) in styles.items():
        subset = gdf_sites[gdf_sites["volume_cat"] == cat]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            c=color,
            marker=marker,
            s=60,
            edgecolor="black",
            linewidth=0.5,
            label=f"{cat} Volume"
        )

    for _, row in gdf_sites.sort_values("patient_count", ascending=False).head(5).iterrows():
        ax.annotate(
            f"{row[SITE_NAME_COL]} ({int(row['patient_count'])})",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->")
        )

    ax.set_title(f"Patient Distribution by {level_name}", fontsize=18, fontweight="bold")
    ax.legend()
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def generate_text_site_report(df):
    """Generate text site report."""
    if UPAZILA_COL_CSV not in df.columns:
        print("Upazila column not found.")
        return

    report = (
        df.groupby([UPAZILA_COL_CSV, SITE_NAME_COL])[PATIENT_ID_COL]
        .nunique()
        .reset_index(name="unique_patients")
        .sort_values([UPAZILA_COL_CSV, "unique_patients"], ascending=[True, False])
    )

    print("\n=== Service Site List ===")
    print(report.to_string(index=False))


def run_spatial_analysis(df_final, base_map_dir):
    """Run spatial analysis."""
    df_clean = clean_coordinates(df_final)
    df_sites_agg = aggregate_sites(df_clean)
    gdf_sites = build_sites_geodataframe(df_sites_agg)

    for level, cfg in ADM_SHP_FILES.items():
        try:
            plot_admin_level_spatial(level, cfg, gdf_sites, base_map_dir)
        except Exception as e:
            print(f"❌ Failed for {level}: {e}")

    generate_text_site_report(df_final)
