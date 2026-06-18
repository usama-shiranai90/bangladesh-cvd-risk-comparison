import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_csv_safe(path):
    """Load CSV safe."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
    return None

def find_available_datasets(base_dir):
    """Recursively find CSV files in the resource directory."""
    datasets = {}
    if not os.path.exists(base_dir):
        return datasets
        
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                name = file
                if name in datasets:
                    name = f"{file} ({os.path.basename(root)})"
                datasets[name] = os.path.join(root, file)
    return datasets

@st.cache_data
def get_merged_data(df_main, df_sites):
    """Get merged data."""
    if df_main is None:
        return None

    if df_sites is None or "site_id" not in df_main.columns:
        return df_main

    sites_copy = df_sites.copy()
    keep_site_cols = [
        "site_id",
        "project_title",
        "site_title",
        "location_type",
        "division_name",
        "district_name",
        "upazila_name",
        "site_latitude",
        "site_longitude",
    ]
    keep_site_cols = [c for c in keep_site_cols if c in sites_copy.columns]

    merged = df_main.merge(
        sites_copy[keep_site_cols],
        on="site_id",
        how="left",
        suffixes=("", "_site"),
    )

    if "location_type" in merged.columns:
        merged["location_type"] = (
            merged["location_type"]
            .astype("string")
            .str.strip()
            .str.title()
        )
        print()
        merged["urban_rural"] = (
            merged["location_type"]
            .map({"Urban": "Urban", "Rural": "Rural","Semi-urban":"Semi-urban"})
            .fillna("Semi-urban")
        )

    return merged
