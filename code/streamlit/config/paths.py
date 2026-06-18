"""
Path Configuration Module

Centralizes all file paths, directory structures, and data versioning.
Supports multiple data versions and environment-based path resolution.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PathConfig:
    """
    Centralized path configuration for CVD data and resources.
    
    Attributes:
        data_version: Current data version (e.g., 'v2.1')
        project_root: Root directory of the project
        parent_dir: Parent directory containing both streamlit and cvd folders
    """
    
    data_version: str = "v2.1"
    
    def __post_init__(self):
        """Initialize computed paths after dataclass initialization."""
        current_file = Path(__file__).resolve()
        self.streamlit_dir = current_file.parent.parent
        self.parent_dir = self.streamlit_dir.parent
        self.cvd_base = self.parent_dir / "cvd"
        
    
    @property
    def resource_dir(self) -> Path:
        """Path to cvd/resource directory."""
        return self.cvd_base / "resource"
    
    @property
    def analyzed_dir(self) -> Path:
        """Path to analyzed data directory for current version."""
        return self.resource_dir / "analyzed" / self.data_version
    
    @property
    def raw_dir(self) -> Path:
        """Path to raw data directory."""
        return self.resource_dir / "raw"
    
    @property
    def assets_dir(self) -> Path:
        """Path to assets directory."""
        return self.streamlit_dir / "assets"
    
    
    @property
    def sites_file(self) -> Path:
        """Path to service sites geographical data."""
        return self.resource_dir / "service_site_with_geographical_points.csv"
    
    
    def get_analyzed_data_path(self, filename: str) -> Path:
        """Get path to a file in the analyzed data directory."""
        return self.analyzed_dir / filename
    
    def get_resource_path(self, filename: str) -> Path:
        """Get path to a file in the resource directory."""
        return self.resource_dir / filename
    
    def get_raw_data_path(self, filename: str) -> Path:
        """Get path to a file in the raw data directory."""
        return self.raw_dir / filename
    
    def get_asset_path(self, filename: str) -> Path:
        """Get path to an asset file."""
        return self.assets_dir / filename
    
    
    @property
    def dataset_mapping(self) -> Dict[str, Path]:
        """Complete mapping of all standard datasets."""
        return {
            "sites": self.sites_file,
            
            "nonlab": self.get_analyzed_data_path("cvd_nonlab.csv"),
            "lab": self.get_analyzed_data_path("cvd_lab.csv"),
            "paired": self.get_analyzed_data_path("cvd_paired.csv"),
            "who_nonlab": self.get_analyzed_data_path("cvd_who_nonlab_domain.csv"),
            "who_lab": self.get_analyzed_data_path("cvd_who_lab_domain.csv"),
        }
    
    def get_dataset_path(self, dataset_key: str) -> Optional[Path]:
        """Get path for a specific dataset by key."""
        return self.dataset_mapping.get(dataset_key)
    
    def set_data_version(self, version: str) -> None:
        """Change the active data version."""
        self.data_version = version
        self.__post_init__()
    
    
    def get_cvd_paths(self) -> tuple:
        """Legacy method for backward compatibility."""
        return self.resource_dir, self.analyzed_dir
    
    
    def validate_paths(self) -> Dict[str, bool]:
        """Check if critical paths exist."""
        paths_to_check = {
            "resource_dir": self.resource_dir,
            "analyzed_dir": self.analyzed_dir,
            "sites_file": self.sites_file,
            "streamlit_dir": self.streamlit_dir,
        }
        
        return {
            name: path.exists() 
            for name, path in paths_to_check.items()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PathConfig(version={self.data_version}, "
            f"streamlit_dir={self.streamlit_dir}, "
            f"analyzed_dir={self.analyzed_dir})"
        )


def get_paths(version: Optional[str] = None) -> PathConfig:
    """Factory function to get PathConfig instance."""
    config = PathConfig()
    if version:
        config.set_data_version(version)
    return config
