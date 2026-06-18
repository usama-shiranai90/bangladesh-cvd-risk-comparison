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
    
    # Default data version - can be overridden
    data_version: str = "v2.1"
    
    def __post_init__(self):
        """Initialize computed paths after dataclass initialization."""
        # Auto-detect project structure
        current_file = Path(__file__).resolve()
        self.streamlit_dir = current_file.parent.parent
        self.parent_dir = self.streamlit_dir.parent
        self.cvd_base = self.parent_dir / "cvd"
        
    # ==================== Base Directories ====================
    
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
    
    # ==================== Specific Data Files ====================
    
    @property
    def sites_file(self) -> Path:
        """Path to service sites geographical data."""
        return self.resource_dir / "service_site_with_geographical_points.csv"
    
    # ==================== Helper Methods ====================
    
    def get_analyzed_data_path(self, filename: str) -> Path:
        """
        Get path to a file in the analyzed data directory.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Full path to the file
            
        Example:
            >>> path = paths.get_analyzed_data_path('cvd_paired.csv')
        """
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
    
    # ==================== Dataset Definitions ====================
    
    @property
    def dataset_mapping(self) -> Dict[str, Path]:
        """
        Complete mapping of all standard datasets.
        
        Returns:
            Dictionary mapping dataset keys to file paths
        """
        return {
            # Site information
            "sites": self.sites_file,
            
            # Analyzed datasets (current version)
            "nonlab": self.get_analyzed_data_path("cvd_nonlab.csv"),
            "lab": self.get_analyzed_data_path("cvd_lab.csv"),
            "paired": self.get_analyzed_data_path("cvd_paired.csv"),
            "who_nonlab": self.get_analyzed_data_path("cvd_who_nonlab_domain.csv"),
            "who_lab": self.get_analyzed_data_path("cvd_who_lab_domain.csv"),
        }
    
    def get_dataset_path(self, dataset_key: str) -> Optional[Path]:
        """
        Get path for a specific dataset by key.
        
        Args:
            dataset_key: Key from dataset_mapping
            
        Returns:
            Path object if key exists, None otherwise
        """
        return self.dataset_mapping.get(dataset_key)
    
    def set_data_version(self, version: str) -> None:
        """
        Change the active data version.
        
        Args:
            version: Version string (e.g., 'v2.0', 'v2.1')
            
        Example:
            >>> paths.set_data_version('v2.0')
        """
        self.data_version = version
        # Reinitialize to update paths
        self.__post_init__()
    
    # ==================== Legacy Compatibility ====================
    
    def get_cvd_paths(self) -> tuple:
        """
        Legacy method for backward compatibility.
        
        Returns:
            Tuple of (base_res, base_an) as Path objects
        """
        return self.resource_dir, self.analyzed_dir
    
    # ==================== Validation ====================
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Check if critical paths exist.
        
        Returns:
            Dictionary mapping path names to existence status
        """
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


# Convenience function for quick access
def get_paths(version: Optional[str] = None) -> PathConfig:
    """
    Factory function to get PathConfig instance.
    
    Args:
        version: Optional data version override
        
    Returns:
        PathConfig instance
    """
    config = PathConfig()
    if version:
        config.set_data_version(version)
    return config
