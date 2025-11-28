#!/usr/bin/env python3
"""
Data source configuration for TCP preprocessing pipeline.

Supports dual data sources:
- Option A: datalad/git-annex (existing workflow)
- Option B: fMRIPrep output with custom parcellation

Author: Ian Philip Eglin
Date: 2025-01-28
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path


class DataSourceType(Enum):
    """Data source types supported by the pipeline"""
    DATALAD = "datalad"
    FMRIPREP = "fmriprep"


@dataclass
class DataSourceConfig:
    """
    Configuration for data source selection in TCP pipeline.

    Supports two modes:
    - DATALAD: Use existing datalad/git-annex workflow with ds005237
    - FMRIPREP: Use local fMRIPrep output with custom parcellation

    Attributes:
        source_type: Type of data source (datalad or fmriprep)
        dataset_path: Path to TCP dataset (for datalad mode)
        fmriprep_root: Root directory of fMRIPrep output (for fmriprep mode)
        parcellated_output_dir: Where to save parcellated .h5 files (for fmriprep mode)
        default_task: Default task to process (default: 'hammer')
        run_range: Range of run numbers to process (default: 1-9)
        phenotype_source: Source for phenotype data (always 'datalad')
    """
    source_type: DataSourceType

    # Option A (datalad) specific
    dataset_path: Optional[Path] = None

    # Option B (fmriprep) specific
    fmriprep_root: Optional[Path] = None
    parcellated_output_dir: Optional[Path] = None
    default_task: str = "hammer"
    run_range: Tuple[int, int] = (1, 9)  # runs 01-09

    # Shared phenotype handling
    phenotype_source: str = "datalad"  # Always from original dataset

    def __post_init__(self):
        """Convert string paths to Path objects and validate source_type"""
        # Convert string to enum if needed
        if isinstance(self.source_type, str):
            self.source_type = DataSourceType(self.source_type.lower())

        # Convert string paths to Path objects
        if self.dataset_path and isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)

        if self.fmriprep_root and isinstance(self.fmriprep_root, str):
            self.fmriprep_root = Path(self.fmriprep_root)

        if self.parcellated_output_dir and isinstance(self.parcellated_output_dir, str):
            self.parcellated_output_dir = Path(self.parcellated_output_dir)

    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration based on source type.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.source_type == DataSourceType.DATALAD:
            # Validate datalad configuration
            if self.dataset_path is None:
                # Try to get default dataset path
                try:
                    self.dataset_path = get_tcp_dataset_path()
                except Exception as e:
                    return False, f"Dataset path not specified and could not get default: {e}"

            if not self.dataset_path.exists():
                return False, f"Dataset path does not exist: {self.dataset_path}"

            if not (self.dataset_path / '.git').exists():
                return False, f"Dataset path is not a git repository: {self.dataset_path}"

            return True, "Datalad configuration valid"

        elif self.source_type == DataSourceType.FMRIPREP:
            # Validate fMRIPrep configuration
            if self.fmriprep_root is None:
                return False, "fMRIPrep root directory not specified"

            if not self.fmriprep_root.exists():
                return False, f"fMRIPrep root directory does not exist: {self.fmriprep_root}"

            if self.parcellated_output_dir is None:
                return False, "Parcellated output directory not specified"

            # Phenotype source validation
            if self.phenotype_source != "datalad":
                return False, "Phenotype source must be 'datalad' (phenotypes not in fMRIPrep output)"

            # Ensure dataset_path is set for phenotype fetching
            if self.dataset_path is None:
                try:
                    self.dataset_path = get_tcp_dataset_path()
                except Exception as e:
                    return False, f"Dataset path required for phenotype fetching: {e}"

            return True, "fMRIPrep configuration valid"

        return False, f"Unknown source type: {self.source_type}"

    def get_timeseries_output_path(self) -> Path:
        """
        Get path where timeseries .h5 files will be stored.

        Returns:
            Path to timeseries directory
        """
        if self.source_type == DataSourceType.DATALAD:
            return self.dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated"
        else:
            return self.parcellated_output_dir

    def get_phenotype_source_path(self) -> Path:
        """
        Get path to phenotype data source.

        Returns:
            Path to dataset containing phenotype files
        """
        # Phenotypes always come from datalad dataset
        if self.dataset_path is None:
            self.dataset_path = get_tcp_dataset_path()
        return self.dataset_path

    def get_description(self) -> str:
        """Get human-readable description of configuration"""
        if self.source_type == DataSourceType.DATALAD:
            return f"Datalad source: {self.dataset_path}"
        else:
            return (f"fMRIPrep source: {self.fmriprep_root}\n"
                   f"  Parcellated output: {self.parcellated_output_dir}\n"
                   f"  Task: {self.default_task}\n"
                   f"  Run range: {self.run_range[0]}-{self.run_range[1]}")


def create_datalad_config(dataset_path: Optional[Path] = None) -> DataSourceConfig:
    """
    Create configuration for datalad data source (Option A).

    Args:
        dataset_path: Path to TCP dataset (uses default if None)

    Returns:
        DataSourceConfig for datalad mode
    """
    if dataset_path is None:
        dataset_path = get_tcp_dataset_path()

    return DataSourceConfig(
        source_type=DataSourceType.DATALAD,
        dataset_path=dataset_path
    )


def create_fmriprep_config(
    fmriprep_root: Path,
    parcellated_output_dir: Path,
    dataset_path: Optional[Path] = None,
    task: str = "hammer",
    run_range: Tuple[int, int] = (1, 9)
) -> DataSourceConfig:
    """
    Create configuration for fMRIPrep data source (Option B).

    Args:
        fmriprep_root: Root directory of fMRIPrep output
        parcellated_output_dir: Where to save parcellated .h5 files
        dataset_path: Path to TCP dataset for phenotypes (uses default if None)
        task: Default task to process
        run_range: Range of run numbers to process

    Returns:
        DataSourceConfig for fMRIPrep mode
    """
    if dataset_path is None:
        dataset_path = get_tcp_dataset_path()

    return DataSourceConfig(
        source_type=DataSourceType.FMRIPREP,
        fmriprep_root=fmriprep_root,
        parcellated_output_dir=parcellated_output_dir,
        dataset_path=dataset_path,
        default_task=task,
        run_range=run_range,
        phenotype_source="datalad"
    )


if __name__ == "__main__":
    # Test configuration creation and validation
    print("Testing DataSourceConfig...")

    # Test datalad config
    print("\n=== Datalad Configuration ===")
    datalad_config = create_datalad_config()
    is_valid, message = datalad_config.validate()
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")
    print(f"Description: {datalad_config.get_description()}")

    # Test fMRIPrep config
    print("\n=== fMRIPrep Configuration ===")
    fmriprep_config = create_fmriprep_config(
        fmriprep_root=Path("/cluster/projects/example/fmriprep-25.1.4"),
        parcellated_output_dir=Path("Data/fmriprep_parcellated"),
        task="hammer",
        run_range=(1, 9)
    )
    is_valid, message = fmriprep_config.validate()
    print(f"Valid: {is_valid}")
    print(f"Message: {message}")
    print(f"Description: {fmriprep_config.get_description()}")
