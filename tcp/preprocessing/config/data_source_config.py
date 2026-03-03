#!/usr/bin/env python3
"""
Data Source Configuration for TCP Preprocessing Pipeline

Supports multiple data sources for fMRI time series:
- DATALAD: Internal datalad dataset with pre-parcellated .h5 files
- HCP: External HCP-preprocessed NIFTI files requiring parcellation
- COMBINED: Mix of DATALAD and HCP subjects with configurable duplicate resolution

Author: Ian Philip Eglin
Date: 2025-12-22
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


DATA_SOURCE_FMRIPREP = "fmriprep_parcellation"


class DataSourceType(Enum):
    """Type of data source for fMRI timeseries"""
    DATALAD = "datalad"
    HCP = "hcp"
    COMBINED = "combined"


@dataclass
class DataSourceConfig:
    """
    Configuration for multi-source data integration.

    Attributes:
        source_type: Type of data source (DATALAD, HCP, or COMBINED)
        dataset_path: Path to datalad dataset (for phenotype data and/or fMRI data)
        hcp_root: Path to HCP output directory (e.g., /hcp_output)
        hcp_parcellated_output: Directory to store parcellated HCP .h5 files
        duplicate_resolution: Strategy for handling subjects in both sources
                            ('prefer_hcp', 'prefer_datalad', or 'error')
        default_task: Default task name (e.g., 'hammer')
    """
    source_type: DataSourceType

    # DATALAD fields
    dataset_path: Optional[Path] = None

    # HCP fields
    hcp_root: Optional[Path] = None
    hcp_parcellated_output: Optional[Path] = None

    # fmriprep fields
    fmriprep_root: Optional[Path] = None
    fmriprep_parcellated_output: Optional[Path] = None

    # COMBINED mode configuration
    duplicate_resolution: str = "prefer_hcp"  # 'prefer_hcp', 'prefer_datalad', or 'error'

    # Common fields
    default_task: str = "hammer"

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Convert string paths to Path objects
        if self.dataset_path and not isinstance(self.dataset_path, Path):
            self.dataset_path = Path(self.dataset_path)
        if self.hcp_root and not isinstance(self.hcp_root, Path):
            self.hcp_root = Path(self.hcp_root)
        if self.hcp_parcellated_output and not isinstance(self.hcp_parcellated_output, Path):
            self.hcp_parcellated_output = Path(self.hcp_parcellated_output)
        if self.fmriprep_root and not isinstance(self.fmriprep_root, Path):
            self.fmriprep_root = Path(self.fmriprep_root)
        if self.fmriprep_parcellated_output and not isinstance(self.fmriprep_parcellated_output, Path):
            self.fmriprep_parcellated_output = Path(self.fmriprep_parcellated_output)

        # Validate source type requirements
        if self.source_type == DataSourceType.DATALAD:
            if not self.dataset_path:
                raise ValueError("DATALAD source requires dataset_path")

        elif self.source_type == DataSourceType.HCP:
            if not self.hcp_root:
                raise ValueError("HCP source requires hcp_root")
            if not self.hcp_parcellated_output:
                raise ValueError("HCP source requires hcp_parcellated_output")

        elif self.source_type == DataSourceType.COMBINED:
            if not self.dataset_path:
                raise ValueError("COMBINED source requires dataset_path (for phenotype data)")
            if not self.hcp_root:
                raise ValueError("COMBINED source requires hcp_root")
            if not self.hcp_parcellated_output:
                raise ValueError("COMBINED source requires hcp_parcellated_output")

        # Validate duplicate resolution strategy
        valid_strategies = ['prefer_hcp', 'prefer_datalad', 'error']
        if self.duplicate_resolution not in valid_strategies:
            raise ValueError(
                f"Invalid duplicate_resolution: {self.duplicate_resolution}. "
                f"Must be one of: {valid_strategies}"
            )

    def discover_hcp_subjects(self) -> List[str]:
        """
        Scan HCP output directory for subjects with task data.

        Only includes subjects that have actual BOLD .nii.gz files, not just directories.

        Returns:
            List of subject IDs (e.g., ['sub-NDARINVXXXXX', ...])
        """
        if not self.hcp_root or not self.hcp_root.exists():
            return []

        subjects = []
        for subject_dir in sorted(self.hcp_root.glob("sub-*")):
            results_dir = subject_dir / "MNINonLinear" / "Results"
            if results_dir.exists():
                # Check for task directories with actual BOLD files
                task_dirs = list(results_dir.glob(f"task-{self.default_task}AP_run-*_bold"))
                has_valid_task = False

                for task_dir in task_dirs:
                    # Verify that the BOLD .nii.gz file actually exists
                    bold_file = task_dir / f"{task_dir.name}.nii.gz"
                    if bold_file.exists() and bold_file.is_file():
                        has_valid_task = True
                        break

                if has_valid_task:
                    subjects.append(subject_dir.name)  # e.g., "sub-NDARINVXXXXX"

        return subjects

    def validate_hcp_structure(self, subject_id: str) -> bool:
        """
        Validate that a subject has the expected HCP directory structure.

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')

        Returns:
            True if structure is valid, False otherwise
        """
        if not self.hcp_root or not self.hcp_root.exists():
            return False

        # Handle subject_id with or without "sub-" prefix
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        subject_dir = self.hcp_root / subject_id
        results_dir = subject_dir / "MNINonLinear" / "Results"

        return results_dir.exists()

    def get_hcp_bold_path(self, subject_id: str, task: str = None, run: int = 1) -> Optional[Path]:
        """
        Get path to HCP BOLD file for a subject.

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')
            task: Task name (default: use self.default_task)
            run: Run number (default: 1)

        Returns:
            Path to BOLD file, or None if not found
        """
        if not self.hcp_root:
            return None

        if task is None:
            task = self.default_task

        # Handle subject_id with or without "sub-" prefix
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        subject_dir = self.hcp_root / subject_id
        results_dir = subject_dir / "MNINonLinear" / "Results"

        task_dir = results_dir / f"task-{task}AP_run-0{run}_bold"
        bold_file = task_dir / f"task-{task}AP_run-0{run}_bold.nii.gz"

        return bold_file if bold_file.exists() else None

    def get_fmriprep_bold_path(self, subject_id: str, task: str = None, run: int = 1) -> Optional[Path]:
        """
        Get path to fmriprep BOLD file for a subject.

        fmriprep 25.1.4 BIDS path pattern:
        {fmriprep_root}/sub-{id}/func/sub-{id}_task-{task}AP_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')
            task: Task name (default: use self.default_task)
            run: Run number (default: 1)

        Returns:
            Path to BOLD file if it exists, None otherwise
        """
        if not self.fmriprep_root:
            return None

        if task is None:
            task = self.default_task

        # Handle subject_id with or without "sub-" prefix
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        func_dir = self.fmriprep_root / subject_id / "func"
        bold_file = (
            func_dir /
            f"{subject_id}_task-{task}AP_run-{run:02d}"
            f"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
        return bold_file if bold_file.exists() else None

    def discover_fmriprep_subjects(self) -> List[str]:
        """
        Scan fmriprep output directory for subjects with BOLD data.

        Only includes subjects that have the actual desc-preproc_bold.nii.gz file,
        not just a func/ directory (avoids partial fmriprep outputs).

        Returns:
            List of subject IDs (e.g., ['sub-NDARINVXXXXX', ...])
        """
        if not self.fmriprep_root or not self.fmriprep_root.exists():
            return []

        subjects = []
        for subject_dir in sorted(self.fmriprep_root.glob("sub-*")):
            func_dir = subject_dir / "func"
            if not func_dir.exists():
                continue

            # Verify the full BIDS filename including all required entities
            bold_pattern = f"{subject_dir.name}_task-{self.default_task}AP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            bold_file = func_dir / bold_pattern
            if bold_file.exists() and bold_file.is_file():
                subjects.append(subject_dir.name)

        return subjects

    def validate_fmriprep_structure(self, subject_id: str) -> bool:
        """
        Validate that a subject has the expected fmriprep directory structure.

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')

        Returns:
            True if func/ directory exists, False otherwise
        """
        if not self.fmriprep_root or not self.fmriprep_root.exists():
            return False

        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        func_dir = self.fmriprep_root / subject_id / "func"
        return func_dir.exists()

    def is_fmriprep_enabled(self) -> bool:
        """Check if fmriprep data source is configured"""
        return self.fmriprep_root is not None

    def is_combined_mode(self) -> bool:
        """Check if running in COMBINED mode"""
        return self.source_type == DataSourceType.COMBINED

    def is_hcp_enabled(self) -> bool:
        """Check if HCP data source is enabled"""
        return self.source_type in [DataSourceType.HCP, DataSourceType.COMBINED]

    def __str__(self) -> str:
        """String representation for logging"""
        lines = [
            f"DataSourceConfig:",
            f"  Source type: {self.source_type.value}",
            f"  Default task: {self.default_task}",
        ]

        if self.dataset_path:
            lines.append(f"  Dataset path: {self.dataset_path}")

        if self.hcp_root:
            lines.append(f"  HCP root: {self.hcp_root}")
            lines.append(f"  HCP parcellated output: {self.hcp_parcellated_output}")

        if self.source_type == DataSourceType.COMBINED:
            lines.append(f"  Duplicate resolution: {self.duplicate_resolution}")

        return "\n".join(lines)


# Factory functions for creating configurations

def create_datalad_config(dataset_path: Path, default_task: str = "hammer") -> DataSourceConfig:
    """
    Create configuration for DATALAD-only mode.

    Args:
        dataset_path: Path to datalad dataset
        default_task: Default task name

    Returns:
        DataSourceConfig for DATALAD mode
    """
    return DataSourceConfig(
        source_type=DataSourceType.DATALAD,
        dataset_path=dataset_path,
        default_task=default_task
    )


def create_hcp_config(hcp_root: Path,
                     parcellated_output: Path,
                     default_task: str = "hammer") -> DataSourceConfig:
    """
    Create configuration for HCP-only mode.

    Args:
        hcp_root: Path to HCP output directory
        parcellated_output: Directory to store parcellated .h5 files
        default_task: Default task name

    Returns:
        DataSourceConfig for HCP mode
    """
    return DataSourceConfig(
        source_type=DataSourceType.HCP,
        hcp_root=hcp_root,
        hcp_parcellated_output=parcellated_output,
        default_task=default_task
    )


def create_combined_config(dataset_path: Path,
                          hcp_root: Path,
                          hcp_parcellated_output: Path,
                          duplicate_resolution: str = "prefer_hcp",
                          default_task: str = "hammer") -> DataSourceConfig:
    """
    Create configuration for COMBINED mode (DATALAD + HCP).

    Args:
        dataset_path: Path to datalad dataset (for phenotype data)
        hcp_root: Path to HCP output directory
        hcp_parcellated_output: Directory to store parcellated .h5 files
        duplicate_resolution: How to handle duplicates ('prefer_hcp', 'prefer_datalad', 'error')
        default_task: Default task name

    Returns:
        DataSourceConfig for COMBINED mode
    """
    return DataSourceConfig(
        source_type=DataSourceType.COMBINED,
        dataset_path=dataset_path,
        hcp_root=hcp_root,
        hcp_parcellated_output=hcp_parcellated_output,
        duplicate_resolution=duplicate_resolution,
        default_task=default_task
    )


def create_fmriprep_config(fmriprep_root: Path,
                           parcellated_output: Path,
                           default_task: str = "hammer") -> DataSourceConfig:
    """
    Create configuration for fmriprep-only mode.

    Uses DataSourceType.HCP internally to maintain compatibility with
    existing HCPParcellator and HCPParcellationRunner code that checks
    source_type. This is intentional: the parcellation engine is unchanged;
    only the path logic is updated.

    Args:
        fmriprep_root: Path to fmriprep output directory (e.g., fmriprep-25.1.4/)
        parcellated_output: Directory to store parcellated .h5 files
        default_task: Default task name

    Returns:
        DataSourceConfig configured for fmriprep path resolution
    """
    return DataSourceConfig(
        source_type=DataSourceType.HCP,
        fmriprep_root=fmriprep_root,
        fmriprep_parcellated_output=parcellated_output,
        # Maintain hcp_root and hcp_parcellated_output for compatibility
        # until runner scripts are updated in Plan 04
        hcp_root=fmriprep_root,
        hcp_parcellated_output=parcellated_output,
        default_task=default_task
    )


if __name__ == "__main__":
    # Example usage
    from config.paths import get_tcp_dataset_path

    print("=" * 60)
    print("Data Source Configuration Examples")
    print("=" * 60)

    # DATALAD mode
    print("\n1. DATALAD Mode:")
    datalad_config = create_datalad_config(
        dataset_path=get_tcp_dataset_path()
    )
    print(datalad_config)

    # HCP mode
    print("\n2. HCP Mode:")
    hcp_config = create_hcp_config(
        hcp_root=Path("/cluster/projects/itea_lille-ie/Transdiagnostic/output/hcp_output"),
        parcellated_output=Path("Data/hcp_parcellated")
    )
    print(hcp_config)

    # COMBINED mode
    print("\n3. COMBINED Mode:")
    combined_config = create_combined_config(
        dataset_path=get_tcp_dataset_path(),
        hcp_root=Path("/cluster/projects/itea_lille-ie/Transdiagnostic/output/hcp_output"),
        hcp_parcellated_output=Path("Data/hcp_parcellated"),
        duplicate_resolution="prefer_hcp"
    )
    print(combined_config)

    # Test HCP subject discovery
    if combined_config.hcp_root.exists():
        print("\n4. HCP Subject Discovery:")
        subjects = combined_config.discover_hcp_subjects()
        print(f"Found {len(subjects)} HCP subjects with {combined_config.default_task} task data")
        if subjects:
            print(f"First 5: {subjects[:5]}")
