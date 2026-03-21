#!/usr/bin/env python3
"""
HCP Parcellation Engine for TCP Pipeline

Parcellates HCP-preprocessed BOLD data into 434-ROI timeseries matching
the format expected by downstream processing pipeline.

Atlas composition:
- 400 cortical parcels (Yeo2011 17-Network)
- 32 subcortical parcels (Tian S2 3T)
- 2 cerebellar regions (Buckner 7-network, aggregated)

Author: Ian Philip Eglin
Date: 2025-01-28
"""

import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from joblib import Parallel, delayed
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_parcellations_path
from tcp.preprocessing.config.data_source_config import DATA_SOURCE_FMRIPREP
from tcp.preprocessing.utils.participants_filter import load_participants_file, apply_participants_filter


class HCPParcellator:
    """Parcellate HCP outputs to 434-ROI timeseries"""

    EXPECTED_TOTAL_PARCELS = 434
    CORTICAL_PARCELS = 400
    SUBCORTICAL_PARCELS = 32
    CEREBELLAR_REGIONS = 2

    def __init__(self, fmriprep_root: Path, verbose: bool = True):
        """
        Initialize parcellator with atlas paths and fmriprep root.

        Args:
            fmriprep_root: Root directory of fmriprep output (e.g., fmriprep-25.1.4/)
            verbose: Print detailed progress information
        """
        self.verbose = verbose
        self.fmriprep_root = Path(fmriprep_root)
        # Maintain backward-compatible alias so any code still using self.hcp_root continues to work
        self.hcp_root = self.fmriprep_root
        self.parcellations_base = get_parcellations_path()

        # Atlas file paths (at project root)
        self.cortical_atlas = self.parcellations_base / "cortical" / "yeo17" / "400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz"
        self.subcortical_atlas = self.parcellations_base / "subcortical" / "tian" / "Tian_Subcortex_S2_3T.nii"

        # TODO: Add Buckner cerebellar atlas when available
        # For now, we use a placeholder to maintain the 434-parcel structure
        self.cerebellar_atlas = None

        # Validate atlas files exist
        self._validate_atlases()

        # Validate fmriprep root exists
        if not self.fmriprep_root.exists():
            raise FileNotFoundError(f"fmriprep root directory not found: {self.fmriprep_root}")

        if self.verbose:
            print(f"Initialized HCPParcellator")
            print(f"  fmriprep root: {self.fmriprep_root}")
            print(f"  Cortical atlas: {self.cortical_atlas.name} ({self.CORTICAL_PARCELS} parcels)")
            print(f"  Subcortical atlas: {self.subcortical_atlas.name} ({self.SUBCORTICAL_PARCELS} parcels)")
            if self.cerebellar_atlas:
                print(f"  Cerebellar atlas: {self.cerebellar_atlas.name} ({self.CEREBELLAR_REGIONS} regions)")
            else:
                print(f"  Cerebellar atlas: PLACEHOLDER ({self.CEREBELLAR_REGIONS} regions - zeros)")

    def _validate_atlases(self):
        """Validate that required atlas files exist"""
        if not self.cortical_atlas.exists():
            raise FileNotFoundError(f"Cortical atlas not found: {self.cortical_atlas}")
        if not self.subcortical_atlas.exists():
            raise FileNotFoundError(f"Subcortical atlas not found: {self.subcortical_atlas}")

    def parcellate_bold(self, bold_nifti_path: Path) -> np.ndarray:
        """
        Parcellate single BOLD run into 434-ROI timeseries.

        Args:
            bold_nifti_path: Path to HCP preprocessed BOLD NIfTI

        Returns:
            Array of shape (434, timepoints) with ROI timeseries
        """
        if self.verbose:
            print(f"  Parcellating: {bold_nifti_path.name}")

        # Load BOLD image
        bold_img = load_img(str(bold_nifti_path))
        n_timepoints = bold_img.shape[-1]

        # 1. Cortical parcellation (400 parcels)
        cortical_ts = self._parcellate_cortical(bold_img)
        if cortical_ts.shape[0] != self.CORTICAL_PARCELS:
            raise ValueError(
                f"Cortical parcellation produced {cortical_ts.shape[0]} parcels, "
                f"expected {self.CORTICAL_PARCELS}"
            )

        # 2. Subcortical parcellation (32 parcels)
        subcortical_ts = self._parcellate_subcortical(bold_img)
        if subcortical_ts.shape[0] != self.SUBCORTICAL_PARCELS:
            raise ValueError(
                f"Subcortical parcellation produced {subcortical_ts.shape[0]} parcels, "
                f"expected {self.SUBCORTICAL_PARCELS}"
            )

        # 3. Cerebellar parcellation (2 regions)
        cerebellar_ts = self._parcellate_cerebellar(bold_img)
        if cerebellar_ts.shape[0] != self.CEREBELLAR_REGIONS:
            raise ValueError(
                f"Cerebellar parcellation produced {cerebellar_ts.shape[0]} regions, "
                f"expected {self.CEREBELLAR_REGIONS}"
            )

        # Combine all parcellations
        combined_ts = np.vstack([cortical_ts, subcortical_ts, cerebellar_ts])

        # Validate output shape
        if combined_ts.shape[0] != self.EXPECTED_TOTAL_PARCELS:
            raise ValueError(
                f"Parcellation produced {combined_ts.shape[0]} ROIs, "
                f"expected {self.EXPECTED_TOTAL_PARCELS}"
            )

        if combined_ts.shape[1] != n_timepoints:
            raise ValueError(
                f"Timepoint mismatch: {combined_ts.shape[1]} vs {n_timepoints}"
            )

        if self.verbose:
            print(f"    Cortical: {cortical_ts.shape[0]} parcels")
            print(f"    Subcortical: {subcortical_ts.shape[0]} parcels")
            print(f"    Cerebellar: {cerebellar_ts.shape[0]} regions")
            print(f"    Combined: {combined_ts.shape[0]} ROIs × {combined_ts.shape[1]} timepoints")

        return combined_ts

    def _parcellate_cortical(self, bold_img) -> np.ndarray:
        """
        Parcellate cortical regions using Yeo17 400-parcel atlas.

        Returns:
            Array of shape (400, timepoints)
        """
        masker = NiftiLabelsMasker(
            labels_img=str(self.cortical_atlas),
            standardize=False,
            detrend=False,
            resampling_target="data"
        )
        # fit_transform returns (timepoints, parcels) - transpose to (parcels, timepoints)
        return masker.fit_transform(bold_img).T

    def _parcellate_subcortical(self, bold_img) -> np.ndarray:
        """
        Parcellate subcortical regions using Tian S2 atlas.

        Returns:
            Array of shape (32, timepoints)
        """
        masker = NiftiLabelsMasker(
            labels_img=str(self.subcortical_atlas),
            standardize=False,
            detrend=False,
            resampling_target="data"
        )
        return masker.fit_transform(bold_img).T

    def _parcellate_cerebellar(self, bold_img) -> np.ndarray:
        """
        Parcellate cerebellar regions using Buckner 7-network atlas.

        Returns:
            Array of shape (2, timepoints)
        """
        n_timepoints = bold_img.shape[-1]

        if self.cerebellar_atlas is None:
            # Placeholder: return zeros with shape (2, timepoints)
            if self.verbose:
                print("    WARNING: Using zeros placeholder for cerebellar atlas")
            return np.zeros((2, n_timepoints))

        # Extract 7 cerebellar network timeseries
        masker = NiftiLabelsMasker(
            labels_img=str(self.cerebellar_atlas),
            standardize=False,
            detrend=False,
            resampling_target="data"
        )
        ts_7networks = masker.fit_transform(bold_img).T  # (7, timepoints)

        # Average networks to create 2 regions
        anterior = ts_7networks[0:3, :].mean(axis=0, keepdims=True)  # (1, timepoints)
        posterior = ts_7networks[3:7, :].mean(axis=0, keepdims=True)  # (1, timepoints)

        return np.vstack([anterior, posterior])  # (2, timepoints)

    def discover_bold_files(self, subject_id: str, task: str = "hammer") -> List[Path]:
        """
        Find BOLD files in fmriprep 25.1.4 directory structure.

        fmriprep structure:
        {fmriprep_root}/sub-{id}/func/
            sub-{id}_task-{task}AP_run-{run:02d}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')
            task: Task name (default: 'hammer')

        Returns:
            List of paths to BOLD NIFTI files
        """
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'

        func_dir = self.fmriprep_root / subject_id / "func"

        if not func_dir.exists():
            if self.verbose:
                print(f"  WARNING: func directory not found: {func_dir}")
            return []

        bold_files = []

        # For hammer task, only run 1 is expected
        if task == "hammer":
            run_nums = [1]
        else:
            run_nums = range(1, 10)

        for run in run_nums:
            bold_file = (
                func_dir /
                f"{subject_id}_task-{task}AP_run-{run:02d}"
                f"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
            if bold_file.exists():
                bold_files.append(bold_file)

        return bold_files

    def parcellate_subject(self,
                          subject_id: str,
                          task: str = "hammer",
                          output_dir: Optional[Path] = None) -> Path:
        """
        Parcellate all runs for a subject and save to .h5 file.

        Args:
            subject_id: Subject ID (e.g., 'sub-NDARINVXXXXX')
            task: Task name (default: 'hammer')
            output_dir: Directory to save .h5 output

        Returns:
            Path to saved .h5 file
        """
        # Find all matching BOLD files
        bold_files = self.discover_bold_files(subject_id, task)

        if not bold_files:
            raise FileNotFoundError(
                f"No BOLD files found for {subject_id}, task={task}\n"
                f"Searched in: {self.fmriprep_root / subject_id / 'func'}"
            )

        if self.verbose:
            print(f"\nProcessing {subject_id}: {len(bold_files)} runs found")

        # Parcellate each run
        run_timeseries = {}
        for bold_path in sorted(bold_files):
            # Parse run number from filename to produce a normalised, stable H5 key
            # Pattern: ..._run-{NN}_..._bold.nii.gz -> task-{task}_run-{NN}
            # NOTE: `import re` is at the top of the file — do NOT re-import here
            stem = bold_path.name
            run_match = re.search(r'run-(\d+)', stem)
            run_num = run_match.group(1) if run_match else "01"
            normalised_key = f"task-{task}_run-{run_num}"
            ts = self.parcellate_bold(bold_path)
            run_timeseries[normalised_key] = ts

        # Save to .h5 file
        output_path = self._save_h5(run_timeseries, subject_id, task, output_dir)

        if self.verbose:
            print(f"  Saved: {output_path}")

        return output_path

    def parcellate_subjects_parallel(self,
                                    subject_ids: List[str],
                                    task: str = "hammer",
                                    output_dir: Optional[Path] = None,
                                    n_jobs: int = 4) -> Dict[str, Path]:
        """
        Parcellate multiple subjects in parallel using joblib.

        Args:
            subject_ids: List of subject IDs to process
            task: Task name
            output_dir: Output directory for .h5 files
            n_jobs: Number of parallel jobs (default: 4)

        Returns:
            Dictionary mapping subject_id to output .h5 path
        """
        print(f"\nParallel parcellation of {len(subject_ids)} subjects ({n_jobs} jobs)")
        print(f"Task: {task}")

        # Create a non-verbose parcellator for parallel processing
        def process_one_subject(subject_id):
            """Process single subject (for parallel execution)"""
            try:
                parcellator = HCPParcellator(fmriprep_root=self.fmriprep_root, verbose=False)
                output_path = parcellator.parcellate_subject(
                    subject_id=subject_id,
                    task=task,
                    output_dir=output_dir
                )
                return subject_id, output_path, None
            except Exception as e:
                return subject_id, None, str(e)

        # Run parallel processing with progress bar
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_one_subject)(subject_id)
            for subject_id in tqdm(subject_ids, desc="Parcellating subjects")
        )

        # Collect results
        output_paths = {}
        errors = {}

        for subject_id, output_path, error in results:
            if error is None:
                output_paths[subject_id] = output_path
            else:
                errors[subject_id] = error

        # Print summary
        print(f"\nParcellation complete:")
        print(f"  Successful: {len(output_paths)}/{len(subject_ids)}")
        print(f"  Failed: {len(errors)}/{len(subject_ids)}")

        if errors:
            print("\nErrors:")
            for subject_id, error in errors.items():
                print(f"  {subject_id}: {error}")

        return output_paths

    def _save_h5(self,
                run_timeseries: Dict[str, np.ndarray],
                subject_id: str,
                task: str,
                output_dir: Optional[Path]) -> Path:
        """
        Save parcellated timeseries to .h5 file.

        Format matches datalad .h5 files for compatibility.

        Args:
            run_timeseries: Dictionary of run_name -> timeseries array
            subject_id: Subject ID in BIDS format
            task: Task name
            output_dir: Output directory

        Returns:
            Path to saved .h5 file
        """
        if output_dir is None:
            output_dir = Path.cwd() / "hcp_parcellated"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure subject_id is in BIDS format
        if not subject_id.startswith('sub-'):
            subject_id = f"sub-{subject_id}"

        # Output filename in BIDS format: sub-{subject_id}_task-{task}_parcellated.h5
        output_path = output_dir / f"{subject_id}_task-{task}_parcellated.h5"

        with h5py.File(output_path, 'w') as f:
            # Store each run as a separate dataset
            for run_name, ts_data in run_timeseries.items():
                f.create_dataset(run_name, data=ts_data, compression="gzip")

            # Add metadata attributes
            f.attrs['subject_id'] = subject_id
            f.attrs['task'] = task
            f.attrs['n_rois'] = self.EXPECTED_TOTAL_PARCELS
            f.attrs['n_runs'] = len(run_timeseries)
            f.attrs['source'] = DATA_SOURCE_FMRIPREP
            f.attrs['cortical_atlas'] = self.cortical_atlas.name
            f.attrs['subcortical_atlas'] = self.subcortical_atlas.name
            f.attrs['cerebellar_note'] = 'Placeholder zeros (atlas not implemented)'

        return output_path


def main():
    """CLI for parcellating HCP data"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Parcellate fmriprep BOLD data into 434-ROI timeseries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parcellate single subject
  python hcp_parcellation.py \\
    --fmriprep-root /cluster/projects/.../fmriprep-25.1.4 \\
    --subject-id sub-NDARINVXXXXX \\
    --output-dir Data/fmriprep_parcellated

  # Parcellate multiple subjects in parallel
  python hcp_parcellation.py \\
    --fmriprep-root /cluster/projects/.../fmriprep-25.1.4 \\
    --subject-ids sub-NDARINV001 sub-NDARINV002 sub-NDARINV003 \\
    --output-dir Data/fmriprep_parcellated \\
    --n-jobs 4
        """
    )

    parser.add_argument('--fmriprep-root', type=Path, required=True,
                       help='Root directory of fmriprep output')
    parser.add_argument('--subject-id', type=str,
                       help='Subject ID (e.g., sub-NDARINVXXXXX) for single subject')
    parser.add_argument('--subject-ids', type=str, nargs='+',
                       help='Multiple subject IDs for parallel processing')
    parser.add_argument('--task', type=str, default='hammer',
                       help='Task name (default: hammer)')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for .h5 files')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4, only used with --subject-ids)')
    parser.add_argument('--participants-file', type=Path, default=None,
                       help='Optional path to participants.txt file. When provided, only subjects '
                            'listed in this file will be parcellated (intersected with --subject-ids '
                            'if both given). Format: one subject ID per line, # comments supported.')

    args = parser.parse_args()

    # Validate arguments
    if args.subject_id and args.subject_ids:
        parser.error("Cannot specify both --subject-id and --subject-ids")
    if not args.subject_id and not args.subject_ids:
        parser.error("Must specify either --subject-id or --subject-ids")

    if args.subject_id:
        # Single subject mode
        parcellator = HCPParcellator(fmriprep_root=args.fmriprep_root, verbose=True)
        output_path = parcellator.parcellate_subject(
            subject_id=args.subject_id,
            task=args.task,
            output_dir=args.output_dir
        )
        print(f"\nSuccess! Parcellated timeseries saved to: {output_path}")

    else:
        # Parallel mode
        parcellator = HCPParcellator(fmriprep_root=args.fmriprep_root, verbose=True)
        subject_ids = args.subject_ids
        if args.participants_file is not None:
            participants_subjects = load_participants_file(args.participants_file)
            subject_ids = apply_participants_filter(
                participants_subjects=participants_subjects,
                discovered_subjects=subject_ids
            )
        output_paths = parcellator.parcellate_subjects_parallel(
            subject_ids=subject_ids,
            task=args.task,
            output_dir=args.output_dir,
            n_jobs=args.n_jobs
        )
        print(f"\nSuccess! Parcellated {len(output_paths)} subjects")
        print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
