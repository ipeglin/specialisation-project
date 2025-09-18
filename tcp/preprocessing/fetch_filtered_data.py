#!/usr/bin/env python3
"""
Datalad data fetcher for filtered TCP dataset subjects.

Uses datalad get to selectively fetch data for only the subjects that passed
filtering, avoiding unnecessary downloads of excluded subjects' data.

Author: Ian Philip Eglin
Date: 2025-09-17
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd
import argparse
import logging
from datetime import datetime

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path

class DataladDataFetcher:
    """Manages selective data fetching for filtered subjects using datalad get"""

    def __init__(self, dataset_path: Path, included_subjects_path: Path, dry_run: bool = False):
        self.dataset_path = Path(dataset_path)
        self.included_subjects_path = Path(included_subjects_path)
        self.dry_run = dry_run
        self.logger = self._setup_logging()

        # Data type patterns for task-related, anatomical, and phenotypic data (no rest scans)
        self.data_patterns = {
            'raw_nifti': '{}/func/{}_task-*_bold.nii.gz',
            'json_metadata': '{}/func/{}_task-*_bold.json',
            'events': '{}/func/{}_task-*_events.tsv',
            'anatomical': '{}/anat/{}_T1w.nii.gz',
            'anatomical_t2w': '{}/anat/{}_T2w.nii.gz',
            'anatomical_json': '{}/anat/{}_T1w.json',
            'anatomical_t2w_json': '{}/anat/{}_T2w.json',
            'physio': '{}/func/{}_task-*_physio.tsv.gz',
            'behavioral': '{}/beh/{}_task-*_beh.tsv',
            'derivatives_fmriprep_func': 'derivatives/fmriprep/{}/func/{}_task-*_space-*_desc-*_bold.nii.gz',
            'derivatives_fmriprep_anat': 'derivatives/fmriprep/{}/anat/{}_*T1w*.nii.gz',
            'derivatives_fmriprep_anat_t2w': 'derivatives/fmriprep/{}/anat/{}_*T2w*.nii.gz',
            'derivatives_fmriprep_figures': 'derivatives/fmriprep/{}/figures/{}_*',
            'derivatives_mriqc_func': 'derivatives/mriqc/{}/func/{}_task-*_*.html',
            'derivatives_mriqc_anat': 'derivatives/mriqc/{}/anat/{}_*T1w*.html',
            'phenotype': 'phenotype/*.tsv',
            'participants': 'participants.tsv',
            'dataset_description': 'dataset_description.json'
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the data fetching process with error handling"""
        # Setup logger
        logger = logging.getLogger('datalad_fetcher')
        logger.setLevel(logging.INFO)

        # Console handler (always works)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Try to add file handler, but don't fail if there are I/O issues
        try:
            log_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'fetch_data_{timestamp}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Logging to: {log_file}")
        except (OSError, IOError) as e:
            print(f"WARNING: Could not create log file due to I/O error: {e}")
            print("Continuing with console logging only...")

        return logger

    def load_included_subjects(self) -> Set[str]:
        """Load the list of included subjects from the filtering pipeline output"""
        self.logger.info(f"Loading included subjects from {self.included_subjects_path}")

        included_subjects = set()

        # Look for patient and control CSV files
        #
        patient_file = self.included_subjects_path / 'patient_subjects.csv'
        control_file = self.included_subjects_path / 'control_subjects.csv'

        for file_path, group_name in [(patient_file, 'patients'), (control_file, 'controls')]:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Try both possible column names
                    id_column = None
                    if 'subject_id' in df.columns:
                        id_column = 'subject_id'
                    elif 'participant_id' in df.columns:
                        id_column = 'participant_id'

                    if id_column:
                        subjects = df[id_column].tolist()
                        included_subjects.update(subjects)
                        self.logger.info(f"Loaded {len(subjects)} {group_name} from {file_path.name}")
                    else:
                        self.logger.warning(f"No 'subject_id' or 'participant_id' column found in {file_path}")
                except Exception as e:
                    self.logger.error(f"Error reading {file_path}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path}")

        self.logger.info(f"Total included subjects: {len(included_subjects)}")
        return included_subjects

    def check_dataset_status(self) -> bool:
        """Check if the dataset path is a valid datalad repository"""
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset path does not exist: {self.dataset_path}")
            return False

        git_dir = self.dataset_path / '.git'
        if not git_dir.exists():
            self.logger.error(f"Dataset is not a git repository: {self.dataset_path}")
            return False

        # Check if it's a datalad dataset
        try:
            result = subprocess.run(
                ['datalad', 'status'],
                cwd=self.dataset_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                self.logger.error(f"Dataset is not a datalad repository: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout checking datalad status")
            return False
        except FileNotFoundError:
            self.logger.error("datalad command not found. Please install datalad.")
            return False

        return True

    def is_file_downloaded(self, file_path: Path) -> bool:
        """Check if a file has been downloaded (not a git-annex symlink)"""
        try:
            full_path = self.dataset_path / file_path
            if not full_path.exists():
                return False

            # Check if it's a symlink pointing to git-annex
            if full_path.is_symlink():
                link_target = str(full_path.readlink())
                if '.git/annex/objects' in link_target:
                    # It's still a git-annex symlink, not downloaded
                    return False

            # Check if it's an actual file with content
            return full_path.stat().st_size > 0

        except (OSError, IOError):
            return False

    def find_actual_files(self, subject_ids: Set[str], data_types: List[str]) -> Dict[str, List[str]]:
        """Find actual files that exist for the given subjects and data types"""
        import glob

        subject_files = {}
        global_files = []  # Files that are dataset-wide, not subject-specific

        # Handle global (non-subject-specific) files first
        global_data_types = ['phenotype', 'participants', 'dataset_description']
        for data_type in data_types:
            if data_type in global_data_types and data_type in self.data_patterns:
                pattern = self.data_patterns[data_type]
                full_pattern = str(self.dataset_path / pattern)
                matching_files = glob.glob(full_pattern)
                relative_files = [str(Path(f).relative_to(self.dataset_path)) for f in matching_files]
                global_files.extend(relative_files)
                self.logger.info(f"Found {len(relative_files)} {data_type} files (dataset-wide)")

        # Handle subject-specific files
        for subject_id in subject_ids:
            files = []
            for data_type in data_types:
                if data_type in self.data_patterns and data_type not in global_data_types:
                    pattern = self.data_patterns[data_type].format(subject_id, subject_id)
                    # Use glob to find actual files matching the pattern
                    full_pattern = str(self.dataset_path / pattern)
                    matching_files = glob.glob(full_pattern)
                    # Convert back to relative paths
                    relative_files = [str(Path(f).relative_to(self.dataset_path)) for f in matching_files]
                    files.extend(relative_files)
                    self.logger.debug(f"Found {len(relative_files)} {data_type} files for {subject_id}")
                elif data_type not in global_data_types and data_type not in self.data_patterns:
                    self.logger.warning(f"Unknown data type: {data_type}")

            # Add global files to the first subject only to avoid duplicates
            if subject_id == list(subject_ids)[0]:
                files.extend(global_files)

            if files:
                subject_files[subject_id] = files
                self.logger.info(f"Found {len(files)} total files for {subject_id}")
            else:
                self.logger.warning(f"No files found for {subject_id}")

        return subject_files

    def fetch_data(self, subject_ids: Set[str], data_types: List[str] = None) -> Dict[str, bool]:
        """Fetch data for specified subjects and data types using datalad get"""
        if data_types is None:
            data_types = ['raw_nifti', 'events']  # Default to essential data

        self.logger.info(f"Starting data fetch for {len(subject_ids)} subjects")
        self.logger.info(f"Data types: {', '.join(data_types)}")

        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual data will be fetched")

        # Find actual files that exist
        subject_files = self.find_actual_files(subject_ids, data_types)
        fetch_results = {}

        # Calculate overall statistics
        total_files = sum(len(files) for files in subject_files.values())
        files_already_downloaded = 0
        files_to_download = 0

        for subject_id, file_list in subject_files.items():
            for file_path in file_list:
                if self.is_file_downloaded(Path(file_path)):
                    files_already_downloaded += 1
                else:
                    files_to_download += 1

        try:
            self.logger.info(f"Overall progress: {files_already_downloaded}/{total_files} files already downloaded")
            self.logger.info(f"Need to download: {files_to_download} files")
        except (OSError, IOError):
            print(f"Overall progress: {files_already_downloaded}/{total_files} files already downloaded")
            print(f"Need to download: {files_to_download} files")

        # Track progress
        total_subjects = len(subject_files)
        current_subject = 0

        for subject_id, file_list in subject_files.items():
            current_subject += 1
            try:
                self.logger.info(f"[{current_subject}/{total_subjects}] Fetching data for subject: {subject_id} ({len(file_list)} files)")
            except (OSError, IOError):
                print(f"[{current_subject}/{total_subjects}] Fetching data for subject: {subject_id} ({len(file_list)} files)")

            success = True
            files_to_download = []
            files_already_downloaded = 0

            # First pass: check which files need downloading
            for file_path in file_list:
                if self.is_file_downloaded(Path(file_path)):
                    files_already_downloaded += 1
                else:
                    files_to_download.append(file_path)

            if files_already_downloaded > 0:
                try:
                    self.logger.info(f"  {files_already_downloaded} files already downloaded")
                except (OSError, IOError):
                    print(f"  {files_already_downloaded} files already downloaded")

            if len(files_to_download) == 0:
                try:
                    self.logger.info(f"  All files already downloaded for {subject_id}")
                except (OSError, IOError):
                    print(f"  All files already downloaded for {subject_id}")
                continue

            try:
                self.logger.info(f"  Downloading {len(files_to_download)} files...")
            except (OSError, IOError):
                print(f"  Downloading {len(files_to_download)} files...")

            # Second pass: download files that need it
            for i, file_path in enumerate(files_to_download, 1):
                try:
                    if self.dry_run:
                        try:
                            self.logger.info(f"  [{i}/{len(files_to_download)}] [DRY RUN] Would fetch: {file_path}")
                        except (OSError, IOError):
                            print(f"  [{i}/{len(files_to_download)}] [DRY RUN] Would fetch: {file_path}")
                        continue

                    # Use datalad get to fetch the specific file
                    cmd = ['datalad', 'get', file_path]
                    try:
                        self.logger.info(f"  [{i}/{len(files_to_download)}] Downloading: {Path(file_path).name}")
                    except (OSError, IOError):
                        print(f"  [{i}/{len(files_to_download)}] Downloading: {Path(file_path).name}")

                    result = subprocess.run(
                        cmd,
                        cwd=self.dataset_path,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout per pattern
                    )

                    if result.returncode == 0:
                        try:
                            self.logger.info(f"    ✓ Downloaded successfully")
                        except (OSError, IOError):
                            print(f"    ✓ Downloaded successfully")
                    else:
                        stderr_msg = result.stderr.strip() if result.stderr else "No error message"
                        stdout_msg = result.stdout.strip() if result.stdout else "No output"
                        try:
                            self.logger.warning(f"    ✗ Download failed: {stderr_msg}")
                        except (OSError, IOError):
                            print(f"    ✗ Download failed: {stderr_msg}")
                        # Don't mark as failure if files don't exist (common for optional data)
                        if "No such file" not in stderr_msg and "not found" not in stderr_msg and "nothing to get" not in stdout_msg.lower():
                            success = False

                except subprocess.TimeoutExpired:
                    try:
                        self.logger.error(f"    ✗ Timeout downloading {Path(file_path).name}")
                    except (OSError, IOError):
                        print(f"    ✗ Timeout downloading {Path(file_path).name}")
                    success = False
                except Exception as e:
                    try:
                        self.logger.error(f"    ✗ Error downloading {Path(file_path).name}: {e}")
                    except (OSError, IOError):
                        print(f"    ✗ Error downloading {Path(file_path).name}: {e}")
                    success = False

            fetch_results[subject_id] = success

        # Handle subjects with no files found
        for subject_id in subject_ids:
            if subject_id not in fetch_results:
                fetch_results[subject_id] = False
                self.logger.warning(f"No files found for subject: {subject_id}")

        return fetch_results

    def generate_summary_report(self, fetch_results: Dict[str, bool], subject_ids: Set[str],
                              data_types: List[str]) -> Path:
        """Generate a summary report of the data fetching process"""
        report_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'fetch_report_{timestamp}.json'

        successful_fetches = sum(1 for success in fetch_results.values() if success)

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'included_subjects_path': str(self.included_subjects_path),
            'dry_run': self.dry_run,
            'total_subjects': len(subject_ids),
            'data_types': data_types,
            'successful_fetches': successful_fetches,
            'failed_fetches': len(fetch_results) - successful_fetches,
            'success_rate': (successful_fetches / len(fetch_results)) * 100 if fetch_results else 0,
            'fetch_results': fetch_results,
            'subject_ids': sorted(list(subject_ids))
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Summary report saved to: {report_file}")
        return report_file

def main():
    """Main execution function"""
    # Define available data types for argument parsing
    available_data_types = [
        'raw_nifti', 'json_metadata', 'events', 'anatomical', 'anatomical_t2w',
        'anatomical_json', 'anatomical_t2w_json', 'physio', 'behavioral',
        'derivatives_fmriprep_func', 'derivatives_fmriprep_anat', 'derivatives_fmriprep_anat_t2w',
        'derivatives_fmriprep_figures', 'derivatives_mriqc_func', 'derivatives_mriqc_anat',
        'phenotype', 'participants', 'dataset_description'
    ]

    parser = argparse.ArgumentParser(description='Fetch data for filtered TCP subjects using datalad')
    parser.add_argument('--data-types', nargs='+',
                       choices=available_data_types,
                       default=['raw_nifti', 'json_metadata', 'events', 'anatomical', 'anatomical_json',
                                'derivatives_fmriprep_func', 'derivatives_fmriprep_anat', 'derivatives_mriqc_func',
                                'derivatives_mriqc_anat', 'phenotype', 'participants', 'dataset_description'],
                       help='Data types to fetch (default: all task-related, anatomical, and phenotypic data)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be fetched without actually fetching')
    parser.add_argument('--dataset-path', type=Path,
                       help='Override dataset path (default: from config)')
    parser.add_argument('--included-path', type=Path,
                       help='Override included subjects path (default: from config)')

    args = parser.parse_args()

    # Get paths
    dataset_path = args.dataset_path or get_tcp_dataset_path()
    included_path = args.included_path or get_script_output_path('tcp_preprocessing', 'filter_subjects', 'included')

    print("TCP Dataset Data Fetcher")
    print("========================")
    print(f"Dataset path: {dataset_path}")
    print(f"Included subjects path: {included_path}")
    print(f"Data types: {', '.join(args.data_types)}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize fetcher
    fetcher = DataladDataFetcher(dataset_path, included_path, dry_run=args.dry_run)

    # Check dataset status
    if not fetcher.check_dataset_status():
        print("ERROR: Dataset validation failed. Please check the dataset path and datalad installation.")
        return 1

    # Load included subjects
    subject_ids = fetcher.load_included_subjects()
    if not subject_ids:
        print("ERROR: No included subjects found. Please check the filtering pipeline output.")
        return 1

    # Fetch data
    fetch_results = fetcher.fetch_data(subject_ids, args.data_types)

    # Generate report
    report_file = fetcher.generate_summary_report(fetch_results, subject_ids, args.data_types)

    # Print summary
    successful = sum(1 for success in fetch_results.values() if success)
    total = len(fetch_results)

    print(f"\n=== FETCH COMPLETE ===")
    print(f"Processed {total} subjects")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    print(f"Report saved to: {report_file}")

    if args.dry_run:
        print("\nThis was a dry run. Use --no-dry-run to actually fetch the data.")

    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
