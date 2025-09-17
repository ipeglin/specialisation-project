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

        # Data type patterns for selective fetching (no sessions in this dataset)
        self.data_patterns = {
            'raw_nifti': '{}/func/{}_task-*_bold.nii.gz',
            'derivatives_fmriprep': 'derivatives/fmriprep/{}/func/{}_task-*_*bold*.nii.gz', 
            'derivatives_mriqc': 'derivatives/mriqc/{}/func/{}_task-*_*.html',
            'events': '{}/func/{}_task-*_events.tsv',
            'anatomical': '{}/anat/{}_T1w.nii.gz',
            'derivatives_anat': 'derivatives/fmriprep/{}/anat/{}_*T1w*.nii.gz'
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the data fetching process"""
        # Create logs directory
        log_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        logger = logging.getLogger('datalad_fetcher')
        logger.setLevel(logging.INFO)

        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'fetch_data_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

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

    def find_actual_files(self, subject_ids: Set[str], data_types: List[str]) -> Dict[str, List[str]]:
        """Find actual files that exist for the given subjects and data types"""
        import glob
        
        subject_files = {}
        
        for subject_id in subject_ids:
            files = []
            for data_type in data_types:
                if data_type in self.data_patterns:
                    pattern = self.data_patterns[data_type].format(subject_id, subject_id)
                    # Use glob to find actual files matching the pattern
                    full_pattern = str(self.dataset_path / pattern)
                    matching_files = glob.glob(full_pattern)
                    # Convert back to relative paths
                    relative_files = [str(Path(f).relative_to(self.dataset_path)) for f in matching_files]
                    files.extend(relative_files)
                    self.logger.debug(f"Found {len(relative_files)} {data_type} files for {subject_id}")
                else:
                    self.logger.warning(f"Unknown data type: {data_type}")
            
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

        for subject_id, file_list in subject_files.items():
            self.logger.info(f"Fetching data for subject: {subject_id} ({len(file_list)} files)")

            success = True
            for file_path in file_list:
                try:
                    if self.dry_run:
                        self.logger.info(f"[DRY RUN] Would fetch: {file_path}")
                        continue

                    # Use datalad get to fetch the specific file
                    cmd = ['datalad', 'get', file_path]
                    self.logger.info(f"Running: {' '.join(cmd)}")

                    result = subprocess.run(
                        cmd,
                        cwd=self.dataset_path,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout per pattern
                    )

                    if result.returncode == 0:
                        self.logger.info(f"Successfully fetched: {file_path}")
                    else:
                        stderr_msg = result.stderr.strip() if result.stderr else "No error message"
                        stdout_msg = result.stdout.strip() if result.stdout else "No output"
                        self.logger.warning(f"Failed to fetch {file_path}")
                        self.logger.warning(f"  Return code: {result.returncode}")
                        self.logger.warning(f"  STDERR: {stderr_msg}")
                        self.logger.warning(f"  STDOUT: {stdout_msg}")
                        # Don't mark as failure if files don't exist (common for optional data)
                        if "No such file" not in stderr_msg and "not found" not in stderr_msg and "nothing to get" not in stdout_msg.lower():
                            success = False

                except subprocess.TimeoutExpired:
                    self.logger.error(f"Timeout fetching {file_path}")
                    success = False
                except Exception as e:
                    self.logger.error(f"Error fetching {file_path}: {e}")
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
        'raw_nifti', 'derivatives_fmriprep', 'derivatives_mriqc',
        'events', 'anatomical', 'derivatives_anat'
    ]

    parser = argparse.ArgumentParser(description='Fetch data for filtered TCP subjects using datalad')
    parser.add_argument('--data-types', nargs='+',
                       choices=available_data_types,
                       default=['raw_nifti', 'events'],
                       help='Data types to fetch (default: raw_nifti events)')
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
