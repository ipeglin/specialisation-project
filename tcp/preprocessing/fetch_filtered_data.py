#!/usr/bin/env python3
"""
Datalad data fetcher for filtered TCP dataset subjects.

Uses datalad get to selectively fetch data for only the subjects that passed
filtering, avoiding unnecessary downloads of excluded subjects' data.

REFACTORED: Now uses pre-computed file mappings from map_subject_files.py for
efficient fetching. Group-agnostic - processes all filtered subjects equally.

Author: Ian Philip Eglin
Date: 2025-09-17
Updated: 2025-10-16
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import argparse
import logging
from datetime import datetime

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path

class DataladDataFetcher:
    """Manages selective data fetching for filtered subjects using datalad get

    Refactored to use pre-computed file mappings from map_subject_files.py,
    eliminating runtime file discovery and improving efficiency.
    """

    def __init__(self, dataset_path: Path, file_mapping_path: Path, dry_run: bool = False):
        self.dataset_path = Path(dataset_path)
        self.file_mapping_path = Path(file_mapping_path)
        self.dry_run = dry_run
        self.logger = self._setup_logging()

        # Data type mapping for filtering which files to fetch
        self.data_type_categories = {
            'raw_nifti': ['raw_nifti'],
            'events': ['events'],
            'json_metadata': ['json_metadata'],
            'anatomical': ['anatomical', 'anatomical_json'],
            'timeseries': ['timeseries']
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

    def load_file_mapping(self) -> Tuple[Dict, List[str]]:
        """Load pre-computed file mapping from map_subject_files.py output"""
        self.logger.info(f"Loading file mapping from {self.file_mapping_path}")

        if not self.file_mapping_path.exists():
            raise FileNotFoundError(
                f"File mapping not found: {self.file_mapping_path}\n"
                f"Please run map_subject_files.py first."
            )

        try:
            with open(self.file_mapping_path, 'r') as f:
                mapping_data = json.load(f)

            subject_file_mapping = mapping_data.get('subjects', {})
            global_files = mapping_data.get('global_files', [])

            self.logger.info(f"Loaded file mapping for {len(subject_file_mapping)} subjects")
            self.logger.info(f"Found {len(global_files)} global files")

            return subject_file_mapping, global_files

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file mapping: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading file mapping: {e}")

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

    def filter_files_by_data_types(self, subject_file_mapping: Dict,
                                    data_types: List[str]) -> Dict[str, List[str]]:
        """Filter the file mapping based on requested data types

        Args:
            subject_file_mapping: Complete file mapping for all subjects
            data_types: List of data type categories to include

        Returns:
            Filtered mapping with only requested data types
        """
        filtered_mapping = {}

        for subject_id, file_map in subject_file_mapping.items():
            files = []

            for data_type in data_types:
                if data_type in file_map:
                    # Handle nested structure (hammer/stroop or t1w/t2w)
                    if isinstance(file_map[data_type], dict):
                        for task_or_scan, file_list in file_map[data_type].items():
                            files.extend(file_list)
                    else:
                        files.extend(file_map[data_type])

            if files:
                filtered_mapping[subject_id] = files
                self.logger.debug(f"Filtered {len(files)} files for {subject_id}")
            else:
                self.logger.warning(f"No files matching requested data types for {subject_id}")

        self.logger.info(f"Filtered file mapping: {len(filtered_mapping)} subjects with requested data types")
        return filtered_mapping

    def fetch_data(self, subject_file_mapping: Dict, global_files: List[str],
                   data_types: List[str] = None) -> Dict[str, bool]:
        """Fetch data using pre-computed file mapping from map_subject_files.py

        Args:
            subject_file_mapping: Pre-computed file mapping for all subjects
            global_files: List of dataset-wide files to fetch
            data_types: List of data type categories to fetch (default: essential data)

        Returns:
            Dictionary mapping subject_id to fetch success boolean
        """
        if data_types is None:
            data_types = ['raw_nifti', 'events']  # Default to essential data

        self.logger.info(f"Starting data fetch for {len(subject_file_mapping)} subjects")
        self.logger.info(f"Data types: {', '.join(data_types)}")

        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual data will be fetched")

        # Filter files by requested data types
        subject_files = self.filter_files_by_data_types(subject_file_mapping, data_types)
        fetch_results = {}

        # Calculate overall statistics
        total_subject_files = sum(len(files) for files in subject_files.values())
        total_files = total_subject_files + len(global_files)
        files_already_downloaded = 0
        files_to_download = 0

        # Check subject-specific files
        for subject_id, file_list in subject_files.items():
            for file_path in file_list:
                if self.is_file_downloaded(Path(file_path)):
                    files_already_downloaded += 1
                else:
                    files_to_download += 1

        # Check global files
        for file_path in global_files:
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

        # Handle global (dataset-wide) files
        if global_files:
            try:
                self.logger.info(f"Processing {len(global_files)} dataset-wide files (phenotype, participants, etc.)")
            except (OSError, IOError):
                print(f"Processing {len(global_files)} dataset-wide files (phenotype, participants, etc.)")

            global_files_to_download = []
            global_files_already_downloaded = 0

            # Check which global files need downloading
            for file_path in global_files:
                if self.is_file_downloaded(Path(file_path)):
                    global_files_already_downloaded += 1
                else:
                    global_files_to_download.append(file_path)

            if global_files_already_downloaded > 0:
                try:
                    self.logger.info(f"  {global_files_already_downloaded} dataset-wide files already downloaded")
                except (OSError, IOError):
                    print(f"  {global_files_already_downloaded} dataset-wide files already downloaded")

            if len(global_files_to_download) > 0:
                try:
                    self.logger.info(f"  Downloading {len(global_files_to_download)} dataset-wide files...")
                except (OSError, IOError):
                    print(f"  Downloading {len(global_files_to_download)} dataset-wide files...")

                # Download global files
                for i, file_path in enumerate(global_files_to_download, 1):
                    try:
                        if self.dry_run:
                            try:
                                self.logger.info(f"  [{i}/{len(global_files_to_download)}] [DRY RUN] Would fetch: {file_path}")
                            except (OSError, IOError):
                                print(f"  [{i}/{len(global_files_to_download)}] [DRY RUN] Would fetch: {file_path}")
                            continue

                        # Use datalad get to fetch the specific file
                        cmd = ['datalad', 'get', file_path]
                        try:
                            self.logger.info(f"  [{i}/{len(global_files_to_download)}] Downloading: {Path(file_path).name}")
                        except (OSError, IOError):
                            print(f"  [{i}/{len(global_files_to_download)}] Downloading: {Path(file_path).name}")

                        result = subprocess.run(
                            cmd,
                            cwd=self.dataset_path,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout per file
                        )

                        if result.returncode == 0:
                            try:
                                self.logger.info(f"    ✓ Downloaded successfully")
                            except (OSError, IOError):
                                print(f"    ✓ Downloaded successfully")
                        else:
                            stderr_msg = result.stderr.strip() if result.stderr else "No error message"
                            try:
                                self.logger.warning(f"    ✗ Download failed: {stderr_msg}")
                            except (OSError, IOError):
                                print(f"    ✗ Download failed: {stderr_msg}")

                    except subprocess.TimeoutExpired:
                        try:
                            self.logger.error(f"    ✗ Timeout downloading {Path(file_path).name}")
                        except (OSError, IOError):
                            print(f"    ✗ Timeout downloading {Path(file_path).name}")
                    except Exception as e:
                        try:
                            self.logger.error(f"    ✗ Error downloading {Path(file_path).name}: {e}")
                        except (OSError, IOError):
                            print(f"    ✗ Error downloading {Path(file_path).name}: {e}")
            else:
                try:
                    self.logger.info(f"  All dataset-wide files already downloaded")
                except (OSError, IOError):
                    print(f"  All dataset-wide files already downloaded")

        # Handle subjects with no files found after filtering
        all_subject_ids = list(subject_file_mapping.keys())
        for subject_id in all_subject_ids:
            if subject_id not in fetch_results:
                fetch_results[subject_id] = False
                self.logger.warning(f"No files found for subject: {subject_id}")

        return fetch_results

    def generate_summary_report(self, fetch_results: Dict[str, bool],
                              data_types: List[str]) -> Path:
        """Generate a summary report of the data fetching process"""
        report_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f'fetch_report_{timestamp}.json'

        successful_fetches = sum(1 for success in fetch_results.values() if success)
        subject_ids = list(fetch_results.keys())

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'file_mapping_path': str(self.file_mapping_path),
            'dry_run': self.dry_run,
            'total_subjects': len(subject_ids),
            'data_types': data_types,
            'successful_fetches': successful_fetches,
            'failed_fetches': len(fetch_results) - successful_fetches,
            'success_rate': (successful_fetches / len(fetch_results)) * 100 if fetch_results else 0,
            'fetch_results': fetch_results,
            'subject_ids': sorted(subject_ids),
            'note': 'Group-agnostic fetching - all filtered subjects processed equally'
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Summary report saved to: {report_file}")
        return report_file

def detect_file_mapping_path() -> Path:
    """Automatically detect the path to file mapping from map_subject_files.py output"""

    # Look for file mapping from map_subject_files step
    mapping_dir = get_script_output_path('tcp_preprocessing', 'map_subject_files')
    mapping_file = mapping_dir / 'subject_file_mapping.json'

    if mapping_file.exists():
        print(f"✓ Found file mapping: {mapping_file}")
        return mapping_file

    # No file mapping found
    raise FileNotFoundError(
        "File mapping not found. Please run the preprocessing pipeline:\n"
        "  1. initialize_dataset.py (if dataset not cloned)\n"
        "  2. validate_subjects.py (required)\n"
        "  3. fetch_global_data.py (required for phenotype filtering)\n"
        "  4. filter_phenotype.py (optional, for diagnosis filtering)\n"
        "  5. filter_subjects.py (required, for task data filtering)\n"
        "  6. map_subject_files.py (required, creates file path mapping)\n"
        "\n"
        f"Expected file: {mapping_file}"
    )

def main():
    """Main execution function"""
    # Define available data types for argument parsing (matching map_subject_files output structure)
    available_data_types = [
        'raw_nifti', 'events', 'json_metadata', 'anatomical', 'anatomical_json', 'timeseries'
    ]

    parser = argparse.ArgumentParser(
        description='Fetch data for filtered TCP subjects using datalad',
        epilog='This script uses pre-computed file mappings from map_subject_files.py'
    )
    parser.add_argument('--data-types', nargs='+',
                       choices=available_data_types,
                       default=['raw_nifti', 'events', 'json_metadata', 'anatomical', 'anatomical_json'],
                       help='Data type categories to fetch (default: raw NIFTI, events, JSON metadata, and anatomical scans)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be fetched without actually fetching')
    parser.add_argument('--dataset-path', type=Path,
                       help='Override dataset path (default: from config)')
    parser.add_argument('--file-mapping', type=Path,
                       help='Override file mapping path (auto-detected by default)')

    args = parser.parse_args()

    # Get paths
    dataset_path = args.dataset_path or get_tcp_dataset_path()

    # Auto-detect file mapping path if not provided
    if args.file_mapping:
        file_mapping_path = args.file_mapping
        print(f"Using specified file mapping: {file_mapping_path}")
    else:
        file_mapping_path = detect_file_mapping_path()

    print("TCP Dataset Data Fetcher (REFACTORED)")
    print("=" * 50)
    print(f"Dataset path: {dataset_path}")
    print(f"File mapping: {file_mapping_path}")
    print(f"Data types: {', '.join(args.data_types)}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Initialize fetcher
    fetcher = DataladDataFetcher(dataset_path, file_mapping_path, dry_run=args.dry_run)

    # Check dataset status
    if not fetcher.check_dataset_status():
        print("ERROR: Dataset validation failed. Please check the dataset path and datalad installation.")
        return 1

    # Load file mapping
    try:
        subject_file_mapping, global_files = fetcher.load_file_mapping()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1

    if not subject_file_mapping:
        print("ERROR: No subjects found in file mapping. Please check map_subject_files.py output.")
        return 1

    # Fetch data
    fetch_results = fetcher.fetch_data(subject_file_mapping, global_files, args.data_types)

    # Generate report
    report_file = fetcher.generate_summary_report(fetch_results, args.data_types)

    # Print summary
    successful = sum(1 for success in fetch_results.values() if success)
    total = len(fetch_results)

    print(f"\n{'=' * 50}")
    print(f"FETCH COMPLETE")
    print(f"{'=' * 50}")
    print(f"Processed {total} subjects (group-agnostic)")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    print(f"Report saved to: {report_file}")

    if args.dry_run:
        print("\nThis was a dry run. Remove --dry-run flag to actually fetch the data.")

    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())
