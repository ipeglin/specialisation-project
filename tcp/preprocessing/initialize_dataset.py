#!/usr/bin/env python3
"""
TCP Dataset Initialization Script

Checks if the TCP dataset is properly installed and initializes it if needed.
This script ensures the dataset is available before proceeding with preprocessing.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path, get_script_output_path

# Import unicode compatibility utilities for cross-platform support
utils_dir = Path(__file__).parent / 'utils'
sys.path.insert(0, str(utils_dir))
from unicode_compat import CHECK, CROSS, WARNING, SUCCESS, ERROR, RUNNING, SEARCH, DOWNLOAD, DOCUMENT, LIGHTBULB, ARROW_RIGHT

class DatasetInitializer:
    """Manages TCP dataset initialization and validation"""
    
    def __init__(self, dataset_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'initialize_dataset')
        self.dataset_url = "https://github.com/OpenNeuroDatasets/ds005237.git"
        
        # Expected dataset structure
        self.required_files = [
            "participants.tsv",
            "dataset_description.json",
            "README",
            "CHANGES"
        ]
        
        self.required_dirs = [
            "phenotype",
            ".git"
        ]
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
    
    def check_datalad_available(self) -> bool:
        """Check if datalad is installed and available"""
        try:
            result = subprocess.run(
                ['datalad', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"{CHECK} DataLad available: {result.stdout.strip()}")
                return True
            else:
                print(f"{CROSS} DataLad check failed: {result.stderr}")
                return False
        except FileNotFoundError:
            print(f"{CROSS} DataLad not found. Please install datalad: pip install datalad")
            return False
        except subprocess.TimeoutExpired:
            print(f"{CROSS} DataLad check timed out")
            return False
    
    def check_git_available(self) -> bool:
        """Check if git is installed and available"""
        try:
            result = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"{CHECK} Git available: {result.stdout.strip()}")
                return True
            else:
                print(f"{CROSS} Git check failed: {result.stderr}")
                return False
        except FileNotFoundError:
            print(f"{CROSS} Git not found. Please install git")
            return False
        except subprocess.TimeoutExpired:
            print(f"{CROSS} Git check timed out")
            return False
    
    def dataset_exists(self) -> bool:
        """Check if dataset directory exists"""
        exists = self.dataset_path.exists()
        symbol = CHECK if exists else CROSS
        print(f"{symbol} Dataset directory exists: {exists}")
        return exists
    
    def is_git_repository(self) -> bool:
        """Check if dataset is a git repository"""
        git_dir = self.dataset_path / '.git'
        is_git = git_dir.exists()
        symbol = CHECK if is_git else CROSS
        print(f"{symbol} Is git repository: {is_git}")
        return is_git
    
    def is_datalad_dataset(self) -> bool:
        """Check if dataset is a datalad dataset"""
        if not self.dataset_exists() or not self.is_git_repository():
            return False
        
        try:
            result = subprocess.run(
                ['datalad', 'status'],
                cwd=self.dataset_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            is_datalad = result.returncode == 0
            symbol = CHECK if is_datalad else CROSS
            print(f"{symbol} Is datalad dataset: {is_datalad}")
            if not is_datalad and result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return is_datalad
        except subprocess.TimeoutExpired:
            print(f"{CROSS} Datalad status check timed out")
            return False
        except Exception as e:
            print(f"{CROSS} Datalad status check failed: {e}")
            return False
    
    def check_dataset_structure(self) -> Dict[str, bool]:
        """Check if required files and directories are present"""
        structure_status = {}
        
        print("Checking dataset structure...")
        
        # Check required files
        for file_name in self.required_files:
            file_path = self.dataset_path / file_name
            exists = file_path.exists()
            structure_status[f"file_{file_name}"] = exists
            symbol = CHECK if exists else CROSS
            print(f"  {symbol} {file_name}: {exists}")

        # Check required directories
        for dir_name in self.required_dirs:
            dir_path = self.dataset_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            structure_status[f"dir_{dir_name}"] = exists
            symbol = CHECK if exists else CROSS
            print(f"  {symbol} {dir_name}/: {exists}")

        # Check for subject directories
        subject_dirs = list(self.dataset_path.glob("sub-*"))
        structure_status["has_subjects"] = len(subject_dirs) > 0
        has_subjects = len(subject_dirs) > 0
        symbol = CHECK if has_subjects else CROSS
        print(f"  {symbol} Subject directories: {len(subject_dirs)} found")
        
        return structure_status
    
    def validate_existing_dataset(self) -> Tuple[bool, Dict]:
        """Validate existing dataset installation"""
        print("\n=== Validating Existing Dataset ===")
        
        validation_results = {
            "dataset_exists": self.dataset_exists(),
            "is_git_repo": False,
            "is_datalad_dataset": False,
            "structure_check": {}
        }
        
        if validation_results["dataset_exists"]:
            validation_results["is_git_repo"] = self.is_git_repository()
            
            if validation_results["is_git_repo"]:
                validation_results["is_datalad_dataset"] = self.is_datalad_dataset()
                validation_results["structure_check"] = self.check_dataset_structure()
        
        is_valid = (
            validation_results["dataset_exists"] and
            validation_results["is_git_repo"] and
            validation_results["is_datalad_dataset"] and
            all(validation_results["structure_check"].values())
        )
        
        symbol = CHECK if is_valid else CROSS
        status = 'PASSED' if is_valid else 'FAILED'
        print(f"\n{symbol} Dataset validation: {status}")
        return is_valid, validation_results
    
    def clone_dataset(self) -> bool:
        """Clone the TCP dataset using datalad install"""
        print(f"\n=== Cloning Dataset ===")
        print(f"Installing dataset from: {self.dataset_url}")
        print(f"Target directory: {self.dataset_path}")

        # Ensure parent directory exists
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use datalad install to clone the dataset
            cmd = ['datalad', 'install', str(self.dataset_path), '-s', self.dataset_url]
            print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for cloning
            )

            if result.returncode == 0:
                print(f"{CHECK} Dataset cloned successfully")
                print(f"Output: {result.stdout}")
                return True
            else:
                print(f"{CROSS} Dataset cloning failed")
                print(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"{CROSS} Dataset cloning timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"{CROSS} Dataset cloning failed with exception: {e}")
            return False

    def reinstall_dataset_content(self) -> bool:
        """Re-fetch all dataset content using datalad get -r"""
        print(f"\n=== Force Reinstalling Dataset Content ===")
        print(f"This will fetch all missing files without re-cloning the repository")
        print(f"Dataset directory: {self.dataset_path}")

        if not self.dataset_exists():
            print(f"{CROSS} Dataset directory does not exist: {self.dataset_path}")
            print(f"Cannot reinstall content. Please run without --force-reinstall to clone the dataset first.")
            return False

        if not self.is_git_repository():
            print(f"{CROSS} Dataset is not a git repository: {self.dataset_path}")
            print(f"Cannot reinstall content. Directory exists but is not a valid git repository.")
            return False

        try:
            # Use datalad get -r to recursively fetch all content
            # The -r flag makes it recursive, fetching all missing files
            # Datalad will skip files that are already downloaded
            cmd = ['datalad', 'get', '-r', '.']
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {self.dataset_path}")
            print(f"\nThis may take a while depending on dataset size and missing files...")
            print(f"(Datalad will automatically skip already-downloaded files)")

            result = subprocess.run(
                cmd,
                cwd=self.dataset_path,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for large datasets
            )

            if result.returncode == 0:
                print(f"{CHECK} Dataset content reinstalled successfully")
                if result.stdout:
                    # Print summary of what was fetched
                    lines = result.stdout.strip().split('\n')
                    fetched_count = sum(1 for line in lines if 'get(ok)' in line.lower())
                    if fetched_count > 0:
                        print(f"  {ARROW_RIGHT} Fetched {fetched_count} missing files")
                    else:
                        print(f"  {ARROW_RIGHT} All files were already present")
                return True
            else:
                print(f"{CROSS} Dataset content reinstall failed")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                # Even if some files fail, this might be acceptable for optional files
                # Check if it's a partial success
                if result.stdout and 'get(ok)' in result.stdout.lower():
                    print(f"{WARNING} Some files were fetched, but the operation completed with errors")
                    print(f"This may be acceptable if only optional files failed")
                    return True
                return False

        except subprocess.TimeoutExpired:
            print(f"{CROSS} Dataset content reinstall timed out (>2 hours)")
            print("The dataset may be very large. Consider fetching specific subdirectories instead.")
            return False
        except Exception as e:
            print(f"{CROSS} Dataset content reinstall failed with exception: {e}")
            return False
    
    def initialize_dataset(self, force_reinstall: bool = False) -> Tuple[bool, Dict]:
        """Main dataset initialization workflow"""
        print("=== TCP Dataset Initialization ===\n")

        # Check prerequisites
        if not self.check_datalad_available():
            return False, {"error": "DataLad not available"}

        if not self.check_git_available():
            return False, {"error": "Git not available"}

        # Check if dataset already exists and is valid
        is_valid, validation_results = self.validate_existing_dataset()

        # Handle force reinstall mode
        if force_reinstall:
            print(f"\n{RUNNING} Force reinstall mode enabled")

            if not self.dataset_exists():
                print("Dataset does not exist. Cloning first...")
                clone_success = self.clone_dataset()
                if not clone_success:
                    return False, {"error": "Dataset cloning failed"}

            # Force reinstall all content
            reinstall_success = self.reinstall_dataset_content()

            if not reinstall_success:
                return False, {
                    "error": "Dataset content reinstall failed",
                    "validation_results": validation_results
                }

            # Re-validate after reinstall
            print(f"\n{SEARCH} Validating dataset after reinstall...")
            is_valid, validation_results = self.validate_existing_dataset()

            if is_valid:
                print(f"\n{SUCCESS} Dataset reinstallation completed successfully!")
                return True, validation_results
            else:
                print(f"\n{WARNING} Dataset reinstalled but validation has warnings")
                print("This may be acceptable if only optional files are missing")
                return True, {
                    "warning": "Validation incomplete but reinstall succeeded",
                    "validation_results": validation_results
                }

        # Normal mode (not force reinstall)
        if is_valid:
            print(f"\n{CHECK} Dataset is already properly initialized!")
            print("Tip: Use --force-reinstall to re-fetch any missing files")
            return True, validation_results

        # If dataset exists but is invalid, ask for action
        if self.dataset_exists():
            print(f"\n{WARNING} Dataset directory exists but is invalid: {self.dataset_path}")
            print("This could be:")
            print("- A partial/failed installation")
            print("- A different dataset")
            print("- A corrupted installation")
            print("\nRecommendations:")
            print(f"  1. Use --force-reinstall to fetch missing files")
            print(f"  2. Remove {self.dataset_path} and re-run to start fresh")
            return False, {
                "error": "Invalid existing dataset",
                "validation_results": validation_results,
                "recommendation": f"Use --force-reinstall or remove {self.dataset_path} and re-run"
            }

        # Clone the dataset
        print(f"\n{DOWNLOAD} Dataset not found. Installing...")
        clone_success = self.clone_dataset()

        if not clone_success:
            return False, {"error": "Dataset cloning failed"}

        # Validate the newly cloned dataset
        print(f"\n{SEARCH} Validating newly installed dataset...")
        is_valid, validation_results = self.validate_existing_dataset()

        if is_valid:
            print(f"\n{SUCCESS} Dataset initialization completed successfully!")
            return True, validation_results
        else:
            print(f"\n{ERROR} Dataset initialization failed - validation failed after cloning")
            return False, {
                "error": "Post-clone validation failed",
                "validation_results": validation_results
            }
    
    def generate_report(self, success: bool, results: Dict) -> Path:
        """Generate initialization report"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'initialization_report_{timestamp}.json'
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "dataset_url": self.dataset_url,
            "initialization_success": success,
            "results": results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{DOCUMENT} Report saved to: {report_file}")
        return report_file

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize TCP dataset and optionally force reinstall of all content'
    )
    parser.add_argument(
        '--force-reinstall',
        action='store_true',
        help='Force reinstall of all dataset content, fetching any missing files without re-cloning'
    )
    parser.add_argument(
        '--dataset-path',
        type=Path,
        help='Override dataset path (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Override output directory (default: from config)'
    )

    args = parser.parse_args()

    print("TCP Dataset Initialization")
    print("=" * 50)

    # Initialize the dataset
    initializer = DatasetInitializer(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
    success, results = initializer.initialize_dataset(force_reinstall=args.force_reinstall)
    
    # Generate report
    report_file = initializer.generate_report(success, results)
    
    # Print summary
    print(f"\n{'=' * 50}")
    print(f"INITIALIZATION {'COMPLETED' if success else 'FAILED'}")
    print(f"{'=' * 50}")
    
    if success:
        print(f"{SUCCESS} Dataset is ready at: {initializer.dataset_path}")
        print(f"{DOCUMENT} Report: {report_file}")
        print(f"\nNext steps:")
        print(f"  1. Run validate_subjects.py to check subject data")
        print(f"  2. Run fetch_global_data.py to get participants.tsv")
        return 0
    else:
        print(f"{ERROR} Initialization failed")
        print(f"{DOCUMENT} Error report: {report_file}")

        if "recommendation" in results:
            print(f"\n{LIGHTBULB} Recommendation: {results['recommendation']}")

        return 1

if __name__ == "__main__":
    sys.exit(main())