#!/usr/bin/env python3
"""
TCP Global Data Fetcher

Fetches dataset-wide files needed for filtering (participants.tsv, phenotype files).
This enables filtering decisions before downloading large MRI files.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path, get_script_output_path

class GlobalDataFetcher:
    """Fetches global dataset files needed for subject filtering"""
    
    def __init__(self, dataset_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'fetch_global_data')
        
        # Global files needed for filtering
        self.global_files = {
            'participants': {
                'path': 'participants.tsv',
                'description': 'Main participants file with basic demographics',
                'required': True
            },
            'phenotype_demos': {
                'path': 'phenotype/demos.tsv',
                'description': 'Demographic and diagnostic information',
                'required': True
            },
            'dataset_description': {
                'path': 'dataset_description.json',
                'description': 'Dataset metadata',
                'required': False
            },
            'readme': {
                'path': 'README',
                'description': 'Dataset documentation',
                'required': False
            }
        }
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
    
    def validate_dataset(self) -> bool:
        """Validate that dataset is properly initialized"""
        if not self.dataset_path.exists():
            print(f"✗ Dataset directory does not exist: {self.dataset_path}")
            return False
        
        git_dir = self.dataset_path / '.git'
        if not git_dir.exists():
            print(f"✗ Dataset is not a git repository: {self.dataset_path}")
            return False
        
        print(f"✓ Dataset directory validated: {self.dataset_path}")
        return True
    
    def check_datalad_available(self) -> bool:
        """Check if datalad is available"""
        try:
            result = subprocess.run(
                ['datalad', '--version'],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def is_file_downloaded(self, file_path: Path) -> bool:
        """Check if a file has been downloaded (not a git-annex symlink)"""
        full_path = self.dataset_path / file_path
        
        if not full_path.exists():
            return False
        
        # Check if it's a symlink pointing to git-annex
        if full_path.is_symlink():
            link_target = str(full_path.readlink())
            if '.git/annex/objects' in link_target:
                return False
        
        # Check if it has actual content
        try:
            return full_path.stat().st_size > 0
        except (OSError, IOError):
            return False
    
    def fetch_file(self, file_info: Dict, file_key: str) -> Tuple[bool, str]:
        """Fetch a single global file using datalad get"""
        file_path = file_info['path']
        full_path = self.dataset_path / file_path
        
        print(f"  Fetching {file_key}: {file_path}")
        
        # Check if already downloaded
        if self.is_file_downloaded(Path(file_path)):
            print(f"    ✓ Already downloaded")
            return True, "Already downloaded"
        
        # Check if file exists in repository (including symlinks)
        if not full_path.exists() and not full_path.is_symlink():
            message = f"File does not exist in repository: {file_path}"
            if file_info['required']:
                print(f"    ✗ {message}")
                return False, message
            else:
                print(f"    ⚠ {message} (optional file)")
                return True, message
        
        try:
            # Use datalad get to fetch the file
            cmd = ['datalad', 'get', file_path]
            result = subprocess.run(
                cmd,
                cwd=self.dataset_path,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per file
            )
            
            if result.returncode == 0:
                print(f"    ✓ Downloaded successfully")
                return True, "Downloaded successfully"
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"    ✗ Download failed: {error_msg}")
                return False, f"Download failed: {error_msg}"
                
        except subprocess.TimeoutExpired:
            message = f"Download timed out for {file_path}"
            print(f"    ✗ {message}")
            return False, message
        except Exception as e:
            message = f"Download error for {file_path}: {e}"
            print(f"    ✗ {message}")
            return False, message
    
    def validate_fetched_files(self) -> Dict[str, Dict]:
        """Validate that fetched files are readable and contain expected data"""
        print("\nValidating fetched files...")
        
        # Import pandas here where it's needed
        try:
            import pandas as pd
        except ImportError:
            print("Warning: pandas not available for TSV validation")
            pd = None
        
        validation_results = {}
        
        for file_key, file_info in self.global_files.items():
            file_path = self.dataset_path / file_info['path']
            
            validation_result = {
                'exists': False,
                'readable': False,
                'valid_format': False,
                'row_count': 0,
                'column_count': 0,
                'columns': [],
                'error': None
            }
            
            print(f"  Validating {file_key}: {file_info['path']}")
            
            # Check existence
            if not file_path.exists():
                validation_result['error'] = "File does not exist"
                validation_results[file_key] = validation_result
                continue
            
            validation_result['exists'] = True
            
            # Try to read the file
            try:
                if file_path.suffix == '.tsv':
                    # Read TSV file
                    if pd is not None:
                        df = pd.read_csv(file_path, sep='\t', nrows=5)  # Read first 5 rows for validation
                        validation_result['readable'] = True
                        validation_result['valid_format'] = True
                        validation_result['column_count'] = len(df.columns)
                        validation_result['columns'] = list(df.columns)
                        
                        # Get full row count
                        full_df = pd.read_csv(file_path, sep='\t')
                        validation_result['row_count'] = len(full_df)
                        
                        print(f"    ✓ Valid TSV: {validation_result['row_count']} rows, {validation_result['column_count']} columns")
                    else:
                        # Fallback validation without pandas
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                        except UnicodeDecodeError:
                            # Try with different encodings
                            try:
                                with open(file_path, 'r', encoding='latin-1') as f:
                                    lines = f.readlines()
                            except:
                                # If all else fails, just check file exists
                                validation_result['readable'] = True
                                validation_result['valid_format'] = True
                                print(f"    ✓ File exists (encoding issues, but readable)")
                                validation_results[file_key] = validation_result
                                continue
                        
                        validation_result['readable'] = True
                        validation_result['valid_format'] = len(lines) > 0
                        validation_result['row_count'] = len(lines) - 1  # minus header
                        if lines:
                            validation_result['columns'] = lines[0].strip().split('\t')
                            validation_result['column_count'] = len(validation_result['columns'])
                        print(f"    ✓ Valid TSV: {validation_result['row_count']} rows, {validation_result['column_count']} columns")
                    
                elif file_path.suffix == '.json':
                    # Read JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    validation_result['readable'] = True
                    validation_result['valid_format'] = True
                    print(f"    ✓ Valid JSON file")
                    
                else:
                    # Read as text file
                    with open(file_path, 'r') as f:
                        content = f.read(1000)  # Read first 1000 chars
                    validation_result['readable'] = True
                    validation_result['valid_format'] = len(content) > 0
                    print(f"    ✓ Readable text file")
                
            except Exception as e:
                validation_result['error'] = str(e)
                print(f"    ✗ Validation failed: {e}")
            
            validation_results[file_key] = validation_result
        
        return validation_results
    
    def fetch_all_files(self) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """Fetch all global files"""
        print("=== Fetching Global Files ===")
        
        if not self.check_datalad_available():
            raise RuntimeError("DataLad is not available. Please install datalad.")
        
        fetch_results = {}
        fetch_messages = {}
        
        for file_key, file_info in self.global_files.items():
            success, message = self.fetch_file(file_info, file_key)
            fetch_results[file_key] = success
            fetch_messages[file_key] = message
        
        return fetch_results, fetch_messages
    
    def create_summary_report(self, fetch_results: Dict[str, bool], 
                            fetch_messages: Dict[str, str],
                            validation_results: Dict[str, Dict]) -> Dict:
        """Create summary report of fetching process"""
        
        successful_fetches = sum(1 for success in fetch_results.values() if success)
        required_files = [k for k, v in self.global_files.items() if v['required']]
        required_successful = sum(1 for k in required_files if fetch_results.get(k, False))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'total_files': len(self.global_files),
            'successful_fetches': successful_fetches,
            'failed_fetches': len(self.global_files) - successful_fetches,
            'required_files_total': len(required_files),
            'required_files_successful': required_successful,
            'all_required_files_fetched': required_successful == len(required_files),
            'fetch_results': fetch_results,
            'fetch_messages': fetch_messages,
            'validation_results': validation_results,
            'files_info': self.global_files
        }
        
        return report
    
    def export_results(self, report: Dict) -> Path:
        """Export fetching results and summary"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting results to {self.output_dir}")
        
        # Export main report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f'global_fetch_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  ✓ Fetch report: {report_file}")
        
        # Create simple status file for pipeline
        status_file = self.output_dir / 'fetch_status.json'
        status = {
            'timestamp': report['timestamp'],
            'all_required_files_fetched': report['all_required_files_fetched'],
            'successful_files': [k for k, v in report['fetch_results'].items() if v],
            'failed_files': [k for k, v in report['fetch_results'].items() if not v]
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"  ✓ Status file: {status_file}")
        
        return self.output_dir
    
    def print_summary(self, report: Dict) -> None:
        """Print summary to console"""
        print(f"\n{'='*60}")
        print(f"GLOBAL DATA FETCH SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total files processed: {report['total_files']}")
        print(f"Successful fetches: {report['successful_fetches']}")
        print(f"Failed fetches: {report['failed_fetches']}")
        print(f"Required files fetched: {report['required_files_successful']}/{report['required_files_total']}")
        
        # Show file-by-file results
        print(f"\nFile-by-file results:")
        for file_key, file_info in self.global_files.items():
            success = report['fetch_results'].get(file_key, False)
            required = " (required)" if file_info['required'] else " (optional)"
            status = "✓" if success else "✗"
            print(f"  {status} {file_key}: {file_info['path']}{required}")
            
            # Show validation info if available
            if file_key in report['validation_results']:
                validation = report['validation_results'][file_key]
                if validation.get('readable', False):
                    if validation.get('row_count', 0) > 0:
                        print(f"      → {validation['row_count']} rows, {validation['column_count']} columns")
                    elif validation.get('valid_format', False):
                        print(f"      → Valid file format")
                elif validation.get('error'):
                    print(f"      → Error: {validation['error']}")
        
        if report['all_required_files_fetched']:
            print(f"\n✅ All required files successfully fetched!")
        else:
            print(f"\n❌ Some required files failed to fetch.")
            failed_required = [k for k in ['participants', 'phenotype_demos'] 
                             if not report['fetch_results'].get(k, False)]
            if failed_required:
                print(f"Failed required files: {failed_required}")

def main():
    """Main execution function"""
    print("TCP Global Data Fetcher")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = GlobalDataFetcher()
    
    # Validate dataset
    if not fetcher.validate_dataset():
        print("❌ Dataset validation failed. Please run initialize_dataset.py first.")
        return 1
    
    try:
        # Fetch all files
        fetch_results, fetch_messages = fetcher.fetch_all_files()
        
        # Validate fetched files
        validation_results = fetcher.validate_fetched_files()
        
        # Create summary report
        report = fetcher.create_summary_report(fetch_results, fetch_messages, validation_results)
        
        # Export results
        output_dir = fetcher.export_results(report)
        
        # Print summary
        fetcher.print_summary(report)
        
        print(f"\n{'='*60}")
        print(f"FETCH COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        
        if report['all_required_files_fetched']:
            print(f"\nNext steps:")
            print(f"  1. Optionally run filter_phenotype.py for diagnosis filtering")
            print(f"  2. Run filter_subjects.py for task data filtering")
            return 0
        else:
            print(f"\n⚠ Some required files could not be fetched.")
            print(f"You may need to check dataset connectivity or file availability.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during global data fetching: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())