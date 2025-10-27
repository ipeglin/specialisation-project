#!/usr/bin/env python3
"""
Downloaded data utilities for TCP processing pipeline.

Provides functions to identify subjects with locally downloaded data
vs. subjects that exist in manifest but haven't been downloaded.

Author: Ian Philip Eglin
Date: 2025-10-27
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path


def get_downloaded_subjects() -> List[str]:
    """
    Get list of subjects that have been downloaded locally.
    
    Returns:
        List of subject IDs that have been sampled and downloaded
        
    Raises:
        FileNotFoundError: If sampled subjects file doesn't exist
    """
    sampling_dir = get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')
    sampled_ids_file = sampling_dir / "sampled_subject_ids.txt"
    
    if not sampled_ids_file.exists():
        raise FileNotFoundError(
            f"Downloaded subjects file not found: {sampled_ids_file}\n"
            f"Run sample_subjects_for_download.py first to generate this file."
        )
    
    downloaded_subjects = []
    with open(sampled_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                downloaded_subjects.append(line)
    
    return downloaded_subjects


def get_downloaded_subjects_details() -> pd.DataFrame:
    """
    Get detailed information about downloaded subjects.
    
    Returns:
        DataFrame with detailed information about each downloaded subject
        
    Raises:
        FileNotFoundError: If sampled subjects CSV doesn't exist
    """
    sampling_dir = get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')
    sampled_csv_file = sampling_dir / "sampled_subjects_for_download.csv"
    
    if not sampled_csv_file.exists():
        raise FileNotFoundError(
            f"Downloaded subjects details not found: {sampled_csv_file}\n"
            f"Run sample_subjects_for_download.py first to generate this file."
        )
    
    return pd.read_csv(sampled_csv_file)


def get_sampling_strategy() -> Dict[str, Any]:
    """
    Get information about the sampling strategy used.
    
    Returns:
        Dictionary with sampling strategy details
        
    Raises:
        FileNotFoundError: If sampling strategy file doesn't exist
    """
    sampling_dir = get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')
    strategy_file = sampling_dir / "sampling_strategy.json"
    
    if not strategy_file.exists():
        raise FileNotFoundError(
            f"Sampling strategy file not found: {strategy_file}\n"
            f"Run sample_subjects_for_download.py first to generate this file."
        )
    
    with open(strategy_file, 'r') as f:
        return json.load(f)


def get_fetch_report() -> Dict[str, Any]:
    """
    Get the latest data fetch report.
    
    Returns:
        Dictionary with fetch results and success rates
        
    Raises:
        FileNotFoundError: If fetch report doesn't exist
    """
    fetch_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
    
    # Find the most recent fetch report
    fetch_reports = list(fetch_dir.glob("fetch_report_*.json"))
    
    if not fetch_reports:
        raise FileNotFoundError(
            f"No fetch reports found in: {fetch_dir}\n"
            f"Run fetch_filtered_data.py first to generate fetch reports."
        )
    
    # Get the most recent report
    latest_report = max(fetch_reports, key=lambda p: p.stat().st_mtime)
    
    with open(latest_report, 'r') as f:
        return json.load(f)


def validate_downloaded_data_availability(subject_ids: List[str], 
                                        data_type: str = 'timeseries',
                                        dataset_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate that downloaded subjects actually have data files available.
    
    Args:
        subject_ids: List of subject IDs to validate
        data_type: Type of data to check ('timeseries', 'raw_nifti', etc.)
        dataset_path: Path to dataset (uses default if None)
        
    Returns:
        Dictionary mapping subject_id to validation results
    """
    if dataset_path is None:
        from config.paths import get_tcp_dataset_path
        dataset_path = get_tcp_dataset_path()
    
    validation_results = {}
    
    for subject_id in subject_ids:
        result = {
            'subject_id': subject_id,
            'has_data': False,
            'data_files': [],
            'errors': []
        }
        
        try:
            if data_type == 'timeseries':
                # Check for processed timeseries data
                from tcp.preprocessing.utils.subject_id_transform import manifest_to_directory_id
                subject_dir_id = manifest_to_directory_id(subject_id)
                timeseries_dir = dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated" / subject_dir_id
                
                if timeseries_dir.exists():
                    data_files = list(timeseries_dir.glob("*_parcellated.h5"))
                    result['has_data'] = len(data_files) > 0
                    result['data_files'] = [str(f) for f in data_files]
                else:
                    result['errors'].append(f"Timeseries directory not found: {timeseries_dir}")
                    
            elif data_type == 'raw_nifti':
                # Check for raw BIDS data
                subject_dir = dataset_path / subject_id
                if subject_dir.exists():
                    func_dir = subject_dir / "func"
                    if func_dir.exists():
                        nifti_files = list(func_dir.glob("*.nii.gz"))
                        result['has_data'] = len(nifti_files) > 0
                        result['data_files'] = [str(f) for f in nifti_files]
                    else:
                        result['errors'].append(f"Func directory not found: {func_dir}")
                else:
                    result['errors'].append(f"Subject directory not found: {subject_dir}")
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        validation_results[subject_id] = result
    
    return validation_results


def get_download_status_report() -> Dict[str, Any]:
    """
    Generate comprehensive report on downloaded data status.
    
    Returns:
        Dictionary with download status, sampling info, and data availability
    """
    try:
        downloaded_subjects = get_downloaded_subjects()
        sampling_strategy = get_sampling_strategy()
        fetch_report = get_fetch_report()
        
        # Validate actual data availability
        timeseries_validation = validate_downloaded_data_availability(
            downloaded_subjects, 'timeseries'
        )
        raw_validation = validate_downloaded_data_availability(
            downloaded_subjects, 'raw_nifti'
        )
        
        # Count successful validations
        timeseries_available = sum(1 for v in timeseries_validation.values() if v['has_data'])
        raw_available = sum(1 for v in raw_validation.values() if v['has_data'])
        
        report = {
            'timestamp': fetch_report.get('timestamp', 'unknown'),
            'download_summary': {
                'total_downloaded': len(downloaded_subjects),
                'fetch_success_rate': fetch_report.get('success_rate', 0),
                'successful_fetches': fetch_report.get('successful_fetches', 0),
                'failed_fetches': fetch_report.get('failed_fetches', 0)
            },
            'sampling_info': {
                'mode': sampling_strategy.get('sampling_details', {}).get('mode', 'unknown'),
                'categories_sampled': sampling_strategy.get('sampling_details', {}).get('categories_sampled', {}),
                'storage_estimate_gb': sampling_strategy.get('storage_estimate', {}).get('total_estimated_gb', 0)
            },
            'data_availability': {
                'timeseries_available': timeseries_available,
                'raw_nifti_available': raw_available,
                'timeseries_success_rate': (timeseries_available / len(downloaded_subjects)) * 100 if downloaded_subjects else 0,
                'raw_nifti_success_rate': (raw_available / len(downloaded_subjects)) * 100 if downloaded_subjects else 0
            },
            'downloaded_subjects': downloaded_subjects,
            'validation_details': {
                'timeseries': timeseries_validation,
                'raw_nifti': raw_validation
            }
        }
        
        return report
        
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': None,
            'download_summary': {'total_downloaded': 0},
            'data_availability': {'timeseries_available': 0, 'raw_nifti_available': 0}
        }


if __name__ == "__main__":
    """Test the downloaded data utilities"""
    print("TCP Downloaded Data Utilities Test")
    print("=" * 40)
    
    try:
        # Test basic functions
        downloaded = get_downloaded_subjects()
        print(f"Downloaded subjects: {len(downloaded)}")
        for subject in downloaded:
            print(f"  - {subject}")
        
        # Test comprehensive report
        print(f"\n" + "=" * 40)
        report = get_download_status_report()
        
        if 'error' in report:
            print(f"Error generating report: {report['error']}")
        else:
            print(f"Download Status Report")
            print(f"  Total downloaded: {report['download_summary']['total_downloaded']}")
            print(f"  Fetch success rate: {report['download_summary']['fetch_success_rate']:.1f}%")
            print(f"  Timeseries available: {report['data_availability']['timeseries_available']}")
            print(f"  Raw NIfTI available: {report['data_availability']['raw_nifti_available']}")
            print(f"  Sampling mode: {report['sampling_info']['mode']}")
        
    except Exception as e:
        print(f"Test failed: {e}")