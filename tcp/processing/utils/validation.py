#!/usr/bin/env python3
"""
Validation utilities for data manifest and file integrity.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .error_handling import ValidationError, ManifestError
from .file_utils import check_file_exists, check_file_integrity


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Validate preprocessing data manifest structure and content.
    
    Args:
        manifest_path: Path to manifest JSON file
        
    Returns:
        Loaded and validated manifest data
        
    Raises:
        ManifestError: If manifest is invalid or corrupt
    """
    if not manifest_path.exists():
        raise ManifestError(f"Manifest file not found: {manifest_path}")
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ManifestError(f"Invalid JSON in manifest: {e}")
    except Exception as e:
        raise ManifestError(f"Cannot read manifest: {e}")
    
    # Validate required top-level keys
    required_keys = ['manifest_metadata', 'subjects', 'analysis_groups', 'path_configuration']
    missing_keys = [key for key in required_keys if key not in manifest]
    if missing_keys:
        raise ManifestError(f"Missing required manifest keys: {missing_keys}")
    
    # Validate metadata structure
    metadata = manifest['manifest_metadata']
    required_metadata = ['created_timestamp', 'total_subjects', 'analysis_groups']
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise ManifestError(f"Missing required metadata: {missing_metadata}")
    
    # Validate subject count consistency
    declared_count = metadata['total_subjects']
    actual_count = len(manifest['subjects'])
    if declared_count != actual_count:
        raise ManifestError(
            f"Subject count mismatch: declared {declared_count}, found {actual_count}"
        )
    
    # Validate analysis groups structure
    analysis_groups = manifest['analysis_groups']
    declared_groups = metadata['analysis_groups']
    
    for group_name in declared_groups:
        if group_name not in analysis_groups:
            raise ManifestError(f"Analysis group '{group_name}' declared but not found")
    
    # Validate subject structure (sample a few subjects)
    subjects = manifest['subjects']
    if subjects:
        # Check first subject for required structure
        first_subject_id = next(iter(subjects))
        subject_data = subjects[first_subject_id]
        
        required_subject_keys = ['demographics', 'classifications', 'files', 'data_availability']
        missing_subject_keys = [key for key in required_subject_keys if key not in subject_data]
        if missing_subject_keys:
            raise ManifestError(
                f"Subject {first_subject_id} missing required keys: {missing_subject_keys}"
            )
    
    return manifest


def validate_file_paths(manifest: Dict[str, Any], base_path: Path, 
                       sample_size: int = 10) -> Dict[str, Any]:
    """
    Validate file paths in manifest against actual filesystem.
    
    Args:
        manifest: Validated manifest data
        base_path: Base path for resolving relative paths
        sample_size: Number of subjects to validate (for performance)
        
    Returns:
        Validation report with statistics
    """
    validation_report = {
        'total_subjects_checked': 0,
        'subjects_with_timeseries': 0,
        'subjects_with_motion': 0,
        'file_validation_errors': [],
        'missing_files': [],
        'accessible_files': [],
        'validation_timestamp': None
    }
    
    from datetime import datetime
    validation_report['validation_timestamp'] = datetime.now().isoformat()
    
    subjects = manifest['subjects']
    
    # Sample subjects to validate (for performance on large datasets)
    subject_ids = list(subjects.keys())
    if len(subject_ids) > sample_size:
        import random
        subject_ids = random.sample(subject_ids, sample_size)
    
    for subject_id in subject_ids:
        validation_report['total_subjects_checked'] += 1
        subject_data = subjects[subject_id]
        
        # Validate timeseries files
        if subject_data['data_availability']['has_timeseries']:
            validation_report['subjects_with_timeseries'] += 1
            timeseries_files = subject_data['files']['timeseries']['available']
            
            for file_path in timeseries_files:
                full_path = base_path / file_path
                if check_file_exists(full_path):
                    validation_report['accessible_files'].append(str(full_path))
                else:
                    validation_report['missing_files'].append(str(full_path))
        
        # Validate motion files
        if subject_data['data_availability']['has_motion']:
            validation_report['subjects_with_motion'] += 1
            motion_files = subject_data['files']['motion']['available']
            
            for file_path in motion_files:
                full_path = base_path / file_path
                if check_file_exists(full_path):
                    validation_report['accessible_files'].append(str(full_path))
                else:
                    validation_report['missing_files'].append(str(full_path))
    
    return validation_report


def validate_subject_data(subject_id: str, subject_data: Dict[str, Any]) -> List[str]:
    """
    Validate individual subject data structure.
    
    Args:
        subject_id: Subject identifier
        subject_data: Subject data from manifest
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['demographics', 'classifications', 'files', 'data_availability']
    for key in required_keys:
        if key not in subject_data:
            errors.append(f"Missing required key: {key}")
    
    # Validate classifications
    if 'classifications' in subject_data:
        classifications = subject_data['classifications']
        required_classifications = ['anhedonia_class', 'anhedonic_status', 'mdd_status']
        
        for classification in required_classifications:
            if classification not in classifications:
                errors.append(f"Missing classification: {classification}")
    
    # Validate files structure
    if 'files' in subject_data:
        files = subject_data['files']
        required_file_types = ['timeseries', 'motion']
        
        for file_type in required_file_types:
            if file_type in files:
                file_data = files[file_type]
                if 'available' not in file_data:
                    errors.append(f"Missing 'available' files list for {file_type}")
                if 'base_path' not in file_data:
                    errors.append(f"Missing 'base_path' for {file_type}")
    
    # Validate data availability flags
    if 'data_availability' in subject_data:
        availability = subject_data['data_availability']
        required_flags = ['has_timeseries', 'has_motion', 'has_phenotype']
        
        for flag in required_flags:
            if flag not in availability:
                errors.append(f"Missing availability flag: {flag}")
            elif not isinstance(availability[flag], bool):
                errors.append(f"Availability flag {flag} should be boolean")
    
    return errors


def validate_analysis_group_consistency(manifest: Dict[str, Any]) -> List[str]:
    """
    Validate consistency between analysis groups and subject memberships.
    
    Args:
        manifest: Validated manifest data
        
    Returns:
        List of consistency errors
    """
    errors = []
    
    analysis_groups = manifest['analysis_groups']
    subjects = manifest['subjects']
    
    # Check that all subjects in analysis groups exist in subjects
    for group_name, subject_list in analysis_groups.items():
        for subject_id in subject_list:
            if subject_id not in subjects:
                errors.append(
                    f"Subject {subject_id} in group '{group_name}' not found in subjects"
                )
    
    # Check that subject group memberships are consistent
    for subject_id, subject_data in subjects.items():
        if 'analysis_group_memberships' in subject_data:
            memberships = subject_data['analysis_group_memberships']
            
            for group_name in memberships:
                if group_name not in analysis_groups:
                    errors.append(
                        f"Subject {subject_id} claims membership in unknown group '{group_name}'"
                    )
                elif subject_id not in analysis_groups[group_name]:
                    errors.append(
                        f"Subject {subject_id} claims membership in '{group_name}' but not listed in group"
                    )
    
    return errors