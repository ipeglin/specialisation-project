#!/usr/bin/env python3
"""
Subject ID transformation utilities for TCP processing.

Handles bidirectional conversion between manifest format and directory format.

Author: Ian Philip Eglin
Date: 2025-10-27
"""

import re
from typing import Optional


def manifest_to_directory_id(subject_id: str) -> str:
    """
    Convert subject ID from manifest format to directory format.
    
    Args:
        subject_id: Subject ID in manifest format (e.g., 'sub-NDARINVBM990HJT')
        
    Returns:
        Subject ID in directory format (e.g., 'NDAR_INVBM990HJT')
        
    Examples:
        >>> manifest_to_directory_id('sub-NDARINVBM990HJT')
        'NDAR_INVBM990HJT'
        >>> manifest_to_directory_id('sub-OTHER_FORMAT123')
        'OTHER_FORMAT123'
    """
    # Remove 'sub-' prefix
    dir_id = subject_id.replace('sub-', '')
    
    # Handle NDAR format: insert underscore after NDAR prefix
    # Transform NDARINVBM990HJT -> NDAR_INVBM990HJT
    if dir_id.startswith('NDAR') and len(dir_id) > 4 and not dir_id.startswith('NDAR_'):
        # Insert underscore after 'NDAR'
        dir_id = 'NDAR_' + dir_id[4:]
    
    return dir_id


def directory_to_manifest_id(directory_id: str) -> str:
    """
    Convert subject ID from directory format to manifest format.
    
    Args:
        directory_id: Subject ID in directory format (e.g., 'NDAR_INVBM990HJT')
        
    Returns:
        Subject ID in manifest format (e.g., 'sub-NDARINVBM990HJT')
        
    Examples:
        >>> directory_to_manifest_id('NDAR_INVBM990HJT')
        'sub-NDARINVBM990HJT'
        >>> directory_to_manifest_id('OTHER_FORMAT123')
        'sub-OTHER_FORMAT123'
    """
    # Handle NDAR format: remove underscore after NDAR prefix
    # Transform NDAR_INVBM990HJT -> NDARINVBM990HJT
    if directory_id.startswith('NDAR_') and len(directory_id) > 5:
        # Remove underscore after 'NDAR'
        manifest_id = 'NDAR' + directory_id[5:]
    else:
        manifest_id = directory_id
    
    # Add 'sub-' prefix
    return f'sub-{manifest_id}'


def validate_subject_id_format(subject_id: str, format_type: str = 'auto') -> bool:
    """
    Validate subject ID format.
    
    Args:
        subject_id: Subject ID to validate
        format_type: 'manifest', 'directory', or 'auto' to detect
        
    Returns:
        True if format is valid
        
    Examples:
        >>> validate_subject_id_format('sub-NDARINVBM990HJT', 'manifest')
        True
        >>> validate_subject_id_format('NDAR_INVBM990HJT', 'directory')
        True
    """
    if format_type == 'auto':
        # Auto-detect format
        if subject_id.startswith('sub-'):
            format_type = 'manifest'
        else:
            format_type = 'directory'
    
    if format_type == 'manifest':
        # Should start with 'sub-' and have at least one character after
        return bool(re.match(r'^sub-.+', subject_id))
    elif format_type == 'directory':
        # Should not start with 'sub-' and have at least one character
        return not subject_id.startswith('sub-') and len(subject_id) > 0
    
    return False


def detect_subject_id_format(subject_id: str) -> str:
    """
    Detect the format of a subject ID.
    
    Args:
        subject_id: Subject ID to analyze
        
    Returns:
        'manifest' if starts with 'sub-', 'directory' otherwise
        
    Examples:
        >>> detect_subject_id_format('sub-NDARINVBM990HJT')
        'manifest'
        >>> detect_subject_id_format('NDAR_INVBM990HJT')
        'directory'
    """
    return 'manifest' if subject_id.startswith('sub-') else 'directory'


def get_conversion_report(subject_ids: list) -> dict:
    """
    Generate a conversion report for a list of subject IDs.
    
    Args:
        subject_ids: List of subject IDs to analyze
        
    Returns:
        Dictionary with conversion statistics and examples
    """
    report = {
        'total_subjects': len(subject_ids),
        'manifest_format': [],
        'directory_format': [],
        'ndar_subjects': [],
        'other_subjects': [],
        'conversion_examples': []
    }
    
    for subject_id in subject_ids:
        format_type = detect_subject_id_format(subject_id)
        
        if format_type == 'manifest':
            report['manifest_format'].append(subject_id)
            dir_id = manifest_to_directory_id(subject_id)
            
            if 'NDAR' in subject_id:
                report['ndar_subjects'].append(subject_id)
            else:
                report['other_subjects'].append(subject_id)
                
            # Add conversion example
            if len(report['conversion_examples']) < 5:
                report['conversion_examples'].append({
                    'manifest': subject_id,
                    'directory': dir_id,
                    'type': 'NDAR' if 'NDAR' in subject_id else 'other'
                })
        else:
            report['directory_format'].append(subject_id)
    
    return report


if __name__ == "__main__":
    # Test the transformation functions
    print("Subject ID Transformation Utility")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        'sub-NDARINVBM990HJT',
        'sub-NDAR1234567890', 
        'sub-NDARABCDEFGHIJ',
        'sub-OTHER_FORMAT123',
        'sub-12345',
    ]
    
    print("\nManifest → Directory conversion:")
    print("-" * 40)
    for manifest_id in test_cases:
        directory_id = manifest_to_directory_id(manifest_id)
        print(f"{manifest_id:25} → {directory_id}")
    
    print("\nDirectory → Manifest conversion:")
    print("-" * 40)
    for manifest_id in test_cases:
        directory_id = manifest_to_directory_id(manifest_id)
        back_to_manifest = directory_to_manifest_id(directory_id)
        print(f"{directory_id:25} → {back_to_manifest}")
    
    print("\nRound-trip test:")
    print("-" * 40)
    for original in test_cases:
        directory = manifest_to_directory_id(original)
        back_to_manifest = directory_to_manifest_id(directory)
        success = original == back_to_manifest
        print(f"{original} → {directory} → {back_to_manifest} [{'✓' if success else '❌'}]")
    
    print(f"\nExpected path format:")
    print(f"  fMRI_timeseries_clean_denoised_GSR_parcellated/{manifest_to_directory_id('sub-NDARINVBM990HJT')}/")
    print(f"  Should match: NDAR_INVBM990HJT")