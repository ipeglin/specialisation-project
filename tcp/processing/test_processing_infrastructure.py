#!/usr/bin/env python3
"""
Test script for TCP processing infrastructure.

Tests cross-platform compatibility, path resolution, and data loading
without running analysis algorithms.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tcp.processing import DataLoader, SubjectManager, ProcessingConfig
    from tcp.processing.utils.validation import validate_manifest
    from config.paths import get_platform_info
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    print("Current working directory:", Path.cwd())
    sys.exit(1)


def test_platform_detection() -> Dict[str, Any]:
    """Test platform detection and path configuration."""
    print("=== Testing Platform Detection ===")
    
    platform_info = get_platform_info()
    
    print(f"Detected platform: {platform_info['detected_platform']}")
    print(f"Hostname: {platform_info['hostname']}")
    print(f"System: {platform_info['system']}")
    
    # Test processing configuration
    config = ProcessingConfig()
    
    print(f"Dataset path: {config.get_dataset_path()}")
    print(f"Processing output path: {config.get_processing_output_path()}")
    print(f"Manifest path: {config.get_manifest_path()}")
    
    # Validate configuration
    validation = config.validate_configuration()
    print(f"Configuration validation: {validation}")
    
    return {
        'platform_info': platform_info,
        'config_validation': validation,
        'paths': {
            'dataset': str(config.get_dataset_path()),
            'processing_output': str(config.get_processing_output_path()),
            'manifest': str(config.get_manifest_path())
        }
    }


def test_manifest_loading() -> Dict[str, Any]:
    """Test manifest loading and validation."""
    print("\n=== Testing Manifest Loading ===")
    
    config = ProcessingConfig()
    manifest_path = config.get_manifest_path()
    
    print(f"Manifest path: {manifest_path}")
    print(f"Manifest exists: {manifest_path.exists()}")
    
    if not manifest_path.exists():
        print("⚠️  Manifest not found - run preprocessing pipeline first")
        return {
            'manifest_found': False,
            'error': 'Manifest file not found'
        }
    
    try:
        # Test manifest validation
        manifest = validate_manifest(manifest_path)
        
        metadata = manifest.get('manifest_metadata', {})
        print(f"✓ Manifest loaded successfully")
        print(f"  Created: {metadata.get('created_timestamp')}")
        print(f"  Total subjects: {metadata.get('total_subjects')}")
        print(f"  Analysis groups: {metadata.get('analysis_groups')}")
        
        return {
            'manifest_found': True,
            'manifest_valid': True,
            'metadata': metadata,
            'subject_count': len(manifest.get('subjects', {})),
            'group_count': len(manifest.get('analysis_groups', {}))
        }
        
    except Exception as e:
        print(f"❌ Manifest validation failed: {e}")
        return {
            'manifest_found': True,
            'manifest_valid': False,
            'error': str(e)
        }


def test_data_loader() -> Dict[str, Any]:
    """Test DataLoader functionality."""
    print("\n=== Testing DataLoader ===")
    
    try:
        # Initialize DataLoader
        config = ProcessingConfig()
        loader = DataLoader(config=config, validate_on_load=True)
        
        # Test basic functionality
        manifest_info = loader.get_manifest_info()
        print(f"✓ DataLoader initialized")
        print(f"  Total subjects: {manifest_info['total_subjects']}")
        print(f"  Data types: {manifest_info['data_types_available']}")
        print(f"  Base path: {manifest_info['base_path']}")
        
        # Test subject access
        all_subjects = loader.get_all_subject_ids()
        print(f"  Subject IDs loaded: {len(all_subjects)}")
        
        # Test group access
        groups = loader.get_analysis_groups()
        print(f"  Analysis groups: {list(groups.keys())}")
        
        # Test subject filtering
        if all_subjects:
            # Test basic filtering
            timeseries_subjects = loader.filter_subjects(
                data_requirements=['timeseries']
            )
            print(f"  Subjects with timeseries: {len(timeseries_subjects)}")
            
            # Test classification filtering
            if groups:
                first_group = list(groups.keys())[0]
                group_subjects = loader.get_subjects_by_group(first_group)
                print(f"  Subjects in '{first_group}': {len(group_subjects)}")
        
        return {
            'loader_initialized': True,
            'manifest_info': manifest_info,
            'total_subjects': len(all_subjects),
            'analysis_groups': list(groups.keys()),
            'test_filtering': len(timeseries_subjects) if all_subjects else 0
        }
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return {
            'loader_initialized': False,
            'error': str(e)
        }


def test_subject_manager() -> Dict[str, Any]:
    """Test SubjectManager functionality."""
    print("\n=== Testing SubjectManager ===")
    
    try:
        # Initialize SubjectManager
        config = ProcessingConfig()
        manager = SubjectManager(config=config)
        
        # Test basic functionality
        all_subjects = manager.get_all_subjects()
        groups = manager.get_analysis_groups()
        
        print(f"✓ SubjectManager initialized")
        print(f"  Total subjects: {len(all_subjects)}")
        print(f"  Analysis groups: {list(groups.keys())}")
        
        test_results = {
            'manager_initialized': True,
            'total_subjects': len(all_subjects),
            'analysis_groups': list(groups.keys()),
            'group_statistics': {},
            'filtering_tests': {}
        }
        
        # Test group statistics
        if groups:
            first_group = list(groups.keys())[0]
            group_stats = manager.get_group_statistics(first_group)
            print(f"  '{first_group}' statistics computed")
            test_results['group_statistics'][first_group] = group_stats['statistics']
        
        # Test advanced filtering
        if all_subjects:
            # Test demographic filtering
            with_timeseries = manager.get_subjects_with_data('timeseries')
            print(f"  Subjects with timeseries data: {len(with_timeseries)}")
            test_results['filtering_tests']['timeseries_subjects'] = len(with_timeseries)
            
            # Test classification filtering
            anhedonic_subjects = manager.get_subjects_by_classification(
                'anhedonic_status', 'anhedonic'
            )
            print(f"  Anhedonic subjects: {len(anhedonic_subjects)}")
            test_results['filtering_tests']['anhedonic_subjects'] = len(anhedonic_subjects)
        
        # Test summary DataFrame
        summary_df = manager.get_summary_dataframe()
        print(f"  Summary DataFrame: {summary_df.shape}")
        test_results['summary_shape'] = list(summary_df.shape)
        
        return test_results
        
    except Exception as e:
        print(f"❌ SubjectManager test failed: {e}")
        return {
            'manager_initialized': False,
            'error': str(e)
        }


def test_file_validation() -> Dict[str, Any]:
    """Test file path validation and accessibility."""
    print("\n=== Testing File Validation ===")
    
    try:
        config = ProcessingConfig()
        loader = DataLoader(config=config)
        
        # Get a sample of subjects for validation
        all_subjects = loader.get_all_subject_ids()
        
        if not all_subjects:
            return {
                'validation_completed': False,
                'error': 'No subjects available for validation'
            }
        
        # Test file validation for a few subjects
        sample_size = min(5, len(all_subjects))
        sample_subjects = all_subjects[:sample_size]
        
        validation_results = {
            'validation_completed': True,
            'sample_size': sample_size,
            'subject_validations': {},
            'summary': {
                'subjects_with_timeseries': 0,
                'subjects_with_motion': 0,
                'total_accessible_files': 0,
                'total_missing_files': 0
            }
        }
        
        for subject_id in sample_subjects:
            # Validate timeseries files
            timeseries_validation = loader.validate_subject_files(subject_id, 'timeseries')
            
            validation_results['subject_validations'][subject_id] = {
                'timeseries': {
                    'total_files': timeseries_validation['total_files'],
                    'accessible': len(timeseries_validation['accessible_files']),
                    'missing': len(timeseries_validation['missing_files'])
                }
            }
            
            # Update summary
            if timeseries_validation['accessible_files']:
                validation_results['summary']['subjects_with_timeseries'] += 1
            
            validation_results['summary']['total_accessible_files'] += len(timeseries_validation['accessible_files'])
            validation_results['summary']['total_missing_files'] += len(timeseries_validation['missing_files'])
        
        print(f"✓ File validation completed for {sample_size} subjects")
        print(f"  Subjects with accessible timeseries: {validation_results['summary']['subjects_with_timeseries']}")
        print(f"  Total accessible files: {validation_results['summary']['total_accessible_files']}")
        print(f"  Total missing files: {validation_results['summary']['total_missing_files']}")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ File validation test failed: {e}")
        return {
            'validation_completed': False,
            'error': str(e)
        }


def main():
    """Run comprehensive infrastructure tests."""
    print("TCP Processing Infrastructure Test Suite")
    print("=" * 50)
    
    test_results = {
        'timestamp': None,
        'platform_test': None,
        'manifest_test': None,
        'data_loader_test': None,
        'subject_manager_test': None,
        'file_validation_test': None,
        'overall_success': False
    }
    
    from datetime import datetime
    test_results['timestamp'] = datetime.now().isoformat()
    
    try:
        # Run all tests
        test_results['platform_test'] = test_platform_detection()
        test_results['manifest_test'] = test_manifest_loading()
        test_results['data_loader_test'] = test_data_loader()
        test_results['subject_manager_test'] = test_subject_manager()
        test_results['file_validation_test'] = test_file_validation()
        
        # Determine overall success
        success_flags = [
            test_results['platform_test'].get('config_validation', {}).get('dataset_path_exists', False),
            test_results['manifest_test'].get('manifest_valid', False),
            test_results['data_loader_test'].get('loader_initialized', False),
            test_results['subject_manager_test'].get('manager_initialized', False),
            test_results['file_validation_test'].get('validation_completed', False)
        ]
        
        test_results['overall_success'] = all(success_flags)
        
        print(f"\n{'='*50}")
        print(f"TEST SUITE SUMMARY")
        print(f"{'='*50}")
        print(f"Platform detection: {'✓' if success_flags[0] else '❌'}")
        print(f"Manifest loading: {'✓' if success_flags[1] else '❌'}")
        print(f"DataLoader: {'✓' if success_flags[2] else '❌'}")
        print(f"SubjectManager: {'✓' if success_flags[3] else '❌'}")
        print(f"File validation: {'✓' if success_flags[4] else '❌'}")
        print(f"\nOverall success: {'✓' if test_results['overall_success'] else '❌'}")
        
        if test_results['overall_success']:
            print(f"\n🎉 TCP processing infrastructure ready for use!")
            
            # Show usage example
            print(f"\nExample usage:")
            print(f"  from tcp.processing import DataLoader, SubjectManager")
            print(f"  ")
            print(f"  # Load data")
            print(f"  loader = DataLoader()")
            print(f"  manager = SubjectManager(loader)")
            print(f"  ")
            print(f"  # Get subjects for analysis")
            print(f"  anhedonic_subjects = manager.filter_subjects(")
            print(f"      groups=['primary_analysis'],")
            print(f"      classifications={{'anhedonic_status': 'anhedonic'}},")
            print(f"      data_requirements=['timeseries']")
            print(f"  )")
        else:
            print(f"\n⚠️  Some components failed - check errors above")
        
        return 0 if test_results['overall_success'] else 1
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())