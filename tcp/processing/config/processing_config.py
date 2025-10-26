#!/usr/bin/env python3
"""
Configuration management for TCP processing pipeline.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Add project root to path to import config  
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import paths after adding to path
try:
    from config.paths import (
        get_tcp_dataset_path, 
        get_tcp_processing_path,
        get_script_output_path,
        get_platform_info
    )
except ImportError:
    # Fallback for testing without full project setup
    def get_tcp_dataset_path():
        return Path("/tmp/test_dataset")
    def get_tcp_processing_path():
        return Path("/tmp/test_processing")
    def get_script_output_path(script_type, script_name=''):
        return Path("/tmp/test_output")
    def get_platform_info():
        return {'detected_platform': 'unknown', 'hostname': 'test'}


class ProcessingConfig:
    """Configuration manager for TCP processing pipeline"""
    
    def __init__(self, 
                 dataset_path: Optional[Path] = None,
                 processing_output_path: Optional[Path] = None,
                 manifest_path: Optional[Path] = None):
        """
        Initialize processing configuration.
        
        Args:
            dataset_path: Override for dataset path
            processing_output_path: Override for processing outputs
            manifest_path: Override for manifest file location
        """
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.processing_output_path = Path(processing_output_path) if processing_output_path else get_tcp_processing_path()
        
        # Default manifest location from preprocessing pipeline
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            preprocessing_integration_path = get_script_output_path('tcp_preprocessing', 'integrate_cross_analysis')
            self.manifest_path = preprocessing_integration_path / "processing_data_manifest.json"
        
        # Platform information
        self.platform_info = get_platform_info()
        
        # Processing pipeline defaults
        self.defaults = {
            'manifest_validation': True,
            'file_validation_sample_size': 50,
            'lazy_loading': True,
            'cache_enabled': True,
            'error_handling': 'strict',
            'cross_platform_paths': True
        }
    
    def get_dataset_path(self) -> Path:
        """Get path to TCP dataset"""
        return self.dataset_path
    
    def get_processing_output_path(self, subpath: str = '') -> Path:
        """Get path for processing outputs"""
        if subpath:
            return self.processing_output_path / subpath
        return self.processing_output_path
    
    def get_manifest_path(self) -> Path:
        """Get path to data manifest"""
        return self.manifest_path
    
    def ensure_output_directories(self) -> None:
        """Create output directories if they don't exist"""
        self.processing_output_path.mkdir(parents=True, exist_ok=True)
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate configuration paths and accessibility.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'dataset_path_exists': self.dataset_path.exists(),
            'dataset_path_readable': self.dataset_path.exists() and self.dataset_path.is_dir(),
            'manifest_path_exists': self.manifest_path.exists(),
            'manifest_path_readable': False,
            'processing_output_writable': False
        }
        
        # Check manifest readability
        if validation['manifest_path_exists']:
            try:
                with open(self.manifest_path, 'r') as f:
                    f.read(1)  # Try to read one character
                validation['manifest_path_readable'] = True
            except (OSError, PermissionError):
                validation['manifest_path_readable'] = False
        
        # Check processing output writability
        try:
            self.processing_output_path.mkdir(parents=True, exist_ok=True)
            test_file = self.processing_output_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            validation['processing_output_writable'] = True
        except (OSError, PermissionError):
            validation['processing_output_writable'] = False
        
        return validation
    
    def get_platform_specific_settings(self) -> Dict[str, Union[str, bool, int]]:
        """
        Get platform-specific configuration settings.
        
        Returns:
            Dictionary with platform-specific settings
        """
        platform = self.platform_info['detected_platform']
        
        settings = {
            'platform': platform,
            'use_memory_mapping': True,  # Generally safe across platforms
            'max_workers': 4,  # Conservative default
            'chunk_size': 1000,
            'temp_dir_cleanup': True
        }
        
        # Platform-specific optimizations
        if platform == 'windows':
            settings.update({
                'path_case_sensitive': False,
                'max_path_length': 260,
                'use_long_path_support': True
            })
        elif platform in ['macos', 'linux', 'idun']:
            settings.update({
                'path_case_sensitive': True,
                'max_path_length': 4096,
                'use_symbolic_links': True
            })
        
        return settings
    
    def get_configuration_summary(self) -> Dict[str, str]:
        """
        Get summary of current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'dataset_path': str(self.dataset_path),
            'processing_output_path': str(self.processing_output_path),
            'manifest_path': str(self.manifest_path),
            'platform': self.platform_info['detected_platform'],
            'hostname': self.platform_info['hostname'],
            'config_valid': all(self.validate_configuration().values())
        }