#!/usr/bin/env python3
"""
Cross-platform path configuration system for specialisation project.

This module automatically detects the current platform and provides appropriate
paths for code, data, output, and temporary files across different environments:
- macOS (M1 MacBook)
- Windows 11 (desktop)
- IDUN cluster (CentOS)

Usage:
    from config.paths import get_data_path, get_output_path, get_code_path

    dataset_path = get_data_path('tcp_dataset')
    output_dir = get_output_path('analysis_results')

Author: Ian Philip Eglin
Date: 2025-09-11
"""

import os
import platform
import socket
import json
from pathlib import Path
from typing import Dict, Optional, Union

class PathConfig:
    """Cross-platform path configuration manager"""

    def __init__(self):
        self.platform = self._detect_platform()
        self.config = self._load_config()
        self.env_overrides = self._load_env_overrides()

    def _detect_platform(self) -> str:
        """Detect the current platform/environment"""
        hostname = socket.gethostname().lower()
        system = platform.system().lower()

        # Check for IDUN cluster
        if 'idun' in hostname or 'login' in hostname or hostname.startswith('idun'):
            return 'idun'

        # Check for specific systems
        if system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        elif system == 'linux':
            # Could be personal Linux or other HPC
            return 'linux'

        # Default fallback
        return 'unknown'

    def _load_config(self) -> Dict:
        """Load default configuration from JSON file"""
        config_file = Path(__file__).parent / 'default_config.json'
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return minimal default if config file doesn't exist yet
            return self._get_minimal_default_config()

    def _get_minimal_default_config(self) -> Dict:
        """Provide minimal default configuration as fallback"""
        return {
            'macos': {
                'code_base': '/Users/ipeglin/Git',
                'data_base': '/Users/ipeglin/Documents/Data',
                'output_base': '/Users/ipeglin/Documents/Analysis',
                'preprocessing_base': '/Users/ipeglin/Documents/Preprocessing',
                'analysis_base': '/Users/ipeglin/Documents/Analysis',
                'figures_base': '/Users/ipeglin/Documents/Figures',
                'models_base': '/Users/ipeglin/Documents/Models',
                'reports_base': '/Users/ipeglin/Documents/Reports',
                'temp_base': '/tmp'
            },
            'windows': {
                'code_base': 'C:/Users/ipeglin/Git',
                'data_base': 'C:/Users/ipeglin/Documents/Data',
                'output_base': 'C:/Users/ipeglin/Documents/Analysis',
                'preprocessing_base': 'C:/Users/ipeglin/Documents/Preprocessing',
                'analysis_base': 'C:/Users/ipeglin/Documents/Analysis',
                'figures_base': 'C:/Users/ipeglin/Documents/Figures',
                'models_base': 'C:/Users/ipeglin/Documents/Models',
                'reports_base': 'C:/Users/ipeglin/Documents/Reports',
                'temp_base': 'C:/temp'
            },
            'idun': {
                'code_base': '/cluster/home/ianpe',
                'data_base': '/cluster/work/ianpe',
                'output_base': '/cluster/work/ianpe/analysis',
                'preprocessing_base': '/cluster/work/ianpe/preprocessing',
                'analysis_base': '/cluster/work/ianpe/analysis',
                'figures_base': '/cluster/work/ianpe/figures',
                'models_base': '/cluster/work/ianpe/models',
                'reports_base': '/cluster/work/ianpe/reports',
                'temp_base': '/cluster/work/ianpe/tmp'
            },
            'linux': {
                'code_base': '/home/ipeglin/git',
                'data_base': '/home/ipeglin/data',
                'output_base': '/home/ipeglin/analysis',
                'preprocessing_base': '/home/ipeglin/preprocessing',
                'analysis_base': '/home/ipeglin/analysis',
                'figures_base': '/home/ipeglin/figures',
                'models_base': '/home/ipeglin/models',
                'reports_base': '/home/ipeglin/reports',
                'temp_base': '/tmp'
            }
        }

    def _load_env_overrides(self) -> Dict:
        """Load environment variable overrides from .env file and system environment"""
        overrides = {}

        # First, try to load from .env file in project root
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                key, value = key.strip(), value.strip()
                                # Set as environment variable so os.getenv can find it
                                os.environ[key] = value
            except Exception as e:
                # Silent fail - if .env can't be read, continue with system env vars only
                pass

        # Check for all path type environment variables (from .env file or system)
        env_vars = [
            ('code_base', 'PROJECT_CODE_BASE'),
            ('data_base', 'PROJECT_DATA_BASE'),
            ('output_base', 'PROJECT_OUTPUT_BASE'),
            ('preprocessing_base', 'PROJECT_PREPROCESSING_BASE'),
            ('analysis_base', 'PROJECT_ANALYSIS_BASE'),
            ('figures_base', 'PROJECT_FIGURES_BASE'),
            ('models_base', 'PROJECT_MODELS_BASE'),
            ('reports_base', 'PROJECT_REPORTS_BASE'),
            ('temp_base', 'PROJECT_TEMP_BASE')
        ]

        for key, env_var in env_vars:
            value = os.getenv(env_var)
            if value:
                overrides[key] = value

        return overrides

    def _get_base_path(self, path_type: str) -> Path:
        """Get base path for a given type (code_base, data_base, etc.)"""
        # Priority: env override > platform config > fallback
        if path_type in self.env_overrides:
            return Path(self.env_overrides[path_type])

        platform_config = self.config.get(self.platform, {})
        if path_type in platform_config:
            return Path(platform_config[path_type])

        # Fallback to current directory
        return Path.cwd()

    def get_code_path(self, subpath: str = '') -> Path:
        """Get path for code/repositories"""
        base = self._get_base_path('code_base')
        return base / subpath if subpath else base

    def get_data_path(self, subpath: str = '') -> Path:
        """Get path for data storage"""
        base = self._get_base_path('data_base')
        return base / subpath if subpath else base

    def get_output_path(self, subpath: str = '') -> Path:
        """Get path for analysis outputs (backward compatibility)"""
        base = self._get_base_path('output_base')
        return base / subpath if subpath else base

    def get_preprocessing_path(self, subpath: str = '') -> Path:
        """Get path for preprocessing outputs"""
        base = self._get_base_path('preprocessing_base')
        return base / subpath if subpath else base

    def get_analysis_path(self, subpath: str = '') -> Path:
        """Get path for analysis outputs"""
        base = self._get_base_path('analysis_base')
        return base / subpath if subpath else base

    def get_figures_path(self, subpath: str = '') -> Path:
        """Get path for figures and visualizations"""
        base = self._get_base_path('figures_base')
        return base / subpath if subpath else base

    def get_models_path(self, subpath: str = '') -> Path:
        """Get path for trained models and checkpoints"""
        base = self._get_base_path('models_base')
        return base / subpath if subpath else base

    def get_reports_path(self, subpath: str = '') -> Path:
        """Get path for reports and documentation"""
        base = self._get_base_path('reports_base')
        return base / subpath if subpath else base

    def get_temp_path(self, subpath: str = '') -> Path:
        """Get path for temporary files"""
        base = self._get_base_path('temp_base')
        return base / subpath if subpath else base

    def get_script_output_path(self, script_type: str, script_name: str = '', subpath: str = '') -> Path:
        """
        Get output path for a specific script type with flexible organization.

        Args:
            script_type: Type of script (e.g., 'tcp_preprocessing', 'mdd_analysis', 'visualization')
            script_name: Optional specific script name for further organization
            subpath: Optional additional subdirectory path

        Returns:
            Path object for the script's output directory
        """
        # Get script pattern configuration
        script_patterns = self.config.get('script_patterns', {})

        if script_type in script_patterns:
            pattern = script_patterns[script_type]
            category = pattern['category']
            subdir = pattern['subdir']
        else:
            # Fallback: try to infer category from script_type name
            if 'preprocess' in script_type.lower():
                category = 'preprocessing'
            elif 'analysis' in script_type.lower():
                category = 'analysis'
            elif any(word in script_type.lower() for word in ['plot', 'vis', 'fig']):
                category = 'figures'
            elif 'model' in script_type.lower():
                category = 'models'
            elif 'report' in script_type.lower():
                category = 'reports'
            else:
                category = 'analysis'  # Default fallback

            subdir = script_type.lower().replace('_', '/')

        # Get the appropriate base path for the category
        category_base = self._get_base_path(f'{category}_base')

        # Build the full path
        full_path = category_base / subdir
        if script_name:
            full_path = full_path / script_name
        if subpath:
            full_path = full_path / subpath

        return full_path

    def ensure_path_exists(self, path: Path) -> Path:
        """Create path if it doesn't exist and return it"""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_platform_info(self) -> Dict:
        """Get information about detected platform and configuration"""
        return {
            'detected_platform': self.platform,
            'hostname': socket.gethostname(),
            'system': platform.system(),
            'config_source': 'default_config.json' if (Path(__file__).parent / 'default_config.json').exists() else 'minimal_default',
            'env_overrides': self.env_overrides,
            'current_paths': {
                'code_base': str(self._get_base_path('code_base')),
                'data_base': str(self._get_base_path('data_base')),
                'output_base': str(self._get_base_path('output_base')),
                'preprocessing_base': str(self._get_base_path('preprocessing_base')),
                'analysis_base': str(self._get_base_path('analysis_base')),
                'figures_base': str(self._get_base_path('figures_base')),
                'models_base': str(self._get_base_path('models_base')),
                'reports_base': str(self._get_base_path('reports_base')),
                'temp_base': str(self._get_base_path('temp_base'))
            }
        }

# Global instance for easy importing
_path_config = PathConfig()

# Convenience functions for easy importing
def get_code_path(subpath: str = '') -> Path:
    """Get path for code/repositories"""
    return _path_config.get_code_path(subpath)

def get_data_path(subpath: str = '') -> Path:
    """Get path for data storage"""
    return _path_config.get_data_path(subpath)

def get_output_path(subpath: str = '') -> Path:
    """Get path for analysis outputs (backward compatibility)"""
    return _path_config.get_output_path(subpath)

def get_preprocessing_path(subpath: str = '') -> Path:
    """Get path for preprocessing outputs"""
    return _path_config.get_preprocessing_path(subpath)

def get_analysis_path(subpath: str = '') -> Path:
    """Get path for analysis outputs"""
    return _path_config.get_analysis_path(subpath)

def get_figures_path(subpath: str = '') -> Path:
    """Get path for figures and visualizations"""
    return _path_config.get_figures_path(subpath)

def get_models_path(subpath: str = '') -> Path:
    """Get path for trained models and checkpoints"""
    return _path_config.get_models_path(subpath)

def get_reports_path(subpath: str = '') -> Path:
    """Get path for reports and documentation"""
    return _path_config.get_reports_path(subpath)

def get_temp_path(subpath: str = '') -> Path:
    """Get path for temporary files"""
    return _path_config.get_temp_path(subpath)

def get_script_output_path(script_type: str, script_name: str = '', subpath: str = '') -> Path:
    """
    Get output path for a specific script type with flexible organization.

    Args:
        script_type: Type of script (e.g., 'tcp_preprocessing', 'mdd_analysis', 'visualization')
        script_name: Optional specific script name for further organization
        subpath: Optional additional subdirectory path

    Returns:
        Path object for the script's output directory

    Examples:
        get_script_output_path('tcp_preprocessing', 'extract_subjects')
        get_script_output_path('mdd_analysis', 'connectivity_analysis', 'results')
        get_script_output_path('visualization', 'brain_plots', 'figures')
    """
    return _path_config.get_script_output_path(script_type, script_name, subpath)

def ensure_path_exists(path: Union[Path, str]) -> Path:
    """Create path if it doesn't exist and return it"""
    path_obj = Path(path) if isinstance(path, str) else path
    return _path_config.ensure_path_exists(path_obj)

def get_platform_info() -> Dict:
    """Get information about detected platform and configuration"""
    return _path_config.get_platform_info()

# Specific paths for TCP dataset analysis
def get_tcp_dataset_path() -> Path:
    """Get path to TCP dataset"""
    return get_data_path('ds005237')

def get_tcp_output_path(subpath: str = 'tcp_analysis') -> Path:
    """Get path for TCP analysis outputs (uses preprocessing category)"""
    return get_script_output_path('tcp_preprocessing', subpath=subpath)

def get_mdd_analysis_path(subpath: str = 'mdd_analysis') -> Path:
    """Get path for MDD analysis outputs (uses analysis category)"""
    return get_script_output_path('mdd_analysis', subpath=subpath)

if __name__ == "__main__":
    # Test and display configuration
    print("=== Path Configuration System ===")
    info = get_platform_info()

    print(f"Detected Platform: {info['detected_platform']}")
    print(f"Hostname: {info['hostname']}")
    print(f"System: {info['system']}")
    print(f"Config Source: {info['config_source']}")

    if info['env_overrides']:
        print(f"Environment Overrides: {info['env_overrides']}")

    print("\nCurrent Path Configuration:")
    for path_type, path_value in info['current_paths'].items():
        print(f"  {path_type}: {path_value}")

    print("\nTCP-specific paths:")
    print(f"  TCP Dataset: {get_tcp_dataset_path()}")
    print(f"  TCP Output: {get_tcp_output_path()}")
    print(f"  MDD Analysis: {get_mdd_analysis_path()}")

    print("\nExample flexible output paths:")
    print(f"  Preprocessing: {get_preprocessing_path('tcp_extract')}")
    print(f"  Analysis: {get_analysis_path('connectivity_study')}")
    print(f"  Figures: {get_figures_path('brain_plots')}")
    print(f"  Models: {get_models_path('neural_networks')}")
    print(f"  Reports: {get_reports_path('final_results')}")

    print("\nExample script-specific paths:")
    print(f"  TCP Preprocessing: {get_script_output_path('tcp_preprocessing', 'extract_subjects')}")
    print(f"  MDD Analysis: {get_script_output_path('mdd_analysis', 'connectivity', 'results')}")
    print(f"  Visualization: {get_script_output_path('visualization', 'brain_plots')}")
