#!/usr/bin/env python3
"""
Dependency checking utilities for TCP processing pipeline.

Author: Ian Philip Eglin
Date: 2025-10-27
"""

import sys
from typing import List, Dict, Any
from pathlib import Path


def check_required_dependencies() -> Dict[str, Any]:
    """
    Check for required dependencies and provide helpful guidance.
    
    Returns:
        Dictionary with dependency status and guidance
    """
    results = {
        'all_available': True,
        'missing_dependencies': [],
        'available_dependencies': [],
        'guidance': []
    }
    
    # Check for pandas
    try:
        import pandas
        results['available_dependencies'].append(f"pandas {pandas.__version__}")
    except ImportError:
        results['missing_dependencies'].append('pandas')
        results['all_available'] = False
    
    # Check for numpy (usually comes with pandas)
    try:
        import numpy
        results['available_dependencies'].append(f"numpy {numpy.__version__}")
    except ImportError:
        results['missing_dependencies'].append('numpy')
        results['all_available'] = False
    
    # Provide guidance based on missing dependencies
    if results['missing_dependencies']:
        results['guidance'].extend([
            "To install missing dependencies:",
            "1. Activate conda environment: 'conda activate masters_thesis'",
            "   OR",
            "2. Install with pip: 'pip install pandas numpy'",
            "   OR", 
            "3. Install with conda: 'conda install pandas numpy'"
        ])
    
    return results


def check_environment_setup() -> Dict[str, Any]:
    """
    Check overall environment setup for TCP processing.
    
    Returns:
        Dictionary with environment status
    """
    results = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'python_executable': sys.executable,
        'working_directory': str(Path.cwd()),
        'environment_status': 'unknown'
    }
    
    # Check if we're in a conda environment
    if 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower():
        results['environment_status'] = 'conda'
    elif 'venv' in sys.executable or 'virtualenv' in sys.executable:
        results['environment_status'] = 'virtual_environment'
    else:
        results['environment_status'] = 'system_python'
    
    return results


def print_dependency_report():
    """Print comprehensive dependency and environment report."""
    print("TCP Processing Pipeline - Dependency Report")
    print("=" * 50)
    
    # Check environment
    env_info = check_environment_setup()
    print(f"\nEnvironment Information:")
    print(f"  Python version: {env_info['python_version']}")
    print(f"  Python executable: {env_info['python_executable']}")
    print(f"  Environment type: {env_info['environment_status']}")
    print(f"  Working directory: {env_info['working_directory']}")
    
    # Check dependencies
    dep_info = check_required_dependencies()
    print(f"\nDependency Status:")
    
    if dep_info['available_dependencies']:
        print(f"  Available:")
        for dep in dep_info['available_dependencies']:
            print(f"    ✓ {dep}")
    
    if dep_info['missing_dependencies']:
        print(f"  Missing:")
        for dep in dep_info['missing_dependencies']:
            print(f"    ❌ {dep}")
    
    if dep_info['guidance']:
        print(f"\nGuidance:")
        for guide in dep_info['guidance']:
            print(f"  {guide}")
    
    # Overall status
    if dep_info['all_available']:
        print(f"\n✓ All dependencies available - full functionality enabled")
    else:
        print(f"\n⚠️  Some dependencies missing - limited functionality")
    
    return dep_info['all_available']


if __name__ == "__main__":
    success = print_dependency_report()
    sys.exit(0 if success else 1)