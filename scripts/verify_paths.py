#!/usr/bin/env python3
"""
Path Configuration Verification Script

This script helps verify that paths are correctly configured for the current environment.
It checks:
- Platform detection
- Environment variable loading
- Path resolution
- Dataset existence

Usage:
    python3 scripts/verify_paths.py
    python3 scripts/verify_paths.py --check-exists  # Also check if paths exist

Author: Ian Philip Eglin
Date: 2025-12-22
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.paths import (
    get_platform_info,
    get_tcp_dataset_path,
    get_data_path,
    get_preprocessing_path,
    get_analysis_path,
    get_fmriprep_output_path,          # add this
)


def check_path_exists(path: Path, name: str) -> bool:
    """Check if a path exists and print status"""
    exists = path.exists()
    status = "EXISTS" if exists else "NOT FOUND"
    icon = "✓" if exists else "✗"
    print(f"  {icon} {name}: {status}")
    return exists


def main():
    parser = argparse.ArgumentParser(
        description="Verify path configuration for current environment"
    )
    parser.add_argument(
        '--check-exists',
        action='store_true',
        help='Check if paths actually exist on the filesystem'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    args = parser.parse_args()

    # Get platform info
    info = get_platform_info()

    if args.json:
        # Add additional path checks
        info['tcp_dataset_path'] = str(get_tcp_dataset_path())
        info['fmriprep_output_path'] = str(get_fmriprep_output_path())

        if args.check_exists:
            info['path_existence'] = {
                'tcp_dataset': get_tcp_dataset_path().exists(),
                'data_base': Path(info['current_paths']['data_base']).exists(),
                'preprocessing_base': Path(info['current_paths']['preprocessing_base']).exists(),
            }

        print(json.dumps(info, indent=2))
        return

    # Pretty print format
    print("=" * 70)
    print("PATH CONFIGURATION VERIFICATION")
    print("=" * 70)

    print(f"\nPlatform Detection:")
    print(f"  Platform: {info['detected_platform']}")
    print(f"  Hostname: {info['hostname']}")
    print(f"  System: {info['system']}")
    print(f"  Config Source: {info['config_source']}")

    if info['env_overrides']:
        print(f"\nEnvironment Overrides Detected:")
        for key, value in info['env_overrides'].items():
            print(f"  {key}: {value}")
    else:
        print(f"\nEnvironment Overrides: None (using defaults)")

    print(f"\nConfigured Paths:")
    for path_type, path_value in info['current_paths'].items():
        print(f"  {path_type}: {path_value}")

    print(f"\nDataset-Specific Paths:")
    tcp_dataset = get_tcp_dataset_path()
    fmriprep_output = get_fmriprep_output_path()
    print(f"  TCP dataset (ds005237): {tcp_dataset}")
    print(f"  fmriprep output (fmriprep-25.1.4): {fmriprep_output}")

    if args.check_exists:
        print(f"\nPath Existence Check:")
        all_exist = True

        all_exist &= check_path_exists(
            Path(info['current_paths']['data_base']),
            "Data base directory"
        )
        all_exist &= check_path_exists(
            tcp_dataset,
            "TCP dataset (ds005237)"
        )
        all_exist &= check_path_exists(
            tcp_dataset / 'participants.tsv',
            "TCP participants.tsv"
        )
        all_exist &= check_path_exists(
            Path(info['current_paths']['preprocessing_base']),
            "Preprocessing base"
        )
        all_exist &= check_path_exists(
            fmriprep_output,
            "fmriprep output (fmriprep-25.1.4)"
        )

        print(f"\n{'='*70}")
        if all_exist:
            print("✓ All critical paths exist!")
        else:
            print("✗ Some paths are missing. See above for details.")
            print("\nRecommendations:")
            print("  1. Check your .env file configuration")
            print("  2. Verify the paths match your actual data locations")
            print("  3. Create missing directories if needed")
            print("  4. Initialize the dataset with: tcp/preprocessing/initialize_dataset.py")

    print(f"\n{'='*70}")
    print("Configuration Summary:")
    if info['env_overrides']:
        print("  Status: Custom configuration via .env file")
    else:
        print(f"  Status: Using platform defaults for '{info['detected_platform']}'")
    print(f"  Dataset expected at: {tcp_dataset}")
    print(f"  fmriprep output at: {fmriprep_output}")


if __name__ == '__main__':
    main()
