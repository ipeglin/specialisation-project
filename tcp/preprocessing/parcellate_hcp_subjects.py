#!/usr/bin/env python3
"""
TCP HCP Subject Parcellation Script

Parcellates HCP subjects that don't have .h5 files yet.
This is a time-intensive operation that can take several hours depending on the number of subjects.

This script:
1. Identifies HCP subjects without .h5 files
2. Parcellates their NIFTI files using HCPParcellator
3. Updates the file mapping with new .h5 paths

Author: Ian Philip Eglin
Date: 2025-12-23
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.config.data_source_config import (
    DataSourceConfig,
    DataSourceType,
    create_datalad_config,
    create_hcp_config,
    create_combined_config
)
from tcp.preprocessing.utils.unicode_compat import CHECK, ERROR, RUNNING


class HCPParcellationRunner:
    """Runner for HCP subject parcellation"""

    def __init__(self, data_source_config: DataSourceConfig):
        self.data_source_config = data_source_config
        self.subject_file_mapping: Optional[Dict] = None
        self.mapping_file = get_script_output_path('tcp_preprocessing', 'map_subject_files') / "subject_file_mapping.json"

        print(f"HCP Parcellation Runner")
        print(f"Data source: {self.data_source_config.source_type.value}")
        if self.data_source_config.is_hcp_enabled():
            print(f"HCP root: {self.data_source_config.hcp_root}")
            print(f"HCP parcellated output: {self.data_source_config.hcp_parcellated_output}")

    def load_subject_file_mapping(self) -> Dict:
        """Load subject file mapping from map_subject_files.py output"""
        print(f"\nLoading subject file mapping from {self.mapping_file}...")

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Subject file mapping not found: {self.mapping_file}\n"
                f"Please run map_subject_files.py first."
            )

        with open(self.mapping_file, 'r') as f:
            mapping_data = json.load(f)

        self.subject_file_mapping = mapping_data.get('subjects', {})
        print(f"{CHECK} Loaded file mappings for {len(self.subject_file_mapping)} subjects")

        return self.subject_file_mapping

    def save_subject_file_mapping(self) -> None:
        """Save updated subject file mapping"""
        print(f"\nSaving updated subject file mapping to {self.mapping_file}...")

        # Load the full mapping file to preserve all metadata
        with open(self.mapping_file, 'r') as f:
            mapping_data = json.load(f)

        # Update the subjects section
        mapping_data['subjects'] = self.subject_file_mapping

        # Write back to file
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)

        print(f"{CHECK} Subject file mapping updated")

    def parcellate_hcp_subjects(self) -> Dict:
        """
        Parcellate HCP subjects that don't have .h5 files yet.

        Returns:
            Dict with parcellation statistics
        """
        if not self.data_source_config.is_hcp_enabled():
            print("HCP data source is not enabled. Skipping parcellation.")
            return {
                'hcp_enabled': False,
                'subjects_processed': 0,
                'successful': 0,
                'failed': 0
            }

        if not self.subject_file_mapping:
            self.load_subject_file_mapping()

        # Find HCP subjects without .h5 files
        hcp_subjects_to_parcellate = []
        for subject_id, file_map in self.subject_file_mapping.items():
            if file_map.get('data_source') == 'hcp':
                # Check if timeseries .h5 files exist
                has_h5 = any(len(files) > 0 for files in file_map.get('timeseries', {}).values())
                if not has_h5:
                    hcp_subjects_to_parcellate.append(subject_id)

        if not hcp_subjects_to_parcellate:
            print(f"\n{CHECK} No HCP subjects need parcellation")
            return {
                'hcp_enabled': True,
                'subjects_processed': 0,
                'successful': 0,
                'failed': 0,
                'subjects_already_parcellated': len([
                    sid for sid, fm in self.subject_file_mapping.items()
                    if fm.get('data_source') == 'hcp'
                ])
            }

        print(f"\n{RUNNING} Parcellating {len(hcp_subjects_to_parcellate)} HCP subjects...")
        print(f"This may take several hours depending on the number of subjects and data size.")

        # Import HCPParcellator
        from tcp.preprocessing.hcp_parcellation import HCPParcellator

        # Initialize parcellator
        parcellator = HCPParcellator(
            hcp_root=self.data_source_config.hcp_root,
            verbose=True
        )

        # Create output directory
        output_dir = self.data_source_config.hcp_parcellated_output
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Parcellate subjects (sequential for now, can be parallelized later)
        successful = 0
        failed = 0
        failed_subjects = []

        for i, subject_id in enumerate(hcp_subjects_to_parcellate, 1):
            print(f"\n[{i}/{len(hcp_subjects_to_parcellate)}] Parcellating subject: {subject_id}")

            try:
                h5_path = parcellator.parcellate_subject(
                    subject_id=subject_id,
                    task=self.data_source_config.default_task,
                    output_dir=output_dir
                )

                # Update file mapping
                task = self.data_source_config.default_task
                if 'timeseries' not in self.subject_file_mapping[subject_id]:
                    self.subject_file_mapping[subject_id]['timeseries'] = {}
                self.subject_file_mapping[subject_id]['timeseries'][task] = [str(h5_path.absolute())]
                successful += 1
                print(f"{CHECK} Successfully parcellated {subject_id}")

            except Exception as e:
                print(f"{ERROR} Failed to parcellate {subject_id}: {e}")
                failed += 1
                failed_subjects.append({'subject_id': subject_id, 'error': str(e)})

        # Save updated file mapping
        self.save_subject_file_mapping()

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"PARCELLATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total subjects processed: {len(hcp_subjects_to_parcellate)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed_subjects:
            print(f"\nFailed subjects:")
            for failure in failed_subjects:
                print(f"  - {failure['subject_id']}: {failure['error']}")

        return {
            'hcp_enabled': True,
            'subjects_processed': len(hcp_subjects_to_parcellate),
            'successful': successful,
            'failed': failed,
            'failed_subjects': failed_subjects
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Parcellate HCP subjects for TCP anhedonia research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script parcellates HCP subjects that don't have .h5 timeseries files yet.
This is a time-intensive operation that can take several hours.

Examples:
  python parcellate_hcp_subjects.py --data-source-type hcp --hcp-root /path/to/hcp --hcp-parcellated-output /path/to/output
  python parcellate_hcp_subjects.py --data-source-type combined --hcp-root /path/to/hcp --hcp-parcellated-output /path/to/output --default-task hammer
        '''
    )

    # Data source configuration
    parser.add_argument('--data-source-type', choices=['datalad', 'hcp', 'combined'],
                       default='datalad',
                       help='Data source type: datalad (default), hcp, or combined')
    parser.add_argument('--hcp-root', type=Path,
                       help='Path to HCP output directory (required for hcp/combined modes)')
    parser.add_argument('--hcp-parcellated-output', type=Path,
                       help='Directory to store parcellated HCP .h5 files (required for hcp/combined modes)')
    parser.add_argument('--duplicate-resolution', choices=['prefer_hcp', 'prefer_datalad', 'error'],
                       default='prefer_hcp',
                       help='How to handle subjects in both datalad and HCP (combined mode only, default: prefer_hcp)')
    parser.add_argument('--default-task', type=str, default='hammer',
                       help='Default task name for HCP data discovery (default: hammer)')

    args = parser.parse_args()

    print("TCP HCP Subject Parcellation")
    print("=" * 50)

    try:
        # Validate HCP arguments
        if args.data_source_type in ['hcp', 'combined']:
            if not args.hcp_root:
                parser.error("--hcp-root is required when --data-source-type is 'hcp' or 'combined'")
            if not args.hcp_parcellated_output:
                parser.error("--hcp-parcellated-output is required when --data-source-type is 'hcp' or 'combined'")

        # Create data source configuration
        dataset_path = get_tcp_dataset_path()

        if args.data_source_type == 'datalad':
            data_source_config = create_datalad_config(
                dataset_path=dataset_path,
                default_task=args.default_task
            )
        elif args.data_source_type == 'hcp':
            data_source_config = create_hcp_config(
                hcp_root=args.hcp_root,
                parcellated_output=args.hcp_parcellated_output,
                default_task=args.default_task
            )
        elif args.data_source_type == 'combined':
            data_source_config = create_combined_config(
                dataset_path=dataset_path,
                hcp_root=args.hcp_root,
                hcp_parcellated_output=args.hcp_parcellated_output,
                duplicate_resolution=args.duplicate_resolution,
                default_task=args.default_task
            )

        # Initialize parcellation runner
        runner = HCPParcellationRunner(data_source_config=data_source_config)

        # Load subject file mapping
        runner.load_subject_file_mapping()

        # Parcellate HCP subjects
        results = runner.parcellate_hcp_subjects()

        # Print completion message
        print(f"\n{'=' * 60}")
        print(f"HCP PARCELLATION COMPLETE")
        print(f"{'=' * 60}")

        if results['hcp_enabled']:
            if results['subjects_processed'] > 0:
                print(f"Successfully parcellated {results['successful']}/{results['subjects_processed']} subjects")
                if results['failed'] > 0:
                    print(f"WARNING: {results['failed']} subjects failed parcellation")
            else:
                print(f"All HCP subjects already have parcellated .h5 files")
        else:
            print(f"HCP data source not enabled - no parcellation needed")

        return 0 if results.get('failed', 0) == 0 else 1

    except Exception as e:
        print(f"{ERROR} Error during HCP parcellation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
