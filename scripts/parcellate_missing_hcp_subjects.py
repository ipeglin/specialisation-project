#!/usr/bin/env python3
"""
Stand-Alone HCP Parcellation Script for Missing Subjects

Parcellates HCP subjects that were skipped during the main preprocessing pipeline.
This utility script is designed to be run manually for subjects that need parcellation
after the initial pipeline execution.

Features:
- Reuses existing HCPParcellator class
- Reads paths from DataSourceConfig
- Skips subjects with existing .h5 files
- Accepts subject IDs from CLI or file
- Generates JSON and text logs for manifest.json updates

Author: Ian Philip Eglin
Date: 2025-12-25
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path
from tcp.preprocessing.config.data_source_config import (
    DataSourceConfig,
    create_hcp_config
)
from tcp.preprocessing.hcp_parcellation import HCPParcellator
from tcp.preprocessing.utils.unicode_compat import CHECK, ERROR, SKIP, RUNNING


class MissingHCPParcellator:
    """Runner for parcellating HCP subjects that were skipped during preprocessing"""

    def __init__(self, data_source_config: DataSourceConfig, output_log_dir: Optional[Path] = None):
        """
        Initialize the parcellator with configuration.

        Args:
            data_source_config: Config object with HCP paths
            output_log_dir: Directory for log outputs (default: script_output_path)
        """
        self.data_source_config = data_source_config
        self.task = data_source_config.default_task

        # Validate HCP mode is enabled
        if not data_source_config.is_hcp_enabled():
            raise ValueError("HCP data source is not enabled in configuration")

        # Set up log directory
        if output_log_dir is None:
            self.log_dir = get_script_output_path('tcp_preprocessing', 'parcellate_missing_hcp_subjects')
        else:
            self.log_dir = output_log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HCPParcellator instance
        self.parcellator = HCPParcellator(
            hcp_root=data_source_config.hcp_root,
            verbose=True
        )

        # Print initialization summary
        print("Missing HCP Parcellator Initialized")
        print(f"  HCP root: {data_source_config.hcp_root}")
        print(f"  HCP parcellated output: {data_source_config.hcp_parcellated_output}")
        print(f"  Task: {self.task}")
        print(f"  Log directory: {self.log_dir}")

    def load_subject_ids(self, cli_ids: Optional[List[str]], subject_file: Optional[Path]) -> List[str]:
        """
        Load subject IDs from either CLI arguments or file input.

        Args:
            cli_ids: List of subject IDs from --subject-ids argument
            subject_file: Path to text file with subject IDs (one per line)

        Returns:
            List of normalized subject IDs

        Raises:
            ValueError: If both or neither input methods are specified
        """
        # Validate mutually exclusive inputs
        if cli_ids and subject_file:
            raise ValueError("Cannot specify both --subject-ids and --subject-file. Choose one.")
        if not cli_ids and not subject_file:
            raise ValueError("Must specify either --subject-ids or --subject-file")

        subject_ids = []

        # Load from CLI arguments
        if cli_ids:
            subject_ids = cli_ids
            print(f"\nLoaded {len(subject_ids)} subject IDs from command line")

        # Load from file
        elif subject_file:
            if not subject_file.exists():
                raise FileNotFoundError(f"Subject file not found: {subject_file}")

            print(f"\nReading subject IDs from: {subject_file}")
            with open(subject_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        subject_ids.append(line)

            print(f"Loaded {len(subject_ids)} subject IDs from file")

        # Normalize subject IDs (ensure "sub-" prefix)
        normalized_ids = []
        for sid in subject_ids:
            if not sid.startswith('sub-'):
                sid = f'sub-{sid}'
            normalized_ids.append(sid)

        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for sid in normalized_ids:
            if sid not in seen:
                seen.add(sid)
                deduplicated.append(sid)

        if len(deduplicated) < len(normalized_ids):
            print(f"Removed {len(normalized_ids) - len(deduplicated)} duplicate subject IDs")

        print(f"Total unique subjects for processing: {len(deduplicated)}")
        return deduplicated

    def check_existing_h5_files(self, subject_ids: List[str]) -> Dict[str, bool]:
        """
        Check which subjects already have .h5 files in the output directory.

        Args:
            subject_ids: List of subject IDs to check

        Returns:
            Dictionary mapping subject_id to has_h5_file boolean
        """
        print(f"\nChecking for existing .h5 files...")
        output_dir = self.data_source_config.hcp_parcellated_output

        existing_status = {}
        for subject_id in subject_ids:
            expected_file = output_dir / f"{subject_id}_task-{self.task}_parcellated.h5"
            existing_status[subject_id] = expected_file.exists()

        # Calculate stats
        already_parcellated = sum(1 for has_h5 in existing_status.values() if has_h5)
        needs_parcellation = len(subject_ids) - already_parcellated

        print(f"Existing file check:")
        print(f"  Already parcellated: {already_parcellated} subjects ({SKIP} will skip)")
        print(f"  Needs parcellation: {needs_parcellation} subjects")

        return existing_status

    def filter_subjects_to_process(self, subject_ids: List[str], existing_h5_status: Dict[str, bool],
                                   force: bool = False) -> Tuple[List[str], List[str]]:
        """
        Separate subjects into "to process" and "already done" lists.

        Args:
            subject_ids: List of all subject IDs
            existing_h5_status: Dictionary of subject_id to has_h5_file boolean
            force: If True, process even subjects with existing .h5 files

        Returns:
            Tuple of (subjects_to_process, subjects_skipped)
        """
        subjects_to_process = []
        subjects_skipped = []

        for subject_id in subject_ids:
            has_h5 = existing_h5_status.get(subject_id, False)
            if has_h5 and not force:
                subjects_skipped.append(subject_id)
            else:
                subjects_to_process.append(subject_id)

        if force and subjects_skipped:
            print(f"\n{RUNNING} Force flag set - will re-parcellate {len(existing_h5_status)} subjects")

        return subjects_to_process, subjects_skipped

    def parcellate_subjects(self, subjects_to_process: List[str]) -> Dict:
        """
        Parcellate subjects using HCPParcellator.

        Args:
            subjects_to_process: List of subject IDs to parcellate

        Returns:
            Results dictionary with:
                - successful: List of successfully parcellated subjects
                - failed: List of dicts with subject_id and error
                - output_paths: Dictionary mapping subject_id to .h5 file path
        """
        results = {
            'successful': [],
            'failed': [],
            'output_paths': {}
        }

        output_dir = self.data_source_config.hcp_parcellated_output
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"PARCELLATING {len(subjects_to_process)} SUBJECTS")
        print(f"{'=' * 60}")

        for i, subject_id in enumerate(subjects_to_process, 1):
            print(f"\n[{i}/{len(subjects_to_process)}] Parcellating subject: {subject_id}")

            try:
                h5_path = self.parcellator.parcellate_subject(
                    subject_id=subject_id,
                    task=self.task,
                    output_dir=output_dir
                )

                results['successful'].append(subject_id)
                results['output_paths'][subject_id] = str(h5_path.absolute())
                print(f"{CHECK} Successfully parcellated {subject_id}")

            except FileNotFoundError as e:
                error_msg = f"No BOLD files found: {e}"
                results['failed'].append({
                    'subject_id': subject_id,
                    'error': error_msg,
                    'error_type': 'FileNotFoundError'
                })
                print(f"{ERROR} Failed to parcellate {subject_id}: {error_msg}")

            except Exception as e:
                error_msg = str(e)
                results['failed'].append({
                    'subject_id': subject_id,
                    'error': error_msg,
                    'error_type': type(e).__name__
                })
                print(f"{ERROR} Failed to parcellate {subject_id}: {error_msg}")

        return results

    def generate_json_log(self, results: Dict, subjects_skipped: List[str]) -> Path:
        """
        Generate JSON log file with subject entries for manifest.json.

        Args:
            results: Results dictionary from parcellate_subjects()
            subjects_skipped: List of skipped subject IDs

        Returns:
            Path to JSON log file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"parcellation_manifest_entries_{timestamp_str}.json"
        log_path = self.log_dir / filename

        # Build log structure
        log_data = {
            'generation_metadata': {
                'timestamp': timestamp.isoformat(),
                'script': 'parcellate_missing_hcp_subjects.py',
                'total_subjects_processed': len(results['successful']) + len(results['failed']),
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'skipped': len(subjects_skipped)
            },
            'subjects_successfully_parcellated': {},
            'subjects_failed': {},
            'subjects_skipped': subjects_skipped,
            'instructions': (
                "Merge entries from 'subjects_successfully_parcellated' into "
                "processing_data_manifest.json under 'subjects' key"
            )
        }

        # Add successful subjects
        for subject_id in results['successful']:
            h5_path = results['output_paths'][subject_id]
            log_data['subjects_successfully_parcellated'][subject_id] = {
                'data_source': 'hcp',
                'files': {
                    'timeseries': {
                        self.task: [h5_path]
                    }
                },
                'parcellation_timestamp': timestamp.isoformat(),
                'note': 'Add this entry to manifest.json subjects section'
            }

        # Add failed subjects
        for failure in results['failed']:
            subject_id = failure['subject_id']
            log_data['subjects_failed'][subject_id] = {
                'error': failure['error'],
                'error_type': failure['error_type'],
                'note': 'Manual intervention required'
            }

        # Write JSON log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n{CHECK} JSON log saved: {log_path}")
        return log_path

    def generate_text_log(self, results: Dict, subjects_skipped: List[str]) -> Path:
        """
        Generate human-readable text summary log.

        Args:
            results: Results dictionary from parcellate_subjects()
            subjects_skipped: List of skipped subject IDs

        Returns:
            Path to text log file
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"parcellation_summary_{timestamp_str}.txt"
        log_path = self.log_dir / filename

        # Build text log
        lines = []
        lines.append("HCP PARCELLATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Script: parcellate_missing_hcp_subjects.py")
        lines.append("")

        lines.append("INPUT SUMMARY")
        lines.append("-" * 60)
        total_requested = len(results['successful']) + len(results['failed']) + len(subjects_skipped)
        lines.append(f"Total subjects requested: {total_requested}")
        lines.append(f"Already parcellated (skipped): {len(subjects_skipped)}")
        lines.append(f"New subjects to parcellate: {len(results['successful']) + len(results['failed'])}")
        lines.append("")

        lines.append("PARCELLATION RESULTS")
        lines.append("-" * 60)
        lines.append(f"Successful: {len(results['successful'])}")
        lines.append(f"Failed: {len(results['failed'])}")
        lines.append("")

        if results['successful']:
            lines.append("SUCCESSFULLY PARCELLATED SUBJECTS")
            lines.append("-" * 60)
            for i, subject_id in enumerate(results['successful'], 1):
                lines.append(f"{i}. {subject_id}")
                lines.append(f"   Output: {results['output_paths'][subject_id]}")
                lines.append("")

        if results['failed']:
            lines.append("FAILED SUBJECTS")
            lines.append("-" * 60)
            for i, failure in enumerate(results['failed'], 1):
                lines.append(f"{i}. {failure['subject_id']}")
                lines.append(f"   Error: {failure['error']}")
                lines.append("")

        if subjects_skipped:
            lines.append("SKIPPED SUBJECTS (Already parcellated)")
            lines.append("-" * 60)
            for i, subject_id in enumerate(subjects_skipped, 1):
                lines.append(f"{i}. {subject_id}")
            lines.append("")

        lines.append("NEXT STEPS")
        lines.append("-" * 60)
        lines.append("1. Review failed subjects and resolve issues")
        lines.append("2. For successful subjects:")
        lines.append("   - Open processing_data_manifest.json")
        lines.append(f"   - Add entries from {filename.replace('summary', 'manifest_entries')}")
        lines.append("   - Verify file paths are correct")
        lines.append("3. Re-run integrate_cross_analysis.py to update manifest if needed")
        lines.append("")

        # Write text log
        with open(log_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"{CHECK} Text log saved: {log_path}")
        return log_path

    def run(self, cli_ids: Optional[List[str]], subject_file: Optional[Path], force: bool = False) -> int:
        """
        Main orchestration method.

        Args:
            cli_ids: List of subject IDs from CLI
            subject_file: Path to subject file
            force: Force re-parcellation even if .h5 exists

        Returns:
            Exit code (0 for success, 1 for errors)
        """
        try:
            # Load subject IDs
            subject_ids = self.load_subject_ids(cli_ids, subject_file)

            # Check existing .h5 files
            existing_h5_status = self.check_existing_h5_files(subject_ids)

            # Filter subjects to process
            subjects_to_process, subjects_skipped = self.filter_subjects_to_process(
                subject_ids, existing_h5_status, force
            )

            # If no subjects to process
            if not subjects_to_process:
                print(f"\n{CHECK} All subjects already parcellated")
                results = {'successful': [], 'failed': [], 'output_paths': {}}
            else:
                # Parcellate subjects
                results = self.parcellate_subjects(subjects_to_process)

            # Generate logs
            json_log_path = self.generate_json_log(results, subjects_skipped)
            text_log_path = self.generate_text_log(results, subjects_skipped)

            # Print summary
            print(f"\n{'=' * 60}")
            print(f"PARCELLATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"Successful: {len(results['successful'])}")
            print(f"Failed: {len(results['failed'])}")
            print(f"Skipped: {len(subjects_skipped)}")
            print(f"\nLog files:")
            print(f"  JSON: {json_log_path}")
            print(f"  Text: {text_log_path}")

            # Return exit code
            if results['failed']:
                return 1
            else:
                return 0

        except Exception as e:
            print(f"\n{ERROR} Error during parcellation: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Parcellate HCP subjects that were skipped during preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This stand-alone script parcellates HCP subjects that don't have .h5 timeseries
files yet. It reuses the existing HCPParcellator logic and generates logs for
manual manifest.json updates.

Examples:
  # Parcellate 3 subjects via CLI
  python scripts/parcellate_missing_hcp_subjects.py \\
    --hcp-root /cluster/projects/.../hcp_output \\
    --hcp-parcellated-output /cluster/work/.../hcp_parcellated \\
    --subject-ids sub-NDARINVXXXXX sub-NDARINVYYYYY sub-NDARINVZZZZZ

  # Parcellate from file
  python scripts/parcellate_missing_hcp_subjects.py \\
    --hcp-root /cluster/projects/.../hcp_output \\
    --hcp-parcellated-output /cluster/work/.../hcp_parcellated \\
    --subject-file missing_subjects.txt

  # Force re-parcellation
  python scripts/parcellate_missing_hcp_subjects.py \\
    --hcp-root /cluster/projects/.../hcp_output \\
    --hcp-parcellated-output /cluster/work/.../hcp_parcellated \\
    --subject-ids sub-NDARINVXXXXX \\
    --force
        '''
    )

    # Required: HCP Configuration
    parser.add_argument('--hcp-root', type=Path, required=True,
                       help='Path to HCP output directory (e.g., /cluster/projects/.../hcp_output)')
    parser.add_argument('--hcp-parcellated-output', type=Path, required=True,
                       help='Directory for parcellated .h5 files')

    # Subject selection (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument('--subject-ids', nargs='+', type=str,
                              help='Space-separated list of subject IDs (e.g., sub-NDARINV001 sub-NDARINV002)')
    subject_group.add_argument('--subject-file', type=Path,
                              help='Path to text file with one subject ID per line (comments starting with # are ignored)')

    # Optional arguments
    parser.add_argument('--task', type=str, default='hammer',
                       help='Task name for HCP data (default: hammer)')
    parser.add_argument('--output-log-dir', type=Path, default=None,
                       help='Override log output directory (default: auto-configured)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-parcellation even if .h5 files already exist')

    args = parser.parse_args()

    print("Stand-Alone HCP Parcellation Script")
    print("=" * 60)

    try:
        # Validate HCP paths exist
        if not args.hcp_root.exists():
            parser.error(f"HCP root directory not found: {args.hcp_root}")

        if args.subject_file and not args.subject_file.exists():
            parser.error(f"Subject file not found: {args.subject_file}")

        # Create data source configuration
        data_source_config = create_hcp_config(
            hcp_root=args.hcp_root,
            parcellated_output=args.hcp_parcellated_output,
            default_task=args.task
        )

        # Initialize parcellator
        parcellator = MissingHCPParcellator(
            data_source_config=data_source_config,
            output_log_dir=args.output_log_dir
        )

        # Run parcellation
        exit_code = parcellator.run(
            cli_ids=args.subject_ids,
            subject_file=args.subject_file,
            force=args.force
        )

        return exit_code

    except Exception as e:
        print(f"{ERROR} Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
