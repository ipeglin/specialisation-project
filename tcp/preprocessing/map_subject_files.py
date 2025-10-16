#!/usr/bin/env python3
"""
TCP Subject File Mapping Script

Maps all data file paths for filtered subjects. This creates a comprehensive mapping
of which files exist for each subject, to be used by fetch_filtered_data.py for
downloading only the necessary files.

This is group-agnostic - it maps files for ALL filtered subjects regardless of
patient/control classification.

Author: Ian Philip Eglin
Date: 2025-10-16
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from datetime import datetime
import glob

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path, get_script_output_path


class SubjectFileMapper:
    """Maps data file paths for filtered subjects"""

    def __init__(self,
                 filtered_subjects_dir: Optional[Path] = None,
                 dataset_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.filtered_subjects_dir = Path(filtered_subjects_dir) if filtered_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_subjects')
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'map_subject_files')

        # File patterns for different data types
        self.file_patterns = {
            'raw_nifti': {
                'hammer': '*task-hammer*_bold.nii.gz',
                'stroop': '*task-stroop*_bold.nii.gz'
            },
            'events': {
                'hammer': '*task-hammer*_events.tsv',
                'stroop': '*task-stroop*_events.tsv'
            },
            'json_metadata': {
                'hammer': '*task-hammer*_bold.json',
                'stroop': '*task-stroop*_bold.json'
            },
            'anatomical': {
                't1w': '*_T1w.nii.gz',
                't2w': '*_T2w.nii.gz'
            },
            'anatomical_json': {
                't1w': '*_T1w.json',
                't2w': '*_T2w.json'
            }
        }

        print(f"Filtered subjects directory: {self.filtered_subjects_dir}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")

    def load_filtered_subjects(self) -> List[str]:
        """Load filtered subject IDs"""
        print("\nLoading filtered subjects...")

        subjects_file = self.filtered_subjects_dir / "task_filtered_subjects.csv"

        if not subjects_file.exists():
            raise FileNotFoundError(
                f"Filtered subjects file not found: {subjects_file}\n"
                f"Please run filter_subjects.py first."
            )

        subjects_df = pd.read_csv(subjects_file)

        if 'subject_id' not in subjects_df.columns:
            raise ValueError("subject_id column not found in filtered subjects data")

        subject_ids = subjects_df['subject_id'].tolist()
        print(f"  Loaded {len(subject_ids)} filtered subjects")

        return subject_ids

    def map_subject_files(self, subject_ids: List[str]) -> Dict[str, Dict]:
        """Map all file paths for each subject"""
        print(f"\nMapping files for {len(subject_ids)} subjects...")

        subject_file_mapping = {}
        timeseries_path = self.dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated"

        for i, subject_id in enumerate(subject_ids, 1):
            if i % 50 == 0 or i == len(subject_ids):
                print(f"  Progress: {i}/{len(subject_ids)} subjects")

            file_map = self._map_single_subject(subject_id, timeseries_path)
            subject_file_mapping[subject_id] = file_map

        print(f"  File mapping complete")
        return subject_file_mapping

    def _map_single_subject(self, subject_id: str, timeseries_path: Path) -> Dict:
        """Map files for a single subject"""
        file_map = {
            'raw_nifti': {'hammer': [], 'stroop': []},
            'events': {'hammer': [], 'stroop': []},
            'json_metadata': {'hammer': [], 'stroop': []},
            'anatomical': {'t1w': [], 't2w': []},
            'anatomical_json': {'t1w': [], 't2w': []},
            'timeseries': {'hammer': [], 'stroop': []}
        }

        # Map raw NIFTI, events, and JSON metadata from func/ directory
        func_dir = self.dataset_path / subject_id / "func"
        if func_dir.exists():
            for data_type in ['raw_nifti', 'events', 'json_metadata']:
                for task in ['hammer', 'stroop']:
                    pattern = self.file_patterns[data_type][task]
                    files = list(func_dir.glob(pattern))
                    # Store relative paths from dataset root
                    file_map[data_type][task] = [
                        str(f.relative_to(self.dataset_path)) for f in files
                    ]

        # Map anatomical scans from anat/ directory
        anat_dir = self.dataset_path / subject_id / "anat"
        if anat_dir.exists():
            for scan_type in ['t1w', 't2w']:
                # NIFTI files
                pattern = self.file_patterns['anatomical'][scan_type]
                files = list(anat_dir.glob(pattern))
                file_map['anatomical'][scan_type] = [
                    str(f.relative_to(self.dataset_path)) for f in files
                ]

                # JSON metadata
                pattern = self.file_patterns['anatomical_json'][scan_type]
                files = list(anat_dir.glob(pattern))
                file_map['anatomical_json'][scan_type] = [
                    str(f.relative_to(self.dataset_path)) for f in files
                ]

        # Map timeseries data (different ID format)
        if timeseries_path.exists():
            # Convert BIDS ID to timeseries ID format
            # sub-NDARINVXXXXX -> NDAR_INVXXXXX
            if subject_id.startswith('sub-NDAR'):
                timeseries_id = subject_id.replace('sub-NDAR', 'NDAR_').replace('INV', 'INV')
                ts_subject_dir = timeseries_path / timeseries_id

                if ts_subject_dir.exists():
                    for task in ['hammer', 'stroop']:
                        # Look for .h5 files containing task name
                        files = list(ts_subject_dir.glob(f"*{task}*.h5"))
                        file_map['timeseries'][task] = [
                            str(f.relative_to(self.dataset_path)) for f in files
                        ]

        return file_map

    def find_global_files(self) -> List[str]:
        """Find dataset-wide files (phenotype, participants, etc.)"""
        print("\nFinding global dataset files...")

        global_files = []

        # Phenotype files
        phenotype_dir = self.dataset_path / "phenotype"
        if phenotype_dir.exists():
            for tsv_file in phenotype_dir.glob("*.tsv"):
                global_files.append(str(tsv_file.relative_to(self.dataset_path)))

        # Participants file
        participants_file = self.dataset_path / "participants.tsv"
        if participants_file.exists():
            global_files.append(str(participants_file.relative_to(self.dataset_path)))

        # Dataset description
        dataset_desc = self.dataset_path / "dataset_description.json"
        if dataset_desc.exists():
            global_files.append(str(dataset_desc.relative_to(self.dataset_path)))

        # README
        readme = self.dataset_path / "README"
        if readme.exists():
            global_files.append(str(readme.relative_to(self.dataset_path)))

        print(f"  Found {len(global_files)} global files")
        return global_files

    def calculate_statistics(self, subject_file_mapping: Dict, global_files: List[str]) -> Dict:
        """Calculate file availability statistics"""
        print("\nCalculating statistics...")

        stats = {
            'total_subjects': len(subject_file_mapping),
            'global_files_count': len(global_files),
            'subjects_with_data': {}
        }

        # Count subjects with each data type
        for data_type in ['raw_nifti', 'events', 'json_metadata', 'anatomical', 'timeseries']:
            for key in subject_file_mapping.values().__iter__().__next__()[data_type].keys():
                stat_key = f'{data_type}_{key}'
                count = sum(
                    1 for subj_map in subject_file_mapping.values()
                    if len(subj_map[data_type][key]) > 0
                )
                stats['subjects_with_data'][stat_key] = count

        # Count missing files
        missing_by_type = {}
        for data_type in ['raw_nifti', 'events', 'timeseries']:
            for task in ['hammer', 'stroop']:
                stat_key = f'{data_type}_{task}'
                missing = sum(
                    1 for subj_map in subject_file_mapping.values()
                    if len(subj_map[data_type][task]) == 0
                )
                if missing > 0:
                    missing_by_type[stat_key] = missing

        stats['missing_files_by_type'] = missing_by_type

        return stats

    def export_results(self, subject_file_mapping: Dict, global_files: List[str], stats: Dict) -> None:
        """Export file mapping results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export main file mapping (used by fetch script)
        mapping_output = {
            'subjects': subject_file_mapping,
            'global_files': global_files,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path),
                'total_subjects': len(subject_file_mapping),
                'total_global_files': len(global_files)
            }
        }

        with open(self.output_dir / "subject_file_mapping.json", 'w') as f:
            json.dump(mapping_output, f, indent=2)
        print(f"  ✓ File mapping: subject_file_mapping.json")

        # 2. Export statistics summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'filtered_subjects_source': str(self.filtered_subjects_dir),
            'statistics': stats,
            'note': 'File mapping is group-agnostic. All filtered subjects included regardless of group.'
        }

        with open(self.output_dir / "file_mapping_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary: file_mapping_summary.json")

        # 3. Export missing files report
        self._export_missing_files_report(subject_file_mapping)

    def _export_missing_files_report(self, subject_file_mapping: Dict) -> None:
        """Export report of subjects missing expected files"""
        missing_data = []

        for subject_id, file_map in subject_file_mapping.items():
            missing_types = []

            # Check for missing critical data types
            for data_type in ['raw_nifti', 'events']:
                for task in ['hammer', 'stroop']:
                    if len(file_map[data_type][task]) == 0:
                        missing_types.append(f'{data_type}_{task}')

            if missing_types:
                missing_data.append({
                    'subject_id': subject_id,
                    'missing_data_types': ', '.join(missing_types)
                })

        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            missing_df.to_csv(self.output_dir / "missing_files_report.csv", index=False)
            print(f"  ✓ Missing files report: missing_files_report.csv ({len(missing_data)} subjects)")
        else:
            print(f"  ✓ No missing files detected")

    def print_summary(self, stats: Dict) -> None:
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"FILE MAPPING SUMMARY")
        print(f"{'=' * 60}")
        print(f"\nTotal subjects mapped: {stats['total_subjects']}")
        print(f"Global files found: {stats['global_files_count']}")

        print(f"\nData availability:")
        for data_type, count in sorted(stats['subjects_with_data'].items()):
            percentage = (count / stats['total_subjects']) * 100 if stats['total_subjects'] > 0 else 0
            print(f"  {data_type}: {count}/{stats['total_subjects']} ({percentage:.1f}%)")

        if stats.get('missing_files_by_type'):
            print(f"\nSubjects missing data:")
            for data_type, count in sorted(stats['missing_files_by_type'].items()):
                print(f"  {data_type}: {count} subjects")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Map data file paths for filtered TCP subjects'
    )
    parser.add_argument('--filtered-subjects-dir', type=Path,
                        help='Override filtered subjects directory (auto-detected by default)')
    parser.add_argument('--dataset-path', type=Path,
                        help='Override dataset path')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Subject File Mapping")
    print("=" * 50)

    try:
        # Initialize mapper
        mapper = SubjectFileMapper(
            filtered_subjects_dir=args.filtered_subjects_dir,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )

        # Load filtered subjects
        subject_ids = mapper.load_filtered_subjects()

        # Map files for each subject
        subject_file_mapping = mapper.map_subject_files(subject_ids)

        # Find global files
        global_files = mapper.find_global_files()

        # Calculate statistics
        stats = mapper.calculate_statistics(subject_file_mapping, global_files)

        # Export results
        mapper.export_results(subject_file_mapping, global_files, stats)

        # Print summary
        mapper.print_summary(stats)

        print(f"\n{'=' * 60}")
        print(f"FILE MAPPING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {mapper.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review subject_file_mapping.json")
        print(f"  2. Run fetch_filtered_data.py to download mapped files")
        print(f"  3. Check missing_files_report.csv if any subjects are missing expected data")

        return 0

    except Exception as e:
        print(f"❌ Error during file mapping: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
