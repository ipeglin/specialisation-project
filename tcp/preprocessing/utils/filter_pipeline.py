#!/usr/bin/env python3
"""
Subject filtering pipeline orchestration for TCP dataset preprocessing.

Manages multiple filters, data loading, and output organization with dependency injection.

Author: Ian Philip Eglin
Date: 2025-09-12
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import json
import logging

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path
from tcp.preprocessing.utils.subject_filters import SubjectFilter, FilterResult


class SubjectFilterPipeline:
    """Orchestrates multiple subject filters with dependency injection"""

    def __init__(self, extracted_data_path: str, output_dir: str):
        self.extracted_data_path = Path(extracted_data_path)
        self.output_dir = Path(output_dir)
        self.filters: List[SubjectFilter] = []

        # Data files from extract_subjects.py
        self.task_filepaths_file = self.extracted_data_path / "task_file_paths.json"
        self.control_subject_file = self.extracted_data_path / "control_subjects.csv"
        self.patient_subject_file = self.extracted_data_path / "patient_subjects.csv"
        self.summary_file = self.extracted_data_path / "summary.json"

        # Loaded data
        self.patient_subjects: Optional[pd.DataFrame] = None
        self.control_subjects: Optional[pd.DataFrame] = None
        self.file_paths: Optional[Dict] = None
        self.original_summary: Optional[Dict] = None

        # Filtering results
        self.filtering_results: Dict[str, Dict[str, FilterResult]] = {
            'patients': {},
            'controls': {}
        }

        print(f"Initializing Subject Filter Pipeline")
        print(f"  Input data: {self.extracted_data_path}")
        print(f"  Output directory: {self.output_dir}")

    def add_filter(self, filter_instance: SubjectFilter) -> 'SubjectFilterPipeline':
        """Add a filter to the pipeline (fluent interface)"""
        self.filters.append(filter_instance)
        print(f"Added filter: {filter_instance.filter_name}")
        return self

    def load_extracted_data(self) -> None:
        """Load data from extract_subjects.py output"""
        print("Loading extracted subject data...")

        # Load subject dataframes
        if self.patient_subject_file.exists():
            self.patient_subjects = pd.read_csv(self.patient_subject_file)
            print(f"  Loaded {len(self.patient_subjects)} patient subjects")
        else:
            raise FileNotFoundError(f"Patient subjects file not found: {self.patient_subject_file}")

        if self.control_subject_file.exists():
            self.control_subjects = pd.read_csv(self.control_subject_file)
            print(f"  Loaded {len(self.control_subjects)} control subjects")
        else:
            raise FileNotFoundError(f"Control subjects file not found: {self.control_subject_file}")

        # Load file paths
        if self.task_filepaths_file.exists():
            with open(self.task_filepaths_file, 'r') as f:
                self.file_paths = json.load(f)
            print(f"  Loaded task file paths for {len(self.file_paths.get('raw_nifti', {}))} subjects")
        else:
            raise FileNotFoundError(f"Task file paths not found: {self.task_filepaths_file}")

        # Load original summary
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.original_summary = json.load(f)
            print(f"  Loaded original summary")

    def apply_filters(self) -> None:
        """Apply all filters to both patient and control groups"""
        if self.patient_subjects is None or self.control_subjects is None:
            raise ValueError("Data must be loaded before applying filters. Call load_extracted_data() first.")

        print(f"\nApplying {len(self.filters)} filters...")

        # Start with original data
        current_patients = self.patient_subjects.copy()
        current_controls = self.control_subjects.copy()

        # Apply filters sequentially
        for filter_instance in self.filters:
            print(f"\nApplying {filter_instance.filter_name}...")

            # Apply to patients
            patient_result = filter_instance.apply(current_patients, self.file_paths)
            self.filtering_results['patients'][filter_instance.filter_name] = patient_result
            current_patients = patient_result.included_subjects

            print(f"  Patients: {len(patient_result.included_subjects)}/{len(patient_result.included_subjects) + len(patient_result.excluded_subjects)} included")

            # Apply to controls
            control_result = filter_instance.apply(current_controls, self.file_paths)
            self.filtering_results['controls'][filter_instance.filter_name] = control_result
            current_controls = control_result.included_subjects

            print(f"  Controls: {len(control_result.included_subjects)}/{len(control_result.included_subjects) + len(control_result.excluded_subjects)} included")

    def get_final_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get final included and excluded subjects for both groups"""
        if not self.filtering_results['patients'] or not self.filtering_results['controls']:
            raise ValueError("Filters must be applied before getting results. Call apply_filters() first.")

        # Get results from last filter applied
        last_patient_result = list(self.filtering_results['patients'].values())[-1]
        last_control_result = list(self.filtering_results['controls'].values())[-1]

        return (
            last_patient_result.included_subjects,
            last_patient_result.excluded_subjects,
            last_control_result.included_subjects,
            last_control_result.excluded_subjects
        )

    def create_filtered_file_paths(self, included_patients: pd.DataFrame, included_controls: pd.DataFrame) -> Dict:
        """Create filtered file paths dictionary containing only included subjects"""
        included_subject_ids = set(included_patients['participant_id'].tolist() +
                                 included_controls['participant_id'].tolist())

        filtered_file_paths = {}
        for data_type, subjects_data in self.file_paths.items():
            filtered_file_paths[data_type] = {
                subj_id: tasks_data
                for subj_id, tasks_data in subjects_data.items()
                if subj_id in included_subject_ids
            }

        return filtered_file_paths

    def create_excluded_file_paths(self, excluded_patients: pd.DataFrame, excluded_controls: pd.DataFrame) -> Dict:
        """Create filtered file paths dictionary containing only excluded subjects"""
        excluded_subject_ids = set(excluded_patients['participant_id'].tolist() +
                                 excluded_controls['participant_id'].tolist())

        filtered_file_paths = {}
        for data_type, subjects_data in self.file_paths.items():
            filtered_file_paths[data_type] = {
                subj_id: tasks_data
                for subj_id, tasks_data in subjects_data.items()
                if subj_id in excluded_subject_ids
            }

        return filtered_file_paths

    def export_results(self) -> Path:
        """Export filtering results to organized output structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # Get final results
        included_patients, excluded_patients, included_controls, excluded_controls = self.get_final_results()

        # Create output directories
        included_dir = self.output_dir / "included"
        excluded_dir = self.output_dir / "excluded"
        included_dir.mkdir(exist_ok=True)
        excluded_dir.mkdir(exist_ok=True)

        # Export included subjects
        included_patients.to_csv(included_dir / "patient_subjects.csv", index=False)
        included_controls.to_csv(included_dir / "control_subjects.csv", index=False)

        included_file_paths = self.create_filtered_file_paths(included_patients, included_controls)
        with open(included_dir / "task_file_paths.json", 'w') as f:
            json.dump(included_file_paths, f, indent=2)

        # Create summary for included subjects
        included_summary = {
            'dataset_info': {
                'total_patients': len(included_patients),
                'total_controls': len(included_controls),
                'patients_with_task_data': int((included_patients.get('has_hammer_raw', pd.Series([False])) |
                                              included_patients.get('has_stroop_raw', pd.Series([False]))).sum()) if len(included_patients) > 0 else 0,
                'controls_with_task_data': int((included_controls.get('has_hammer_raw', pd.Series([False])) |
                                              included_controls.get('has_stroop_raw', pd.Series([False]))).sum()) if len(included_controls) > 0 else 0
            },
            'filtering_applied': [f.get_filter_info() for f in self.filters]
        }

        with open(included_dir / "summary.json", 'w') as f:
            json.dump(included_summary, f, indent=2)

        # Export excluded subjects
        excluded_patients.to_csv(excluded_dir / "patient_subjects.csv", index=False)
        excluded_controls.to_csv(excluded_dir / "control_subjects.csv", index=False)

        excluded_file_paths = self.create_excluded_file_paths(excluded_patients, excluded_controls)
        with open(excluded_dir / "task_file_paths.json", 'w') as f:
            json.dump(excluded_file_paths, f, indent=2)

        # Collect all exclusion reasons
        all_exclusion_reasons = {}
        for group_name in ['patients', 'controls']:
            for filter_name, filter_result in self.filtering_results[group_name].items():
                for subj_id, reason in filter_result.exclusion_reasons.items():
                    if subj_id not in all_exclusion_reasons:
                        all_exclusion_reasons[subj_id] = []
                    all_exclusion_reasons[subj_id].append(f"{filter_name}: {reason}")

        with open(excluded_dir / "exclusion_reasons.json", 'w') as f:
            json.dump(all_exclusion_reasons, f, indent=2)

        # Create overall filtering report
        filtering_report = {
            'original_data': {
                'total_patients': len(self.patient_subjects),
                'total_controls': len(self.control_subjects)
            },
            'final_results': {
                'included_patients': len(included_patients),
                'included_controls': len(included_controls),
                'excluded_patients': len(excluded_patients),
                'excluded_controls': len(excluded_controls)
            },
            'filters_applied': [],
            'processing_summary': {
                'inclusion_rate_patients': len(included_patients) / len(self.patient_subjects) if len(self.patient_subjects) > 0 else 0,
                'inclusion_rate_controls': len(included_controls) / len(self.control_subjects) if len(self.control_subjects) > 0 else 0,
                'overall_inclusion_rate': (len(included_patients) + len(included_controls)) / (len(self.patient_subjects) + len(self.control_subjects))
            }
        }

        # Add detailed filter results
        for filter_instance in self.filters:
            filter_info = {
                'filter_name': filter_instance.filter_name,
                'description': filter_instance.description,
                'criteria': filter_instance.get_criteria_description(),
                'patient_results': self.filtering_results['patients'][filter_instance.filter_name].statistics,
                'control_results': self.filtering_results['controls'][filter_instance.filter_name].statistics
            }
            filtering_report['filters_applied'].append(filter_info)

        with open(self.output_dir / "filtering_report.json", 'w') as f:
            json.dump(filtering_report, f, indent=2)

        print("Export completed:")
        print(f"  - included/: {len(included_patients)} patients, {len(included_controls)} controls")
        print(f"  - excluded/: {len(excluded_patients)} patients, {len(excluded_controls)} controls")
        print("  - filtering_report.json: Complete filtering statistics")

        return self.output_dir
