#!/usr/bin/env python3
"""
TCP Diagnosis Classification Script

Classifies subjects by MDD status using the Group column and diagnosis fields:
- Controls: Group = "GenPop" 
- MDD_Primary: Group = "Patient" AND Primary_Dx contains "MDD"
- MDD_Comorbid: Group = "Patient" AND Non-Primary_Dx contains "MDD" 
- MDD_Past: Group = "Patient" AND (Primary_Dx OR Non-Primary_Dx) contains "past MDD"/"PMDD"

This classification provides the foundation for the SECONDARY, TERTIARY, and 
QUATERNARY analysis groups with different MDD inclusion criteria.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime
import re

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path


class DiagnosisClassificationPipeline:
    """Pipeline for classifying subjects by MDD diagnosis status"""

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 base_subjects_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.base_subjects_dir = Path(base_subjects_dir) if base_subjects_dir else \
            get_script_output_path('tcp_preprocessing', 'filter_base_subjects')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'classify_diagnoses')

        self.subjects_df: Optional[pd.DataFrame] = None
        self.phenotype_data: Dict[str, pd.DataFrame] = {}

        print(f"Dataset path: {self.dataset_path}")
        print(f"Base subjects input: {self.base_subjects_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_base_subjects(self) -> None:
        """Load base filtered subjects"""
        print("Loading base filtered subjects...")

        base_subjects_file = self.base_subjects_dir / "base_filtered_subjects.csv"

        if not base_subjects_file.exists():
            raise FileNotFoundError(
                f"Base filtered subjects file not found: {base_subjects_file}\n"
                f"Please run filter_base_subjects.py first."
            )

        self.subjects_df = pd.read_csv(base_subjects_file)
        print(f"  Loaded {len(self.subjects_df)} base filtered subjects")

        if 'subject_id' not in self.subjects_df.columns:
            raise ValueError("subject_id column not found in base subjects data")

    def load_phenotype_data(self) -> None:
        """Load phenotype data files from dataset"""
        print("Loading phenotype data...")

        phenotype_files = {
            'demos': 'phenotype/demos.tsv'
        }

        for file_key, file_path in phenotype_files.items():
            full_path = self.dataset_path / file_path

            if full_path.exists():
                try:
                    # Handle different file formats - demos.tsv is actually CSV format
                    if 'demos' in file_key:
                        # Special handling for demos file - skip first row and use comma separator
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                df = pd.read_csv(full_path, sep=',', encoding=encoding, skiprows=1)
                                break
                            except UnicodeDecodeError:
                                if encoding == 'cp1252':  # Last encoding to try
                                    raise
                                continue
                    else:
                        # Standard TSV format for other files
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                df = pd.read_csv(full_path, sep='\t', encoding=encoding)
                                break
                            except UnicodeDecodeError:
                                if encoding == 'cp1252':  # Last encoding to try
                                    raise
                                continue
                    self.phenotype_data[file_key] = df
                    print(f"  Loaded {file_key}: {df.shape[0]} rows, {df.shape[1]} columns")

                    # Show relevant columns for diagnosis
                    diag_cols = [col for col in df.columns if any(term in col.lower() for term in ['group', 'dx', 'diagnosis'])]
                    if diag_cols:
                        print(f"    Diagnosis columns: {diag_cols}")

                    # Show unique values in key columns
                    if 'Group' in df.columns:
                        group_counts = df['Group'].value_counts()
                        print(f"    Group distribution: {dict(group_counts)}")

                except Exception as e:
                    print(f"  ERROR: Could not load {file_key} from {full_path}: {e}")
                    raise
            else:
                print(f"  ERROR: Phenotype file not found: {full_path}")
                raise FileNotFoundError(f"Required phenotype file not found: {full_path}")

    def classify_mdd_status(self, subject_row: pd.Series, demos_data: pd.DataFrame) -> str:
        """
        Classify a single subject's MDD status
        
        Args:
            subject_row: Row from subjects dataframe
            demos_data: Demographics/diagnosis dataframe
            
        Returns:
            MDD status: 'Control', 'MDD_Primary', 'MDD_Comorbid', 'MDD_Past', or 'Unknown'
        """
        subject_id = subject_row['subject_id']
        
        # Convert BIDS subject ID to match demographics format
        # sub-NDARINVXXXXX -> NDAR_INVXXXXX
        if subject_id.startswith('sub-NDAR'):
            demo_subject_id = subject_id.replace('sub-NDAR', 'NDAR_')
        else:
            demo_subject_id = subject_id
            
        # Find subject in demographics data
        subject_demo = demos_data[demos_data['subjectkey'] == demo_subject_id]
        
        if len(subject_demo) == 0:
            return 'Unknown'
            
        subject_demo = subject_demo.iloc[0]
        
        # Get relevant fields
        group = str(subject_demo.get('Group', '')).strip()
        primary_dx = str(subject_demo.get('Primary_Dx', '')).strip()
        non_primary_dx = str(subject_demo.get('Non-Primary_Dx', '')).strip()
        
        # Controls: Group = "GenPop"
        if group == 'GenPop':
            return 'Control'
            
        # Patients: Group = "Patient"
        if group == 'Patient':
            # Check for MDD in primary diagnosis
            if 'MDD' in primary_dx.upper():
                # Check if it's past MDD
                if any(term in primary_dx.upper() for term in ['PAST', 'PMDD']):
                    return 'MDD_Past'
                else:
                    return 'MDD_Primary'
                    
            # Check for MDD in non-primary (comorbid) diagnosis
            if 'MDD' in non_primary_dx.upper():
                # Check if it's past MDD
                if any(term in non_primary_dx.upper() for term in ['PAST', 'PMDD']):
                    return 'MDD_Past'
                else:
                    return 'MDD_Comorbid'
                    
            # Patient without MDD diagnosis
            return 'Patient_Other'
            
        return 'Unknown'

    def classify_all_subjects(self) -> Tuple[pd.DataFrame, Dict]:
        """Classify all subjects by MDD status"""
        print("Classifying subjects by MDD diagnosis status...")

        if 'demos' not in self.phenotype_data:
            raise ValueError("Demographics data not loaded")
            
        demos_data = self.phenotype_data['demos']
        
        # Apply classification to all subjects
        classified_subjects = self.subjects_df.copy()
        
        mdd_status_list = []
        for _, subject_row in classified_subjects.iterrows():
            mdd_status = self.classify_mdd_status(subject_row, demos_data)
            mdd_status_list.append(mdd_status)
            
        classified_subjects['mdd_status'] = mdd_status_list
        
        # Add binary patient/control status
        classified_subjects['patient_control'] = classified_subjects['mdd_status'].apply(
            lambda x: 'Control' if x == 'Control' else 'Patient'
        )
        
        # Count subjects in each category
        mdd_counts = classified_subjects['mdd_status'].value_counts()
        print(f"  MDD status distribution:")
        for status, count in mdd_counts.items():
            print(f"    {status}: {count}")
            
        # Create statistics
        statistics = {
            'total_subjects': len(classified_subjects),
            'mdd_status_distribution': mdd_counts.to_dict(),
            'patient_control_distribution': classified_subjects['patient_control'].value_counts().to_dict()
        }
        
        return classified_subjects, statistics

    def create_analysis_ready_groups(self, classified_subjects: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create subject groups for different analysis types"""
        print("Creating analysis-ready diagnosis groups...")
        
        groups = {}
        
        # Secondary analysis: MDD Primary + Controls
        secondary_subjects = classified_subjects[
            classified_subjects['mdd_status'].isin(['Control', 'MDD_Primary'])
        ].copy()
        groups['secondary'] = secondary_subjects
        print(f"  Secondary (MDD Primary + Controls): {len(secondary_subjects)}")
        
        # Tertiary analysis: MDD Primary/Comorbid + Controls
        tertiary_subjects = classified_subjects[
            classified_subjects['mdd_status'].isin(['Control', 'MDD_Primary', 'MDD_Comorbid'])
        ].copy()
        groups['tertiary'] = tertiary_subjects
        print(f"  Tertiary (MDD Primary/Comorbid + Controls): {len(tertiary_subjects)}")
        
        # Quaternary analysis: All MDD types + Controls
        quaternary_subjects = classified_subjects[
            classified_subjects['mdd_status'].isin(['Control', 'MDD_Primary', 'MDD_Comorbid', 'MDD_Past'])
        ].copy()
        groups['quaternary'] = quaternary_subjects
        print(f"  Quaternary (All MDD + Controls): {len(quaternary_subjects)}")
        
        return groups

    def export_results(self, classified_subjects: pd.DataFrame, analysis_groups: Dict[str, pd.DataFrame], statistics: Dict) -> None:
        """Export diagnosis classification results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export all classified subjects
        classified_file = self.output_dir / "diagnosis_classified_subjects.csv"
        classified_subjects.to_csv(classified_file, index=False)
        print(f"  ✓ Classified subjects: {classified_file}")

        # 2. Export subjects by MDD status
        for status in classified_subjects['mdd_status'].unique():
            if pd.notna(status):
                status_subjects = classified_subjects[classified_subjects['mdd_status'] == status]
                status_file = self.output_dir / f"{status.lower()}_subjects.csv"
                status_subjects.to_csv(status_file, index=False)
                print(f"  ✓ {status} subjects: {status_file}")

        # 3. Export analysis groups
        for group_name, group_subjects in analysis_groups.items():
            group_file = self.output_dir / f"{group_name}_analysis_subjects.csv"
            group_subjects.to_csv(group_file, index=False)
            print(f"  ✓ {group_name.title()} analysis group: {group_file}")

        # 4. Export classification summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'base_subjects_source': str(self.base_subjects_dir),
            'classification_criteria': {
                'Control': 'Group = "GenPop"',
                'MDD_Primary': 'Group = "Patient" AND Primary_Dx contains "MDD"',
                'MDD_Comorbid': 'Group = "Patient" AND Non-Primary_Dx contains "MDD"',
                'MDD_Past': 'Group = "Patient" AND (Primary_Dx OR Non-Primary_Dx) contains "past MDD"/"PMDD"'
            },
            'analysis_groups': {
                'secondary': 'MDD Primary + Controls',
                'tertiary': 'MDD Primary/Comorbid + Controls',
                'quaternary': 'All MDD types + Controls'
            },
            'statistics': statistics,
            'note': 'Diagnosis classification for SECONDARY/TERTIARY/QUATERNARY analysis groups'
        }

        summary_file = self.output_dir / "diagnosis_classification_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary: {summary_file}")

    def print_summary(self, statistics: Dict, analysis_groups: Dict[str, pd.DataFrame]) -> None:
        """Print summary to console"""
        print(f"\n{'=' * 60}")
        print(f"DIAGNOSIS CLASSIFICATION SUMMARY")
        print(f"{'=' * 60}")
        
        print(f"\nClassification criteria:")
        print(f"  Control: Group = 'GenPop'")
        print(f"  MDD_Primary: Group = 'Patient' AND Primary_Dx contains 'MDD'")
        print(f"  MDD_Comorbid: Group = 'Patient' AND Non-Primary_Dx contains 'MDD'")
        print(f"  MDD_Past: Group = 'Patient' AND diagnosis contains 'past MDD'/'PMDD'")

        print(f"\nMDD status distribution:")
        total_subjects = statistics['total_subjects']
        for status, count in statistics['mdd_status_distribution'].items():
            percentage = (count / total_subjects) * 100 if total_subjects > 0 else 0
            print(f"  {status}: {count} ({percentage:.1f}%)")

        print(f"\nAnalysis groups:")
        for group_name, group_subjects in analysis_groups.items():
            print(f"  {group_name.title()}: {len(group_subjects)} subjects")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Classify TCP subjects by MDD diagnosis status'
    )
    parser.add_argument('--dataset-path', type=Path,
                        help='Override dataset path')
    parser.add_argument('--base-subjects-dir', type=Path,
                        help='Override base subjects directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Diagnosis Classification")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = DiagnosisClassificationPipeline(
            dataset_path=args.dataset_path,
            base_subjects_dir=args.base_subjects_dir,
            output_dir=args.output_dir
        )

        # Load input data
        pipeline.load_base_subjects()
        pipeline.load_phenotype_data()

        # Classify MDD status
        classified_subjects, statistics = pipeline.classify_all_subjects()

        # Create analysis groups
        analysis_groups = pipeline.create_analysis_ready_groups(classified_subjects)

        # Export results
        pipeline.export_results(classified_subjects, analysis_groups, statistics)

        # Print summary
        pipeline.print_summary(statistics, analysis_groups)

        print(f"\n{'=' * 60}")
        print(f"DIAGNOSIS CLASSIFICATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {pipeline.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run generate_analysis_groups.py to create final analysis datasets")
        print(f"  2. Analysis groups ready: Secondary, Tertiary, Quaternary")

        return 0

    except Exception as e:
        print(f"❌ Error during diagnosis classification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())