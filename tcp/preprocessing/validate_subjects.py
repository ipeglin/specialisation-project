#!/usr/bin/env python3
"""
TCP Subject Validation Script

Validates which subjects have actual data directories and basic structure.
This provides early filtering before downloading large MRI files.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
from datetime import datetime
import re

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_tcp_dataset_path, get_script_output_path

class SubjectValidator:
    """Validates TCP dataset subjects and their basic structure"""
    
    def __init__(self, dataset_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else get_tcp_dataset_path()
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'validate_subjects')
        
        # Expected subject directory structure
        self.expected_subdirs = ['func', 'anat']
        self.optional_subdirs = ['dwi', 'fmap']
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
    
    def validate_dataset_exists(self) -> bool:
        """Check if dataset directory exists"""
        if not self.dataset_path.exists():
            print(f"✗ Dataset directory does not exist: {self.dataset_path}")
            return False
        
        if not self.dataset_path.is_dir():
            print(f"✗ Dataset path is not a directory: {self.dataset_path}")
            return False
        
        print(f"✓ Dataset directory found: {self.dataset_path}")
        return True
    
    def get_expected_subjects_from_participants(self) -> List[str]:
        """Get list of expected subjects from participants.tsv"""
        participants_file = self.dataset_path / "participants.tsv"
        
        if not participants_file.exists():
            print(f"Warning: participants.tsv not found at {participants_file}")
            return []
        
        try:
            participants_df = pd.read_csv(participants_file, sep='\t')
            if 'participant_id' not in participants_df.columns:
                print(f"Warning: participants.tsv missing 'participant_id' column")
                return []
            
            expected_subjects = participants_df['participant_id'].tolist()
            print(f"Found {len(expected_subjects)} subjects in participants.tsv")
            return expected_subjects
        except Exception as e:
            print(f"Error reading participants.tsv: {e}")
            return []
    
    def find_subject_directories(self) -> List[str]:
        """Find all subject directories in the dataset"""
        if not self.validate_dataset_exists():
            return []
        
        # Find all directories matching sub-* pattern
        subject_pattern = "sub-*"
        subject_dirs = [d.name for d in self.dataset_path.glob(subject_pattern) if d.is_dir()]
        
        # Sort for consistent processing
        subject_dirs.sort()
        
        print(f"Found {len(subject_dirs)} subject directories")
        return subject_dirs
    
    def get_all_subjects_to_process(self) -> Tuple[List[str], List[str]]:
        """Get comprehensive list of subjects to process and those missing directories"""
        expected_subjects = self.get_expected_subjects_from_participants()
        existing_dirs = self.find_subject_directories()
        
        # Find subjects missing directories
        existing_dirs_set = set(existing_dirs)
        missing_dirs = [subj for subj in expected_subjects if subj not in existing_dirs_set]
        
        if missing_dirs:
            print(f"Found {len(missing_dirs)} subjects in participants.tsv without directories:")
            for subj in missing_dirs[:5]:  # Show first 5 as example
                print(f"  - {subj}")
            if len(missing_dirs) > 5:
                print(f"  ... and {len(missing_dirs) - 5} more")
        
        # Combine all subjects for processing (dirs + missing)
        all_subjects = list(existing_dirs_set.union(set(expected_subjects)))
        all_subjects.sort()
        
        return all_subjects, missing_dirs
    
    def validate_subject_id_format(self, subject_id: str) -> Tuple[bool, str]:
        """Validate subject ID format (BIDS compliance)"""
        # Expected format: sub-NDARXXXXXXX (TCP dataset specific)
        pattern = r'^sub-NDAR[A-Z0-9]+$'
        
        if re.match(pattern, subject_id):
            return True, "Valid BIDS subject ID format"
        else:
            return False, f"Invalid subject ID format (expected: sub-NDARXXXXXXX, got: {subject_id})"
    
    def validate_subject_structure(self, subject_id: str) -> Dict[str, any]:
        """Validate individual subject directory structure"""
        subject_dir = self.dataset_path / subject_id
        validation_result = {
            'subject_id': subject_id,
            'directory_exists': False,
            'is_directory': False,
            'id_format_valid': False,
            'id_format_reason': '',
            'subdirectories': {},
            'has_func': False,
            'has_anat': False,
            'func_file_count': 0,
            'anat_file_count': 0,
            'validation_status': 'failed',
            'exclusion_reasons': []
        }
        
        # Check directory existence
        if not subject_dir.exists():
            validation_result['exclusion_reasons'].append("Subject directory does not exist")
            return validation_result
        
        validation_result['directory_exists'] = True
        
        if not subject_dir.is_dir():
            validation_result['exclusion_reasons'].append("Subject path is not a directory")
            return validation_result
        
        validation_result['is_directory'] = True
        
        # Validate subject ID format
        id_valid, id_reason = self.validate_subject_id_format(subject_id)
        validation_result['id_format_valid'] = id_valid
        validation_result['id_format_reason'] = id_reason
        
        if not id_valid:
            validation_result['exclusion_reasons'].append(id_reason)
        
        # Check for expected subdirectories
        for subdir in self.expected_subdirs + self.optional_subdirs:
            subdir_path = subject_dir / subdir
            exists = subdir_path.exists() and subdir_path.is_dir()
            validation_result['subdirectories'][subdir] = exists
            
            if subdir == 'func':
                validation_result['has_func'] = exists
                if exists:
                    # Count files in func directory
                    func_files = list(subdir_path.glob("*"))
                    validation_result['func_file_count'] = len(func_files)
            
            elif subdir == 'anat':
                validation_result['has_anat'] = exists
                if exists:
                    # Count files in anat directory
                    anat_files = list(subdir_path.glob("*"))
                    validation_result['anat_file_count'] = len(anat_files)
        
        # Check for required subdirectories
        missing_required = [subdir for subdir in self.expected_subdirs 
                          if not validation_result['subdirectories'].get(subdir, False)]
        
        if missing_required:
            validation_result['exclusion_reasons'].append(
                f"Missing required subdirectories: {missing_required}"
            )
        
        # Determine final validation status
        if (validation_result['directory_exists'] and 
            validation_result['is_directory'] and
            validation_result['id_format_valid'] and
            validation_result['has_func'] and
            validation_result['has_anat']):
            validation_result['validation_status'] = 'valid'
        else:
            validation_result['validation_status'] = 'invalid'
            
            # Add general reason if no specific reasons found
            if not validation_result['exclusion_reasons']:
                validation_result['exclusion_reasons'].append("Failed general validation criteria")
        
        return validation_result
    
    def create_missing_directory_entry(self, subject_id: str) -> Dict[str, any]:
        """Create validation entry for subject missing directory"""
        return {
            'subject_id': subject_id,
            'directory_exists': False,
            'is_directory': False,
            'id_format_valid': True,  # Assume valid since it's in participants.tsv
            'id_format_reason': 'Valid BIDS subject ID format',
            'subdirectories': {},
            'has_func': False,
            'has_anat': False,
            'func_file_count': 0,
            'anat_file_count': 0,
            'validation_status': 'invalid',
            'exclusion_reasons': ['Subject directory does not exist (listed in participants.tsv but no directory found)']
        }
    
    def validate_all_subjects(self) -> Tuple[List[Dict], List[Dict]]:
        """Validate all subjects and separate into valid/invalid"""
        print("\n=== Validating All Subjects ===")
        
        # Get comprehensive list of subjects to process
        all_subjects, missing_dirs = self.get_all_subjects_to_process()
        subject_dirs = self.find_subject_directories()
        
        if not all_subjects:
            print("No subjects found to validate")
            return [], []
        
        valid_subjects = []
        invalid_subjects = []
        
        print(f"Validating {len(all_subjects)} subjects (including {len(missing_dirs)} missing directories)...")
        
        for i, subject_id in enumerate(all_subjects, 1):
            if i % 50 == 0 or i == len(all_subjects):
                print(f"  Progress: {i}/{len(all_subjects)} subjects")
            
            # Check if this subject has a directory
            if subject_id in missing_dirs:
                # Create entry for missing directory
                validation_result = self.create_missing_directory_entry(subject_id)
                invalid_subjects.append(validation_result)
            else:
                # Validate normally
                validation_result = self.validate_subject_structure(subject_id)
                
                if validation_result['validation_status'] == 'valid':
                    valid_subjects.append(validation_result)
                else:
                    invalid_subjects.append(validation_result)
        
        print(f"\nValidation complete:")
        print(f"  ✓ Valid subjects: {len(valid_subjects)}")
        print(f"  ✗ Invalid subjects: {len(invalid_subjects)}")
        print(f"  📁 Missing directories: {len(missing_dirs)}")
        
        return valid_subjects, invalid_subjects
    
    def create_subject_dataframes(self, valid_subjects: List[Dict], invalid_subjects: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert validation results to pandas DataFrames"""
        print("\nCreating subject DataFrames...")
        
        # Create DataFrame for valid subjects
        if valid_subjects:
            valid_df = pd.DataFrame(valid_subjects)
            print(f"  Valid subjects DataFrame: {valid_df.shape}")
        else:
            # Create empty DataFrame with expected columns
            valid_df = pd.DataFrame(columns=[
                'subject_id', 'directory_exists', 'is_directory', 'id_format_valid',
                'has_func', 'has_anat', 'func_file_count', 'anat_file_count',
                'validation_status'
            ])
            print("  Valid subjects DataFrame: empty")
        
        # Create DataFrame for invalid subjects
        if invalid_subjects:
            invalid_df = pd.DataFrame(invalid_subjects)
            print(f"  Invalid subjects DataFrame: {invalid_df.shape}")
        else:
            # Create empty DataFrame with expected columns
            invalid_df = pd.DataFrame(columns=[
                'subject_id', 'directory_exists', 'is_directory', 'id_format_valid',
                'has_func', 'has_anat', 'func_file_count', 'anat_file_count',
                'validation_status', 'exclusion_reasons'
            ])
            print("  Invalid subjects DataFrame: empty")
        
        return valid_df, invalid_df
    
    def generate_validation_summary(self, valid_subjects: List[Dict], invalid_subjects: List[Dict]) -> Dict:
        """Generate summary statistics of validation"""
        total_subjects = len(valid_subjects) + len(invalid_subjects)
        
        # Count exclusion reasons
        exclusion_reason_counts = {}
        for subject in invalid_subjects:
            for reason in subject.get('exclusion_reasons', []):
                exclusion_reason_counts[reason] = exclusion_reason_counts.get(reason, 0) + 1
        
        # Calculate statistics for valid subjects
        valid_stats = {
            'total_valid': len(valid_subjects),
            'avg_func_files': 0,
            'avg_anat_files': 0,
            'subjects_with_func': 0,
            'subjects_with_anat': 0
        }
        
        if valid_subjects:
            valid_stats['avg_func_files'] = sum(s['func_file_count'] for s in valid_subjects) / len(valid_subjects)
            valid_stats['avg_anat_files'] = sum(s['anat_file_count'] for s in valid_subjects) / len(valid_subjects)
            valid_stats['subjects_with_func'] = sum(1 for s in valid_subjects if s['has_func'])
            valid_stats['subjects_with_anat'] = sum(1 for s in valid_subjects if s['has_anat'])
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'total_subjects_found': total_subjects,
            'valid_subjects': len(valid_subjects),
            'invalid_subjects': len(invalid_subjects),
            'validation_rate': len(valid_subjects) / total_subjects if total_subjects > 0 else 0,
            'exclusion_reasons': exclusion_reason_counts,
            'valid_subject_statistics': valid_stats
        }
        
        return summary
    
    def export_results(self, valid_subjects: List[Dict], invalid_subjects: List[Dict]) -> Path:
        """Export validation results to files"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nExporting results to {self.output_dir}")
        
        # Convert to DataFrames
        valid_df, invalid_df = self.create_subject_dataframes(valid_subjects, invalid_subjects)
        
        # Export valid subjects
        valid_file = self.output_dir / "valid_subjects.csv"
        valid_df.to_csv(valid_file, index=False)
        print(f"  ✓ Valid subjects: {valid_file}")
        
        # Export invalid subjects
        invalid_file = self.output_dir / "invalid_subjects.csv"
        invalid_df.to_csv(invalid_file, index=False)
        print(f"  ✓ Invalid subjects: {invalid_file}")
        
        # Generate and export summary
        summary = self.generate_validation_summary(valid_subjects, invalid_subjects)
        summary_file = self.output_dir / "validation_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Summary: {summary_file}")
        
        # Export detailed validation results
        detailed_results = {
            'valid_subjects': valid_subjects,
            'invalid_subjects': invalid_subjects,
            'summary': summary
        }
        
        detailed_file = self.output_dir / "detailed_validation_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"  ✓ Detailed results: {detailed_file}")
        
        return self.output_dir
    
    def print_summary(self, valid_subjects: List[Dict], invalid_subjects: List[Dict]) -> None:
        """Print validation summary to console"""
        total_subjects = len(valid_subjects) + len(invalid_subjects)
        
        print(f"\n{'='*60}")
        print(f"SUBJECT VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total subjects found: {total_subjects}")
        print(f"Valid subjects: {len(valid_subjects)} ({len(valid_subjects)/total_subjects*100:.1f}%)")
        print(f"Invalid subjects: {len(invalid_subjects)} ({len(invalid_subjects)/total_subjects*100:.1f}%)")
        
        if invalid_subjects:
            print(f"\nTop exclusion reasons:")
            exclusion_counts = {}
            for subject in invalid_subjects:
                for reason in subject.get('exclusion_reasons', []):
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            
            # Sort by frequency and show top 5
            sorted_reasons = sorted(exclusion_counts.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons[:5]:
                print(f"  - {reason}: {count} subjects")
        
        if valid_subjects:
            avg_func = sum(s['func_file_count'] for s in valid_subjects) / len(valid_subjects)
            avg_anat = sum(s['anat_file_count'] for s in valid_subjects) / len(valid_subjects)
            print(f"\nValid subjects statistics:")
            print(f"  - Average func files: {avg_func:.1f}")
            print(f"  - Average anat files: {avg_anat:.1f}")
            print(f"  - All have func directory: {all(s['has_func'] for s in valid_subjects)}")
            print(f"  - All have anat directory: {all(s['has_anat'] for s in valid_subjects)}")

def main():
    """Main execution function"""
    print("TCP Subject Validation")
    print("=" * 50)
    
    # Initialize validator
    validator = SubjectValidator()
    
    # Validate all subjects
    valid_subjects, invalid_subjects = validator.validate_all_subjects()
    
    # Export results
    output_dir = validator.export_results(valid_subjects, invalid_subjects)
    
    # Print summary
    validator.print_summary(valid_subjects, invalid_subjects)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run fetch_global_data.py to get participants.tsv")
    print(f"  2. Run filter_subjects.py for task data filtering")
    print(f"  3. Run the anhedonia classification pipeline (filter_base_subjects.py, classify_anhedonia.py, etc.)")
    
    # Return success if we have any valid subjects
    return 0 if valid_subjects else 1

if __name__ == "__main__":
    sys.exit(main())