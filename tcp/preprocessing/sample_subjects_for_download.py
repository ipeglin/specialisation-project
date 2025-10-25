#!/usr/bin/env python3
"""
TCP Subject Sampling for Data Download Script

Implements intelligent sampling strategies for different development scenarios:

DEVELOPMENT MODE: Minimal subjects for testing (1 per category)
- PRIMARY: 1 non-anhedonic, 1 low-anhedonic, 1 high-anhedonic
- SECONDARY: 1 MDD primary + anhedonic, 1 MDD primary + non-anhedonic, 1 control  
- TERTIARY: 1 MDD primary, 1 MDD comorbid, 1 control
- QUATERNARY: 1 MDD primary, 1 MDD comorbid, 1 MDD past, 1 control

PRODUCTION MODE: All subjects from selected analysis groups

CUSTOM MODE: N subjects per subgroup for balanced testing

This dramatically reduces storage requirements for local development while
ensuring representative samples for method testing.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random

# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path


class SubjectSamplingPipeline:
    """Pipeline for intelligent subject sampling for data download"""

    def __init__(self,
                 analysis_groups_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None,
                 random_seed: int = 42):
        self.analysis_groups_dir = Path(analysis_groups_dir) if analysis_groups_dir else \
            get_script_output_path('tcp_preprocessing', 'generate_analysis_groups')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')

        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.analysis_groups: Dict[str, pd.DataFrame] = {}

        print(f"Analysis groups input: {self.analysis_groups_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {random_seed}")

    def load_analysis_groups(self, requested_groups: List[str]) -> None:
        """Load requested analysis groups"""
        print("Loading analysis groups...")

        for group_name in requested_groups:
            group_file = self.analysis_groups_dir / f"{group_name}_analysis_subjects.csv"
            
            if not group_file.exists():
                raise FileNotFoundError(
                    f"Analysis group file not found: {group_file}\n"
                    f"Please run generate_analysis_groups.py first."
                )

            group_data = pd.read_csv(group_file)
            self.analysis_groups[group_name] = group_data
            print(f"  Loaded {group_name.upper()}: {len(group_data)} subjects")

    def prioritize_subjects_by_quality(self, subjects: pd.DataFrame) -> pd.DataFrame:
        """Random sampling (could be enhanced with quality metrics in the future)"""
        return subjects.sample(frac=1).reset_index(drop=True)

    def sample_development_mode(self) -> Tuple[pd.DataFrame, Dict]:
        """Sample minimal subjects for development mode (1 per category)"""
        print("Sampling subjects for DEVELOPMENT mode...")

        sampled_subjects = []
        sampling_details = {}

        for group_name, group_data in self.analysis_groups.items():
            print(f"  Sampling from {group_name.upper()} group...")

            if group_name == 'primary':
                # PRIMARY: 1 from each anhedonia class
                for anhedonia_class in ['non-anhedonic', 'low-anhedonic', 'high-anhedonic']:
                    class_subjects = group_data[group_data['anhedonia_class'] == anhedonia_class]
                    if len(class_subjects) > 0:
                        # Prioritize by quality and select first
                        prioritized = self.prioritize_subjects_by_quality(class_subjects)
                        selected = prioritized.head(1).copy()
                        selected['sampling_category'] = f"{group_name}_{anhedonia_class}"
                        selected['sampling_reason'] = f"Development mode: {anhedonia_class} representative"
                        sampled_subjects.append(selected)
                        print(f"    {anhedonia_class}: {len(selected)} subjects")
                    else:
                        print(f"    WARNING: No {anhedonia_class} subjects found")

            elif group_name == 'secondary':
                # SECONDARY: 1 MDD primary + anhedonic, 1 MDD primary + non-anhedonic, 1 control
                
                # Controls
                controls = group_data[group_data['mdd_status'] == 'Control']
                if len(controls) > 0:
                    selected = self.prioritize_subjects_by_quality(controls).head(1).copy()
                    selected['sampling_category'] = f"{group_name}_control"
                    selected['sampling_reason'] = "Development mode: control representative"
                    sampled_subjects.append(selected)
                    print(f"    Controls: {len(selected)} subjects")

                # MDD Primary + anhedonic
                mdd_anhedonic = group_data[
                    (group_data['mdd_status'] == 'MDD_Primary') & 
                    (group_data['anhedonic_status'] == 'anhedonic')
                ]
                if len(mdd_anhedonic) > 0:
                    selected = self.prioritize_subjects_by_quality(mdd_anhedonic).head(1).copy()
                    selected['sampling_category'] = f"{group_name}_mdd_anhedonic"
                    selected['sampling_reason'] = "Development mode: MDD primary + anhedonic"
                    sampled_subjects.append(selected)
                    print(f"    MDD Primary + anhedonic: {len(selected)} subjects")

                # MDD Primary + non-anhedonic
                mdd_non_anhedonic = group_data[
                    (group_data['mdd_status'] == 'MDD_Primary') & 
                    (group_data['anhedonic_status'] == 'non-anhedonic')
                ]
                if len(mdd_non_anhedonic) > 0:
                    selected = self.prioritize_subjects_by_quality(mdd_non_anhedonic).head(1).copy()
                    selected['sampling_category'] = f"{group_name}_mdd_non_anhedonic"
                    selected['sampling_reason'] = "Development mode: MDD primary + non-anhedonic"
                    sampled_subjects.append(selected)
                    print(f"    MDD Primary + non-anhedonic: {len(selected)} subjects")

            elif group_name == 'tertiary':
                # TERTIARY: 1 MDD primary, 1 MDD comorbid, 1 control
                for mdd_status in ['Control', 'MDD_Primary', 'MDD_Comorbid']:
                    status_subjects = group_data[group_data['mdd_status'] == mdd_status]
                    if len(status_subjects) > 0:
                        selected = self.prioritize_subjects_by_quality(status_subjects).head(1).copy()
                        selected['sampling_category'] = f"{group_name}_{mdd_status.lower()}"
                        selected['sampling_reason'] = f"Development mode: {mdd_status} representative"
                        sampled_subjects.append(selected)
                        print(f"    {mdd_status}: {len(selected)} subjects")

            elif group_name == 'quaternary':
                # QUATERNARY: 1 MDD primary, 1 MDD comorbid, 1 MDD past, 1 control
                for mdd_status in ['Control', 'MDD_Primary', 'MDD_Comorbid', 'MDD_Past']:
                    status_subjects = group_data[group_data['mdd_status'] == mdd_status]
                    if len(status_subjects) > 0:
                        selected = self.prioritize_subjects_by_quality(status_subjects).head(1).copy()
                        selected['sampling_category'] = f"{group_name}_{mdd_status.lower()}"
                        selected['sampling_reason'] = f"Development mode: {mdd_status} representative"
                        sampled_subjects.append(selected)
                        print(f"    {mdd_status}: {len(selected)} subjects")

        # Combine all sampled subjects
        if sampled_subjects:
            final_sampled = pd.concat(sampled_subjects, ignore_index=True)
            
            # Remove duplicates (subjects can be in multiple groups)
            unique_subjects = final_sampled.drop_duplicates(subset=['subject_id']).reset_index(drop=True)
            
            # Create sampling details
            sampling_details = {
                'mode': 'development',
                'total_sampled': len(unique_subjects),
                'total_before_dedup': len(final_sampled),
                'duplicates_removed': len(final_sampled) - len(unique_subjects),
                'categories_sampled': final_sampled['sampling_category'].value_counts().to_dict()
            }
            
            print(f"  Total sampled (before deduplication): {len(final_sampled)}")
            print(f"  Unique subjects selected: {len(unique_subjects)}")
            print(f"  Duplicates removed: {sampling_details['duplicates_removed']}")
            
            return unique_subjects, sampling_details
        else:
            return pd.DataFrame(), {'mode': 'development', 'total_sampled': 0}

    def sample_production_mode(self) -> Tuple[pd.DataFrame, Dict]:
        """Sample all subjects from requested analysis groups"""
        print("Sampling subjects for PRODUCTION mode...")

        all_subjects = []
        
        for group_name, group_data in self.analysis_groups.items():
            group_subjects = group_data.copy()
            group_subjects['sampling_category'] = group_name
            group_subjects['sampling_reason'] = f"Production mode: all {group_name} subjects"
            all_subjects.append(group_subjects)
            print(f"  {group_name.upper()}: {len(group_subjects)} subjects")

        # Combine all subjects
        if all_subjects:
            combined_subjects = pd.concat(all_subjects, ignore_index=True)
            
            # Remove duplicates (subjects can be in multiple groups)
            unique_subjects = combined_subjects.drop_duplicates(subset=['subject_id']).reset_index(drop=True)
            
            sampling_details = {
                'mode': 'production',
                'total_sampled': len(unique_subjects),
                'total_before_dedup': len(combined_subjects),
                'duplicates_removed': len(combined_subjects) - len(unique_subjects),
                'groups_included': list(self.analysis_groups.keys())
            }
            
            print(f"  Total subjects (before deduplication): {len(combined_subjects)}")
            print(f"  Unique subjects selected: {len(unique_subjects)}")
            print(f"  Duplicates removed: {sampling_details['duplicates_removed']}")
            
            return unique_subjects, sampling_details
        else:
            return pd.DataFrame(), {'mode': 'production', 'total_sampled': 0}

    def sample_custom_mode(self, subjects_per_group: int) -> Tuple[pd.DataFrame, Dict]:
        """Sample N subjects per subgroup for custom testing"""
        print(f"Sampling subjects for CUSTOM mode ({subjects_per_group} per subgroup)...")

        sampled_subjects = []
        sampling_details = {'mode': 'custom', 'subjects_per_group': subjects_per_group}

        for group_name, group_data in self.analysis_groups.items():
            print(f"  Sampling from {group_name.upper()} group...")

            if group_name == 'primary':
                # Sample N from each anhedonia class
                for anhedonia_class in group_data['anhedonia_class'].unique():
                    if pd.notna(anhedonia_class):
                        class_subjects = group_data[group_data['anhedonia_class'] == anhedonia_class]
                        n_to_sample = min(subjects_per_group, len(class_subjects))
                        selected = self.prioritize_subjects_by_quality(class_subjects).head(n_to_sample).copy()
                        selected['sampling_category'] = f"{group_name}_{anhedonia_class}"
                        selected['sampling_reason'] = f"Custom mode: {anhedonia_class} (n={subjects_per_group})"
                        sampled_subjects.append(selected)
                        print(f"    {anhedonia_class}: {len(selected)} subjects")

            else:
                # Sample N from each MDD status
                for mdd_status in group_data['mdd_status'].unique():
                    if pd.notna(mdd_status):
                        status_subjects = group_data[group_data['mdd_status'] == mdd_status]
                        n_to_sample = min(subjects_per_group, len(status_subjects))
                        selected = self.prioritize_subjects_by_quality(status_subjects).head(n_to_sample).copy()
                        selected['sampling_category'] = f"{group_name}_{mdd_status}"
                        selected['sampling_reason'] = f"Custom mode: {mdd_status} (n={subjects_per_group})"
                        sampled_subjects.append(selected)
                        print(f"    {mdd_status}: {len(selected)} subjects")

        # Combine and deduplicate
        if sampled_subjects:
            combined_subjects = pd.concat(sampled_subjects, ignore_index=True)
            unique_subjects = combined_subjects.drop_duplicates(subset=['subject_id']).reset_index(drop=True)
            
            sampling_details.update({
                'total_sampled': len(unique_subjects),
                'total_before_dedup': len(combined_subjects),
                'duplicates_removed': len(combined_subjects) - len(unique_subjects)
            })
            
            print(f"  Total subjects (before deduplication): {len(combined_subjects)}")
            print(f"  Unique subjects selected: {len(unique_subjects)}")
            
            return unique_subjects, sampling_details
        else:
            return pd.DataFrame(), sampling_details

    def export_results(self, sampled_subjects: pd.DataFrame, sampling_details: Dict, 
                      requested_groups: List[str]) -> None:
        """Export sampling results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export sampled subjects for download
        sampled_file = self.output_dir / "sampled_subjects_for_download.csv"
        sampled_subjects.to_csv(sampled_file, index=False)
        print(f"  ✓ Sampled subjects: {sampled_file}")

        # 2. Export subject IDs only (for easy reference)
        subject_ids_file = self.output_dir / "sampled_subject_ids.txt"
        with open(subject_ids_file, 'w') as f:
            for subject_id in sampled_subjects['subject_id'].unique():
                f.write(f"{subject_id}\n")
        print(f"  ✓ Subject IDs: {subject_ids_file}")

        # 3. Export detailed sampling strategy
        strategy = {
            'timestamp': datetime.now().isoformat(),
            'analysis_groups_source': str(self.analysis_groups_dir),
            'requested_groups': requested_groups,
            'sampling_details': sampling_details,
            'sampled_subjects_count': len(sampled_subjects),
            'unique_subjects_count': len(sampled_subjects['subject_id'].unique()),
            'storage_estimate': {
                'subjects': len(sampled_subjects['subject_id'].unique()),
                'estimated_gb_per_subject': 1.5,
                'total_estimated_gb': len(sampled_subjects['subject_id'].unique()) * 1.5
            },
            'note': 'Subjects selected for data download based on sampling strategy'
        }

        strategy_file = self.output_dir / "sampling_strategy.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        print(f"  ✓ Strategy: {strategy_file}")

        # 4. Export sampling breakdown by category
        if 'sampling_category' in sampled_subjects.columns:
            breakdown = sampled_subjects['sampling_category'].value_counts().to_dict()
            breakdown_file = self.output_dir / "sampling_breakdown.json"
            with open(breakdown_file, 'w') as f:
                json.dump(breakdown, f, indent=2)
            print(f"  ✓ Breakdown: {breakdown_file}")

    def print_summary(self, sampled_subjects: pd.DataFrame, sampling_details: Dict) -> None:
        """Print sampling summary to console"""
        print(f"\n{'=' * 60}")
        print(f"SUBJECT SAMPLING SUMMARY")
        print(f"{'=' * 60}")

        mode = sampling_details.get('mode', 'unknown')
        print(f"\nSampling mode: {mode.upper()}")
        
        print(f"\nSubjects selected:")
        print(f"  Total unique subjects: {len(sampled_subjects['subject_id'].unique())}")
        print(f"  Before deduplication: {sampling_details.get('total_before_dedup', 'N/A')}")
        print(f"  Duplicates removed: {sampling_details.get('duplicates_removed', 'N/A')}")

        # Storage estimate
        unique_count = len(sampled_subjects['subject_id'].unique())
        estimated_gb = unique_count * 1.5
        print(f"\nEstimated storage requirements:")
        print(f"  {unique_count} subjects × 1.5 GB ≈ {estimated_gb:.1f} GB")

        if 'sampling_category' in sampled_subjects.columns:
            print(f"\nSampling breakdown:")
            breakdown = sampled_subjects['sampling_category'].value_counts()
            for category, count in breakdown.items():
                print(f"  {category}: {count}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Sample TCP subjects for data download based on analysis requirements'
    )
    parser.add_argument('--sample-mode', 
                        choices=['development', 'production', 'custom'],
                        default='development',
                        help='Sampling strategy (default: development)')
    parser.add_argument('--subjects-per-group', type=int, default=5,
                        help='Number of subjects per subgroup (custom mode only)')
    parser.add_argument('--analysis-groups', nargs='+',
                        choices=['primary', 'secondary', 'tertiary', 'quaternary', 'all'],
                        default=['primary'],
                        help='Analysis groups to sample from (default: primary)')
    parser.add_argument('--analysis-groups-dir', type=Path,
                        help='Override analysis groups directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    print("TCP Subject Sampling for Data Download")
    print("=" * 50)

    try:
        # Determine which analysis groups to use
        if 'all' in args.analysis_groups:
            requested_groups = ['primary', 'secondary', 'tertiary', 'quaternary']
        else:
            requested_groups = args.analysis_groups

        print(f"Sampling mode: {args.sample_mode.upper()}")
        print(f"Analysis groups: {', '.join(requested_groups)}")
        if args.sample_mode == 'custom':
            print(f"Subjects per group: {args.subjects_per_group}")

        # Initialize pipeline
        pipeline = SubjectSamplingPipeline(
            analysis_groups_dir=args.analysis_groups_dir,
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )

        # Load analysis groups
        pipeline.load_analysis_groups(requested_groups)

        # Apply sampling strategy
        if args.sample_mode == 'development':
            sampled_subjects, sampling_details = pipeline.sample_development_mode()
        elif args.sample_mode == 'production':
            sampled_subjects, sampling_details = pipeline.sample_production_mode()
        elif args.sample_mode == 'custom':
            sampled_subjects, sampling_details = pipeline.sample_custom_mode(args.subjects_per_group)

        # Export results
        pipeline.export_results(sampled_subjects, sampling_details, requested_groups)

        # Print summary
        pipeline.print_summary(sampled_subjects, sampling_details)

        print(f"\n{'=' * 60}")
        print(f"SUBJECT SAMPLING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {pipeline.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Run map_subject_files.py to map files for selected subjects")
        print(f"  2. Run fetch_filtered_data.py to download data for selected subjects")
        print(f"  3. Begin analysis with your sampled dataset")

        return 0

    except Exception as e:
        print(f"❌ Error during subject sampling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())