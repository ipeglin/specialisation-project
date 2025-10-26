#!/usr/bin/env python3
"""
TCP Cross-Analysis Integration Script

Generates comprehensive cross-tabulation statistics and analysis-ready datasets
that combine anhedonia and diagnosis classifications for research insights.

This script provides:
1. Cross-tabulation of anhedonia × diagnosis combinations
2. Statistical summaries across all analysis groups
3. Data quality and completeness reports
4. Analysis-ready export formats

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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Add project root to path to import config
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_script_output_path, get_tcp_dataset_path
from tcp.preprocessing.utils.unicode_compat import CHECK, ERROR


class CrossAnalysisIntegrator:
    """Integrator for cross-analysis statistics and data preparation"""

    def __init__(self,
                 analysis_groups_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.analysis_groups_dir = Path(analysis_groups_dir) if analysis_groups_dir else \
            get_script_output_path('tcp_preprocessing', 'generate_analysis_groups')
        self.output_dir = Path(output_dir) if output_dir else \
            get_script_output_path('tcp_preprocessing', 'integrate_cross_analysis')

        self.combined_subjects: Optional[pd.DataFrame] = None

        print(f"Analysis groups input: {self.analysis_groups_dir}")
        print(f"Output directory: {self.output_dir}")

    def load_combined_subjects_data(self) -> None:
        """Load the combined subjects data with all classifications"""
        print("Loading combined subjects data...")

        combined_file = self.analysis_groups_dir / "all_subjects_with_classifications.csv"

        if not combined_file.exists():
            raise FileNotFoundError(
                f"Combined subjects file not found: {combined_file}\n"
                f"Please run generate_analysis_groups.py first."
            )

        self.combined_subjects = pd.read_csv(combined_file)
        print(f"  Loaded {len(self.combined_subjects)} subjects with full classifications")

        # Verify required columns
        required_cols = ['subject_id', 'anhedonia_class', 'anhedonic_status', 'mdd_status', 'patient_control']
        missing_cols = [col for col in required_cols if col not in self.combined_subjects.columns]
        if missing_cols:
            print(f"  Available columns: {list(self.combined_subjects.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")

    def generate_cross_tabulations(self) -> Dict:
        """Generate comprehensive cross-tabulation analyses"""
        print("Generating cross-tabulation analyses...")

        cross_tabs = {}

        # 1. Anhedonia class × MDD status (detailed)
        anhedonia_mdd_detailed = pd.crosstab(
            self.combined_subjects['anhedonia_class'],
            self.combined_subjects['mdd_status'],
            margins=True
        )
        cross_tabs['anhedonia_by_mdd_detailed'] = anhedonia_mdd_detailed.to_dict()
        print(f"  {CHECK} Anhedonia × MDD status (detailed)")

        # 2. Anhedonic status × Patient/Control (binary)
        anhedonic_patient_binary = pd.crosstab(
            self.combined_subjects['anhedonic_status'],
            self.combined_subjects['patient_control'],
            margins=True
        )
        cross_tabs['anhedonic_by_patient_binary'] = anhedonic_patient_binary.to_dict()
        print(f"  {CHECK} Anhedonic status × Patient/Control (binary)")

        # 3. Analysis group membership overlap
        analysis_groups = ['primary', 'secondary', 'tertiary', 'quaternary']
        group_membership = {}
        
        for group in analysis_groups:
            group_file = self.analysis_groups_dir / f"{group}_analysis_subjects.csv"
            if group_file.exists():
                group_data = pd.read_csv(group_file)
                group_subjects = set(group_data['subject_id'])
                group_membership[group] = group_subjects
            else:
                group_membership[group] = set()

        # Calculate overlap matrix
        overlap_matrix = {}
        for group1 in analysis_groups:
            overlap_matrix[group1] = {}
            for group2 in analysis_groups:
                overlap = len(group_membership[group1] & group_membership[group2])
                overlap_matrix[group1][group2] = overlap

        cross_tabs['group_overlap_matrix'] = overlap_matrix
        print(f"  {CHECK} Analysis group overlap matrix")

        # 4. Anhedonia distribution within each MDD group
        anhedonia_within_mdd = {}
        for mdd_status in self.combined_subjects['mdd_status'].unique():
            if pd.notna(mdd_status):
                mdd_subset = self.combined_subjects[self.combined_subjects['mdd_status'] == mdd_status]
                anhedonia_dist = mdd_subset['anhedonia_class'].value_counts().to_dict()
                anhedonia_within_mdd[mdd_status] = anhedonia_dist

        cross_tabs['anhedonia_within_mdd_groups'] = anhedonia_within_mdd
        print(f"  {CHECK} Anhedonia distribution within MDD groups")

        return cross_tabs

    def calculate_comprehensive_statistics(self) -> Dict:
        """Calculate comprehensive statistics across all classifications"""
        print("Calculating comprehensive statistics...")

        stats = {
            'total_subjects': len(self.combined_subjects),
            'data_completeness': {},
            'distribution_summaries': {},
            'analysis_group_sizes': {},
            'cross_classification_counts': {}
        }

        # Data completeness
        required_cols = ['subject_id', 'anhedonia_class', 'mdd_status']
        for col in required_cols:
            if col in self.combined_subjects.columns:
                complete_count = self.combined_subjects[col].notna().sum()
                completeness_rate = (complete_count / len(self.combined_subjects)) * 100
                stats['data_completeness'][col] = {
                    'complete': complete_count,
                    'missing': len(self.combined_subjects) - complete_count,
                    'completeness_rate_percent': completeness_rate
                }

        # Distribution summaries
        categorical_cols = ['anhedonia_class', 'anhedonic_status', 'mdd_status', 'patient_control']
        for col in categorical_cols:
            if col in self.combined_subjects.columns:
                distribution = self.combined_subjects[col].value_counts().to_dict()
                stats['distribution_summaries'][col] = distribution

        # Analysis group sizes
        analysis_groups = ['primary', 'secondary', 'tertiary', 'quaternary']
        for group in analysis_groups:
            group_file = self.analysis_groups_dir / f"{group}_analysis_subjects.csv"
            if group_file.exists():
                group_data = pd.read_csv(group_file)
                stats['analysis_group_sizes'][group] = len(group_data)

        # Cross-classification counts
        cross_counts = {}
        for anhedonic_status in ['anhedonic', 'non-anhedonic']:
            cross_counts[anhedonic_status] = {}
            for patient_status in ['Control', 'Patient']:
                subset = self.combined_subjects[
                    (self.combined_subjects['anhedonic_status'] == anhedonic_status) &
                    (self.combined_subjects['patient_control'] == patient_status)
                ]
                cross_counts[anhedonic_status][patient_status] = len(subset)

        stats['cross_classification_counts'] = cross_counts

        return stats

    def create_analysis_ready_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create analysis-ready datasets for common research questions"""
        print("Creating analysis-ready datasets...")

        datasets = {}

        # 1. Anhedonic vs Non-anhedonic (all subjects)
        anhedonic_comparison = self.combined_subjects.copy()
        anhedonic_comparison['analysis_focus'] = 'anhedonic_comparison'
        datasets['anhedonic_vs_non_anhedonic'] = anhedonic_comparison
        print(f"  {CHECK} Anhedonic vs Non-anhedonic: {len(anhedonic_comparison)} subjects")

        # 2. Anhedonic patients vs Controls
        anhedonic_patients_controls = self.combined_subjects[
            ((self.combined_subjects['anhedonic_status'] == 'anhedonic') &
             (self.combined_subjects['patient_control'] == 'Patient')) |
            (self.combined_subjects['patient_control'] == 'Control')
        ].copy()
        anhedonic_patients_controls['analysis_focus'] = 'anhedonic_patients_vs_controls'
        datasets['anhedonic_patients_vs_controls'] = anhedonic_patients_controls
        print(f"  {CHECK} Anhedonic patients vs Controls: {len(anhedonic_patients_controls)} subjects")

        # 3. Non-anhedonic patients vs Controls
        non_anhedonic_patients_controls = self.combined_subjects[
            ((self.combined_subjects['anhedonic_status'] == 'non-anhedonic') &
             (self.combined_subjects['patient_control'] == 'Patient')) |
            (self.combined_subjects['patient_control'] == 'Control')
        ].copy()
        non_anhedonic_patients_controls['analysis_focus'] = 'non_anhedonic_patients_vs_controls'
        datasets['non_anhedonic_patients_vs_controls'] = non_anhedonic_patients_controls
        print(f"  {CHECK} Non-anhedonic patients vs Controls: {len(non_anhedonic_patients_controls)} subjects")

        # 4. Anhedonic vs Non-anhedonic within patients only
        patients_only_anhedonia = self.combined_subjects[
            self.combined_subjects['patient_control'] == 'Patient'
        ].copy()
        patients_only_anhedonia['analysis_focus'] = 'patients_anhedonic_comparison'
        datasets['patients_anhedonic_comparison'] = patients_only_anhedonia
        print(f"  {CHECK} Anhedonic vs Non-anhedonic (patients only): {len(patients_only_anhedonia)} subjects")

        # 5. Anhedonic vs Non-anhedonic within controls only
        controls_only_anhedonia = self.combined_subjects[
            self.combined_subjects['patient_control'] == 'Control'
        ].copy()
        controls_only_anhedonia['analysis_focus'] = 'controls_anhedonic_comparison'
        datasets['controls_anhedonic_comparison'] = controls_only_anhedonia
        print(f"  {CHECK} Anhedonic vs Non-anhedonic (controls only): {len(controls_only_anhedonia)} subjects")

        return datasets

    def export_results(self, cross_tabs: Dict, stats: Dict, analysis_datasets: Dict[str, pd.DataFrame]) -> None:
        """Export cross-analysis results"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results to {self.output_dir}")

        # 1. Export cross-tabulation results
        cross_tabs_file = self.output_dir / "cross_tabulation_analysis.json"
        with open(cross_tabs_file, 'w') as f:
            json.dump(cross_tabs, f, indent=2, cls=NumpyEncoder)
        print(f"  {CHECK} Cross-tabulations: {cross_tabs_file}")

        # 2. Export comprehensive statistics
        stats_file = self.output_dir / "comprehensive_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        print(f"  {CHECK} Statistics: {stats_file}")

        # 3. Export analysis-ready datasets
        for dataset_name, dataset in analysis_datasets.items():
            dataset_file = self.output_dir / f"{dataset_name}_analysis_ready.csv"
            dataset.to_csv(dataset_file, index=False)
            print(f"  {CHECK} {dataset_name} dataset: {dataset_file}")

        # 4. Export human-readable summary report
        self._export_human_readable_report(cross_tabs, stats)

        # 5. Export master summary
        master_summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_groups_source': str(self.analysis_groups_dir),
            'total_subjects_analyzed': len(self.combined_subjects),
            'cross_tabulations': cross_tabs,
            'comprehensive_statistics': stats,
            'analysis_ready_datasets': {
                name: len(dataset) for name, dataset in analysis_datasets.items()
            },
            'note': 'Comprehensive cross-analysis integration for anhedonia research'
        }

        master_file = self.output_dir / "cross_analysis_master_summary.json"
        with open(master_file, 'w') as f:
            json.dump(master_summary, f, indent=2, cls=NumpyEncoder)
        print(f"  {CHECK} Master summary: {master_file}")
        
        # 6. Export comprehensive data manifest for processing pipeline
        self._export_data_manifest(analysis_datasets)

    def _export_data_manifest(self, analysis_datasets: Dict[str, pd.DataFrame]) -> None:
        """Export comprehensive data manifest for processing pipeline"""
        print("  Creating comprehensive data manifest for processing pipeline...")
        
        # Load additional subject data for complete manifest
        manifest_data = self._build_comprehensive_manifest(analysis_datasets)
        
        # Export the manifest
        manifest_file = self.output_dir / "processing_data_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2, cls=NumpyEncoder)
        print(f"  {CHECK} Processing data manifest: {manifest_file}")
        
    def _build_comprehensive_manifest(self, analysis_datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Build comprehensive data manifest with subject files and metadata"""
        from config.paths import get_tcp_dataset_path
        
        dataset_path = get_tcp_dataset_path()
        
        # Base manifest structure
        manifest = {
            "manifest_metadata": {
                "created_timestamp": datetime.now().isoformat(),
                "preprocessing_version": "1.0.0",
                "source_dataset": str(dataset_path),
                "total_subjects": len(self.combined_subjects),
                "analysis_groups": list(analysis_datasets.keys()),
                "data_types_available": ["timeseries", "motion", "phenotype"]
            },
            "path_configuration": {
                "dataset_root": str(dataset_path),
                "preprocessing_root": str(self.output_dir.parent.parent),
                "relative_paths": True,
                "platform_info": "Use config.paths for cross-platform resolution"
            },
            "subjects": {},
            "analysis_groups": {},
            "file_patterns": {
                "timeseries": "fMRI_timeseries_clean_denoised_GSR_parcellated/{subject_id}/*_parcellated.h5",
                "motion": "motion_FD/TCP_FD_*.csv",
                "phenotype": "phenotype/*.tsv"
            }
        }
        
        # Build analysis groups mapping
        for group_name, dataset in analysis_datasets.items():
            manifest["analysis_groups"][group_name] = dataset['subject_id'].tolist()
        
        # Build subject-level manifest
        for _, subject_row in self.combined_subjects.iterrows():
            subject_id = subject_row['subject_id']
            
            # Convert subject ID to directory format (remove 'sub-' prefix for directory lookup)
            subject_dir_id = subject_id.replace('sub-', '')
            
            subject_manifest = {
                "demographics": {
                    "age": subject_row.get('age', None),
                    "sex": subject_row.get('sex', None),
                    "site": subject_row.get('Site', None),
                    "group": subject_row.get('Group', None)
                },
                "classifications": {
                    "anhedonia_class": subject_row.get('anhedonia_class', None),
                    "anhedonic_status": subject_row.get('anhedonic_status', None),
                    "mdd_status": subject_row.get('mdd_status', None),
                    "patient_control": subject_row.get('patient_control', None)
                },
                "phenotype_scores": {
                    "shaps_total": subject_row.get('shaps_total', None)
                },
                "files": {
                    "timeseries": {
                        "base_path": f"fMRI_timeseries_clean_denoised_GSR_parcellated/{subject_dir_id}",
                        "available": [],  # Will be populated by file scanning
                        "patterns": ["*_parcellated.h5"]
                    },
                    "motion": {
                        "base_path": "motion_FD",
                        "available": [],  # Will be populated by file scanning
                        "patterns": ["TCP_FD_*.csv"]
                    }
                },
                "analysis_group_memberships": [],
                "data_availability": {
                    "has_timeseries": False,
                    "has_motion": False,
                    "has_phenotype": True  # All subjects have phenotype data
                },
                "validation_status": "validated"  # All subjects in this pipeline are validated
            }
            
            # Determine analysis group memberships
            for group_name, dataset in analysis_datasets.items():
                if subject_id in dataset['subject_id'].values:
                    subject_manifest["analysis_group_memberships"].append(group_name)
            
            # Check for actual file availability (basic check)
            timeseries_dir = dataset_path / "fMRI_timeseries_clean_denoised_GSR_parcellated" / subject_dir_id
            if timeseries_dir.exists():
                subject_manifest["data_availability"]["has_timeseries"] = True
                # List available timeseries files
                timeseries_files = list(timeseries_dir.glob("*_parcellated.h5"))
                subject_manifest["files"]["timeseries"]["available"] = [
                    f"fMRI_timeseries_clean_denoised_GSR_parcellated/{subject_dir_id}/{f.name}" 
                    for f in timeseries_files
                ]
            
            # Check motion data availability
            motion_dir = dataset_path / "motion_FD"
            if motion_dir.exists():
                motion_files = list(motion_dir.glob("TCP_FD_*.csv"))
                if motion_files:
                    subject_manifest["data_availability"]["has_motion"] = True
                    subject_manifest["files"]["motion"]["available"] = [
                        f"motion_FD/{f.name}" for f in motion_files
                    ]
            
            manifest["subjects"][subject_id] = subject_manifest
        
        return manifest

    def _export_human_readable_report(self, cross_tabs: Dict, stats: Dict) -> None:
        """Export human-readable text report"""
        report_file = self.output_dir / "cross_analysis_report.txt"

        with open(report_file, 'w') as f:
            f.write("TCP CROSS-ANALYSIS INTEGRATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total subjects analyzed: {stats['total_subjects']}\n\n")

            # Distribution summaries
            f.write("DISTRIBUTION SUMMARIES\n")
            f.write("-" * 30 + "\n")
            for category, distribution in stats['distribution_summaries'].items():
                f.write(f"\n{category.upper()}:\n")
                for value, count in distribution.items():
                    percentage = (count / stats['total_subjects']) * 100
                    f.write(f"  {value}: {count} ({percentage:.1f}%)\n")

            # Cross-tabulations
            f.write(f"\nCROSS-TABULATIONS\n")
            f.write("-" * 30 + "\n")

            if 'anhedonic_by_patient_binary' in cross_tabs:
                f.write(f"\nAnhedonic Status × Patient/Control:\n")
                binary_cross = cross_tabs['anhedonic_by_patient_binary']
                for anhedonic_status, patient_counts in binary_cross.items():
                    if anhedonic_status != 'All':
                        f.write(f"  {anhedonic_status}:\n")
                        for patient_status, count in patient_counts.items():
                            if patient_status != 'All':
                                f.write(f"    {patient_status}: {count}\n")

            # Analysis group sizes
            f.write(f"\nANALYSIS GROUP SIZES\n")
            f.write("-" * 30 + "\n")
            for group, size in stats['analysis_group_sizes'].items():
                f.write(f"  {group.upper()}: {size} subjects\n")

        print(f"  {CHECK} Human-readable report: {report_file}")

    def print_summary(self, cross_tabs: Dict, stats: Dict, analysis_datasets: Dict) -> None:
        """Print comprehensive summary to console"""
        print(f"\n{'=' * 60}")
        print(f"CROSS-ANALYSIS INTEGRATION SUMMARY")
        print(f"{'=' * 60}")

        print(f"\nTotal subjects analyzed: {stats['total_subjects']}")

        print(f"\nKey distributions:")
        for category, distribution in stats['distribution_summaries'].items():
            if category in ['anhedonic_status', 'patient_control']:
                print(f"  {category}: {distribution}")

        print(f"\nCross-classification (Anhedonic × Patient/Control):")
        if 'cross_classification_counts' in stats:
            for anhedonic_status, patient_counts in stats['cross_classification_counts'].items():
                print(f"  {anhedonic_status}:")
                for patient_status, count in patient_counts.items():
                    print(f"    {patient_status}: {count}")

        print(f"\nAnalysis-ready datasets created:")
        for dataset_name, dataset in analysis_datasets.items():
            print(f"  {dataset_name}: {len(dataset)} subjects")

        print(f"\nAnalysis group sizes:")
        for group, size in stats['analysis_group_sizes'].items():
            print(f"  {group.upper()}: {size} subjects")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive cross-analysis integration for TCP anhedonia research'
    )
    parser.add_argument('--analysis-groups-dir', type=Path,
                        help='Override analysis groups directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Override output directory')

    args = parser.parse_args()

    print("TCP Cross-Analysis Integration")
    print("=" * 50)

    try:
        # Initialize integrator
        integrator = CrossAnalysisIntegrator(
            analysis_groups_dir=args.analysis_groups_dir,
            output_dir=args.output_dir
        )

        # Load combined subjects data
        integrator.load_combined_subjects_data()

        # Generate cross-tabulations
        cross_tabs = integrator.generate_cross_tabulations()

        # Calculate comprehensive statistics
        stats = integrator.calculate_comprehensive_statistics()

        # Create analysis-ready datasets
        analysis_datasets = integrator.create_analysis_ready_datasets()

        # Export results
        integrator.export_results(cross_tabs, stats, analysis_datasets)

        # Print summary
        integrator.print_summary(cross_tabs, stats, analysis_datasets)

        print(f"\n{'=' * 60}")
        print(f"CROSS-ANALYSIS INTEGRATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Results saved to: {integrator.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review cross_analysis_report.txt for human-readable summary")
        print(f"  2. Use analysis-ready CSV files for your research")
        print(f"  3. Run sample_subjects_for_download.py to select subjects for data fetching")

        return 0

    except Exception as e:
        print(f"{ERROR} Error during cross-analysis integration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())