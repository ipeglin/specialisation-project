#!/usr/bin/env python3
"""
Subject manager for TCP processing pipeline.

Provides advanced subject selection, filtering, and metadata management
without implementing analysis algorithms.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .data_loader import DataLoader
from .config.processing_config import ProcessingConfig
from .utils.error_handling import ValidationError, DataNotFoundError
from .utils.downloaded_data import get_downloaded_subjects, get_download_status_report


class SubjectManager:
    """
    Advanced subject management for TCP processing pipeline.
    
    Provides sophisticated subject selection, filtering, and metadata
    access without implementing analysis algorithms.
    """
    
    def __init__(self, 
                 data_loader: Optional[DataLoader] = None,
                 config: Optional[ProcessingConfig] = None):
        """
        Initialize SubjectManager.
        
        Args:
            data_loader: DataLoader instance (creates default if None)
            config: Processing configuration
        """
        self.config = config if config else ProcessingConfig()
        self.data_loader = data_loader if data_loader else DataLoader(config=self.config)
        
        # Cache for frequently accessed data
        self._subject_cache: Dict[str, Dict[str, Any]] = {}
        self._group_cache: Dict[str, List[str]] = {}
        self._summary_cache: Optional[pd.DataFrame] = None
    
    def get_all_subjects(self) -> List[str]:
        """Get list of all available subject IDs."""
        return self.data_loader.get_all_subject_ids()
    
    def get_analysis_groups(self) -> Dict[str, List[str]]:
        """Get all analysis group definitions."""
        if not self._group_cache:
            self._group_cache = self.data_loader.get_analysis_groups()
        return self._group_cache.copy()
    
    def get_group_subjects(self, group_name: str) -> List[str]:
        """
        Get subjects in specific analysis group.
        
        Args:
            group_name: Name of analysis group
            
        Returns:
            List of subject IDs in group
        """
        groups = self.get_analysis_groups()
        if group_name not in groups:
            available = list(groups.keys())
            raise ValidationError(f"Group '{group_name}' not found. Available: {available}")
        
        return groups[group_name].copy()
    
    def get_subjects_by_classification(self, 
                                     classification_key: str, 
                                     classification_value: Any) -> List[str]:
        """
        Get subjects with specific classification value.
        
        Args:
            classification_key: Classification field name
            classification_value: Required value
            
        Returns:
            List of matching subject IDs
        """
        matching_subjects = []
        
        for subject_id in self.get_all_subjects():
            try:
                metadata = self.get_subject_metadata(subject_id)
                classifications = metadata.get('classifications', {})
                
                if classifications.get(classification_key) == classification_value:
                    matching_subjects.append(subject_id)
            except Exception:
                continue  # Skip subjects with missing data
        
        return matching_subjects
    
    def get_downloaded_subjects(self) -> List[str]:
        """
        Get list of subjects that have been downloaded locally.
        
        Returns:
            List of subject IDs with locally downloaded data
        """
        try:
            return get_downloaded_subjects()
        except FileNotFoundError:
            return []  # Return empty list if no downloaded subjects file exists
    
    def get_subjects_with_data(self, 
                              data_types: Union[str, List[str]], 
                              require_all: bool = True) -> List[str]:
        """
        Get subjects with specific data availability.
        
        Args:
            data_types: Data type(s) to check ('timeseries', 'motion', 'phenotype')
            require_all: If True, require all data types; if False, require any
            
        Returns:
            List of subject IDs with required data
        """
        if isinstance(data_types, str):
            data_types = [data_types]
        
        matching_subjects = []
        
        for subject_id in self.get_all_subjects():
            try:
                metadata = self.get_subject_metadata(subject_id)
                availability = metadata.get('data_availability', {})
                
                has_data = []
                for data_type in data_types:
                    availability_key = f'has_{data_type}'
                    has_data.append(availability.get(availability_key, False))
                
                if require_all and all(has_data):
                    matching_subjects.append(subject_id)
                elif not require_all and any(has_data):
                    matching_subjects.append(subject_id)
                    
            except Exception:
                continue
        
        return matching_subjects
    
    def filter_subjects(self, 
                       base_subjects: Optional[List[str]] = None,
                       groups: Optional[List[str]] = None,
                       classifications: Optional[Dict[str, Any]] = None,
                       data_requirements: Optional[List[str]] = None,
                       demographics: Optional[Dict[str, Any]] = None,
                       downloaded_only: bool = False,
                       custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[str]:
        """
        Apply multiple filters to select subjects.
        
        Args:
            base_subjects: Starting set of subjects (all if None)
            groups: Analysis groups to include
            classifications: Classification criteria
            data_requirements: Required data types
            demographics: Demographic criteria (e.g., {'sex': 'F', 'age': (18, 65)})
            downloaded_only: If True, only include subjects with downloaded data
            custom_filter: Custom filter function that takes subject metadata
            
        Returns:
            List of filtered subject IDs
        """
        # Start with base subjects or all subjects
        if base_subjects is None:
            filtered_subjects = set(self.get_all_subjects())
        else:
            filtered_subjects = set(base_subjects)
        
        # Apply group filter
        if groups:
            group_subjects = set()
            for group in groups:
                group_subjects.update(self.get_group_subjects(group))
            filtered_subjects &= group_subjects
        
        # Apply classification filters
        if classifications:
            for key, value in classifications.items():
                matching = set(self.get_subjects_by_classification(key, value))
                filtered_subjects &= matching
        
        # Apply data requirement filters
        if data_requirements:
            data_subjects = set(self.get_subjects_with_data(data_requirements, require_all=True))
            filtered_subjects &= data_subjects
        
        # Apply downloaded-only filter
        if downloaded_only:
            downloaded_subjects = set(self.get_downloaded_subjects())
            filtered_subjects &= downloaded_subjects
        
        # Apply demographic filters
        if demographics:
            demo_filtered = set()
            for subject_id in filtered_subjects:
                try:
                    metadata = self.get_subject_metadata(subject_id)
                    subject_demographics = metadata.get('demographics', {})
                    
                    matches_demographics = True
                    for key, value in demographics.items():
                        subject_value = subject_demographics.get(key)
                        
                        if isinstance(value, tuple) and len(value) == 2:
                            # Range filter (e.g., age range)
                            if subject_value is None or not (value[0] <= subject_value <= value[1]):
                                matches_demographics = False
                                break
                        else:
                            # Exact match
                            if subject_value != value:
                                matches_demographics = False
                                break
                    
                    if matches_demographics:
                        demo_filtered.add(subject_id)
                        
                except Exception:
                    continue
            
            filtered_subjects &= demo_filtered
        
        # Apply custom filter
        if custom_filter:
            custom_filtered = set()
            for subject_id in filtered_subjects:
                try:
                    metadata = self.get_subject_metadata(subject_id)
                    if custom_filter(metadata):
                        custom_filtered.add(subject_id)
                except Exception:
                    continue
            
            filtered_subjects &= custom_filtered
        
        return sorted(list(filtered_subjects))
    
    def get_subject_files_by_task(self, subject_id: str, data_type: str = 'timeseries',
                                 task_filter: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Get file paths for subject filtered by task type.
        
        Args:
            subject_id: Subject identifier
            data_type: Type of data ('timeseries', 'motion')
            task_filter: Task name(s) to filter by (e.g., 'hammer', ['hammer', 'rest'])
            
        Returns:
            List of file paths matching the task filter
        """
        return self.data_loader.get_subject_files_by_task(subject_id, data_type, task_filter)
    
    def filter_subjects_by_task_availability(self, 
                                           task_filter: Union[str, List[str]],
                                           data_type: str = 'timeseries',
                                           base_subjects: Optional[List[str]] = None) -> List[str]:
        """
        Filter subjects that have data available for specific tasks.
        
        Args:
            task_filter: Task name(s) to filter by (e.g., 'hammer', ['hammer', 'rest'])
            data_type: Type of data to check ('timeseries', 'motion')
            base_subjects: Starting set of subjects (all if None)
            
        Returns:
            List of subject IDs that have data for the specified tasks
        """
        if base_subjects is None:
            base_subjects = self.get_all_subjects()
        
        subjects_with_task_data = []
        
        for subject_id in base_subjects:
            try:
                task_files = self.get_subject_files_by_task(subject_id, data_type, task_filter)
                if task_files:  # Subject has at least one file for the specified task(s)
                    subjects_with_task_data.append(subject_id)
            except Exception:
                continue  # Skip subjects with missing data
        
        return subjects_with_task_data
    
    def get_subject_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Get cached or fresh metadata for subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Subject metadata dictionary
        """
        if subject_id not in self._subject_cache:
            self._subject_cache[subject_id] = self.data_loader.get_subject_metadata(subject_id)
        
        return self._subject_cache[subject_id].copy()
    
    def get_group_statistics(self, group_name: str) -> Dict[str, Any]:
        """
        Get descriptive statistics for analysis group.
        
        Args:
            group_name: Name of analysis group
            
        Returns:
            Dictionary with group statistics
        """
        subjects = self.get_group_subjects(group_name)
        
        if not subjects:
            return {
                'group_name': group_name,
                'total_subjects': 0,
                'statistics': {}
            }
        
        # Gather metadata for all subjects in group
        group_data = []
        for subject_id in subjects:
            try:
                metadata = self.get_subject_metadata(subject_id)
                
                # Flatten relevant fields
                row = {'subject_id': subject_id}
                
                demographics = metadata.get('demographics', {})
                row.update({
                    'age': demographics.get('age'),
                    'sex': demographics.get('sex'),
                    'site': demographics.get('site')
                })
                
                classifications = metadata.get('classifications', {})
                row.update({
                    'anhedonic_status': classifications.get('anhedonic_status'),
                    'mdd_status': classifications.get('mdd_status'),
                    'patient_control': classifications.get('patient_control')
                })
                
                availability = metadata.get('data_availability', {})
                row.update({
                    'has_timeseries': availability.get('has_timeseries', False),
                    'has_motion': availability.get('has_motion', False)
                })
                
                group_data.append(row)
                
            except Exception:
                continue
        
        if not group_data:
            return {
                'group_name': group_name,
                'total_subjects': len(subjects),
                'statistics': {'error': 'No valid subject data found'}
            }
        
        df = pd.DataFrame(group_data)
        
        statistics = {
            'total_subjects': len(subjects),
            'valid_metadata': len(df),
            'demographics': {},
            'classifications': {},
            'data_availability': {}
        }
        
        # Age statistics
        if 'age' in df.columns and df['age'].notna().any():
            age_data = df['age'].dropna()
            statistics['demographics']['age'] = {
                'count': len(age_data),
                'mean': float(age_data.mean()),
                'std': float(age_data.std()),
                'min': float(age_data.min()),
                'max': float(age_data.max())
            }
        
        # Sex distribution
        if 'sex' in df.columns:
            sex_counts = df['sex'].value_counts().to_dict()
            statistics['demographics']['sex_distribution'] = sex_counts
        
        # Site distribution
        if 'site' in df.columns:
            site_counts = df['site'].value_counts().to_dict()
            statistics['demographics']['site_distribution'] = site_counts
        
        # Classification distributions
        for col in ['anhedonic_status', 'mdd_status', 'patient_control']:
            if col in df.columns:
                counts = df[col].value_counts().to_dict()
                statistics['classifications'][col] = counts
        
        # Data availability
        for col in ['has_timeseries', 'has_motion']:
            if col in df.columns:
                statistics['data_availability'][col] = int(df[col].sum())
        
        return {
            'group_name': group_name,
            'statistics': statistics
        }
    
    def compare_groups(self, group1: str, group2: str) -> Dict[str, Any]:
        """
        Compare two analysis groups.
        
        Args:
            group1: First group name
            group2: Second group name
            
        Returns:
            Dictionary with group comparison
        """
        stats1 = self.get_group_statistics(group1)
        stats2 = self.get_group_statistics(group2)
        
        comparison = {
            'group1': group1,
            'group2': group2,
            'group1_size': stats1['statistics'].get('total_subjects', 0),
            'group2_size': stats2['statistics'].get('total_subjects', 0),
            'comparison': {}
        }
        
        # Compare demographics
        if ('demographics' in stats1['statistics'] and 
            'demographics' in stats2['statistics']):
            
            demo1 = stats1['statistics']['demographics']
            demo2 = stats2['statistics']['demographics']
            
            # Age comparison
            if 'age' in demo1 and 'age' in demo2:
                comparison['comparison']['age'] = {
                    'group1_mean': demo1['age']['mean'],
                    'group2_mean': demo2['age']['mean'],
                    'difference': demo1['age']['mean'] - demo2['age']['mean']
                }
            
            # Sex distribution comparison
            if 'sex_distribution' in demo1 and 'sex_distribution' in demo2:
                comparison['comparison']['sex_distribution'] = {
                    'group1': demo1['sex_distribution'],
                    'group2': demo2['sex_distribution']
                }
        
        # Compare data availability
        if ('data_availability' in stats1['statistics'] and 
            'data_availability' in stats2['statistics']):
            
            avail1 = stats1['statistics']['data_availability']
            avail2 = stats2['statistics']['data_availability']
            
            comparison['comparison']['data_availability'] = {
                'group1': avail1,
                'group2': avail2
            }
        
        return comparison
    
    def get_summary_dataframe(self, subject_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comprehensive summary DataFrame.
        
        Args:
            subject_ids: Specific subjects to include (all if None)
            
        Returns:
            DataFrame with subject summary
        """
        if subject_ids is None and self._summary_cache is not None:
            return self._summary_cache.copy()
        
        df = self.data_loader.get_subjects_summary(subject_ids)
        
        if subject_ids is None:
            self._summary_cache = df.copy()
        
        return df
    
    def create_subject_subset(self, 
                             selection_criteria: Dict[str, Any],
                             max_subjects: Optional[int] = None,
                             random_seed: Optional[int] = None) -> List[str]:
        """
        Create balanced subject subset based on criteria.
        
        Args:
            selection_criteria: Criteria for subject selection
            max_subjects: Maximum number of subjects to select
            random_seed: Random seed for reproducible selection
            
        Returns:
            List of selected subject IDs
        """
        # Apply filters to get candidate subjects
        candidates = self.filter_subjects(**selection_criteria)
        
        if max_subjects is None or len(candidates) <= max_subjects:
            return candidates
        
        # Random sampling if more subjects than requested
        if random_seed is not None:
            np.random.seed(random_seed)
        
        selected = np.random.choice(candidates, size=max_subjects, replace=False)
        return sorted(selected.tolist())
    
    def validate_subject_selection(self, subject_ids: List[str]) -> Dict[str, Any]:
        """
        Validate a subject selection for data completeness.
        
        Args:
            subject_ids: List of subject IDs to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_subjects': len(subject_ids),
            'valid_subjects': 0,
            'missing_subjects': [],
            'data_availability': {
                'timeseries': 0,
                'motion': 0,
                'phenotype': 0
            },
            'classification_distribution': {},
            'demographic_summary': {}
        }
        
        valid_subjects = []
        
        for subject_id in subject_ids:
            try:
                metadata = self.get_subject_metadata(subject_id)
                valid_subjects.append(subject_id)
                validation_report['valid_subjects'] += 1
                
                # Count data availability
                availability = metadata.get('data_availability', {})
                for data_type in ['timeseries', 'motion', 'phenotype']:
                    if availability.get(f'has_{data_type}', False):
                        validation_report['data_availability'][data_type] += 1
                
            except DataNotFoundError:
                validation_report['missing_subjects'].append(subject_id)
        
        # Analyze valid subjects
        if valid_subjects:
            df = self.get_summary_dataframe(valid_subjects)
            
            # Classification distributions
            for col in ['anhedonic_status', 'mdd_status', 'patient_control']:
                if col in df.columns:
                    validation_report['classification_distribution'][col] = df[col].value_counts().to_dict()
            
            # Demographic summary
            if 'age' in df.columns:
                age_data = df['age'].dropna()
                if len(age_data) > 0:
                    validation_report['demographic_summary']['age'] = {
                        'mean': float(age_data.mean()),
                        'std': float(age_data.std()),
                        'range': [float(age_data.min()), float(age_data.max())]
                    }
            
            if 'sex' in df.columns:
                validation_report['demographic_summary']['sex_distribution'] = df['sex'].value_counts().to_dict()
        
        return validation_report
    
    def get_download_status(self) -> Dict[str, Any]:
        """
        Get status of downloaded data.
        
        Returns:
            Dictionary with download status and data availability
        """
        try:
            return get_download_status_report()
        except Exception as e:
            return {
                'error': str(e),
                'download_summary': {'total_downloaded': 0},
                'data_availability': {'timeseries_available': 0, 'raw_nifti_available': 0}
            }
    
    def get_subjects_availability_summary(self, subject_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary of subject availability (total vs downloaded vs with data).
        
        Args:
            subject_ids: Specific subjects to analyze (all if None)
            
        Returns:
            Dictionary with availability breakdown
        """
        if subject_ids is None:
            subject_ids = self.get_all_subjects()
        
        downloaded_subjects = set(self.get_downloaded_subjects())
        timeseries_subjects = set(self.get_subjects_with_data('timeseries'))
        
        summary = {
            'total_subjects': len(subject_ids),
            'downloaded_subjects': len(downloaded_subjects & set(subject_ids)),
            'with_timeseries_data': len(timeseries_subjects & set(subject_ids)),
            'downloaded_with_timeseries': len(downloaded_subjects & timeseries_subjects & set(subject_ids)),
            'breakdown': {
                'available_in_manifest': len(set(subject_ids)),
                'downloaded_locally': len(downloaded_subjects & set(subject_ids)),
                'has_timeseries_metadata': len(timeseries_subjects & set(subject_ids)),
                'ready_for_processing': len(downloaded_subjects & timeseries_subjects & set(subject_ids))
            }
        }
        
        # Calculate percentages
        if summary['total_subjects'] > 0:
            summary['percentages'] = {
                'downloaded': (summary['downloaded_subjects'] / summary['total_subjects']) * 100,
                'with_timeseries': (summary['with_timeseries_data'] / summary['total_subjects']) * 100,
                'ready_for_processing': (summary['breakdown']['ready_for_processing'] / summary['total_subjects']) * 100
            }
        else:
            summary['percentages'] = {'downloaded': 0, 'with_timeseries': 0, 'ready_for_processing': 0}
        
        return summary