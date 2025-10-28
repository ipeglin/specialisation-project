#!/usr/bin/env python3
"""
Data loader for TCP processing pipeline with manifest-based file access.

Provides standardized interface for loading neuroimaging data without 
implementing analysis algorithms.

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config.processing_config import ProcessingConfig
from .utils.validation import validate_manifest, validate_file_paths
from .utils.file_utils import resolve_platform_path, check_file_exists, check_file_integrity
from .utils.error_handling import DataNotFoundError, ManifestError, ValidationError


class DataLoader:
    """
    Manifest-based data loader for TCP processing pipeline.
    
    Provides cross-platform file access and subject data management
    without implementing analysis algorithms.
    """
    
    def __init__(self, 
                 manifest_path: Optional[Path] = None,
                 config: Optional[ProcessingConfig] = None,
                 validate_on_load: bool = True):
        """
        Initialize DataLoader with manifest file.
        
        Args:
            manifest_path: Path to processing data manifest
            config: Processing configuration (uses default if None)
            validate_on_load: Whether to validate manifest on loading
        """
        self.config = config if config else ProcessingConfig()
        
        # Use provided manifest path or default from config
        if manifest_path:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.config.get_manifest_path()
        
        self.manifest: Optional[Dict[str, Any]] = None
        self.base_path: Optional[Path] = None
        self._subjects_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load and validate manifest
        self.load_manifest(validate=validate_on_load)
    
    def load_manifest(self, validate: bool = True) -> None:
        """
        Load data manifest from file.
        
        Args:
            validate: Whether to validate manifest structure
            
        Raises:
            ManifestError: If manifest is invalid or missing
        """
        if not self.manifest_path.exists():
            raise ManifestError(f"Manifest file not found: {self.manifest_path}")
        
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise ManifestError(f"Invalid JSON in manifest: {e}")
        except Exception as e:
            raise ManifestError(f"Cannot read manifest: {e}")
        
        # Validate manifest structure
        if validate:
            self.manifest = validate_manifest(self.manifest_path)
        
        # Set base path for file resolution
        path_config = self.manifest.get('path_configuration', {})
        dataset_root = path_config.get('dataset_root')
        
        if dataset_root:
            self.base_path = Path(dataset_root)
        else:
            self.base_path = self.config.get_dataset_path()
        
        print(f"[OK] Loaded manifest with {len(self.manifest['subjects'])} subjects")
        print(f"[OK] Base data path: {self.base_path}")
    
    def get_manifest_info(self) -> Dict[str, Any]:
        """
        Get metadata about loaded manifest.
        
        Returns:
            Dictionary with manifest information
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        metadata = self.manifest.get('manifest_metadata', {})
        return {
            'manifest_path': str(self.manifest_path),
            'created_timestamp': metadata.get('created_timestamp'),
            'total_subjects': metadata.get('total_subjects', 0),
            'analysis_groups': metadata.get('analysis_groups', []),
            'data_types_available': metadata.get('data_types_available', []),
            'base_path': str(self.base_path)
        }
    
    def get_all_subject_ids(self) -> List[str]:
        """
        Get list of all subject IDs in manifest.
        
        Returns:
            List of subject identifiers
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        return list(self.manifest['subjects'].keys())
    
    def get_analysis_groups(self) -> Dict[str, List[str]]:
        """
        Get analysis group definitions.
        
        Returns:
            Dictionary mapping group names to subject lists
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        return self.manifest.get('analysis_groups', {})
    
    def get_subjects_by_group(self, group_name: str) -> List[str]:
        """
        Get subject IDs for specific analysis group.
        
        Args:
            group_name: Name of analysis group
            
        Returns:
            List of subject IDs in group
            
        Raises:
            ValidationError: If group doesn't exist
        """
        groups = self.get_analysis_groups()
        
        if group_name not in groups:
            available_groups = list(groups.keys())
            raise ValidationError(f"Group '{group_name}' not found. Available: {available_groups}")
        
        return groups[group_name]
    
    def get_subject_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Get metadata for specific subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Dictionary with subject metadata
            
        Raises:
            DataNotFoundError: If subject not found
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        subjects = self.manifest['subjects']
        
        if subject_id not in subjects:
            raise DataNotFoundError(f"Subject {subject_id} not found in manifest")
        
        return subjects[subject_id]
    
    def get_subject_files(self, subject_id: str, data_type: str = 'timeseries') -> List[str]:
        """
        Get file paths for subject and data type.
        
        Args:
            subject_id: Subject identifier
            data_type: Type of data ('timeseries', 'motion')
            
        Returns:
            List of file paths
            
        Raises:
            DataNotFoundError: If subject or data type not found
        """
        subject_data = self.get_subject_metadata(subject_id)
        
        files_info = subject_data.get('files', {})
        if data_type not in files_info:
            available_types = list(files_info.keys())
            raise DataNotFoundError(
                f"Data type '{data_type}' not available for {subject_id}. Available: {available_types}"
            )
        
        return files_info[data_type].get('available', [])
    
    def get_subject_files_by_task(self, subject_id: str, data_type: str = 'timeseries', 
                                 task_filter: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Get file paths for subject and data type, optionally filtered by task.
        
        Args:
            subject_id: Subject identifier
            data_type: Type of data ('timeseries', 'motion')
            task_filter: Task name(s) to filter by (e.g., 'hammer', ['hammer', 'rest'])
            
        Returns:
            List of file paths matching the task filter
            
        Raises:
            DataNotFoundError: If subject or data type not found
        """
        # Get all files for the subject and data type
        all_files = self.get_subject_files(subject_id, data_type)
        
        # If no task filter specified, return all files
        if task_filter is None:
            return all_files
        
        # Normalize task_filter to a list
        if isinstance(task_filter, str):
            task_filter = [task_filter]
        
        # Filter files based on task names in the file paths
        filtered_files = []
        for file_path in all_files:
            # Extract task information from filename
            # Look for patterns like "task-hammer", "task-stroop", "task-rest"
            filename = Path(file_path).name.lower()
            
            # Check if any of the requested tasks appear in the filename
            for task in task_filter:
                task_pattern = f"task-{task.lower()}"
                if task_pattern in filename:
                    filtered_files.append(file_path)
                    break  # Don't add the same file multiple times
        
        return filtered_files
    
    def resolve_file_path(self, relative_path: str) -> Path:
        """
        Resolve relative file path to absolute path.
        
        Args:
            relative_path: Relative path from manifest
            
        Returns:
            Resolved absolute path
        """
        if not self.base_path:
            raise ManifestError("No base path configured")
        
        return resolve_platform_path(self.base_path, relative_path)
    
    def validate_subject_files(self, subject_id: str, data_type: str = 'timeseries') -> Dict[str, Any]:
        """
        Validate file availability for subject.
        
        Args:
            subject_id: Subject identifier
            data_type: Type of data to validate
            
        Returns:
            Validation report for subject files
        """
        file_paths = self.get_subject_files(subject_id, data_type)
        
        validation_report = {
            'subject_id': subject_id,
            'data_type': data_type,
            'total_files': len(file_paths),
            'accessible_files': [],
            'missing_files': [],
            'integrity_checks': []
        }
        
        for relative_path in file_paths:
            full_path = self.resolve_file_path(relative_path)
            
            if check_file_exists(full_path):
                validation_report['accessible_files'].append(str(full_path))
                
                # Basic integrity check
                try:
                    integrity = check_file_integrity(full_path)
                    validation_report['integrity_checks'].append({
                        'file': str(full_path),
                        'size_mb': integrity['size_mb'],
                        'readable': integrity['readable'],
                        'format': integrity['format']
                    })
                except Exception as e:
                    validation_report['integrity_checks'].append({
                        'file': str(full_path),
                        'error': str(e)
                    })
            else:
                validation_report['missing_files'].append(str(full_path))
        
        return validation_report
    
    def filter_subjects(self, 
                       groups: Optional[List[str]] = None,
                       classifications: Optional[Dict[str, Any]] = None,
                       data_requirements: Optional[List[str]] = None) -> List[str]:
        """
        Filter subjects based on criteria.
        
        Args:
            groups: Analysis groups to include
            classifications: Classification criteria (e.g., {'anhedonic_status': 'anhedonic'})
            data_requirements: Required data types (e.g., ['timeseries', 'motion'])
            
        Returns:
            List of subject IDs matching criteria
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        all_subjects = self.get_all_subject_ids()
        filtered_subjects = set(all_subjects)
        
        # Filter by analysis groups
        if groups:
            group_subjects = set()
            for group in groups:
                group_subjects.update(self.get_subjects_by_group(group))
            filtered_subjects &= group_subjects
        
        # Filter by classifications
        if classifications:
            for subject_id in list(filtered_subjects):
                subject_data = self.get_subject_metadata(subject_id)
                subject_classifications = subject_data.get('classifications', {})
                
                for key, value in classifications.items():
                    if subject_classifications.get(key) != value:
                        filtered_subjects.discard(subject_id)
                        break
        
        # Filter by data availability
        if data_requirements:
            for subject_id in list(filtered_subjects):
                subject_data = self.get_subject_metadata(subject_id)
                availability = subject_data.get('data_availability', {})
                
                for data_type in data_requirements:
                    availability_key = f'has_{data_type}'
                    if not availability.get(availability_key, False):
                        filtered_subjects.discard(subject_id)
                        break
        
        return sorted(list(filtered_subjects))
    
    def get_subjects_summary(self, subject_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary DataFrame of subjects and their metadata.
        
        Args:
            subject_ids: Specific subjects to include (all if None)
            
        Returns:
            DataFrame with subject summary information
        """
        if not self.manifest:
            raise ManifestError("No manifest loaded")
        
        if subject_ids is None:
            subject_ids = self.get_all_subject_ids()
        
        summary_data = []
        
        for subject_id in subject_ids:
            try:
                subject_data = self.get_subject_metadata(subject_id)
                
                row = {'subject_id': subject_id}
                
                # Add demographics
                demographics = subject_data.get('demographics', {})
                row.update({
                    'age': demographics.get('age'),
                    'sex': demographics.get('sex'),
                    'site': demographics.get('site'),
                    'group': demographics.get('group')
                })
                
                # Add classifications
                classifications = subject_data.get('classifications', {})
                row.update({
                    'anhedonia_class': classifications.get('anhedonia_class'),
                    'anhedonic_status': classifications.get('anhedonic_status'),
                    'mdd_status': classifications.get('mdd_status'),
                    'patient_control': classifications.get('patient_control')
                })
                
                # Add data availability
                availability = subject_data.get('data_availability', {})
                row.update({
                    'has_timeseries': availability.get('has_timeseries', False),
                    'has_motion': availability.get('has_motion', False),
                    'has_phenotype': availability.get('has_phenotype', False)
                })
                
                # Add group memberships
                memberships = subject_data.get('analysis_group_memberships', [])
                row['analysis_groups'] = ', '.join(memberships) if memberships else ''
                
                summary_data.append(row)
                
            except Exception as e:
                print(f"Warning: Could not process subject {subject_id}: {e}")
                continue
        
        return pd.DataFrame(summary_data)