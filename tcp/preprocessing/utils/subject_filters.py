#!/usr/bin/env python3
"""
Subject filtering system with dependency injection for TCP dataset preprocessing.

Provides extensible, configurable filtering of subjects based on task data availability
while maintaining data integrity and traceability.

Author: Ian Philip Eglin
Date: 2025-09-12
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class FilterAction(Enum):
    """Defines what action to take with filtered subjects"""
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class FilterResult:
    """Results from applying a filter to subjects"""
    included_subjects: pd.DataFrame
    excluded_subjects: pd.DataFrame
    criteria_description: str
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)


class SubjectFilter(ABC):
    """Abstract base class for subject filtering with dependency injection"""

    def __init__(self, filter_name: str, description: str):
        self.filter_name = filter_name
        self.description = description

    @abstractmethod
    def apply(self, subjects_df: pd.DataFrame, file_paths: Dict, **kwargs) -> FilterResult:
        """
        Apply filter to subjects dataframe

        Args:
            subjects_df: DataFrame with subject information
            file_paths: Dictionary with task file paths per subject
            **kwargs: Additional filter-specific parameters

        Returns:
            FilterResult with included/excluded subjects and metadata
        """
        pass

    @abstractmethod
    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        pass

    def get_filter_info(self) -> Dict[str, str]:
        """Get filter metadata"""
        return {
            "name": self.filter_name,
            "description": self.description,
            "criteria": self.get_criteria_description()
        }


class TaskAvailabilityFilter(SubjectFilter):
    """Filter subjects based on task data availability"""

    def __init__(self,
                 required_tasks: List[str] = None,
                 require_all_tasks: bool = False,
                 data_type: str = 'raw_nifti',
                 data_types: List[str] = None):
        """
        Initialize task availability filter

        Args:
            required_tasks: List of tasks that must be present (e.g., ['hammer', 'stroop'])
            require_all_tasks: If True, subject must have ALL tasks. If False, subject needs ANY task
            data_type: Type of data to check ('raw_nifti', 'timeseries', etc.) - used when data_types is None
            data_types: List of data types to check. If provided, overrides data_type parameter
        """
        self.required_tasks = required_tasks or ['hammer', 'stroop']
        self.require_all_tasks = require_all_tasks
        
        # Support both single data_type and multiple data_types
        if data_types is not None:
            self.data_types = data_types
            self.data_type = data_types[0]  # Keep for backward compatibility
        else:
            self.data_types = [data_type]
            self.data_type = data_type

        filter_name = "TaskAvailabilityFilter"
        data_type_desc = ", ".join(self.data_types) if len(self.data_types) > 1 else self.data_types[0]
        description = f"Filter subjects based on {data_type_desc} task data availability"
        super().__init__(filter_name, description)

    def apply(self, subjects_df: pd.DataFrame, file_paths: Dict, **kwargs) -> FilterResult:
        """Apply task availability filtering"""
        subjects_df = subjects_df.copy()
        exclusion_reasons = {}

        # Create task availability columns if they don't exist
        for task in self.required_tasks:
            task_col = f'has_{task}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
            if task_col not in subjects_df.columns:
                # Calculate task availability from file_paths
                subjects_df[task_col] = subjects_df['participant_id'].apply(
                    lambda subj_id: self._has_task_data(subj_id, task, file_paths)
                )

        # Apply filtering logic
        if self.require_all_tasks:
            # Subject must have ALL required tasks
            task_mask = self._get_all_tasks_mask(subjects_df)
            logic_description = f"all of {self.required_tasks}"
        else:
            # Subject must have AT LEAST ONE required task
            task_mask = self._get_any_task_mask(subjects_df)
            logic_description = f"at least one of {self.required_tasks}"

        # Split subjects
        included_subjects = subjects_df[task_mask].copy()
        excluded_subjects = subjects_df[~task_mask].copy()

        # Generate exclusion reasons
        for _, subject in excluded_subjects.iterrows():
            subj_id = subject['participant_id']
            missing_tasks = []
            for task in self.required_tasks:
                task_col = f'has_{task}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
                if not subject.get(task_col, False):
                    missing_tasks.append(task)

            if self.require_all_tasks:
                reason = f"Missing required tasks: {missing_tasks}"
            else:
                reason = f"No task data available for any of: {self.required_tasks}"
            exclusion_reasons[subj_id] = reason

        # Calculate statistics
        statistics = {
            "total_subjects": len(subjects_df),
            "included_subjects": len(included_subjects),
            "excluded_subjects": len(excluded_subjects),
            "inclusion_rate": len(included_subjects) / len(subjects_df) if len(subjects_df) > 0 else 0,
            "task_breakdown": self._calculate_task_breakdown(subjects_df)
        }

        data_type_desc = ", ".join(self.data_types) if len(self.data_types) > 1 else self.data_types[0]
        criteria_description = f"Subjects must have {logic_description} ({data_type_desc} data)"

        return FilterResult(
            included_subjects=included_subjects,
            excluded_subjects=excluded_subjects,
            criteria_description=criteria_description,
            exclusion_reasons=exclusion_reasons,
            statistics=statistics
        )

    def _has_task_data(self, subject_id: str, task: str, file_paths: Dict) -> bool:
        """Check if subject has data for specific task across all configured data types"""
        for data_type in self.data_types:
            if data_type not in file_paths:
                continue

            if subject_id not in file_paths[data_type]:
                continue

            task_files = file_paths[data_type][subject_id].get(task, [])
            if len(task_files) > 0:
                return True
        
        return False

    def _get_all_tasks_mask(self, subjects_df: pd.DataFrame) -> pd.Series:
        """Get mask for subjects with ALL required tasks"""
        task_columns = []
        for task in self.required_tasks:
            task_col = f'has_{task}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
            task_columns.append(task_col)

        # Subject must have all tasks
        return subjects_df[task_columns].all(axis=1)

    def _get_any_task_mask(self, subjects_df: pd.DataFrame) -> pd.Series:
        """Get mask for subjects with ANY of the required tasks"""
        task_columns = []
        for task in self.required_tasks:
            task_col = f'has_{task}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
            task_columns.append(task_col)

        # Subject must have at least one task
        return subjects_df[task_columns].any(axis=1)

    def _calculate_task_breakdown(self, subjects_df: pd.DataFrame) -> Dict[str, int]:
        """Calculate breakdown of task availability"""
        breakdown = {}
        for task in self.required_tasks:
            task_col = f'has_{task}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
            breakdown[f"{task}_available"] = int(subjects_df[task_col].sum())

        # Calculate combinations
        if len(self.required_tasks) == 2:
            task1_col = f'has_{self.required_tasks[0]}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")
            task2_col = f'has_{self.required_tasks[1]}_{self.data_type.replace("_", "_")}'.replace("_nifti", "_raw")

            breakdown["both_tasks"] = int((subjects_df[task1_col] & subjects_df[task2_col]).sum())
            breakdown["only_first_task"] = int((subjects_df[task1_col] & ~subjects_df[task2_col]).sum())
            breakdown["only_second_task"] = int((~subjects_df[task1_col] & subjects_df[task2_col]).sum())
            breakdown["no_tasks"] = int((~subjects_df[task1_col] & ~subjects_df[task2_col]).sum())

        return breakdown

    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        logic = "all" if self.require_all_tasks else "any"
        data_type_desc = ", ".join(self.data_types) if len(self.data_types) > 1 else self.data_types[0]
        return f"Subjects must have {logic} of these tasks: {self.required_tasks} ({data_type_desc} data)"
