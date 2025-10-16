#!/usr/bin/env python3
"""
Phenotype filtering system with dependency injection for TCP dataset preprocessing.

Provides extensible, configurable filtering of subjects based on phenotype data
(demographics, diagnoses, clinical measures) while maintaining data integrity.

Author: Ian Philip Eglin
Date: 2025-09-23
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import re


class FilterAction(Enum):
    """Defines what action to take with filtered subjects"""
    INCLUDE = "include"
    EXCLUDE = "exclude"


@dataclass
class PhenotypeFilterResult:
    """Results from applying a phenotype filter to subjects"""
    included_subjects: pd.DataFrame
    excluded_subjects: pd.DataFrame
    criteria_description: str
    inclusion_reasons: Dict[str, str] = field(default_factory=dict)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)


class PhenotypeFilter(ABC):
    """Abstract base class for phenotype filtering with dependency injection"""

    def __init__(self, filter_name: str, description: str):
        self.filter_name = filter_name
        self.description = description

    @abstractmethod
    def apply(self, subjects_df: pd.DataFrame, phenotype_data: Dict[str, pd.DataFrame], **kwargs) -> PhenotypeFilterResult:
        """
        Apply filter to subjects dataframe using phenotype data

        Args:
            subjects_df: DataFrame with subject information (from validate_subjects.py)
            phenotype_data: Dictionary of phenotype DataFrames (e.g., 'demos', 'assessment')
            **kwargs: Additional filter-specific parameters

        Returns:
            PhenotypeFilterResult with included/excluded subjects and metadata
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


class ColumnValueFilter(PhenotypeFilter):
    """Generic filter for column values in phenotype data"""

    def __init__(self, 
                 phenotype_file: str,
                 column_name: str,
                 filter_values: Union[str, List[str]],
                 match_type: str = 'exact',
                 case_sensitive: bool = False,
                 action: FilterAction = FilterAction.INCLUDE):
        """
        Initialize column value filter

        Args:
            phenotype_file: Name of phenotype file (e.g., 'demos', 'assessment')
            column_name: Name of column to filter on
            filter_values: Value(s) to match (string or list of strings)
            match_type: 'exact', 'contains', 'startswith', 'endswith', 'regex'
            case_sensitive: Whether matching should be case sensitive
            action: Whether to INCLUDE or EXCLUDE matching subjects
        """
        self.phenotype_file = phenotype_file
        self.column_name = column_name
        self.filter_values = [filter_values] if isinstance(filter_values, str) else filter_values
        self.match_type = match_type
        self.case_sensitive = case_sensitive
        self.action = action

        filter_name = f"ColumnValueFilter_{phenotype_file}_{column_name}"
        description = f"Filter subjects based on {column_name} in {phenotype_file}.tsv"
        super().__init__(filter_name, description)

    def _matches_criteria(self, value: Any) -> bool:
        """Check if a value matches the filter criteria"""
        if pd.isna(value):
            return False

        value_str = str(value)
        if not self.case_sensitive:
            value_str = value_str.lower()
            comparison_values = [str(v).lower() for v in self.filter_values]
        else:
            comparison_values = [str(v) for v in self.filter_values]

        if self.match_type == 'exact':
            return value_str in comparison_values
        elif self.match_type == 'contains':
            return any(comp_val in value_str for comp_val in comparison_values)
        elif self.match_type == 'startswith':
            return any(value_str.startswith(comp_val) for comp_val in comparison_values)
        elif self.match_type == 'endswith':
            return any(value_str.endswith(comp_val) for comp_val in comparison_values)
        elif self.match_type == 'regex':
            return any(re.search(comp_val, value_str) for comp_val in comparison_values)
        else:
            raise ValueError(f"Unknown match_type: {self.match_type}")

    def apply(self, subjects_df: pd.DataFrame, phenotype_data: Dict[str, pd.DataFrame], **kwargs) -> PhenotypeFilterResult:
        """Apply column value filtering"""
        subjects_df = subjects_df.copy()

        # Check if required phenotype file exists
        if self.phenotype_file not in phenotype_data:
            raise ValueError(f"Phenotype file '{self.phenotype_file}' not found in phenotype_data. "
                           f"Available files: {list(phenotype_data.keys())}")

        phenotype_df = phenotype_data[self.phenotype_file]

        # Check if required column exists
        if self.column_name not in phenotype_df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in {self.phenotype_file}.tsv. "
                           f"Available columns: {list(phenotype_df.columns)}")

        # Merge subjects with phenotype data on participant_id
        if 'participant_id' not in phenotype_df.columns:
            # Try to find the subject ID column
            possible_id_columns = ['subject_id', 'Subject', 'ID', 'subjectkey']
            id_column = None
            for col in possible_id_columns:
                if col in phenotype_df.columns:
                    id_column = col
                    break
            
            if id_column is None:
                raise ValueError(f"No subject ID column found in {self.phenotype_file}.tsv. "
                               f"Looked for: participant_id, {possible_id_columns}")
            
            # Rename the column to match
            phenotype_df = phenotype_df.copy()
            phenotype_df['participant_id'] = phenotype_df[id_column]

        # Convert subject_id format if needed (sub-NDARXXX -> NDARXXX or vice versa)
        subjects_ids = set(subjects_df['subject_id'].tolist())
        phenotype_ids = set(phenotype_df['participant_id'].tolist())
        
        # Check if we need to convert formats
        if not subjects_ids.intersection(phenotype_ids):
            # Try multiple conversion strategies
            subjects_df = subjects_df.copy()
            
            if subjects_df['subject_id'].iloc[0].startswith('sub-'):
                # Try multiple conversions for BIDS format subjects
                # 1. Convert sub-NDARXXX to NDARXXX
                converted_ids = subjects_df['subject_id'].str.replace('sub-', '')
                
                # 2. If still no match, try sub-NDAR-XXX to NDAR_XXX (hyphens to underscores)
                if not set(converted_ids).intersection(phenotype_ids):
                    converted_ids = subjects_df['subject_id'].str.replace('sub-NDAR', 'NDAR_').str.replace('-', '_')
                
                subjects_df['participant_id_for_merge'] = converted_ids
            else:
                # Convert NDARXXX to sub-NDARXXX
                subjects_df['participant_id_for_merge'] = 'sub-' + subjects_df['subject_id']
        else:
            subjects_df['participant_id_for_merge'] = subjects_df['subject_id']

        # Merge dataframes
        merged_df = subjects_df.merge(
            phenotype_df[['participant_id', self.column_name]], 
            left_on='participant_id_for_merge', 
            right_on='participant_id', 
            how='left'
        )

        # Apply filtering criteria
        inclusion_reasons = {}
        exclusion_reasons = {}
        
        # Check each subject
        matches_criteria = merged_df[self.column_name].apply(self._matches_criteria)
        
        if self.action == FilterAction.INCLUDE:
            # Include subjects that match criteria
            included_mask = matches_criteria
            included_subjects = merged_df[included_mask].copy()
            excluded_subjects = merged_df[~included_mask].copy()
            
            # Generate reasons
            for _, subject in included_subjects.iterrows():
                subj_id = subject['subject_id']
                value = subject[self.column_name]
                inclusion_reasons[subj_id] = f"{self.column_name}='{value}' matches criteria"
            
            for _, subject in excluded_subjects.iterrows():
                subj_id = subject['subject_id']
                value = subject[self.column_name]
                if pd.isna(value):
                    exclusion_reasons[subj_id] = f"{self.column_name} is missing/NaN"
                else:
                    exclusion_reasons[subj_id] = f"{self.column_name}='{value}' does not match criteria"
        
        else:  # FilterAction.EXCLUDE
            # Exclude subjects that match criteria
            excluded_mask = matches_criteria
            included_subjects = merged_df[~excluded_mask].copy()
            excluded_subjects = merged_df[excluded_mask].copy()
            
            # Generate reasons
            for _, subject in included_subjects.iterrows():
                subj_id = subject['subject_id']
                value = subject[self.column_name]
                inclusion_reasons[subj_id] = f"{self.column_name}='{value}' does not match exclusion criteria"
            
            for _, subject in excluded_subjects.iterrows():
                subj_id = subject['subject_id']
                value = subject[self.column_name]
                exclusion_reasons[subj_id] = f"{self.column_name}='{value}' matches exclusion criteria"

        # Remove merge columns
        columns_to_keep = [col for col in subjects_df.columns if col != 'participant_id_for_merge']
        included_subjects = included_subjects[columns_to_keep]
        excluded_subjects = excluded_subjects[columns_to_keep]

        # Calculate statistics
        total_subjects = len(merged_df)
        value_counts = merged_df[self.column_name].value_counts(dropna=False)
        
        statistics = {
            "total_subjects": total_subjects,
            "included_subjects": len(included_subjects),
            "excluded_subjects": len(excluded_subjects),
            "inclusion_rate": len(included_subjects) / total_subjects if total_subjects > 0 else 0,
            "value_distribution": value_counts.to_dict(),
            "missing_values": int(merged_df[self.column_name].isna().sum()),
            "unique_values": int(merged_df[self.column_name].nunique(dropna=False))
        }

        criteria_description = self.get_criteria_description()

        return PhenotypeFilterResult(
            included_subjects=included_subjects,
            excluded_subjects=excluded_subjects,
            criteria_description=criteria_description,
            inclusion_reasons=inclusion_reasons,
            exclusion_reasons=exclusion_reasons,
            statistics=statistics
        )

    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        action_text = "include" if self.action == FilterAction.INCLUDE else "exclude"
        values_text = ", ".join([f"'{v}'" for v in self.filter_values])
        
        return (f"{action_text.capitalize()} subjects where {self.column_name} "
                f"{self.match_type} {values_text} (case {'sensitive' if self.case_sensitive else 'insensitive'})")


class PrimaryDiagnosisFilter(ColumnValueFilter):
    """Specialized filter for Primary_Dx column to identify MDD subjects"""

    def __init__(self, include_mdd: bool = True, include_control: bool = True):
        """
        Initialize Primary Diagnosis filter for MDD identification

        Args:
            include_mdd: Whether to include subjects with MDD diagnosis
            include_control: Whether to include subjects with control diagnosis (999)
        """
        self.include_mdd = include_mdd
        self.include_control = include_control

        # Define filter values based on what to include
        filter_values = []
        if include_control:
            filter_values.append("999")  # Control code
        if include_mdd:
            filter_values.append("MDD")  # MDD diagnosis

        super().__init__(
            phenotype_file='demos',
            column_name='Primary_Dx',
            filter_values=filter_values,
            match_type='contains',  # Use contains to catch variations like "MDD, recurrent"
            case_sensitive=False,
            action=FilterAction.INCLUDE
        )

        # Override filter name and description
        self.filter_name = "PrimaryDiagnosisFilter"
        if include_mdd and include_control:
            self.description = "Filter for MDD patients and controls based on Primary_Dx"
        elif include_mdd:
            self.description = "Filter for MDD patients only based on Primary_Dx"
        elif include_control:
            self.description = "Filter for control subjects only based on Primary_Dx"
        else:
            self.description = "Filter that excludes all subjects (no valid diagnosis criteria)"

    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        criteria_parts = []
        if self.include_control:
            criteria_parts.append("Primary_Dx = '999' (controls)")
        if self.include_mdd:
            criteria_parts.append("Primary_Dx contains 'MDD' (patients)")

        if criteria_parts:
            return f"Include subjects with: {' OR '.join(criteria_parts)}"
        else:
            return "Exclude all subjects (no valid criteria specified)"

class ShapsCompletionFilter(ColumnValueFilter):
    """Specialized filter for shaps_total column to identify subjects that have not completed the Snaith-Hamilton Please Scale(SHAPS) questionnaire"""

    def __init__(self, exclude_incomplete: bool = True):
        """
        Initialize Shaps Completion filter for all subjects

        Args:
            exclude_incomplete: Whether to exclude subjects with "Missing/NK/NA" questionnaires (999)
        """
        self.exclude_incomplete = exclude_incomplete

        # Define filter values based on what to include
        filter_values = []
        if exclude_incomplete:
            filter_values.append("999")

        super().__init__(
            phenotype_file='shaps01',
            column_name='shaps_total',
            filter_values=filter_values,
            match_type='contains',
            case_sensitive=False,
            action=FilterAction.EXCLUDE
        )

        # Override filter name and description
        self.filter_name = "SHAPSCompletionFilter"
        if not exclude_incomplete:
            self.description = "Filter for subjects entered in the SHAPS questionnaire, but not necessarily completed it"
        else: 
            self.description = "Filter for subjects who have not completed the SHAPS questionnaire"

    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        criteria_parts = []
        if self.exclude_incomplete:
            criteria_parts.append("shaps_total = '999' (missing questionnaire)")

        if criteria_parts:
            return f"Exclude subjects exactly matching: {' OR '.join(criteria_parts)}"
        else:
            return "Include all subjects (no valid criteria specified)"

class AgeRangeFilter(PhenotypeFilter):
    """Filter subjects based on age range"""

    def __init__(self, 
                 min_age: Optional[float] = None, 
                 max_age: Optional[float] = None,
                 age_column: str = 'age',
                 phenotype_file: str = 'demos'):
        """
        Initialize age range filter

        Args:
            min_age: Minimum age (inclusive)
            max_age: Maximum age (inclusive)
            age_column: Name of age column
            phenotype_file: Name of phenotype file containing age data
        """
        self.min_age = min_age
        self.max_age = max_age
        self.age_column = age_column
        self.phenotype_file = phenotype_file

        filter_name = "AgeRangeFilter"
        description = f"Filter subjects based on age range ({age_column} in {phenotype_file})"
        super().__init__(filter_name, description)

    def apply(self, subjects_df: pd.DataFrame, phenotype_data: Dict[str, pd.DataFrame], **kwargs) -> PhenotypeFilterResult:
        """Apply age range filtering"""
        subjects_df = subjects_df.copy()

        # Check if required phenotype file exists
        if self.phenotype_file not in phenotype_data:
            raise ValueError(f"Phenotype file '{self.phenotype_file}' not found in phenotype_data")

        phenotype_df = phenotype_data[self.phenotype_file]

        # Check if age column exists
        if self.age_column not in phenotype_df.columns:
            raise ValueError(f"Age column '{self.age_column}' not found in {self.phenotype_file}.tsv")

        # Handle subject ID matching (similar to ColumnValueFilter)
        if 'participant_id' not in phenotype_df.columns:
            possible_id_columns = ['subject_id', 'Subject', 'ID', 'subjectkey']
            id_column = None
            for col in possible_id_columns:
                if col in phenotype_df.columns:
                    id_column = col
                    break
            
            if id_column is None:
                raise ValueError(f"No subject ID column found in {self.phenotype_file}.tsv")
            
            phenotype_df = phenotype_df.copy()
            phenotype_df['participant_id'] = phenotype_df[id_column]

        # Handle subject ID format conversion (same logic as ColumnValueFilter)
        subjects_ids = set(subjects_df['subject_id'].tolist())
        phenotype_ids = set(phenotype_df['participant_id'].tolist())
        
        if not subjects_ids.intersection(phenotype_ids):
            # Try multiple conversion strategies
            subjects_df = subjects_df.copy()
            
            if subjects_df['subject_id'].iloc[0].startswith('sub-'):
                # Try multiple conversions for BIDS format subjects
                # 1. Convert sub-NDARXXX to NDARXXX
                converted_ids = subjects_df['subject_id'].str.replace('sub-', '')
                
                # 2. If still no match, try sub-NDAR-XXX to NDAR_XXX (hyphens to underscores)
                if not set(converted_ids).intersection(phenotype_ids):
                    converted_ids = subjects_df['subject_id'].str.replace('sub-NDAR', 'NDAR_').str.replace('-', '_')
                
                subjects_df['participant_id_for_merge'] = converted_ids
            else:
                # Convert NDARXXX to sub-NDARXXX
                subjects_df['participant_id_for_merge'] = 'sub-' + subjects_df['subject_id']
        else:
            subjects_df['participant_id_for_merge'] = subjects_df['subject_id']

        # Merge dataframes
        merged_df = subjects_df.merge(
            phenotype_df[['participant_id', self.age_column]], 
            left_on='participant_id_for_merge', 
            right_on='participant_id', 
            how='left'
        )

        # Apply age filtering
        inclusion_reasons = {}
        exclusion_reasons = {}

        # Create age filter mask
        age_mask = pd.Series([True] * len(merged_df))
        
        if self.min_age is not None:
            age_mask &= (merged_df[self.age_column] >= self.min_age)
        
        if self.max_age is not None:
            age_mask &= (merged_df[self.age_column] <= self.max_age)

        # Handle NaN ages
        age_mask &= merged_df[self.age_column].notna()

        included_subjects = merged_df[age_mask].copy()
        excluded_subjects = merged_df[~age_mask].copy()

        # Generate reasons
        for _, subject in included_subjects.iterrows():
            subj_id = subject['subject_id']
            age = subject[self.age_column]
            inclusion_reasons[subj_id] = f"Age {age} is within range"

        for _, subject in excluded_subjects.iterrows():
            subj_id = subject['subject_id']
            age = subject[self.age_column]
            if pd.isna(age):
                exclusion_reasons[subj_id] = "Age is missing/NaN"
            elif self.min_age is not None and age < self.min_age:
                exclusion_reasons[subj_id] = f"Age {age} below minimum {self.min_age}"
            elif self.max_age is not None and age > self.max_age:
                exclusion_reasons[subj_id] = f"Age {age} above maximum {self.max_age}"
            else:
                exclusion_reasons[subj_id] = f"Age {age} outside specified range"

        # Remove merge columns
        columns_to_keep = [col for col in subjects_df.columns if col != 'participant_id_for_merge']
        included_subjects = included_subjects[columns_to_keep]
        excluded_subjects = excluded_subjects[columns_to_keep]

        # Calculate statistics
        total_subjects = len(merged_df)
        valid_ages = merged_df[self.age_column].dropna()
        
        statistics = {
            "total_subjects": total_subjects,
            "included_subjects": len(included_subjects),
            "excluded_subjects": len(excluded_subjects),
            "inclusion_rate": len(included_subjects) / total_subjects if total_subjects > 0 else 0,
            "age_statistics": {
                "mean_age": float(valid_ages.mean()) if len(valid_ages) > 0 else None,
                "median_age": float(valid_ages.median()) if len(valid_ages) > 0 else None,
                "min_age": float(valid_ages.min()) if len(valid_ages) > 0 else None,
                "max_age": float(valid_ages.max()) if len(valid_ages) > 0 else None,
                "std_age": float(valid_ages.std()) if len(valid_ages) > 0 else None
            },
            "missing_ages": int(merged_df[self.age_column].isna().sum())
        }

        criteria_description = self.get_criteria_description()

        return PhenotypeFilterResult(
            included_subjects=included_subjects,
            excluded_subjects=excluded_subjects,
            criteria_description=criteria_description,
            inclusion_reasons=inclusion_reasons,
            exclusion_reasons=exclusion_reasons,
            statistics=statistics
        )

    def get_criteria_description(self) -> str:
        """Get human-readable description of filter criteria"""
        if self.min_age is not None and self.max_age is not None:
            return f"Age between {self.min_age} and {self.max_age} years (inclusive)"
        elif self.min_age is not None:
            return f"Age >= {self.min_age} years"
        elif self.max_age is not None:
            return f"Age <= {self.max_age} years"
        else:
            return "No age restrictions (include all ages)"