#!/usr/bin/env python3
"""
TCP Preprocessing Pipeline Orchestrator

Runs the complete TCP anhedonia-focused preprocessing pipeline in the correct order with
error handling, progress tracking, and resumable execution.

The pipeline follows this sequence:
1. Dataset initialization and validation
2. Subject filtering (task fMRI + SHAPS completion)
3. Anhedonia classification (SHAPS scores)
4. Diagnosis classification (MDD status)
5. Analysis group generation (Primary/Secondary/Tertiary/Quaternary)
6. Subject sampling and file mapping
7. Cross-analysis integration and data download

Author: Ian Philip Eglin
Date: 2025-10-25
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path BEFORE importing local modules
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import local modules (after path is set up)
from config.paths import get_script_output_path
from tcp.preprocessing.utils.unicode_compat import (
    CHECK,
    CROSS,
    ERROR,
    PARTY,
    RUNNING,
    SKIP,
    SUCCESS,
)


class PipelineStep(Enum):
    """Pipeline step identifiers"""
    INITIALIZE_DATASET = "initialize_dataset"
    VALIDATE_SUBJECTS = "validate_subjects"
    FETCH_GLOBAL_DATA = "fetch_global_data"
    FILTER_SUBJECTS = "filter_subjects"
    FILTER_BASE_SUBJECTS = "filter_base_subjects"
    CLASSIFY_ANHEDONIA = "classify_anhedonia"
    CLASSIFY_DIAGNOSES = "classify_diagnoses"
    GENERATE_ANALYSIS_GROUPS = "generate_analysis_groups"
    SAMPLE_SUBJECTS = "sample_subjects"
    PARCELLATE_FMRIPREP = "parcellate_fmriprep"  # NEW - Option B only
    MAP_SUBJECT_FILES = "map_subject_files"
    INTEGRATE_CROSS_ANALYSIS = "integrate_cross_analysis"
    FETCH_FILTERED_DATA = "fetch_filtered_data"

class StepStatus(Enum):
    """Step execution status"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TCPPipeline:
    """TCP preprocessing pipeline orchestrator"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else get_script_output_path('tcp_preprocessing', 'pipeline')
        self.scripts_dir = Path(__file__).parent

        # Pipeline step definitions
        self.steps = {
            PipelineStep.INITIALIZE_DATASET: {
                'script': 'initialize_dataset.py',
                'description': 'Initialize and clone TCP dataset',
                'required': True,
                'estimated_time': '10-15 minutes',
                'timeout': 900  # 15 minutes
            },
            PipelineStep.VALIDATE_SUBJECTS: {
                'script': 'validate_subjects.py',
                'description': 'Validate subject directory structure',
                'required': True,
                'estimated_time': '2-5 minutes',
                'timeout': 900  # 15 minutes
            },
            PipelineStep.FETCH_GLOBAL_DATA: {
                'script': 'fetch_global_data.py',
                'description': 'Fetch participants.tsv and phenotype files',
                'required': True,
                'estimated_time': '1-3 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.FILTER_SUBJECTS: {
                'script': 'filter_subjects.py',
                'description': 'Filter subjects with task fMRI data (group-agnostic)',
                'required': True,
                'estimated_time': '5-15 minutes',
                'timeout': 1800  # 30 minutes
            },
            PipelineStep.FILTER_BASE_SUBJECTS: {
                'script': 'filter_base_subjects.py',
                'description': 'Apply universal inclusion criteria (SHAPS completion)',
                'required': True,
                'estimated_time': '1-2 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.CLASSIFY_ANHEDONIA: {
                'script': 'classify_anhedonia.py',
                'description': 'Classify subjects by anhedonia severity (SHAPS scores)',
                'required': True,
                'estimated_time': '1-2 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.CLASSIFY_DIAGNOSES: {
                'script': 'classify_diagnoses.py',
                'description': 'Classify subjects by MDD diagnosis status',
                'required': True,
                'estimated_time': '1-2 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.GENERATE_ANALYSIS_GROUPS: {
                'script': 'generate_analysis_groups.py',
                'description': 'Generate 4 analysis groups (Primary/Secondary/Tertiary/Quaternary)',
                'required': True,
                'estimated_time': '1-2 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.SAMPLE_SUBJECTS: {
                'script': 'sample_subjects_for_download.py',
                'description': 'Sample subjects for data download (development vs production)',
                'required': False,
                'estimated_time': '30 seconds',
                'timeout': 300  # 5 minutes
            },
            PipelineStep.PARCELLATE_FMRIPREP: {
                'script': 'fmriprep_parcellation.py',
                'description': 'Parcellate fMRIPrep BOLD data to 434-ROI timeseries (fMRIPrep mode only)',
                'required': False,  # Only required for fMRIPrep mode
                'estimated_time': '1-3 hours (depends on number of subjects and parallelization)',
                'timeout': 14400  # 4 hours
            },
            PipelineStep.MAP_SUBJECT_FILES: {
                'script': 'map_subject_files.py',
                'description': 'Map file paths for sampled/filtered subjects',
                'required': True,
                'estimated_time': '1-10 minutes',
                'timeout': 1200  # 20 minutes
            },
            PipelineStep.INTEGRATE_CROSS_ANALYSIS: {
                'script': 'integrate_cross_analysis.py',
                'description': 'Generate cross-analysis statistics and datasets',
                'required': False,
                'estimated_time': '1-2 minutes',
                'timeout': 600  # 10 minutes
            },
            PipelineStep.FETCH_FILTERED_DATA: {
                'script': 'fetch_filtered_data.py',
                'description': 'Download MRI data for selected subjects',
                'required': True,
                'estimated_time': '2-10 hours',
                'timeout': 36000  # 10 hours
            }
        }

        # Pipeline state
        self.step_status = {step: StepStatus.NOT_STARTED for step in PipelineStep}
        self.step_results = {}
        self.pipeline_start_time = None
        self.pipeline_end_time = None

        print(f"TCP Pipeline Orchestrator")
        print(f"Output directory: {self.output_dir}")

    def load_pipeline_state(self) -> bool:
        """Load existing pipeline state if available"""
        state_file = self.output_dir / 'pipeline_state.json'

        if not state_file.exists():
            return False

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Convert string keys back to enums
            for step_name, status_name in state.get('step_status', {}).items():
                try:
                    step = PipelineStep(step_name)
                    status = StepStatus(status_name)
                    self.step_status[step] = status
                except ValueError:
                    continue

            self.step_results = state.get('step_results', {})

            if 'pipeline_start_time' in state:
                self.pipeline_start_time = datetime.fromisoformat(state['pipeline_start_time'])

            print(f"{CHECK} Loaded existing pipeline state")
            return True

        except Exception as e:
            print(f"WARNING: Could not load pipeline state: {e}")
            return False

    def save_pipeline_state(self) -> None:
        """Save current pipeline state"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        state = {
            'step_status': {step.value: status.value for step, status in self.step_status.items()},
            'step_results': self.step_results,
            'pipeline_start_time': self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
            'pipeline_end_time': self.pipeline_end_time.isoformat() if self.pipeline_end_time else None,
            'last_updated': datetime.now().isoformat()
        }

        state_file = self.output_dir / 'pipeline_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def check_step_completed(self, step: PipelineStep, **kwargs) -> bool:
        """Check if a step has been completed successfully"""
        if step == PipelineStep.INITIALIZE_DATASET:
            # Check if dataset exists and is valid
            try:
                from tcp.preprocessing.initialize_dataset import DatasetInitializer
                initializer = DatasetInitializer()
                is_valid, _ = initializer.validate_existing_dataset()
                return is_valid
            except:
                return False

        elif step == PipelineStep.VALIDATE_SUBJECTS:
            # Check if validation output exists
            validation_dir = get_script_output_path('tcp_preprocessing', 'validate_subjects')
            return (validation_dir / 'valid_subjects.csv').exists()

        elif step == PipelineStep.FETCH_GLOBAL_DATA:
            # Check if global data has been fetched
            global_dir = get_script_output_path('tcp_preprocessing', 'fetch_global_data')
            status_file = global_dir / 'fetch_status.json'
            if status_file.exists():
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    return status.get('all_required_files_fetched', False)
                except:
                    return False
            return False

        elif step == PipelineStep.FILTER_SUBJECTS:
            # Check if subject filtering output exists (group-agnostic task filtering)
            filter_dir = get_script_output_path('tcp_preprocessing', 'filter_subjects')
            return (filter_dir / 'task_filtered_subjects.csv').exists()

        elif step == PipelineStep.FILTER_BASE_SUBJECTS:
            # Check if base subject filtering output exists (universal inclusion criteria)
            base_dir = get_script_output_path('tcp_preprocessing', 'filter_base_subjects')
            return (base_dir / 'base_filtered_subjects.csv').exists()

        elif step == PipelineStep.CLASSIFY_ANHEDONIA:
            # Check if anhedonia classification output exists
            anhedonia_dir = get_script_output_path('tcp_preprocessing', 'classify_anhedonia')
            return (anhedonia_dir / 'anhedonia_classified_subjects.csv').exists()

        elif step == PipelineStep.CLASSIFY_DIAGNOSES:
            # Check if diagnosis classification output exists
            diagnoses_dir = get_script_output_path('tcp_preprocessing', 'classify_diagnoses')
            return (diagnoses_dir / 'diagnosis_classified_subjects.csv').exists()

        elif step == PipelineStep.GENERATE_ANALYSIS_GROUPS:
            # Check if analysis groups generation output exists
            groups_dir = get_script_output_path('tcp_preprocessing', 'generate_analysis_groups')
            return (groups_dir / 'primary_analysis_subjects.csv').exists()

        elif step == PipelineStep.SAMPLE_SUBJECTS:
            # Check if subject sampling output exists (optional step)
            sample_dir = get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')
            return (sample_dir / 'sampled_subjects_for_download.csv').exists()

        elif step == PipelineStep.PARCELLATE_FMRIPREP:
            # Check if fMRIPrep parcellation output exists (fMRIPrep mode only)
            # This is considered complete if parcellated output directory has .h5 files
            parcellated_dir = kwargs.get('parcellated_output_dir')
            if parcellated_dir and Path(parcellated_dir).exists():
                h5_files = list(Path(parcellated_dir).glob('*.h5'))
                return len(h5_files) > 0
            return False

        elif step == PipelineStep.MAP_SUBJECT_FILES:
            # Check if file mapping output exists
            mapping_dir = get_script_output_path('tcp_preprocessing', 'map_subject_files')
            return (mapping_dir / 'subject_file_mapping.json').exists()

        elif step == PipelineStep.INTEGRATE_CROSS_ANALYSIS:
            # Check if cross-analysis integration output exists (optional step)
            cross_dir = get_script_output_path('tcp_preprocessing', 'integrate_cross_analysis')
            return (cross_dir / 'cross_analysis_master_summary.json').exists()

        elif step == PipelineStep.FETCH_FILTERED_DATA:
            # Check if MRI data fetching was completed
            fetch_dir = get_script_output_path('tcp_preprocessing', 'fetch_filtered_data')
            # Look for recent fetch report
            reports = list(fetch_dir.glob('fetch_report_*.json'))
            return len(reports) > 0

        return False

    def update_step_status(self, **kwargs) -> None:
        """Update step status based on output files"""
        for step in PipelineStep:
            if self.step_status[step] == StepStatus.NOT_STARTED:
                if self.check_step_completed(step, **kwargs):
                    self.step_status[step] = StepStatus.COMPLETED
                    print(f"  {CHECK} {step.value}: Already completed")

    def run_step(self, step: PipelineStep, dry_run: bool = False, **kwargs) -> bool:
        """Run a single pipeline step"""
        step_info = self.steps[step]
        script_path = self.scripts_dir / step_info['script']

        print(f"\n{'='*60}")
        print(f"RUNNING STEP: {step.value}")
        print(f"{'='*60}")
        print(f"Description: {step_info['description']}")
        print(f"Estimated time: {step_info['estimated_time']}")
        print(f"Script: {script_path}")

        if dry_run:
            print(f"[DRY RUN] Would execute: python {script_path}")
            return True

        # Mark step as running
        self.step_status[step] = StepStatus.RUNNING
        self.save_pipeline_state()

        try:
            # Build command
            cmd = [sys.executable, str(script_path)]

            # Add step-specific arguments
            if step == PipelineStep.SAMPLE_SUBJECTS:
                # Add subject sampling arguments if provided
                sample_mode = kwargs.get('sample_mode', 'development')
                cmd.extend(['--sample-mode', sample_mode])

                analysis_groups = kwargs.get('analysis_groups', ['primary'])
                if analysis_groups:
                    cmd.extend(['--analysis-groups'] + analysis_groups)

            elif step == PipelineStep.PARCELLATE_FMRIPREP:
                # Add fMRIPrep parcellation arguments
                fmriprep_root = kwargs.get('fmriprep_root')
                parcellated_output_dir = kwargs.get('parcellated_output_dir')

                if not fmriprep_root or not parcellated_output_dir:
                    raise ValueError("fMRIPrep parcellation requires --fmriprep-root and --parcellated-output-dir")

                # Convert paths to forward slashes to avoid escape sequence issues
                fmriprep_root_str = str(Path(fmriprep_root).as_posix())
                parcellated_output_str = str(Path(parcellated_output_dir).as_posix())

                cmd.extend(['--fmriprep-root', fmriprep_root_str])
                cmd.extend(['--output-dir', parcellated_output_str])

                # Add task and run range if specified
                if 'task' in kwargs:
                    cmd.extend(['--task', kwargs['task']])
                if 'run_start' in kwargs:
                    cmd.extend(['--run-start', str(kwargs['run_start'])])
                if 'run_end' in kwargs:
                    cmd.extend(['--run-end', str(kwargs['run_end'])])
                if 'n_jobs' in kwargs:
                    cmd.extend(['--n-jobs', str(kwargs['n_jobs'])])

                # Load subject IDs from sampled subjects file
                subject_ids = kwargs.get('parcellate_subject_ids', [])
                if not subject_ids:
                    # Auto-load from sampled_subjects_for_download.csv
                    import pandas as pd
                    sample_dir = get_script_output_path('tcp_preprocessing', 'sample_subjects_for_download')
                    sample_file = sample_dir / 'sampled_subjects_for_download.csv'

                    if sample_file.exists():
                        df = pd.read_csv(sample_file)
                        # Use subject_id column (BIDS format: sub-NDARINVXXXXX)
                        subject_ids = df['subject_id'].tolist()
                        print(f"  Loaded {len(subject_ids)} subject IDs from {sample_file.name}")
                    else:
                        raise ValueError(f"No subject IDs provided and sample file not found: {sample_file}")

                if subject_ids:
                    cmd.extend(['--subject-ids'] + subject_ids)

            elif step == PipelineStep.FETCH_FILTERED_DATA:
                # Add data fetching arguments if provided
                if 'data_types' in kwargs:
                    cmd.extend(['--data-types'] + kwargs['data_types'])
                if 'tasks' in kwargs:
                    cmd.extend(['--tasks'] + kwargs['tasks'])
                if kwargs.get('fetch_dry_run', False):
                    cmd.append('--dry-run')

            # Execute step
            step_timeout = step_info.get('timeout', 7200)  # Default 2 hours if not specified
            print(f"Executing: {' '.join(cmd)}")
            print(f"Timeout: {step_timeout} seconds ({step_timeout/3600:.1f} hours)")

            # Use real-time output for long-running steps (initialization and data fetching)
            if step in [PipelineStep.INITIALIZE_DATASET, PipelineStep.FETCH_FILTERED_DATA]:
                # Allow real-time output to show progress during long operations
                if step == PipelineStep.INITIALIZE_DATASET:
                    print("Starting dataset initialization with real-time output...")
                    print("(This may take 10-15 minutes for cloning...)")
                else:
                    print("Starting data fetch with real-time output...")

                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Show real-time output
                    text=True,
                    timeout=step_timeout
                )
                # Create result object compatible with existing code
                step_result = {
                    'command': ' '.join(cmd),
                    'return_code': result.returncode,
                    'stdout': f"Real-time output mode - check console above",
                    'stderr': f"Real-time output mode - errors shown in console",
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Use captured output for other steps
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=step_timeout
                )
                # Store results
                step_result = {
                    'command': ' '.join(cmd),
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }

            self.step_results[step.value] = step_result

            if result.returncode == 0:
                print(f"{CHECK} Step completed successfully")
                self.step_status[step] = StepStatus.COMPLETED
                self.save_pipeline_state()
                return True
            else:
                print(f"{CROSS} Step failed with return code {result.returncode}")
                # Only show stderr if it was captured (not in real-time mode)
                if step_result.get('stderr') and step_result['stderr'] != "Real-time output mode - errors shown in console":
                    print(f"STDERR: {step_result['stderr']}")
                self.step_status[step] = StepStatus.FAILED
                self.save_pipeline_state()
                return False

        except subprocess.TimeoutExpired:
            step_timeout = step_info.get('timeout', 7200)
            print(f"{CROSS} Step timed out after {step_timeout} seconds ({step_timeout/3600:.1f} hours)")
            self.step_status[step] = StepStatus.FAILED
            self.save_pipeline_state()
            return False
        except Exception as e:
            print(f"{CROSS} Step failed with exception: {e}")
            self.step_status[step] = StepStatus.FAILED
            self.save_pipeline_state()
            return False

    def run_pipeline(self,
                    start_from: Optional[PipelineStep] = None,
                    stop_at: Optional[PipelineStep] = None,
                    skip_optional: bool = False,
                    dry_run: bool = False,
                    **kwargs) -> bool:
        """Run the complete pipeline"""

        print(f"\n{'='*60}")
        print(f"TCP PREPROCESSING PIPELINE")
        print(f"{'='*60}")

        # Load existing state
        self.load_pipeline_state()

        # Update status based on existing files
        print("Checking existing pipeline state...")
        self.update_step_status(**kwargs)

        if self.pipeline_start_time is None:
            self.pipeline_start_time = datetime.now()

        # Determine steps to run based on data source
        data_source = kwargs.get('data_source', 'datalad')

        if data_source == 'fmriprep':
            # fMRIPrep mode: Skip INITIALIZE_DATASET and FETCH_FILTERED_DATA, add PARCELLATE_FMRIPREP
            steps_to_run = [
                PipelineStep.VALIDATE_SUBJECTS,
                PipelineStep.FETCH_GLOBAL_DATA,  # Still need phenotypes from datalad
                PipelineStep.FILTER_SUBJECTS,
                PipelineStep.FILTER_BASE_SUBJECTS,
                PipelineStep.CLASSIFY_ANHEDONIA,
                PipelineStep.CLASSIFY_DIAGNOSES,
                PipelineStep.GENERATE_ANALYSIS_GROUPS,
                PipelineStep.SAMPLE_SUBJECTS,  # Optional
                PipelineStep.PARCELLATE_FMRIPREP,  # NEW - parcellate fMRIPrep data
                PipelineStep.MAP_SUBJECT_FILES,
                PipelineStep.INTEGRATE_CROSS_ANALYSIS,  # Optional
                # Skip FETCH_FILTERED_DATA (data already local after parcellation)
            ]
        else:
            # Datalad mode: Original workflow
            steps_to_run = list(PipelineStep)
            # Remove PARCELLATE_FMRIPREP from datalad workflow
            if PipelineStep.PARCELLATE_FMRIPREP in steps_to_run:
                steps_to_run.remove(PipelineStep.PARCELLATE_FMRIPREP)

        if start_from:
            start_index = steps_to_run.index(start_from) if start_from in steps_to_run else 0
            steps_to_run = steps_to_run[start_index:]

        if stop_at:
            stop_index = steps_to_run.index(stop_at) if stop_at in steps_to_run else len(steps_to_run) - 1
            steps_to_run = steps_to_run[:stop_index + 1]

        if skip_optional:
            steps_to_run = [step for step in steps_to_run if self.steps[step]['required']]

        print(f"\nPipeline plan:")
        for i, step in enumerate(steps_to_run, 1):
            status = self.step_status[step]
            required = "Required" if self.steps[step]['required'] else "Optional"
            status_symbol = CHECK if status == StepStatus.COMPLETED else "o"
            print(f"  {i}. {status_symbol} {step.value} ({required}) - {status.value}")

        if dry_run:
            print(f"\n[DRY RUN] Pipeline execution plan complete")
            return True

        # Run steps
        print(f"\nStarting pipeline execution...")
        success = True

        for step in steps_to_run:
            # Skip completed steps
            if self.step_status[step] == StepStatus.COMPLETED:
                print(f"\n{SKIP} Skipping {step.value} (already completed)")
                continue

            # Skip optional steps if requested
            if skip_optional and not self.steps[step]['required']:
                print(f"\n{SKIP} Skipping {step.value} (optional step)")
                self.step_status[step] = StepStatus.SKIPPED
                continue

            # Run the step
            step_success = self.run_step(step, dry_run=False, **kwargs)

            if not step_success:
                if self.steps[step]['required']:
                    print(f"\n{ERROR} Required step {step.value} failed. Pipeline cannot continue.")
                    success = False
                    break
                else:
                    print(f"\nWARNING: Optional step {step.value} failed. Continuing pipeline.")
                    self.step_status[step] = StepStatus.FAILED

        # Mark pipeline completion
        self.pipeline_end_time = datetime.now()
        self.save_pipeline_state()

        # Print final summary
        self.print_pipeline_summary()

        return success

    def print_pipeline_summary(self) -> None:
        """Print pipeline execution summary"""
        print(f"\n{'='*60}")
        print(f"PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")

        # Calculate timing
        if self.pipeline_start_time and self.pipeline_end_time:
            duration = self.pipeline_end_time - self.pipeline_start_time
            print(f"Total execution time: {duration}")

        # Show step results
        completed_steps = sum(1 for status in self.step_status.values() if status == StepStatus.COMPLETED)
        failed_steps = sum(1 for status in self.step_status.values() if status == StepStatus.FAILED)
        skipped_steps = sum(1 for status in self.step_status.values() if status == StepStatus.SKIPPED)

        print(f"Steps completed: {completed_steps}")
        print(f"Steps failed: {failed_steps}")
        print(f"Steps skipped: {skipped_steps}")

        print(f"\nStep-by-step results:")
        for step in PipelineStep:
            status = self.step_status[step]
            status_symbol = {
                StepStatus.COMPLETED: SUCCESS,
                StepStatus.FAILED: ERROR,
                StepStatus.SKIPPED: SKIP,
                StepStatus.NOT_STARTED: "o",
                StepStatus.RUNNING: RUNNING
            }.get(status, "?")

            print(f"  {status_symbol} {step.value}: {status.value}")

        # Show next steps
        if failed_steps == 0:
            print(f"\n{PARTY} Pipeline completed successfully!")
            print(f"Your TCP dataset is ready for analysis.")
            print(f"Check the output directories for filtered subjects and downloaded data.")
        else:
            print(f"\nWARNING: Pipeline completed with {failed_steps} failed steps.")
            print(f"Check the logs and re-run failed steps if needed.")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run TCP preprocessing pipeline')
    parser.add_argument('--start-from', choices=[s.value for s in PipelineStep],
                       help='Start pipeline from specific step')
    parser.add_argument('--stop-at', choices=[s.value for s in PipelineStep],
                       help='Stop pipeline at specific step')
    parser.add_argument('--skip-optional', action='store_true',
                       help='Skip optional steps (e.g., phenotype filtering)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')
    parser.add_argument('--output-dir', type=Path,
                       help='Override pipeline output directory')

    # Phenotype filtering options
    parser.add_argument('--include-mdd', action='store_true', default=True,
                       help='Include MDD subjects in phenotype filtering')
    parser.add_argument('--include-controls', action='store_true', default=True,
                       help='Include control subjects in phenotype filtering')
    parser.add_argument('--min-age', type=float,
                       help='Minimum age for phenotype filtering')
    parser.add_argument('--max-age', type=float,
                       help='Maximum age for phenotype filtering')

    # Subject sampling options
    parser.add_argument('--sample-mode',
                       choices=['development', 'production', 'custom'],
                       default='development',
                       help='Sampling strategy: development (~15GB), production (~300GB), or custom (default: development)')
    parser.add_argument('--analysis-groups', nargs='+',
                       choices=['primary', 'secondary', 'tertiary', 'quaternary', 'all'],
                       default=['primary'],
                       help='Analysis groups to include in sampling (default: primary only)')

    # Data fetching options
    parser.add_argument('--data-types', nargs='+',
                       default=['raw_nifti', 'events', 'json_metadata', 'anatomical', 'anatomical_json', 'timeseries'],
                       help='Data types to fetch (default: ALL data types). Use this flag to restrict to specific data types only.')
    parser.add_argument('--tasks', nargs='+',
                       choices=['hammer', 'stroop'],
                       default=['hammer'],
                       help='Tasks to fetch (default: hammer only). Use this to include stroop or other tasks.')
    parser.add_argument('--fetch-dry-run', action='store_true',
                       help='Dry run for data fetching step')

    # Data source options (NEW - for fMRIPrep integration)
    parser.add_argument('--data-source',
                       choices=['datalad', 'fmriprep'],
                       default='datalad',
                       help='Data source type: datalad (existing workflow) or fmriprep (custom parcellation)')
    parser.add_argument('--fmriprep-root', type=Path,
                       help='Root directory of fMRIPrep output (required for fmriprep mode)')
    parser.add_argument('--parcellated-output-dir', type=Path,
                       help='Output directory for parcellated .h5 files (required for fmriprep mode)')
    parser.add_argument('--run-start', type=int, default=1,
                       help='First run number for parcellation (default: 1)')
    parser.add_argument('--run-end', type=int, default=9,
                       help='Last run number for parcellation (default: 9)')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel jobs for parcellation (default: 4)')

    args = parser.parse_args()

    # Convert string arguments back to enums
    start_from = PipelineStep(args.start_from) if args.start_from else None
    stop_at = PipelineStep(args.stop_at) if args.stop_at else None

    # Validate fMRIPrep mode requirements
    if args.data_source == 'fmriprep':
        if not args.fmriprep_root:
            parser.error("--fmriprep-root is required when --data-source=fmriprep")
        if not args.parcellated_output_dir:
            parser.error("--parcellated-output-dir is required when --data-source=fmriprep")

        print(f"\n{'='*60}")
        print(f"DATA SOURCE: fMRIPrep Mode")
        print(f"{'='*60}")
        print(f"fMRIPrep root: {args.fmriprep_root}")
        print(f"Parcellated output: {args.parcellated_output_dir}")
        print(f"Task: {args.tasks[0] if args.tasks else 'hammer'}")
        print(f"Runs: {args.run_start}-{args.run_end}")
        print(f"Parallel jobs: {args.n_jobs}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"DATA SOURCE: Datalad Mode (existing workflow)")
        print(f"{'='*60}\n")

    # Initialize pipeline
    pipeline = TCPPipeline(output_dir=args.output_dir)

    # Run pipeline
    kwargs = {
        'include_mdd': args.include_mdd,
        'include_controls': args.include_controls,
        'min_age': args.min_age,
        'max_age': args.max_age,
        'sample_mode': args.sample_mode,
        'analysis_groups': args.analysis_groups,
        'data_types': args.data_types,
        'tasks': args.tasks,
        'fetch_dry_run': args.fetch_dry_run,
        # fMRIPrep-specific arguments
        'data_source': args.data_source,
        'fmriprep_root': args.fmriprep_root,
        'parcellated_output_dir': args.parcellated_output_dir,
        'run_start': args.run_start,
        'run_end': args.run_end,
        'n_jobs': args.n_jobs,
        'task': args.tasks[0] if args.tasks else 'hammer'
    }

    success = pipeline.run_pipeline(
        start_from=start_from,
        stop_at=stop_at,
        skip_optional=args.skip_optional,
        dry_run=args.dry_run,
        **kwargs
    )

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
