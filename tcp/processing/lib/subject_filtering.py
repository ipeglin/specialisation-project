
from tcp.utils.file_utils import is_git_annex_pointer


def get_accessible_subjects_from_file(subjects, subject_manager, file_loader, datatype='timeseries', task='hammer', required_data_source=None):
    """
    Get list of subjects with accessible files for a given task.

    Args:
        subjects: List of subject IDs to check
        subject_manager: SubjectManager instance
        file_loader: DataLoader instance
        datatype: Type of data to check (default: 'timeseries')
        task: Task name (default: 'hammer')
        required_data_source: Optional data source filter ('hcp', 'datalad', or None for any)
                             CRITICAL: Use this to prevent mixing HCP and datalad data

    Returns:
        List of subject IDs that have accessible data from the required source
    """
    accessible_subjects = []
    for subject_id in subjects:
        try:
            # CRITICAL: Check data source FIRST to prevent cross-source contamination
            if required_data_source is not None:
                subject_metadata = subject_manager.get_subject_metadata(subject_id)
                actual_data_source = subject_metadata.get('data_source', 'datalad')

                if actual_data_source != required_data_source:
                    print(f"    Skipping {subject_id} - wrong data source (expected {required_data_source}, got {actual_data_source})")
                    continue

            # Get only hammer task files
            hammer_files = subject_manager.get_subject_files_by_task(subject_id, datatype, task)
            if hammer_files:
                # Check if at least one hammer file is actually downloaded (not a git-annex symlink)
                first_file = file_loader.resolve_file_path(hammer_files[0])
                if not is_git_annex_pointer(first_file):
                    accessible_subjects.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer file exists but not downloaded (git-annex symlink)")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")

    return accessible_subjects
