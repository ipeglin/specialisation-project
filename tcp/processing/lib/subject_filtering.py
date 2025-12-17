
from tcp.utils.file_utils import is_git_annex_pointer


def get_accessible_subjects_from_file(subjects, subject_manager, file_loader, datatype='timeseries', task='hammer'):
    accessible_subjects = []
    for subject_id in subjects:
        try:
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
