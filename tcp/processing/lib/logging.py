def write_analysis_log(output_dir, groups_config, all_results, low_anhedonic_subjects, high_anhedonic_subjects, timestamp=None):
    """
    Write comprehensive analysis log file tracking all analyzed subjects organized by groups.

    Args:
        output_dir: Directory to save log file
        groups_config: List of tuples (group_name, subject_ids)
        all_results: Dictionary mapping subject_id to processing results
        low_anhedonic_subjects: List of low anhedonic subject IDs
        high_anhedonic_subjects: List of high anhedonic subject IDs
        timestamp: Optional timestamp string for log filename

    Returns:
        Path to created log file
    """
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = output_dir / f'analysis_log_{timestamp}.txt'

    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FUNCTIONAL CONNECTIVITY ANALYSIS LOG\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log File: {log_file.name}\n\n")

        total_attempted = len(all_results)
        total_success = sum(1 for r in all_results.values() if r.get('success'))
        total_failed = total_attempted - total_success

        f.write("="*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total subjects attempted: {total_attempted}\n")
        f.write(f"Successfully processed: {total_success}\n")
        f.write(f"Failed: {total_failed}\n")
        f.write(f"Success rate: {total_success/total_attempted*100:.1f}%\n\n")

        f.write("="*80 + "\n")
        f.write("SUBJECTS BY GROUP\n")
        f.write("="*80 + "\n\n")

        for group_name, group_subjects in groups_config:
            successful_in_group = [sid for sid in group_subjects if all_results.get(sid, {}).get('success')]
            failed_in_group = [sid for sid in group_subjects if sid in all_results and not all_results[sid].get('success')]

            f.write(f"{group_name.upper()}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Total in group: {len(group_subjects)}\n")
            f.write(f"Successfully processed: {len(successful_in_group)}\n")
            f.write(f"Failed: {len(failed_in_group)}\n\n")

            if successful_in_group:
                f.write(f"Successfully processed subjects ({len(successful_in_group)}):\n")
                for sid in sorted(successful_in_group):
                    subgroup = ""
                    if sid in low_anhedonic_subjects:
                        subgroup = " [LOW-ANHEDONIC]"
                    elif sid in high_anhedonic_subjects:
                        subgroup = " [HIGH-ANHEDONIC]"
                    f.write(f"  - {sid}{subgroup}\n")
                f.write("\n")

            if failed_in_group:
                f.write(f"Failed subjects ({len(failed_in_group)}):\n")
                for sid in sorted(failed_in_group):
                    error_msg = all_results[sid].get('error', 'Unknown error')
                    f.write(f"  - {sid}: {error_msg}\n")
                f.write("\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("GROUP AVERAGES - INCLUDED SUBJECTS\n")
        f.write("="*80 + "\n\n")

        for group_name, group_subjects in groups_config:
            successful_in_group = [sid for sid in group_subjects if all_results.get(sid, {}).get('success')]

            f.write(f"{group_name.upper()} (N={len(successful_in_group)})\n")
            f.write(f"{'-'*80}\n")

            if successful_in_group:
                for sid in sorted(successful_in_group):
                    subgroup = ""
                    if sid in low_anhedonic_subjects:
                        subgroup = " [LOW-ANHEDONIC]"
                    elif sid in high_anhedonic_subjects:
                        subgroup = " [HIGH-ANHEDONIC]"
                    f.write(f"  {sid}{subgroup}\n")
            else:
                f.write("  None\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("END OF LOG\n")
        f.write("="*80 + "\n")

    return log_file