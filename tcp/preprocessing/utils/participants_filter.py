#!/usr/bin/env python3
"""
Participants File Filter for TCP Parcellation Pipeline

Loads a participants.txt file and intersects it with discovered fmriprep subjects.
Hard error if any listed subject is missing from fmriprep output.

Author: Ian Philip Eglin
Date: 2026-03-21
"""

from pathlib import Path
from typing import List

from tcp.preprocessing.utils.unicode_compat import CHECK, WARNING


def load_participants_file(participants_path: Path) -> List[str]:
    """
    Load and normalise subject IDs from a participants file.

    Reads one subject ID per line. Lines starting with '#' are skipped as
    comments. Inline '#' comments are stripped. Empty lines are skipped.
    Subject IDs are normalised to include the 'sub-' prefix. Duplicates are
    removed while preserving the original order.

    Args:
        participants_path: Path to the participants.txt file.

    Returns:
        List of normalised, deduplicated subject IDs.

    Raises:
        FileNotFoundError: If the participants file does not exist.
    """
    if not participants_path.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_path}")

    raw_ids: List[str] = []
    with open(participants_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if '#' in line:
                line = line.split('#')[0].strip()
            if not line:
                continue
            raw_ids.append(line)

    # Normalise: ensure 'sub-' prefix
    normalised: List[str] = []
    for sid in raw_ids:
        if not sid.startswith('sub-'):
            sid = f'sub-{sid}'
        normalised.append(sid)

    # Deduplicate preserving order
    seen = set()
    deduplicated: List[str] = []
    for sid in normalised:
        if sid not in seen:
            seen.add(sid)
            deduplicated.append(sid)

    if len(deduplicated) < len(normalised):
        print(f"Loaded {len(deduplicated)} subjects from participants file (after deduplication)")
    else:
        print(f"Loaded {len(deduplicated)} subjects from participants file")

    return deduplicated


def apply_participants_filter(
    participants_subjects: List[str],
    discovered_subjects: List[str],
) -> List[str]:
    """
    Intersect a participants list with discovered fmriprep subjects.

    Subjects in participants_subjects that are NOT in discovered_subjects
    cause a hard error listing all missing IDs.
    Subjects in discovered_subjects but NOT in participants_subjects are
    silently excluded with a warning log.

    Args:
        participants_subjects: Output of load_participants_file().
        discovered_subjects: Output of discover_fmriprep_subjects() or
            any subject discovery returning a list of subject ID strings.

    Returns:
        Filtered list of subject IDs in participants file order.

    Raises:
        ValueError: If one or more participants file subjects are not found
            in the discovered subjects.
    """
    discovered_set = set(discovered_subjects)

    missing = [s for s in participants_subjects if s not in discovered_set]
    if missing:
        raise ValueError(
            f"{len(missing)} subject(s) from participants file not found in fmriprep output:\n"
            + "\n".join(f"  - {s}" for s in missing)
            + f"\n\nExpected fmriprep output at the configured fmriprep_root. "
            + f"Ensure preprocessing is complete for all listed subjects."
        )

    filtered = [s for s in participants_subjects if s in discovered_set]

    excluded_count = len(discovered_set) - len(filtered)
    if excluded_count > 0:
        print(f"{WARNING} {excluded_count} fmriprep subject(s) excluded by participants filter")

    print(f"{CHECK} Participants filter applied: {len(filtered)} subjects selected")
    return filtered
