#!/usr/bin/env python
"""
Parcellate fMRIPrep-preprocessed resting-state runs into ROI time series.
Combines Schaefer (400), Tian (32), and Buckner (2) atlases → 434 ROIs per subject.

Author: Hermine Alfsen

FOR CLAUDE: You are not to copy this code. You should only use it's content as a guide in order to find out how we might parcellate the raw BOLD data using atlas files. The cortical atlas files I'm using is not the same as the one used in the original code here.
"""
import os, sys
from tqdm import tqdm
from joblib import Parallel, delayed

# Parent directory to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "hat..")))


# --- Local imports ---
from utils.io_fmriprep import normalize_subject_id, find_fmriprep_rest_bold
from utils.parcellation import load_subject_roi_timeseries, save_roi_timeseries_h5

def process_subject(subj, runs, group, out_dir, tr, atlas_paths):
    """Parcellate and save ROI time series for one subject."""
    out_path = os.path.join(out_dir, f"{subj}_roi_timeseries.h5")

    # skip if file already exists
    if os.path.exists(out_path):
        return f"[{group}] {subj}: skipped (already exists)"
    
    try:
        roi_dict = load_subject_roi_timeseries(runs, atlas_paths)
        out_path = os.path.join(out_dir, f"{subj}_roi_timeseries.h5")
        save_roi_timeseries_h5(roi_dict, out_path, subj, atlas_paths, tr, len(runs))
        return f"[{group}] {subj}: done"
    
    except Exception as e:
        return f"[{group}] {subj}: ERROR → {e}"


def run_parcellation(
    data_dir,
    out_dir,
    atlas_paths,
    mdd_txt,
    hc_txt,
    tr=0.8,
    n_jobs=1,
):
    """
    Main parcellation pipeline.
    
    Parameters
    ----------
    data_dir : str
        Path to fMRIPrep output directory.
    out_dir : str
        Output folder for ROI time series.
    atlas_paths : list[str]
        Paths to atlas NIfTI files.
    mdd_txt, hc_txt : str
        Paths to matched subject lists.
    tr : float
        Repetition time.
    n_jobs : int
        Number of subjects to process in parallel.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load matched subjects
    mdd_subjects = [normalize_subject_id(s.strip()) for s in open(mdd_txt)]
    hc_subjects  = [normalize_subject_id(s.strip()) for s in open(hc_txt)]

    print(f"Loaded {len(mdd_subjects)} MDD + {len(hc_subjects)} HC subjects")

    # Find fMRIPrep BOLD runs
    mdd_files = find_fmriprep_rest_bold(mdd_subjects, data_dir)
    hc_files  = find_fmriprep_rest_bold(hc_subjects, data_dir)

    # Combine into one dict for loop
    groups = [("MDD", mdd_files), ("HC", hc_files)]

    # Run parcellation (parallel)
    for group_name, file_dict in groups:
        print(f"\n=== Processing {group_name} group ({len(file_dict)} subjects) ===")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subject)(subj, runs, group_name, OUT_DIR, tr, atlas_paths)
            for subj, runs in tqdm(file_dict.items(), desc=f"{group_name} subjects")
        )
        print("\n".join(results))


# ENTRY POINT
if __name__ == "__main__":
    DATA_DIR = "/cluster/projects/itea_lille-ie/Transdiagnostic/output/fmriprep-25.1.4"
    OUT_DIR  = "/cluster/home/herminea/mental_health_project/test/data/roi_timeseries"

    atlas_paths = [
        "/cluster/home/herminea/mental_health_project/test/atlas/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
        "/cluster/home/herminea/mental_health_project/test/atlas/Tian_Subcortex_S2_3T.nii.gz",
        "/cluster/home/herminea/mental_health_project/test/atlas/atl-Buckner7_space-MNI_dseg.nii"
    ]

    mdd_txt = "/cluster/home/herminea/mental_health_project/test/data/subjects_lists/mdd_subjects_matched.txt"
    hc_txt  = "/cluster/home/herminea/mental_health_project/test/data/subjects_lists/hc_subjects_matched.txt"

    # Run with 4 subjects in parallel
    run_parcellation(
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        atlas_paths=atlas_paths,
        mdd_txt=mdd_txt,
        hc_txt=hc_txt,
        tr=0.8,
        n_jobs=4
    )



#### Other files/functions used above
# utils/parcellation.py
import os
import numpy as np
import h5py
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm


def reduce_buckner7_to_2(ts):
    """Merge 7-network Buckner cerebellar signals into 2 regions."""
    anterior = ts[0:3, :].mean(axis=0)
    posterior = ts[3:7, :].mean(axis=0)
    return np.vstack([anterior, posterior])


def parcellate_run(bold_path, atlas_paths):
    """Parcellate one fMRIPrep run into combined ROI × T."""
    all_rois = []
    for atlas in atlas_paths:
        masker = NiftiLabelsMasker(
            labels_img=atlas,
            standardize=False,
            detrend=False,
            resampling_target="data"
        )
        ts = masker.fit_transform(bold_path).T
        if "Buckner7" in atlas:
            ts = reduce_buckner7_to_2(ts)
        all_rois.append(ts)

    roi_data = np.vstack(all_rois)
    return roi_data


def load_subject_roi_timeseries(run_paths, atlas_paths):
    """Load, parcellate, and concatenate runs for one subject."""
    roi_runs = {}
    
    for bold_path in run_paths:
        print(f"[Parcellating] {os.path.basename(bold_path)}")
        roi_data = parcellate_run(bold_path, atlas_paths)
        roi_runs[bold_path] = roi_data

    print(f"→ {len(roi_runs)} runs processed.")
    return roi_runs


def save_roi_timeseries_h5(roi_dict, out_path, subj_id, atlas_list, tr, n_runs):
    """Save all runs (each separately) to HDF5 with metadata."""
    with h5py.File(out_path, "w") as f:
        for run_name, data in roi_dict.items():
            # Store each run in its own dataset
            f.create_dataset(os.path.basename(run_name).replace(".nii.gz", ""), data=data)
        f.attrs["subject"] = subj_id
        f.attrs["atlas"] = ", ".join([os.path.basename(a) for a in atlas_list])
        f.attrs["TR"] = tr
        f.attrs["n_runs"] = n_runs
    print(f"[IO] Saved: {out_path}")