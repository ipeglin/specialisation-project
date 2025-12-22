# Brain Parcellation Atlases

This directory contains brain parcellation atlases required for HCP data parcellation.

## Required Atlas Files

### Cortical Parcellation
**Location:** `cortical/yeo17/`
**Required File:** `400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz`

- **Description:** Yeo 400-parcel 17-network cortical parcellation
- **Space:** FSLMNI152 2mm
- **Parcels:** 400 cortical regions
- **Status:** ⚠️ **FILE MISSING** - Download required

**Download Instructions:**
1. Download from: [Yeo 2011 Parcellation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering)
2. Place the `.nii.gz` file in `cortical/yeo17/` directory

### Subcortical Parcellation
**Location:** `subcortical/tian/`
**Required File:** `Tian_Subcortex_S2_3T.nii`

- **Description:** Tian subcortical atlas (Scale 2, 3T)
- **Parcels:** 32 subcortical regions
- **Status:** ⚠️ **FILE MISSING** - Download required

**Download Instructions:**
1. Download from: [Tian Subcortical Atlas](https://github.com/yetianmed/subcortex)
2. Place the `.nii` file (not `.dscalar.nii`) in `subcortical/tian/` directory

### Cerebellar Parcellation
**Location:** `cerebellar/buckner/` (when available)
**Required File:** TBD

- **Description:** Buckner 7-network cerebellar parcellation
- **Parcels:** 2 aggregated cerebellar regions (anterior/posterior)
- **Status:** ⚠️ **NOT YET IMPLEMENTED** - Placeholder zeros used

## Atlas Summary

Total expected parcels: **434**
- Cortical: 400 parcels (Yeo 17-network)
- Subcortical: 32 parcels (Tian S2)
- Cerebellar: 2 regions (Buckner - placeholder)

## Usage

The HCP parcellation pipeline (`tcp/preprocessing/hcp_parcellation.py`) uses these atlases to extract ROI timeseries from HCP-preprocessed BOLD data.

```python
from tcp.preprocessing.hcp_parcellation import HCPParcellator
from config.paths import get_parcellations_path

# Atlas paths are automatically resolved
parcellator = HCPParcellator(hcp_root="/path/to/hcp_output")
```

## Notes

- Atlas files are stored in volumetric NIfTI format (`.nii` or `.nii.gz`)
- All atlases must be in MNI152 2mm space to match HCP data
- The HCP parcellator will validate atlas files exist on initialization

## Troubleshooting

**Error: "Cortical atlas not found"**
- Ensure `400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz` exists in `cortical/yeo17/`

**Error: "Subcortical atlas not found"**
- Ensure `Tian_Subcortex_S2_3T.nii` exists in `subcortical/tian/`

**Warning: "Using zeros placeholder for cerebellar atlas"**
- This is expected - cerebellar atlas implementation is pending
- Parcellation will complete with zeros for cerebellar regions (ROIs 433-434)
