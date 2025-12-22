#!/usr/bin/env python3
"""
Validate and help download required brain parcellation atlases.

This script checks if required atlas files are present and provides
instructions for downloading missing files.

Author: Ian Philip Eglin
Date: 2025-12-22
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.paths import get_parcellations_path


class AtlasValidator:
    """Validate presence of required brain atlas files"""

    def __init__(self):
        self.parcellations_base = get_parcellations_path()

        # Define required atlas files
        self.required_atlases = {
            'cortical': {
                'name': 'Yeo 400-parcel cortical atlas',
                'path': self.parcellations_base / 'cortical' / 'yeo17' / '400Parcels_Yeo2011_17Networks_FSLMNI152_2mm.nii.gz',
                'parcels': 400,
                'download_url': 'https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152',
                'notes': 'Download the 400-parcel 17-network parcellation in FSLMNI152 2mm space'
            },
            'subcortical': {
                'name': 'Tian subcortical atlas',
                'path': self.parcellations_base / 'subcortical' / 'tian' / 'Tian_Subcortex_S2_3T.nii',
                'parcels': 32,
                'download_url': 'https://github.com/yetianmed/subcortex',
                'notes': 'Download the volumetric .nii file (Scale 2, 3T), NOT the .dscalar.nii surface file'
            }
        }

    def check_atlas(self, atlas_type: str) -> bool:
        """Check if a specific atlas file exists"""
        atlas_info = self.required_atlases[atlas_type]
        return atlas_info['path'].exists()

    def validate_all(self) -> dict:
        """Validate all required atlas files"""
        results = {}

        print("=" * 80)
        print("Brain Parcellation Atlas Validation")
        print("=" * 80)
        print(f"\nParcellations directory: {self.parcellations_base}")
        print(f"Directory exists: {self.parcellations_base.exists()}")

        all_present = True

        for atlas_type, atlas_info in self.required_atlases.items():
            print(f"\n{'-' * 80}")
            print(f"Atlas: {atlas_info['name']}")
            print(f"Type: {atlas_type}")
            print(f"Expected path: {atlas_info['path']}")
            print(f"Parcels: {atlas_info['parcels']}")

            exists = self.check_atlas(atlas_type)
            results[atlas_type] = exists

            if exists:
                file_size = atlas_info['path'].stat().st_size / (1024 * 1024)  # MB
                print(f"Status: ✓ FOUND ({file_size:.1f} MB)")
            else:
                print(f"Status: ✗ MISSING")
                print(f"\nDownload instructions:")
                print(f"  1. Visit: {atlas_info['download_url']}")
                print(f"  2. {atlas_info['notes']}")
                print(f"  3. Place file at: {atlas_info['path']}")
                all_present = False

        print(f"\n{'=' * 80}")
        if all_present:
            print("✓ All required atlas files are present!")
            print("\nYou can now run HCP parcellation:")
            print("  python tcp/preprocessing/hcp_parcellation.py --help")
        else:
            print("✗ Some atlas files are missing")
            print("\nPlease download the missing files before running HCP parcellation.")
            print("See instructions above for each missing file.")

        print(f"{'=' * 80}\n")

        return results

    def print_summary(self):
        """Print a summary of atlas requirements"""
        print("\n" + "=" * 80)
        print("Atlas Summary")
        print("=" * 80)
        print("\nTotal expected parcels: 434")
        print("  - Cortical:    400 parcels (Yeo 17-network)")
        print("  - Subcortical:  32 parcels (Tian S2)")
        print("  - Cerebellar:    2 regions (Placeholder - not yet implemented)")
        print("\nAtlas format: Volumetric NIfTI (.nii or .nii.gz)")
        print("Atlas space:  MNI152 2mm")
        print("=" * 80 + "\n")


def main():
    """Main validation function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate brain parcellation atlas files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tcp/parcellations/validate_atlases.py
  python tcp/parcellations/validate_atlases.py --summary
        """
    )
    parser.add_argument('--summary', action='store_true',
                       help='Print atlas summary information')

    args = parser.parse_args()

    validator = AtlasValidator()

    if args.summary:
        validator.print_summary()

    results = validator.validate_all()

    # Exit with error code if any atlas is missing
    if not all(results.values()):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
