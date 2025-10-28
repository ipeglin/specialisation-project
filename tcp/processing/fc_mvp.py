#!/usr/bin/env python3
"""Functional Connectivity MVP Script"""

import sys
from pathlib import Path

# Add project root to path (same fix as test script)
project_root = Path(__file__).parent.parent.parent

# Clear any conflicting paths and ensure clean import environment
paths_to_remove = [
    str(Path.cwd()),
    str(Path(__file__).parent.parent),  # tcp directory
    str(Path(__file__).parent),         # tcp/processing directory#
]

for path in paths_to_remove:
    while path in sys.path:
        sys.path.remove(path)

# Insert project root at the beginning
sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
import numpy as np

from tcp.processing import DataLoader, SubjectManager
from tcp.processing.roi import CorticalAtlasLookup, ROIExtractionService


def main():
    """Main function for FC MVP analysis"""
    print("=== Functional Connectivity MVP ===")
    
    # Initialize data infrastructure
    loader = DataLoader()
    manager = SubjectManager(data_loader=loader)
    
    print(f"✓ Loaded manifest with {len(loader.get_all_subject_ids())} subjects")
    
    # Get available analysis groups
    groups = loader.get_analysis_groups()
    print(f"Available groups: {list(groups.keys())}")
    
    # Show data availability summary
    availability = manager.get_subjects_availability_summary()
    print(f"\nData Availability Summary:")
    print(f"  Total subjects in manifest: {availability['total_subjects']}")
    print(f"  Downloaded locally: {availability['downloaded_subjects']}")
    print(f"  With timeseries metadata: {availability['with_timeseries_data']}")
    print(f"  Ready for processing: {availability['breakdown']['ready_for_processing']}")
    
    # Decide on processing mode
    use_downloaded_only = availability['downloaded_subjects'] > 0
    if use_downloaded_only:
        print(f"\nUsing DOWNLOADED-ONLY mode ({availability['downloaded_subjects']} subjects)")
        print("  This ensures all subjects have locally available data files")
    else:
        print(f"\nUsing ALL-AVAILABLE mode ({availability['with_timeseries_data']} subjects)")
        print("  Warning: Some subjects may not have locally downloaded data")
    
    # Use a valid group name from the manifest
    # The manifest shows: anhedonic_vs_non_anhedonic, anhedonic_patients_vs_controls, etc.
    group_name = 'anhedonic_vs_non_anhedonic'  # Use actual group from manifest
    
    # Get subjects for analysis
    low_anhedonic_subjects = manager.filter_subjects(
        groups=[group_name],
        classifications={'anhedonic_status': 'anhedonic',
                         'anhedonia_class': 'low-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )
    high_anhedonic_subjects = manager.filter_subjects(
        groups=[group_name],
        classifications={'anhedonic_status': 'anhedonic',
                         'anhedonia_class': 'high-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )
    anhedonic_subjects = low_anhedonic_subjects + high_anhedonic_subjects
    
    non_anhedonic_subjects = manager.filter_subjects(  # Fixed typo
        groups=[group_name],
        classifications={'anhedonic_status': 'non-anhedonic'},
        data_requirements=['timeseries'],
        downloaded_only=use_downloaded_only,
    )
    
    print(f"\nSubject Selection:")
    print(f"  Anhedonic subjects: {len(anhedonic_subjects)}")
    print(f"\tLOW: {len(low_anhedonic_subjects)}")
    print(f"\tHIGH: {len(high_anhedonic_subjects)}")
    print(f"  Non-anhedonic subjects: {len(non_anhedonic_subjects)}")
    
    # Validate file access for processing
    print(f"\nValidating file access:")
    accessible_anhedonic = []
    accessible_non_anhedonic = []
    
    # Check anhedonic subjects (hammer task only)
    for subject_id in anhedonic_subjects:
        try:
            # Get only hammer task files
            hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
            if hammer_files:
                # Check if at least one hammer file actually exists
                first_file = loader.resolve_file_path(hammer_files[0])
                if first_file.exists():
                    accessible_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer files listed but not accessible")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")
    
    # Check non-anhedonic subjects (hammer task only)
    for subject_id in non_anhedonic_subjects:
        try:
            # Get only hammer task files
            hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
            if hammer_files:
                first_file = loader.resolve_file_path(hammer_files[0])
                if first_file.exists():
                    accessible_non_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer files listed but not accessible")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")
    
    # Report final accessible counts
    print(f"\nFinal Processing Summary:")
    print(f"  Anhedonic subjects (accessible): {len(accessible_anhedonic)}")
    print(f"  Non-anhedonic subjects (accessible): {len(accessible_non_anhedonic)}")
    print(f"  Total ready for FC analysis (hammer task only): {len(accessible_anhedonic) + len(accessible_non_anhedonic)}")
    
    # Show example hammer task file paths for first few accessible subjects
    if accessible_anhedonic:
        print(f"\nExample accessible hammer task file paths:")
        for subject_id in accessible_anhedonic[:2]:  # Show first 2
            print(f"\n  Subject: {subject_id}")
            try:
                # Get only hammer task files
                hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
                for file_path in hammer_files[:2]:  # Show first 2 hammer files
                    full_path = loader.resolve_file_path(file_path)
                    print(f"    {full_path}")
            except Exception as e:
                print(f"    Error: {e}")
    
    """
    MVP: Only check data for a single subject.
    After completing the data extraction for a single subject, this may be implemented in a way that loops over all downloaded subjects
    """
    # Read data from .h5 files
    first_subject_id = accessible_anhedonic[0]
    hammer_files = manager.get_subject_files_by_task(first_subject_id, 'timeseries', 'hammer')
    first_subject_id_hammer_file = loader.resolve_file_path(hammer_files[0])
    
    data = None
    with h5py.File(first_subject_id_hammer_file, 'r') as file:
      a_group_key = list(file.keys())[0]
    
      # Getting the data
      data = np.asarray(file[a_group_key])
      print(f"Found data with shape: {data.shape}")
      
      # Segmenting into timeseries groups (cortical, subcortical, cerebellum; samples [1-400], [401-432] and [433-434], respectively)
      cortical_timeseries = data[:400] # using hMRF atlas (https://www.sciencedirect.com/science/article/pii/S1053811923001568?via%3Dihub)
      cortical_L, cortical_R = cortical_timeseries[:200], cortical_timeseries[200:]
      cortical_homotopic_pairs = np.asarray(list(zip(cortical_L, cortical_R))) # Combine into L/R homotopic pairs of ROIs
      subcortical_timeseries = data[400:432] # “scale II” resolution atlas by Tian and colleagues (https://www.nature.com/articles/s41593-020-00711-6#code-availability)
      cerebellum_timeseries = data[432:] # using Buckner et al. atlas (https://journals.physiology.org/doi/full/10.1152/jn.00339.2011)
      print("Found parcels:")
      print(f"Cortical: {cortical_timeseries.shape}\n\tLEFT Hemisphere: {cortical_L.shape}\n\tRIGHT Hemisphere: {cortical_R.shape}\n\tHomotopic Pairs: {cortical_homotopic_pairs.shape}\nSubcortical: {subcortical_timeseries.shape}\nCerebellum: {cerebellum_timeseries.shape}")
      
      """
        Get ROI parcel indeces by searching for ROIs by lines in cortical LUT file
        
        ROI format for each timeseries spans two lines:
        ```
        <7|17>networks_<L|R>H_<network_name>_<ROI_abbreviation>[_subarea_number] # first line
        <parcel_idx> <red> <green> <blue> <redundant_value> # second line
        ```
        
        Example:
        ```
        17networks_LH_TempPar_IPL_1
        1 12 48 255 255
        ```
        Meaning that parcel 1 (parcel_idx) for a space matches to Yeo17 (17networks) is on the left hemisphere (LH) and is assigned to the TemporalPariental network. The data point contains the first subarea (_1) for the Inferior Parietal Lobule (IPL). This datapoint is visually coloured with RGB (12,48,255) when opened in a viewing program.
      """
      # Initialize modular ROI extraction system
      cortical_lut_file = Path(__file__).parent / 'parcellations/hcp/yeo17/400Parcels_Yeo2011_17Networks_info.txt'
      cortical_atlas = CorticalAtlasLookup(cortical_lut_file)
      roi_extractor = ROIExtractionService(cortical_atlas)
      
      # Define ROIs of interest
      cortical_ROIs = [
        'PFCm',  # medial PFC
        'PFCv',  # ventral PFC
      ]
      
      # Validate ROI coverage before extraction
      validation_result = roi_extractor.validate_roi_coverage(cortical_timeseries, cortical_ROIs)
      print(f"\nROI Validation Results:")
      print(f"  Valid ROIs: {validation_result['valid_rois']}")
      print(f"  Invalid ROIs: {validation_result['invalid_rois']}")
      print(f"  Coverage issues: {validation_result['coverage_issues']}")
      print(f"  Atlas: {validation_result['atlas_info']['name']} ({validation_result['atlas_info']['total_parcels']} parcels)")
      
      # Extract ROI timeseries data
      if validation_result['valid_rois'] and not validation_result['coverage_issues']:
          roi_timeseries = roi_extractor.extract_roi_timeseries(
              cortical_timeseries, 
              cortical_ROIs, 
              aggregation_method='all'
          )
          
          # Display extraction results
          extraction_summary = roi_extractor.get_extraction_summary(cortical_ROIs, roi_timeseries)
          print(f"\nROI Extraction Summary:")
          print(f"  Requested: {extraction_summary['requested_rois']}")
          print(f"  Extracted: {extraction_summary['extracted_rois']}")
          print(f"  Atlas indexing: {extraction_summary['atlas_indexing']}")
          
          # Show details for each extracted ROI
          for roi_name, details in extraction_summary['roi_details'].items():
              print(f"\n  {roi_name}:")
              print(f"    Timeseries shape: {details['timeseries_shape']}")
              print(f"    Parcel count: {details['parcel_count']}")
              print(f"    Hemispheres: {details['hemispheres']}")
              print(f"    Networks: {details['networks']}")
              
              # Show first few timepoints as example
              timeseries_data = roi_timeseries[roi_name]
              print(f"    Sample timepoints: {timeseries_data[:5]}")
              
          # Demonstrate network-specific extraction if supported
          if roi_extractor.supports_network_queries():
              print(f"\n=== Network-Specific Analysis ===")
              
              # Get available networks
              available_networks = roi_extractor.atlas_lookup.get_available_networks()
              print(f"Available networks: {sorted(available_networks)}")
              
              # Get network breakdown for our ROIs
              network_breakdown = roi_extractor.get_network_breakdown_summary(cortical_ROIs)
              if network_breakdown:
                  print(f"\nNetwork breakdown:")
                  for roi_name, networks in network_breakdown.items():
                      print(f"  {roi_name}:")
                      for network, details in networks.items():
                          print(f"    {network}: {details['parcel_count']} parcels")
              
              # Extract network-specific timeseries (default: keep all parcels)
              network_timeseries = roi_extractor.extract_roi_timeseries_by_network(
                  cortical_timeseries, 
                  cortical_ROIs,
                  aggregation_method='all'
              )
              
              print(f"\nNetwork-specific extraction results:")
              for roi_name, networks in network_timeseries.items():
                  print(f"  {roi_name}:")
                  for network, timeseries in networks.items():
                      # Show just first 3 timepoints from first parcel for consistent output
                      if timeseries.ndim == 1:
                          sample_data = timeseries[:3]
                      else:
                          sample_data = timeseries[0, :3]  # First parcel, first 3 timepoints
                      print(f"    {network}: shape {timeseries.shape}, sample: {sample_data}")
      else:
          print("ROI extraction skipped due to validation issues")
      
    
    return {
        'anhedonic_subjects': accessible_anhedonic,
        'non_anhedonic_subjects': accessible_non_anhedonic,
        'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        'summary': {
            'total_accessible': len(accessible_anhedonic) + len(accessible_non_anhedonic),
            'anhedonic_count': len(accessible_anhedonic),
            'non_anhedonic_count': len(accessible_non_anhedonic)
        }
    }


if __name__ == '__main__':
    main()