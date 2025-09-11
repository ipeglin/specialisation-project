#!/usr/bin/env python3
"""
Example script demonstrating the flexible cross-platform path configuration system.
This shows how different types of scripts automatically get organized into appropriate directories.

Author: Ian Philip Eglin
Date: 2025-09-11
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.paths import (
    get_script_output_path, 
    get_preprocessing_path,
    get_analysis_path, 
    get_figures_path,
    get_models_path,
    get_reports_path,
    ensure_path_exists,
    get_platform_info
)

def demonstrate_flexible_paths():
    """Demonstrate how different script types get organized into appropriate directories"""
    
    print("=== Flexible Output Path Demonstration ===\n")
    
    # Show platform info
    info = get_platform_info()
    print(f"Platform: {info['detected_platform']}")
    print(f"Current configuration source: {info['config_source']}\n")
    
    # 1. Preprocessing scripts automatically go to preprocessing directory
    print("1. PREPROCESSING SCRIPTS:")
    tcp_preprocess = get_script_output_path('tcp_preprocessing', 'clean_data')
    print(f"   TCP preprocessing: {tcp_preprocess}")
    
    data_cleaning = get_script_output_path('data_cleaning', 'subjects')
    print(f"   Data cleaning: {data_cleaning}")
    
    # 2. Analysis scripts automatically go to analysis directory
    print("\n2. ANALYSIS SCRIPTS:")
    mdd_analysis = get_script_output_path('mdd_analysis', 'connectivity_study')
    print(f"   MDD analysis: {mdd_analysis}")
    
    statistical_analysis = get_script_output_path('statistical_analysis', 'results')
    print(f"   Statistical analysis: {statistical_analysis}")
    
    # 3. Visualization scripts automatically go to figures directory
    print("\n3. VISUALIZATION SCRIPTS:")
    brain_plots = get_script_output_path('visualization', 'brain_connectivity')
    print(f"   Brain plots: {brain_plots}")
    
    model_plots = get_script_output_path('model_visualization', 'training_curves')
    print(f"   Model plots: {model_plots}")
    
    # 4. Model scripts automatically go to models directory
    print("\n4. MODEL TRAINING SCRIPTS:")
    neural_net = get_script_output_path('model_training', 'neural_networks')
    print(f"   Neural networks: {neural_net}")
    
    classifier = get_script_output_path('classification_model', 'random_forest')
    print(f"   Classifier: {classifier}")
    
    # 5. Report scripts automatically go to reports directory
    print("\n5. REPORT SCRIPTS:")
    final_report = get_script_output_path('report_generation', 'final_results')
    print(f"   Final report: {final_report}")
    
    # 6. Direct category access
    print("\n6. DIRECT CATEGORY ACCESS:")
    print(f"   Preprocessing dir: {get_preprocessing_path('custom_preprocessing')}")
    print(f"   Analysis dir: {get_analysis_path('custom_analysis')}")
    print(f"   Figures dir: {get_figures_path('custom_plots')}")
    print(f"   Models dir: {get_models_path('custom_models')}")
    print(f"   Reports dir: {get_reports_path('custom_reports')}")
    
    print("\n=== Creating Example Directories ===\n")
    
    # Create a few example directories to show it works
    example_paths = [
        get_script_output_path('tcp_preprocessing', 'example_output'),
        get_script_output_path('mdd_analysis', 'example_results'),
        get_script_output_path('visualization', 'example_plots')
    ]
    
    for path in example_paths:
        ensure_path_exists(path)
        print(f"✓ Created: {path}")
        
        # Create a simple file to show it works
        example_file = path / "example.txt"
        with open(example_file, 'w') as f:
            f.write(f"Example output from script in {path.name}\n")
            f.write(f"Generated on platform: {info['detected_platform']}\n")
        
        print(f"  └─ Created example file: {example_file.name}")
    
    print(f"\n🎉 Flexible output system demonstration complete!")
    print(f"All scripts automatically organize their outputs into appropriate directories.")

if __name__ == "__main__":
    demonstrate_flexible_paths()