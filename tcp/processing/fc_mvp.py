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
import seaborn as sns
from scipy import stats

from tcp.processing import DataLoader, SubjectManager
from tcp.processing.roi import (CorticalAtlasLookup, ROIExtractionService,
                                SubCorticalAtlasLookup)


def is_actual_file(file_path: Path) -> bool:
    """
    Check if a file is actually downloaded (not a git-annex symlink).

    Works cross-platform (Windows, macOS, Linux) by checking:
    1. File exists
    2. If it's a symlink, verify the target exists
    3. File has actual content (size > 1KB, symlinks are tiny)

    Args:
        file_path: Path to check

    Returns:
        True if file is actually available for reading, False if it's a symlink stub
    """
    if not file_path.exists():
        return False

    # Check if it's a symlink
    if file_path.is_symlink():
        # On Windows, symlinks might point to git-annex objects
        try:
            # Check if symlink target exists and is accessible
            resolved = file_path.resolve(strict=True)
            # Verify it's not just a symlink to an annex object path
            if '.git/annex/objects' in str(resolved):
                # This is a git-annex symlink, check if target actually exists
                if not resolved.exists():
                    return False
        except (OSError, RuntimeError):
            return False

    # Check file size - git-annex symlinks are very small (<1KB)
    # Real H5 files should be much larger
    try:
        size = file_path.stat().st_size
        if size < 1024:  # Less than 1KB = likely a symlink or empty
            return False
    except OSError:
        return False

    return True


def compute_fc_matrix(timeseries_dict, roi_names=None):
    """
    Compute functional connectivity (Pearson correlation) matrix between ROI timeseries.
    
    Args:
        timeseries_dict: Dictionary mapping ROI names to timeseries arrays
        roi_names: Optional list to specify order of ROIs in matrix
        
    Returns:
        tuple: (correlation_matrix, roi_labels, p_values_matrix)
    """
    if roi_names is None:
        roi_names = list(timeseries_dict.keys())
    
    # Collect all timeseries
    timeseries_list = []
    roi_labels = []
    
    for roi_name in roi_names:
        if roi_name in timeseries_dict:
            ts = timeseries_dict[roi_name]
            if ts.size > 0:  # Check if timeseries is not empty
                timeseries_list.append(ts)
                roi_labels.append(roi_name)
    
    if len(timeseries_list) < 2:
        print(f"[WARNING] Need at least 2 ROI timeseries for FC computation, got {len(timeseries_list)}")
        return None, roi_labels, None
    
    # Stack timeseries for correlation computation
    stacked_timeseries = np.vstack(timeseries_list)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(stacked_timeseries)
    
    # Compute p-values for correlations
    n_rois = len(roi_labels)
    p_values = np.ones((n_rois, n_rois))
    
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            # Compute p-value for correlation
            corr_coef, p_val = stats.pearsonr(timeseries_list[i], timeseries_list[j])
            p_values[i, j] = p_val
            p_values[j, i] = p_val  # Symmetric matrix
    
    return corr_matrix, roi_labels, p_values


def analyze_connectivity_patterns(corr_matrix, roi_labels, p_values=None, alpha=0.05):
    """
    Extract and analyze specific connectivity patterns from correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels corresponding to matrix rows/columns  
        p_values: Optional p-values matrix for significance testing
        alpha: Significance threshold
        
    Returns:
        dict: Dictionary with different connectivity pattern results
    """
    results = {
        'interhemispheric': {},
        'cross_regional': {},
        'ipsilateral': {},
        'contralateral': {},
        'all_pairwise': {}
    }
    
    n_rois = len(roi_labels)
    
    # Extract all pairwise correlations
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            roi1, roi2 = roi_labels[i], roi_labels[j]
            corr_val = corr_matrix[i, j]
            p_val = p_values[i, j] if p_values is not None else None
            is_significant = p_val < alpha if p_val is not None else None
            
            pair_key = f"{roi1}_{roi2}"
            results['all_pairwise'][pair_key] = {
                'correlation': corr_val,
                'p_value': p_val,
                'significant': is_significant
            }
            
            # Categorize by connectivity pattern
            
            # Extract hemisphere and region info
            roi1_parts = roi1.split('_')
            roi2_parts = roi2.split('_')
            
            # Interhemispheric (same region, different hemispheres)
            if len(roi1_parts) >= 2 and len(roi2_parts) >= 2:
                roi1_region = roi1_parts[0]
                roi1_hemi = roi1_parts[1] if len(roi1_parts) > 1 else 'unknown'
                roi2_region = roi2_parts[0] 
                roi2_hemi = roi2_parts[1] if len(roi2_parts) > 1 else 'unknown'
                
                if roi1_region == roi2_region and roi1_hemi != roi2_hemi:
                    results['interhemispheric'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'region': roi1_region
                    }
                
                # Cross-regional (different regions)
                elif roi1_region != roi2_region:
                    results['cross_regional'][pair_key] = {
                        'correlation': corr_val,
                        'p_value': p_val,
                        'significant': is_significant,
                        'regions': f"{roi1_region}_{roi2_region}"
                    }
                    
                    # Ipsilateral (same hemisphere, different regions)
                    if roi1_hemi == roi2_hemi:
                        results['ipsilateral'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemisphere': roi1_hemi,
                            'regions': f"{roi1_region}_{roi2_region}"
                        }
                    
                    # Contralateral (different hemisphere, different regions)
                    elif roi1_hemi != roi2_hemi:
                        results['contralateral'][pair_key] = {
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': is_significant,
                            'hemispheres': f"{roi1_hemi}_{roi2_hemi}",
                            'regions': f"{roi1_region}_{roi2_region}"
                        }
    
    return results


def plot_fc_results(corr_matrix, roi_labels, p_values=None, connectivity_patterns=None, alpha=0.05):
    """
    Create visualizations for functional connectivity results.
    
    Args:
        corr_matrix: Correlation matrix
        roi_labels: ROI labels
        p_values: Optional p-values matrix
        connectivity_patterns: Optional results from analyze_connectivity_patterns
        alpha: Significance threshold for marking significant correlations
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Correlation matrix heatmap
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Create significance mask if p-values available
    if p_values is not None:
        mask = p_values >= alpha
    else:
        mask = None
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=roi_labels,
                yticklabels=roi_labels,
                center=0,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                mask=mask,
                ax=ax1)
    ax1.set_title('Functional Connectivity Matrix\n(Pearson Correlations)')
    
    if connectivity_patterns is not None:
        # 2. Interhemispheric connectivity
        ax2 = plt.subplot(2, 3, 2)
        inter_corrs = [v['correlation'] for v in connectivity_patterns['interhemispheric'].values()]
        inter_labels = [k.replace('_', '\n') for k in connectivity_patterns['interhemispheric'].keys()]
        
        if inter_corrs:
            bars = ax2.bar(range(len(inter_corrs)), inter_corrs, color='lightblue', edgecolor='navy')
            ax2.set_xticks(range(len(inter_labels)))
            ax2.set_xticklabels(inter_labels, rotation=45, ha='right')
            ax2.set_ylabel('Correlation')
            ax2.set_title('Interhemispheric\nConnectivity')
            ax2.set_ylim(-1, 1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Mark significant correlations
            if p_values is not None:
                for i, (k, v) in enumerate(connectivity_patterns['interhemispheric'].items()):
                    if v['significant']:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.8)
        else:
            ax2.text(0.5, 0.5, 'No interhemispheric\nconnections found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Interhemispheric\nConnectivity')
        
        # 3. Cross-regional connectivity
        ax3 = plt.subplot(2, 3, 3)
        cross_corrs = [v['correlation'] for v in connectivity_patterns['cross_regional'].values()]
        cross_labels = [k.replace('_', '\n') for k in connectivity_patterns['cross_regional'].keys()]
        
        if cross_corrs:
            bars = ax3.bar(range(len(cross_corrs)), cross_corrs, color='lightgreen', edgecolor='darkgreen')
            ax3.set_xticks(range(len(cross_labels)))
            ax3.set_xticklabels(cross_labels, rotation=45, ha='right')
            ax3.set_ylabel('Correlation')
            ax3.set_title('Cross-Regional\nConnectivity')
            ax3.set_ylim(-1, 1)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Mark significant correlations
            if p_values is not None:
                for i, (k, v) in enumerate(connectivity_patterns['cross_regional'].items()):
                    if v['significant']:
                        bars[i].set_color('red')
                        bars[i].set_alpha(0.8)
        else:
            ax3.text(0.5, 0.5, 'No cross-regional\nconnections found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cross-Regional\nConnectivity')
        
        # 4. Ipsilateral vs Contralateral
        ax4 = plt.subplot(2, 3, 5)
        ipsi_corrs = [v['correlation'] for v in connectivity_patterns['ipsilateral'].values()]
        contra_corrs = [v['correlation'] for v in connectivity_patterns['contralateral'].values()]
        
        if ipsi_corrs or contra_corrs:
            width = 0.35
            ipsi_mean = np.mean(ipsi_corrs) if ipsi_corrs else 0
            contra_mean = np.mean(contra_corrs) if contra_corrs else 0
            
            bars = ax4.bar(['Ipsilateral', 'Contralateral'], 
                          [ipsi_mean, contra_mean],
                          color=['orange', 'purple'], 
                          alpha=0.7)
            ax4.set_ylabel('Mean Correlation')
            ax4.set_title('Ipsilateral vs\nContralateral Connectivity')
            ax4.set_ylim(-1, 1)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add error bars if multiple connections
            if len(ipsi_corrs) > 1:
                ax4.errorbar([0], [ipsi_mean], yerr=[np.std(ipsi_corrs)], 
                           color='black', capsize=5)
            if len(contra_corrs) > 1:
                ax4.errorbar([1], [contra_mean], yerr=[np.std(contra_corrs)], 
                           color='black', capsize=5)
        else:
            ax4.text(0.5, 0.5, 'No ipsilateral/contralateral\nconnections found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Ipsilateral vs\nContralateral Connectivity')
        
        # 5. Summary statistics
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('off')
        
        summary_text = "FC Summary Statistics:\n\n"
        summary_text += f"Total connections: {len(connectivity_patterns['all_pairwise'])}\n"
        summary_text += f"Interhemispheric: {len(connectivity_patterns['interhemispheric'])}\n"
        summary_text += f"Cross-regional: {len(connectivity_patterns['cross_regional'])}\n"
        summary_text += f"Ipsilateral: {len(connectivity_patterns['ipsilateral'])}\n"
        summary_text += f"Contralateral: {len(connectivity_patterns['contralateral'])}\n\n"
        
        if p_values is not None:
            sig_count = sum(1 for v in connectivity_patterns['all_pairwise'].values() 
                           if v['significant'])
            summary_text += f"Significant (p<{alpha}): {sig_count}\n"
        
        # Strongest connections
        all_pairs = connectivity_patterns['all_pairwise']
        if all_pairs:
            strongest = max(all_pairs.items(), key=lambda x: abs(x[1]['correlation']))
            summary_text += f"\nStrongest connection:\n{strongest[0]}: r={strongest[1]['correlation']:.3f}"
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig


def compare_fc_between_groups(group1_fc_results, group2_fc_results, group1_name="Group1", group2_name="Group2"):
    """
    Compare functional connectivity patterns between two groups.
    
    Args:
        group1_fc_results: List of FC results dictionaries for group 1
        group2_fc_results: List of FC results dictionaries for group 2  
        group1_name: Name for group 1 (e.g., "Anhedonic")
        group2_name: Name for group 2 (e.g., "Non-anhedonic")
        
    Returns:
        dict: Statistical comparison results
    """
    from scipy import stats as scipy_stats

    # Extract connectivity patterns for each group
    def extract_group_connectivity(fc_results_list):
        group_patterns = {
            'interhemispheric': [],
            'cross_regional': [],
            'ipsilateral': [], 
            'contralateral': [],
            'all_pairwise': []
        }
        
        for fc_result in fc_results_list:
            if fc_result and 'connectivity_patterns' in fc_result:
                patterns = fc_result['connectivity_patterns']
                for pattern_type in group_patterns.keys():
                    if pattern_type in patterns:
                        correlations = [v['correlation'] for v in patterns[pattern_type].values()]
                        group_patterns[pattern_type].extend(correlations)
        
        return group_patterns
    
    group1_patterns = extract_group_connectivity(group1_fc_results)
    group2_patterns = extract_group_connectivity(group2_fc_results)
    
    # Perform statistical comparisons
    comparison_results = {}
    
    for pattern_type in group1_patterns.keys():
        g1_values = group1_patterns[pattern_type]
        g2_values = group2_patterns[pattern_type]
        
        if len(g1_values) > 0 and len(g2_values) > 0:
            # Perform t-test
            t_stat, p_val = scipy_stats.ttest_ind(g1_values, g2_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(g1_values)-1)*np.var(g1_values) + (len(g2_values)-1)*np.var(g2_values)) / (len(g1_values)+len(g2_values)-2))
            cohens_d = (np.mean(g1_values) - np.mean(g2_values)) / pooled_std if pooled_std > 0 else 0
            
            comparison_results[pattern_type] = {
                'group1_mean': np.mean(g1_values),
                'group1_std': np.std(g1_values),
                'group1_n': len(g1_values),
                'group2_mean': np.mean(g2_values),
                'group2_std': np.std(g2_values), 
                'group2_n': len(g2_values),
                'ttest_statistic': t_stat,
                'ttest_pvalue': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05
            }
        else:
            comparison_results[pattern_type] = {
                'group1_mean': np.mean(g1_values) if g1_values else np.nan,
                'group1_std': np.std(g1_values) if g1_values else np.nan,
                'group1_n': len(g1_values),
                'group2_mean': np.mean(g2_values) if g2_values else np.nan,
                'group2_std': np.std(g2_values) if g2_values else np.nan,
                'group2_n': len(g2_values),
                'ttest_statistic': np.nan,
                'ttest_pvalue': np.nan,
                'cohens_d': np.nan,
                'significant': False,
                'note': 'Insufficient data for comparison'
            }
    
    return {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'comparisons': comparison_results,
        'group1_patterns': group1_patterns,
        'group2_patterns': group2_patterns
    }


def create_research_summary(fc_results, subject_info=None):
    """
    Create a structured summary for research reporting.
    
    Args:
        fc_results: FC analysis results
        subject_info: Optional subject information
        
    Returns:
        dict: Structured research summary
    """
    if not fc_results:
        return {'error': 'No FC results available'}
    
    summary = {
        'sample_info': subject_info if subject_info else {},
        'methodology': {
            'roi_extraction': 'Hemisphere-specific extraction using atlas-based parcellation',
            'fc_computation': 'Pearson correlation between ROI mean timeseries',
            'significance_testing': 'Two-tailed correlation with p<0.05 threshold'
        },
        'results': {}
    }
    
    if 'connectivity_patterns' in fc_results:
        patterns = fc_results['connectivity_patterns']
        
        # Key findings
        summary['results']['key_findings'] = {
            'total_connections_tested': len(patterns.get('all_pairwise', {})),
            'significant_connections': sum(1 for v in patterns.get('all_pairwise', {}).values() 
                                         if v.get('significant', False)),
            'strongest_connection': None,
            'interhemispheric_connectivity': {},
            'cross_regional_connectivity': {}
        }
        
        # Find strongest connection
        all_pairs = patterns.get('all_pairwise', {})
        if all_pairs:
            strongest = max(all_pairs.items(), key=lambda x: abs(x[1]['correlation']))
            summary['results']['key_findings']['strongest_connection'] = {
                'pair': strongest[0],
                'correlation': strongest[1]['correlation'],
                'p_value': strongest[1]['p_value'],
                'significant': strongest[1]['significant']
            }
        
        # Interhemispheric summary
        inter_connections = patterns.get('interhemispheric', {})
        if inter_connections:
            summary['results']['key_findings']['interhemispheric_connectivity'] = {
                'vmPFC_hemispheric_correlation': inter_connections.get('vmPFC_RH_vmPFC_LH', {}).get('correlation'),
                'AMY_hemispheric_correlation': inter_connections.get('AMY_rh_AMY_lh', {}).get('correlation'),
                'significant_interhemispheric': sum(1 for v in inter_connections.values() if v.get('significant', False))
            }
        
        # Cross-regional summary
        cross_connections = patterns.get('cross_regional', {})
        if cross_connections:
            summary['results']['key_findings']['cross_regional_connectivity'] = {
                'vmPFC_AMY_connections': {pair: stats_dict['correlation'] 
                                        for pair, stats_dict in cross_connections.items()},
                'significant_cross_regional': sum(1 for v in cross_connections.values() if v.get('significant', False))
            }
        
        # Statistical summary
        all_correlations = [v['correlation'] for v in all_pairs.values()]
        if all_correlations:
            summary['results']['statistical_summary'] = {
                'mean_correlation': np.mean(all_correlations),
                'std_correlation': np.std(all_correlations),
                'range': [np.min(all_correlations), np.max(all_correlations)],
                'median_correlation': np.median(all_correlations)
            }
    
    return summary


def process_subject(subject_id, manager, loader, cortical_atlas, subcortical_atlas, 
                   cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
                   verbose=True):
    """
    Process a single subject for ROI extraction and functional connectivity analysis.
    
    Args:
        subject_id: Subject identifier
        manager: SubjectManager instance
        loader: DataLoader instance 
        cortical_atlas: CorticalAtlasLookup instance
        subcortical_atlas: SubCorticalAtlasLookup instance
        cortical_roi_extractor: ROIExtractionService for cortical data
        subcortical_roi_extractor: ROIExtractionService for subcortical data
        cortical_ROIs: List of cortical ROI names
        subcortical_ROIs: List of subcortical ROI names
        verbose: Whether to print detailed output
        
    Returns:
        dict: Subject analysis results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*60}")
    
    try:
        # Get subject files
        hammer_files = manager.get_subject_files_by_task(subject_id, 'timeseries', 'hammer')
        if not hammer_files:
            return {
                'subject_id': subject_id,
                'error': 'No hammer task files found',
                'success': False
            }
        
        subject_file = loader.resolve_file_path(hammer_files[0])
        
        # Read data from .h5 file
        data = None
        with h5py.File(subject_file, 'r') as file:
            a_group_key = list(file.keys())[0]
            data = np.asarray(file[a_group_key])
            
        if verbose:
            print(f"Found data with shape: {data.shape}")
        
        # Segment into timeseries groups
        cortical_timeseries = data[:400]
        cortical_R, cortical_L = cortical_timeseries[:200], cortical_timeseries[200:]
        cortical_homotopic_pairs = np.asarray(list(zip(cortical_L, cortical_R)))
        subcortical_timeseries = data[400:432]
        cerebellum_timeseries = data[432:]
        
        if verbose:
            print("Found parcels:")
            print(f"Cortical: {cortical_timeseries.shape}")
            print(f"\tLEFT Hemisphere: {cortical_L.shape}")
            print(f"\tRIGHT Hemisphere: {cortical_R.shape}")
            print(f"\tHomotopic Pairs: {cortical_homotopic_pairs.shape}")
            print(f"Subcortical: {subcortical_timeseries.shape}")
            print(f"Cerebellum: {cerebellum_timeseries.shape}")
        
        # ROI Validation and Extraction
        cortical_validation_result = cortical_roi_extractor.validate_roi_coverage(cortical_timeseries, cortical_ROIs)
        subcortical_validation_result = subcortical_roi_extractor.validate_roi_coverage(subcortical_timeseries, subcortical_ROIs)
        
        if verbose:
            print(f"\nCORTICAL ROI Validation Results:")
            print(f"  Valid ROIs: {cortical_validation_result['valid_rois']}")
            print(f"  Invalid ROIs: {cortical_validation_result['invalid_rois']}")
            
            print(f"\nSUBCORTICAL ROI Validation Results:")
            print(f"  Valid ROIs: {subcortical_validation_result['valid_rois']}")
            print(f"  Invalid ROIs: {subcortical_validation_result['invalid_rois']}")
        
        # Extract ROI timeseries
        cortical_roi_timeseries = None
        if cortical_validation_result['valid_rois'] and not cortical_validation_result['coverage_issues']:
            cortical_roi_timeseries = cortical_roi_extractor.extract_roi_timeseries(
                cortical_timeseries, 
                cortical_ROIs, 
                aggregation_method='all'
            )
        
        subcortical_roi_timeseries = None
        if subcortical_validation_result['valid_rois'] and not subcortical_validation_result['coverage_issues']:
            subcortical_roi_timeseries = subcortical_roi_extractor.extract_roi_timeseries(
                subcortical_timeseries,
                subcortical_ROIs,
                aggregation_method='all'
            )
        
        # Hemisphere-specific extraction
        cortical_right_timeseries = None
        cortical_left_timeseries = None
        subcortical_right_timeseries = None
        subcortical_left_timeseries = None
        vmPFC_right = None
        vmPFC_left = None
        amy_right = None
        amy_left = None
        
        cortical_valid_rois = cortical_validation_result['valid_rois']
        
        if cortical_roi_extractor.supports_hemisphere_queries() and cortical_valid_rois:
            if verbose:
                print(f"\n=== HEMISPHERE-SPECIFIC CORTICAL EXTRACTION ===")
            
            cortical_right_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                cortical_timeseries, 
                cortical_valid_rois, 
                hemisphere='RH',
                aggregation_method='mean'
            )
            
            cortical_left_timeseries = cortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                cortical_timeseries, 
                cortical_valid_rois, 
                hemisphere='LH',
                aggregation_method='mean'
            )
            
            if verbose:
                print(f"RIGHT hemisphere extraction results:")
                for roi_name, timeseries in cortical_right_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")
                
                print(f"LEFT hemisphere extraction results:")
                for roi_name, timeseries in cortical_left_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")
            
            # Construct mean vmPFC signal - handle both aggregation methods  
            if cortical_right_timeseries['PFCm'].ndim == 1:
                # Mean aggregation: PFCm and PFCv are already 1D arrays, average them
                vmPFC_right = np.mean([cortical_right_timeseries['PFCm'], 
                                     cortical_right_timeseries['PFCv']], axis=0)
                vmPFC_left = np.mean([cortical_left_timeseries['PFCm'], 
                                    cortical_left_timeseries['PFCv']], axis=0)
            else:
                # 'All' aggregation: PFCm and PFCv are 2D arrays, stack and average
                vmPFC_right = np.mean(np.vstack([cortical_right_timeseries['PFCm'], 
                                               cortical_right_timeseries['PFCv']]), axis=0)
                vmPFC_left = np.mean(np.vstack([cortical_left_timeseries['PFCm'], 
                                              cortical_left_timeseries['PFCv']]), axis=0)
            
            if verbose:
                print(f"Mean vmPFC signal extraction results:")
                print(f"  RIGHT: shape {vmPFC_right.shape}")
                print(f"  LEFT: shape {vmPFC_left.shape}")
        
        subcortical_valid_rois = subcortical_validation_result['valid_rois']
        
        if subcortical_roi_extractor.supports_hemisphere_queries() and subcortical_valid_rois:
            if verbose:
                print(f"\n=== HEMISPHERE-SPECIFIC SUBCORTICAL EXTRACTION ===")
            
            subcortical_right_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                subcortical_timeseries,
                subcortical_valid_rois,
                hemisphere='rh',
                aggregation_method='mean'
            )
            
            subcortical_left_timeseries = subcortical_roi_extractor.extract_roi_timeseries_by_hemisphere(
                subcortical_timeseries,
                subcortical_valid_rois,
                hemisphere='lh', 
                aggregation_method='mean'
            )
            
            if verbose:
                print(f"RIGHT hemisphere extraction results:")
                for roi_name, timeseries in subcortical_right_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")
                        
                print(f"LEFT hemisphere extraction results:")
                for roi_name, timeseries in subcortical_left_timeseries.items():
                    if timeseries.size > 0:
                        print(f"  {roi_name}: shape {timeseries.shape}")
            
            # Construct mean AMY signal - handle both aggregation methods
            if subcortical_right_timeseries['AMY'].ndim == 1:
                # Mean aggregation: AMY is already a 1D array (493,), use directly
                amy_right = subcortical_right_timeseries['AMY']
                amy_left = subcortical_left_timeseries['AMY']
            else:
                # 'All' aggregation: AMY is 2D (N, 493), need to average
                amy_right = np.mean(subcortical_right_timeseries['AMY'], axis=0)
                amy_left = np.mean(subcortical_left_timeseries['AMY'], axis=0)
            
            if verbose:
                print(f"Mean AMY signal extraction results:")
                print(f"  RIGHT: shape {amy_right.shape}")
                print(f"  LEFT: shape {amy_left.shape}")
        
        # Functional connectivity analysis
        fc_results = None
        missing_timeseries = any([v is None for v in [cortical_right_timeseries, cortical_left_timeseries, 
                                                     subcortical_right_timeseries, subcortical_left_timeseries]])
        
        if not missing_timeseries and vmPFC_right is not None and amy_right is not None:
            if verbose:
                print(f"\n=== FUNCTIONAL CONNECTIVITY ANALYSIS ===")
            
            fc_timeseries = {
                'vmPFC_RH': vmPFC_right,
                'vmPFC_LH': vmPFC_left,
                'AMY_rh': amy_right,
                'AMY_lh': amy_left
            }
            
            fc_matrix, fc_labels, fc_pvalues = compute_fc_matrix(fc_timeseries)
            
            if fc_matrix is not None:
                if verbose:
                    print(f"FC Matrix shape: {fc_matrix.shape}")
                    print(f"ROI labels: {fc_labels}")
                
                connectivity_patterns = analyze_connectivity_patterns(fc_matrix, fc_labels, fc_pvalues)
                
                if verbose:
                    print(f"\nConnectivity Pattern Analysis:")
                    print(f"  Total pairwise connections: {len(connectivity_patterns['all_pairwise'])}")
                    print(f"  Interhemispheric connections: {len(connectivity_patterns['interhemispheric'])}")
                    print(f"  Cross-regional connections: {len(connectivity_patterns['cross_regional'])}")
                
                fc_results = {
                    'fc_matrix': fc_matrix,
                    'fc_labels': fc_labels,
                    'fc_pvalues': fc_pvalues,
                    'connectivity_patterns': connectivity_patterns,
                    'timeseries_used': fc_timeseries
                }
        
        return {
            'subject_id': subject_id,
            'success': True,
            'data_shape': data.shape,
            'roi_extraction_results': {
                'cortical': {
                    'atlas_name': cortical_atlas.atlas_name,
                    'roi_timeseries': cortical_roi_timeseries,
                    'requested_rois': cortical_ROIs,
                    'extraction_successful': cortical_roi_timeseries is not None,
                    'hemisphere_specific': {
                        'right_hemisphere': cortical_right_timeseries,
                        'left_hemisphere': cortical_left_timeseries,
                        'supports_hemisphere_queries': cortical_roi_extractor.supports_hemisphere_queries()
                    }
                },
                'subcortical': {
                    'atlas_name': subcortical_atlas.atlas_name,
                    'roi_timeseries': subcortical_roi_timeseries,
                    'requested_rois': subcortical_ROIs,
                    'extraction_successful': subcortical_roi_timeseries is not None,
                    'hemisphere_specific': {
                        'right_hemisphere': subcortical_right_timeseries,
                        'left_hemisphere': subcortical_left_timeseries,
                        'supports_hemisphere_queries': subcortical_roi_extractor.supports_hemisphere_queries()
                    }
                }
            },
            'functional_connectivity': fc_results,
            'mean_signals': {
                'vmPFC_right': vmPFC_right,
                'vmPFC_left': vmPFC_left,
                'amy_right': amy_right,
                'amy_left': amy_left
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to process subject {subject_id}: {str(e)}")
        return {
            'subject_id': subject_id,
            'error': str(e),
            'success': False
        }


def main():
    """Main function for FC MVP analysis"""
    print("=== Functional Connectivity MVP ===")
    
    # ===== CONFIGURATION FOR MULTI-SUBJECT ANALYSIS =====
    LIMIT_SUBJECTS = True  # Set False for full analysis
    MAX_SUBJECTS_PER_GROUP = 2  # Limit when testing (only used if LIMIT_SUBJECTS=True)
    SHOW_INDIVIDUAL_PLOTS = True  # Show individual subject plots
    VERBOSE_SUBJECT_OUTPUT = True  # Detailed per-subject printing
    SHOW_GROUP_SUMMARY = True  # Show aggregated group analysis
    
    print(f"Configuration:")
    print(f"  Subject limiting: {'ENABLED' if LIMIT_SUBJECTS else 'DISABLED'}")
    if LIMIT_SUBJECTS:
        print(f"  Max subjects per group: {MAX_SUBJECTS_PER_GROUP}")
    print(f"  Individual plots: {'ENABLED' if SHOW_INDIVIDUAL_PLOTS else 'DISABLED'}")
    print(f"  Verbose output: {'ENABLED' if VERBOSE_SUBJECT_OUTPUT else 'DISABLED'}")
    print()
    
    # Initialize data infrastructure
    loader = DataLoader()
    manager = SubjectManager(data_loader=loader)
    
    print(f"[OK] Loaded manifest with {len(loader.get_all_subject_ids())} subjects")
    
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
                # Check if at least one hammer file is actually downloaded (not a git-annex symlink)
                first_file = loader.resolve_file_path(hammer_files[0])
                if is_actual_file(first_file):
                    accessible_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer file exists but not downloaded (git-annex symlink)")
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
                if is_actual_file(first_file):
                    accessible_non_anhedonic.append(subject_id)
                else:
                    print(f"    Warning: {subject_id} - hammer file exists but not downloaded (git-annex symlink)")
            else:
                print(f"    Warning: {subject_id} - no hammer task files available")
        except Exception as e:
            print(f"    Error accessing {subject_id}: {e}")
    
    # Report final accessible counts
    print(f"\nFinal Processing Summary:")
    print(f"  Anhedonic subjects (accessible): {len(accessible_anhedonic)}")
    print(f"  Non-anhedonic subjects (accessible): {len(accessible_non_anhedonic)}")
    print(f"  Total ready for FC analysis (hammer task only): {len(accessible_anhedonic) + len(accessible_non_anhedonic)}")

    # Check if we have any accessible subjects
    if len(accessible_anhedonic) == 0 and len(accessible_non_anhedonic) == 0:
        print(f"\n[ERROR] No subjects have actually downloaded timeseries files!")
        print(f"   The files exist as git-annex symlinks but haven't been fetched yet.")
        print(f"\n   To download timeseries data, run:")
        print(f"   cd {loader.base_path}")
        print(f"   datalad get fMRI_timeseries_clean_denoised_GSR_parcellated/")
        print(f"\n   Or use the preprocessing pipeline with timeseries data type enabled.")
        return {
            'error': 'No downloaded timeseries files',
            'anhedonic_subjects': [],
            'non_anhedonic_subjects': [],
            'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        }
    
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
    
    # ===== ATLAS INITIALIZATION =====
    print(f"\n{'='*50}")
    print(f"INITIALIZING ATLAS SYSTEMS")
    print(f"{'='*50}")
    
    # Initialize modular ROI extraction system
    cortical_lut_file = Path(__file__).parent / 'parcellations/cortical/yeo17/400Parcels_Yeo2011_17Networks_info.txt'
    subcortical_lut_file = Path(__file__).parent / 'parcellations/subcortical/tian/Tian_Subcortex_S2_3T_label.txt'
    cortical_atlas = CorticalAtlasLookup(cortical_lut_file)
    subcortical_atlas = SubCorticalAtlasLookup(subcortical_lut_file)
    cortical_roi_extractor = ROIExtractionService(cortical_atlas)
    subcortical_roi_extractor = ROIExtractionService(subcortical_atlas)
    
    # Define ROIs of interest
    cortical_ROIs = [
        'PFCm',  # medial PFC
        'PFCv',  # ventral PFC
    ]
    
    subcortical_ROIs = [
        'AMY',  # whole amygdala
    ]
    
    print(f"Initialized atlases:")
    print(f"  Cortical: {cortical_atlas.atlas_name} ({cortical_atlas.total_parcels} parcels)")
    print(f"  Subcortical: {subcortical_atlas.atlas_name} ({subcortical_atlas.total_parcels} parcels)")
    print(f"ROIs of interest:")
    print(f"  Cortical: {cortical_ROIs}")
    print(f"  Subcortical: {subcortical_ROIs}")
    
    # ===== MULTI-SUBJECT PROCESSING =====
    print(f"\n{'='*80}")
    print(f"STARTING MULTI-SUBJECT ANALYSIS")
    print(f"{'='*80}")
    
    # Apply subject limiting if enabled
    if LIMIT_SUBJECTS:
        anhedonic_subjects_to_process = accessible_anhedonic[:MAX_SUBJECTS_PER_GROUP]
        non_anhedonic_subjects_to_process = accessible_non_anhedonic[:MAX_SUBJECTS_PER_GROUP]
        print(f"LIMITING ENABLED: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")
    else:
        anhedonic_subjects_to_process = accessible_anhedonic
        non_anhedonic_subjects_to_process = accessible_non_anhedonic
        print(f"FULL ANALYSIS: Processing {len(anhedonic_subjects_to_process)} anhedonic + {len(non_anhedonic_subjects_to_process)} non-anhedonic subjects")
    
    # Process all subjects
    anhedonic_results = {}
    non_anhedonic_results = {}
    
    # Process anhedonic subjects
    print(f"\n{'='*50}")
    print(f"PROCESSING ANHEDONIC SUBJECTS ({len(anhedonic_subjects_to_process)})")
    print(f"{'='*50}")
    
    for i, subject_id in enumerate(anhedonic_subjects_to_process, 1):
        print(f"\n[{i}/{len(anhedonic_subjects_to_process)}] Processing anhedonic subject: {subject_id}")
        
        subject_result = process_subject(
            subject_id, manager, loader, cortical_atlas, subcortical_atlas,
            cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
            verbose=VERBOSE_SUBJECT_OUTPUT
        )
        
        anhedonic_results[subject_id] = subject_result
        
        if subject_result['success']:
            print(f"    ✅ Success: {subject_id}")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")
    
    # Process non-anhedonic subjects   
    print(f"\n{'='*50}")
    print(f"PROCESSING NON-ANHEDONIC SUBJECTS ({len(non_anhedonic_subjects_to_process)})")
    print(f"{'='*50}")
    
    for i, subject_id in enumerate(non_anhedonic_subjects_to_process, 1):
        print(f"\n[{i}/{len(non_anhedonic_subjects_to_process)}] Processing non-anhedonic subject: {subject_id}")
        
        subject_result = process_subject(
            subject_id, manager, loader, cortical_atlas, subcortical_atlas,
            cortical_roi_extractor, subcortical_roi_extractor, cortical_ROIs, subcortical_ROIs,
            verbose=VERBOSE_SUBJECT_OUTPUT
        )
        
        non_anhedonic_results[subject_id] = subject_result
        
        if subject_result['success']:
            print(f"    ✅ Success: {subject_id}")
        else:
            print(f"    ❌ Failed: {subject_id} - {subject_result.get('error', 'Unknown error')}")
    
    # ===== RESULTS SUMMARY =====
    print(f"\n{'='*80}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    anhedonic_success = sum(1 for r in anhedonic_results.values() if r['success'])
    non_anhedonic_success = sum(1 for r in non_anhedonic_results.values() if r['success'])
    total_success = anhedonic_success + non_anhedonic_success
    total_processed = len(anhedonic_results) + len(non_anhedonic_results)
    
    print(f"Successfully processed: {total_success}/{total_processed} subjects")
    print(f"  Anhedonic: {anhedonic_success}/{len(anhedonic_results)}")
    print(f"  Non-anhedonic: {non_anhedonic_success}/{len(non_anhedonic_results)}")
    
    # Collect FC results for group comparison
    anhedonic_fc_results = []
    non_anhedonic_fc_results = []
    
    for subject_id, result in anhedonic_results.items():
        if result['success'] and result.get('functional_connectivity'):
            anhedonic_fc_results.append(result['functional_connectivity'])
    
    for subject_id, result in non_anhedonic_results.items():
        if result['success'] and result.get('functional_connectivity'):
            non_anhedonic_fc_results.append(result['functional_connectivity'])
    
    print(f"FC analysis available:")
    print(f"  Anhedonic: {len(anhedonic_fc_results)} subjects")
    print(f"  Non-anhedonic: {len(non_anhedonic_fc_results)} subjects")
    
    # ===== GROUP COMPARISON ANALYSIS =====
    group_comparison_results = None
    if len(anhedonic_fc_results) > 0 and len(non_anhedonic_fc_results) > 0:
        print(f"\n{'='*80}")
        print(f"GROUP COMPARISON ANALYSIS")
        print(f"{'='*80}")
        
        group_comparison_results = compare_fc_between_groups(
            anhedonic_fc_results, 
            non_anhedonic_fc_results,
            group1_name="Anhedonic",
            group2_name="Non-anhedonic"
        )
        
        print(f"Statistical Comparisons (Anhedonic vs Non-anhedonic):")
        for pattern_type, comparison in group_comparison_results['comparisons'].items():
            if not comparison.get('note'):  # Skip patterns with insufficient data
                print(f"\n{pattern_type.upper()}:")
                print(f"  Anhedonic: M={comparison['group1_mean']:.3f}, SD={comparison['group1_std']:.3f}, N={comparison['group1_n']}")
                print(f"  Non-anhedonic: M={comparison['group2_mean']:.3f}, SD={comparison['group2_std']:.3f}, N={comparison['group2_n']}")
                print(f"  t({comparison['group1_n']+comparison['group2_n']-2})={comparison['ttest_statistic']:.3f}, p={comparison['ttest_pvalue']:.3f}")
                print(f"  Cohen's d={comparison['cohens_d']:.3f}, Significant={'Yes' if comparison['significant'] else 'No'}")
        
    else:
        print(f"\n[WARNING] Insufficient FC data for group comparison")
        print(f"  Need at least 1 subject per group with successful FC analysis")
    
    # ===== INDIVIDUAL SUBJECT PLOTS =====
    individual_plots_created = 0
    if SHOW_INDIVIDUAL_PLOTS and total_success <= 3:  # Only show individual plots for ≤3 subjects
        print(f"\n{'='*80}")
        print(f"CREATING INDIVIDUAL SUBJECT PLOTS")
        print(f"{'='*80}")
        
        all_results = {**anhedonic_results, **non_anhedonic_results}
        for subject_id, result in all_results.items():
            if result['success'] and result.get('functional_connectivity'):
                print(f"Creating plot for {subject_id}...")
                fc_data = result['functional_connectivity']
                
                if fc_data['fc_matrix'] is not None:
                    # Create subject-specific plot
                    fc_fig = plot_fc_results(
                        fc_data['fc_matrix'], 
                        fc_data['fc_labels'], 
                        fc_data['fc_pvalues'], 
                        fc_data['connectivity_patterns']
                    )
                    fc_fig.suptitle(f'Functional Connectivity: {subject_id}', fontsize=16)
                    individual_plots_created += 1
        
        print(f"Created {individual_plots_created} individual subject plots")
    elif total_success > 3:
        print(f"\n[INFO] Skipping individual plots (too many subjects: {total_success})")
        print(f"      Individual plots only shown for ≤3 subjects")
    else:
        print(f"\n[INFO] No individual plots created (no successful subjects)")
    
    # ===== RETURN MULTI-SUBJECT RESULTS =====
    return {
        'anhedonic_subjects': anhedonic_subjects_to_process,
        'non_anhedonic_subjects': non_anhedonic_subjects_to_process,
        'processing_mode': 'downloaded_only' if use_downloaded_only else 'all_available',
        'configuration': {
            'limit_subjects': LIMIT_SUBJECTS,
            'max_subjects_per_group': MAX_SUBJECTS_PER_GROUP if LIMIT_SUBJECTS else None,
            'show_individual_plots': SHOW_INDIVIDUAL_PLOTS,
            'verbose_subject_output': VERBOSE_SUBJECT_OUTPUT
        },
        'summary': {
            'total_processed': total_processed,
            'total_successful': total_success,
            'anhedonic_processed': len(anhedonic_results),
            'anhedonic_successful': anhedonic_success,
            'non_anhedonic_processed': len(non_anhedonic_results),
            'non_anhedonic_successful': non_anhedonic_success,
            'individual_plots_created': individual_plots_created
        },
        'subject_results': {
            'anhedonic': anhedonic_results,
            'non_anhedonic': non_anhedonic_results
        },
        'group_analysis': {
            'anhedonic_fc_count': len(anhedonic_fc_results),
            'non_anhedonic_fc_count': len(non_anhedonic_fc_results),
            'group_comparison': group_comparison_results
        }
    }


if __name__ == '__main__':
    SHOW_PLOTS = True
    main()
    
    if SHOW_PLOTS:
        plt.show()