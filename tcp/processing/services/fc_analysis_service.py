"""
Functional Connectivity Analysis Service.

This service handles static FC computation (Pearson correlations) and
connectivity pattern analysis.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


class FCAnalysisService:
    """Service for computing static functional connectivity.

    This service encapsulates:
    - FC matrix computation (Pearson correlations)
    - Statistical testing (p-values)
    - Connectivity pattern analysis
    """

    def compute_static_fc(
        self,
        channels: np.ndarray,
        labels: List[str],
        verbose: bool = False
    ) -> Dict:
        """Compute static functional connectivity matrix and patterns.

        Args:
            channels: Array of timeseries data (n_channels x n_timepoints)
            labels: List of channel label strings
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing:
                - 'static_fc_matrix': Correlation matrix
                - 'static_fc_labels': ROI labels
                - 'static_fc_pvalues': P-values matrix
                - 'static_connectivity_patterns': Analyzed connectivity patterns
                - 'channel_label_map': Label identity mapping
        """
        # Create FC timeseries dictionary
        fc_timeseries = {}
        for i, channel_label in enumerate(labels):
            fc_timeseries[channel_label] = channels[i]

        # Compute FC matrix
        fc_matrix, fc_labels, fc_pvalues = self._compute_fc_matrix(fc_timeseries)

        if fc_matrix is None:
            return None

        if verbose:
            print(f"FC Matrix shape: {fc_matrix.shape}")
            print(f"ROI labels (alphabetical): {sorted(fc_labels)}")

        # Analyze connectivity patterns
        connectivity_patterns = self._analyze_connectivity_patterns(
            fc_matrix,
            fc_labels,
            fc_pvalues
        )

        if verbose:
            print(f"Interhemispheric connections (ALL): {connectivity_patterns['interhemispheric'].keys()}")
            print(f"\nConnectivity Pattern Analysis:")
            print(f"  Total pairwise connections: {len(connectivity_patterns['all_pairwise'])}")
            print(f"  Interhemispheric connections: {len(connectivity_patterns['interhemispheric'])}")
            print(f"  Cross-regional connections: {len(connectivity_patterns['cross_regional'])}")

        # Create channel label map
        channel_label_map = {label: label for label in labels}

        return {
            'static_fc_matrix': fc_matrix,
            'static_fc_labels': fc_labels,
            'static_fc_pvalues': fc_pvalues,
            'static_connectivity_patterns': connectivity_patterns,
            'channel_label_map': channel_label_map
        }

    def _compute_fc_matrix(
        self,
        timeseries_dict: Dict[str, np.ndarray],
        roi_names: List[str] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Compute functional connectivity matrix using Pearson correlation.

        Args:
            timeseries_dict: Dictionary mapping ROI names to timeseries arrays
            roi_names: Optional list of ROI names (if None, uses dict keys)

        Returns:
            Tuple of (correlation_matrix, roi_labels, p_values)
        """
        if roi_names is None:
            roi_names = list(timeseries_dict.keys())

        n_rois = len(roi_names)

        # Initialize correlation and p-value matrices
        corr_matrix = np.zeros((n_rois, n_rois))
        p_values = np.ones((n_rois, n_rois))

        # Compute pairwise correlations
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_values[i, j] = 0.0
                else:
                    ts_i = timeseries_dict[roi_names[i]]
                    ts_j = timeseries_dict[roi_names[j]]

                    # Pearson correlation
                    r, p = stats.pearsonr(ts_i, ts_j)
                    corr_matrix[i, j] = r
                    p_values[i, j] = p

        return corr_matrix, roi_names, p_values

    def _analyze_connectivity_patterns(
        self,
        corr_matrix: np.ndarray,
        roi_labels: List[str],
        p_values: np.ndarray = None,
        alpha: float = 0.05
    ) -> Dict:
        """Analyze connectivity patterns from correlation matrix.

        Args:
            corr_matrix: Correlation matrix
            roi_labels: ROI labels
            p_values: Optional p-values matrix
            alpha: Significance threshold

        Returns:
            Dictionary with categorized connectivity patterns
        """
        n_rois = len(roi_labels)

        patterns = {
            'all_pairwise': {},
            'interhemispheric': {},
            'cross_regional': {},
            'ipsilateral': {},
            'contralateral': {}
        }

        # Analyze all pairwise connections
        for i in range(n_rois):
            for j in range(i + 1, n_rois):  # Upper triangle only
                label_i = roi_labels[i]
                label_j = roi_labels[j]

                correlation = corr_matrix[i, j]
                significant = p_values[i, j] < alpha if p_values is not None else True

                connection_key = f"{label_i}_{label_j}"
                connection_info = {
                    'correlation': correlation,
                    'significant': significant,
                    'roi_1': label_i,
                    'roi_2': label_j
                }

                if p_values is not None:
                    connection_info['p_value'] = p_values[i, j]

                # Store in all_pairwise
                patterns['all_pairwise'][connection_key] = connection_info

                # Categorize connection type
                # Interhemispheric: same region, different hemispheres
                if self._is_interhemispheric(label_i, label_j):
                    patterns['interhemispheric'][connection_key] = connection_info

                # Cross-regional: different regions
                if self._is_cross_regional(label_i, label_j):
                    patterns['cross_regional'][connection_key] = connection_info

                    # Further classify as ipsilateral or contralateral
                    if self._same_hemisphere(label_i, label_j):
                        patterns['ipsilateral'][connection_key] = connection_info
                    else:
                        patterns['contralateral'][connection_key] = connection_info

        return patterns

    def _is_interhemispheric(self, label_1: str, label_2: str) -> bool:
        """Check if connection is interhemispheric (same region, different hemispheres)."""
        # Extract region name (before first _RH_ or _LH_)
        region_1 = label_1.split('_RH_')[0].split('_LH_')[0]
        region_2 = label_2.split('_RH_')[0].split('_LH_')[0]

        # Same region, different hemispheres
        if region_1 == region_2:
            has_rh_1 = '_RH_' in label_1 or label_1.endswith('-rh')
            has_lh_1 = '_LH_' in label_1 or label_1.endswith('-lh')
            has_rh_2 = '_RH_' in label_2 or label_2.endswith('-rh')
            has_lh_2 = '_LH_' in label_2 or label_2.endswith('-lh')

            return (has_rh_1 and has_lh_2) or (has_lh_1 and has_rh_2)

        return False

    def _is_cross_regional(self, label_1: str, label_2: str) -> bool:
        """Check if connection is cross-regional (different regions)."""
        region_1 = label_1.split('_RH_')[0].split('_LH_')[0].split('-')[0]
        region_2 = label_2.split('_RH_')[0].split('_LH_')[0].split('-')[0]

        return region_1 != region_2

    def _same_hemisphere(self, label_1: str, label_2: str) -> bool:
        """Check if both ROIs are in the same hemisphere."""
        has_rh_1 = '_RH_' in label_1 or label_1.endswith('-rh')
        has_lh_1 = '_LH_' in label_1 or label_1.endswith('-lh')
        has_rh_2 = '_RH_' in label_2 or label_2.endswith('-rh')
        has_lh_2 = '_LH_' in label_2 or label_2.endswith('-lh')

        return (has_rh_1 and has_rh_2) or (has_lh_1 and has_lh_2)
