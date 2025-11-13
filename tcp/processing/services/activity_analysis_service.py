"""
Activity Analysis Service for signal envelope computation.

This service handles Hilbert transform and envelope analysis for fMRI timeseries.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import signal


class ActivityAnalysisService:
    """Service for computing activity envelopes via Hilbert transform.

    This service encapsulates the signal processing for activity analysis:
    - Hilbert transform
    - Analytic signal computation
    - Envelope extraction (magnitude of analytic signal)
    - Low-pass filtering of envelopes
    """

    # Dataset-specific constants (specific to this fMRI dataset)
    TR = 0.8  # Repetition Time [seconds]
    FILTER_ORDER = 2
    CUTOFF_FREQUENCY_HZ = 0.2

    def compute_envelope_analysis(
        self,
        channels: np.ndarray,
        labels: List[str],
        verbose: bool = False
    ) -> Dict:
        """Perform Hilbert transform and envelope computation.

        Args:
            channels: Array of timeseries data (n_channels x n_timepoints)
            labels: List of channel label strings
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing:
                - 'all_channels': Original timeseries
                - 'analytic_signal': Complex analytic signal
                - 'hilbert_transforms': Hilbert transforms
                - 'analytic_envelope': Raw envelope (magnitude of analytic signal)
                - 'smoothed_envelope': LP-filtered envelope
                - 'timeseries_used': Dictionary mapping labels to timeseries
                - 'filtered_timeseries': Filtered analytic signal
                - 'filter_used': Filter coefficients (b, a)
                - 'channel_label_map': Label identity mapping
                - 'channel_labels': List of all channel labels
        """
        # Create timeseries dictionary
        activity_timeseries = {}
        for i, channel_label in enumerate(labels):
            activity_timeseries[channel_label] = channels[i]

        # Perform Hilbert transform on each timeseries
        hilbert_transforms = signal.hilbert(channels)

        # Derive analytic signals: z(t) = x(t) + j * H{x(t)}
        analytic_timeseries = channels + 1j * hilbert_transforms

        # Compute envelope of analytic signal (activity measure)
        analytic_envelope = np.abs(analytic_timeseries)

        # Apply low-pass filter for smoothing envelope
        sampling_rate = 1 / self.TR  # 1.25 Hz
        nyquist_frequency = 0.5 * sampling_rate  # 0.625 Hz

        # Digital filter critical frequencies must be 0 < Wn < 1
        normalized_cutoff = self.CUTOFF_FREQUENCY_HZ / nyquist_frequency
        b, a = signal.butter(self.FILTER_ORDER, normalized_cutoff, btype='low', analog=False)

        filtered_timeseries = signal.lfilter(b, a, analytic_timeseries)
        filtered_envelope = np.abs(filtered_timeseries)

        if verbose:
            print(f"All Channels shape: {channels.shape}")
            print(f"Hilbert Transformed channels shape: {hilbert_transforms.shape}")
            print(f"Analytic channels shape: {analytic_timeseries.shape}")

        # Create channel label map (identity mapping for display)
        channel_label_map = {label: label for label in labels}

        return {
            'all_channels': channels,
            'analytic_signal': analytic_timeseries,
            'hilbert_transforms': hilbert_transforms,
            'analytic_envelope': analytic_envelope,
            'smoothed_envelope': filtered_envelope,
            'timeseries_used': activity_timeseries,
            'filtered_timeseries': filtered_timeseries,
            'filter_used': (b, a),
            'channel_label_map': channel_label_map,
            'channel_labels': labels
        }
