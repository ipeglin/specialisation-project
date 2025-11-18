#!/usr/bin/env python3
"""
MVMD Implementation

This implementation properly handles the reconstruction constraint and works
with real signals while using analytic representation only for bandwidth computation.

The subsequenct script is a class refactoring of an original implementation by
Dmocrito [1], who implements the MVMD optimization problem introduced in [2].

I do not claim to have ownership of this code, and have simply rewritten it to match
my subjective preferences and the overall code formatting of this project.
The original code provided in [1] is open to public use through the MIT license.

Author: Dmocrito
Date: 05-03-2025

Refactored by: Ian Philip Eglin
Date: 18-11-2025

---------------------------------------------------------------------
[1] Dmocrito (2025) mvmd, GitHub (https://github.com/Dmocrito/mvmd/tree/main)
[2] N. Rehman and H. Aftab (2019) Multivariate Variational Mode Decomposition, IEEE Transactions on Signal Processing
"""

from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray


class MVMD:
    """
    Corrected Multivariate Variational Mode Decomposition (MVMD) Implementation

    Based on: N. Rehman and H. Aftab, "Multivariate Variational Mode Decomposition"

    Parameters:
        alpha : float, default=2000
            Bandwidth constraint parameter
        tolerance : float, default=1e-3
            Stopping criterion for the dual ascent
        init : str, default='uniform'
            Frequency distribution initialization ('uniform', 'linear' or 'log')
        tau : float, default=1e-2
            Time-step of the dual ascent (use 0 for noise-slack)
        max_iter : int, default=500
            Maximum number of iterations allowed
        sampling_rate : float, default=1
            Sampling rate of input signal used for normalization
        verbose : bool, default=False
            Print information about the covergence of the algorithm
    """

    def __init__(
        self,
        config = None,
    ):
        if config is None:
            config = {
                'alpha': 2000,
                'tolerance': 1e-3,
                'init': 'uniform',
                'tau': 1e-2,
                'sampling_rate': 1,
                'max_iter': 500,
                'verbose': False,
            }

        self.config = config
        config_res = self._verify_configuration()
        if not config_res['success']:
            raise ValueError(f"Configuration verification failed, got {config_res['message']}")

        # Results
        self.modes_ = None
        self.center_frequencies_ = None
        self.n_iter_ = None

    @staticmethod
    def _fourier_transform(time_signal: NDArray, pad_mode='symmetric') -> NDArray:
        sample_count = time_signal.shape[1]

        match pad_mode:
            case None:
                signal = time_signal
            case _:
                signal = np.pad(time_signal, pad_width=((0, 0), (sample_count // 2, sample_count - sample_count // 2)), mode=pad_mode)

        freq_signal = np.fft.fft(signal, axis=1)[:, :sample_count + 1]

        return freq_signal

    @staticmethod
    def _inv_fourier_transform(freq_signal: NDArray, extended=False):
        channel_count, freq_sample_count = freq_signal.shape
        reduced_sample_count = freq_sample_count - 1

        # Hermitian-symmetric assuming freq_signal is real
        full_freq_signal = np.zeros((channel_count, 2 * reduced_sample_count), dtype=complex)

        full_freq_signal[:, reduced_sample_count:] = freq_signal[:, :reduced_sample_count]
        full_freq_signal[:, :reduced_sample_count] = np.conj(freq_signal[:, reduced_sample_count:0:-1])

        # Inverse FFT
        shifted_time_signal = np.fft.ifftshift(full_freq_signal, axes=1)
        time_signal = np.real(np.fft.ifft(shifted_time_signal, axis=1))

        if extended:
            return time_signal

        return time_signal[:, (reduced_sample_count // 2):(3 * reduced_sample_count // 2)]

    def decompose(self, time_signal: NDArray, num_modes: int):
        """
        Run the MVMD algorithm on time-domain signal

        All signals and variables should be considered as frequency-domain
            representations unless explicitly prefixed with 'time_', in contrast
            to the original code by Dmocrito using suffic '_hat'.

        Parameters:
            time_signal: ndarray
                Any time-domain signal to decompose
            num_modes: int
                The K number of modes to decompose the signal into
        """
        self._validate_input(time_signal)

        channel_count, sample_count = time_signal.shape

        if self.config['verbose']:
            print(f"\nParameters:")
            print(f"Signal: {channel_count}-channel with {sample_count} sample(s)")
            print(f" Modes: Decomposing into {num_modes}")

        # Initiate frequencies
        center_freqs = np.zeros((self.config['max_iter'] + 1, num_modes))

        match self.config['init']:
            case 'linear':
                center_freqs[0, :] = np.linspace(0, 0.5, num_modes)
            case 'log':
                center_freqs[0, :] = np.logspace(-3, 0, num_modes)
            case _:
                center_freqs[0, :] = 0 # 'uniform'

        freq_sample_count = sample_count + 1
        freqs = np.linspace(0, 0.5, freq_sample_count)

        # Compute frequency-domain signal
        signal = MVMD._fourier_transform(time_signal)

        # Initiate modes
        modes = np.zeros((num_modes, channel_count, freq_sample_count), dtype=complex)

        # === MVMD algorithm ===
        lagrangian = np.zeros((self.config['max_iter'] + 1, channel_count, freq_sample_count), dtype=complex)
        # Langrangian multiplier lambda
        residual_difference = self.config['tolerance'] + np.finfo(float).eps # Stopping criterion

        iteration_count = 0

        while iteration_count < self.config['max_iter'] and residual_difference > self.config['tolerance']:
            residual_difference = 0

            for k in range(num_modes):
                aux_modes = np.copy(modes[k, :, :])
                modes[k, :, :] = 0 # Remove contribution from previous iterations

                modes[k, :, :] = signal - np.sum(modes, axis=0) - 0.5 * lagrangian[iteration_count, :, :]
                modes[k, :, :] /= 1 + self.config['alpha'] * (freqs - center_freqs[iteration_count, k]) ** 2

                residual_difference += np.sum(np.abs(modes[k, :, :] - aux_modes) ** 2)

                mode_power_spectra = np.abs(modes[k, :, :]) ** 2

                center_freqs[iteration_count+1, k] = np.sum(np.dot(mode_power_spectra, freqs))
                center_freqs[iteration_count+1, k] /= np.sum(mode_power_spectra)

                # Dual ascent
                lagrangian[iteration_count+1, :, :] = (
                    lagrangian[iteration_count, :, :] + self.config['tau'] * (np.sum(modes, axis=0) - signal)
                )

            iteration_count += 1

            # Convergence modulation
            residual_difference /= sample_count

        # Normalize w.r.t. sampling rate
        center_freqs = center_freqs[:iteration_count, :] / self.config['sampling_rate']

        # Order frequencies
        idx = np.argsort(center_freqs[-1, :])
        center_freqs = center_freqs[:, idx]

        # Signal reconstruction
        time_modes = np.array([MVMD._inv_fourier_transform(mode) for mode in modes])

        # Order modes
        time_modes = time_modes[idx, :, :]

        # Store results to instance
        self.modes_ = {
            'time_modes': time_modes,
            'freq_modes': modes,
        }
        self.center_frequencies_ = center_freqs
        self.n_iter_ = iteration_count

        return {
            'success': True,
            'message': f'Successfully decomposed signal with shape {time_signal.shape}',
            'time_modes': time_modes,
            'freq_modes': modes,
            'center_freqs': center_freqs,
        }

    def _verify_configuration(self):
        if self.config is None:
            return {
                'success': False,
                'message': 'Configuration is None',
            }

        return {
            'success': True,
            'message': 'Configuration valid'
        }

    def _validate_input(self, signal: NDArray) -> None:
        """Validate input."""
        if signal is None:
            raise TypeError("Signal is None")
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be numpy array")
        if signal.ndim != 2:
            raise ValueError("Signal must be 2D (channels × samples)")
        if signal.shape[1] < 4:
            raise ValueError("Need at least 4 samples")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("\n" + "="*80)
    print(" MVMD Testing and Validation ".center(80, "="))
    print("="*80)

    # ===== Test 1: Simple Multi-tone Signal =====
    print("\n=== Test 1: Simple Multi-tone Signal ===")

    # Generate test signal with 3 known frequency components
    N = 1000
    fs = 1000  # Sampling rate (Hz)
    t = np.arange(N) / fs

    # Define 3 modes with different frequencies
    f1, f2, f3 = 10, 50, 120  # Hz

    # Mode 1: Low frequency (different amplitudes per channel)
    mode1_ch1 = 1.0 * np.sin(2 * np.pi * f1 * t)
    mode1_ch2 = 0.8 * np.sin(2 * np.pi * f1 * t + np.pi/4)

    # Mode 2: Medium frequency
    mode2_ch1 = 0.5 * np.sin(2 * np.pi * f2 * t)
    mode2_ch2 = 0.6 * np.sin(2 * np.pi * f2 * t - np.pi/6)

    # Mode 3: High frequency
    mode3_ch1 = 0.3 * np.sin(2 * np.pi * f3 * t)
    mode3_ch2 = 0.4 * np.sin(2 * np.pi * f3 * t + np.pi/3)

    # Combine modes
    channel1 = mode1_ch1 + mode2_ch1 + mode3_ch1
    channel2 = mode1_ch2 + mode2_ch2 + mode3_ch2

    signal = np.vstack([channel1, channel2])

    print(f"Signal shape: {signal.shape}")
    print(f"True frequencies: {f1} Hz, {f2} Hz, {f3} Hz")

    # Apply MVMD
    config = {
        'alpha': 2000,
        'init': 'uniform',
        'sampling_rate': fs,
        'tolerance': 1e-6,
        'tau': 0,
        'max_iter': 500,
        'verbose': True,
    }

    mvmd = MVMD(config)
    result = mvmd.decompose(signal, num_modes=3)

    time_modes = result['time_modes']
    center_freqs = result['center_freqs'][-1, :]  # Final frequencies

    print(f"\nExtracted center frequencies: {center_freqs} Hz")
    print(f"Modes shape: {time_modes.shape}")

    # Compute reconstruction error
    reconstructed = np.sum(time_modes, axis=0)
    recon_error = np.linalg.norm(signal - reconstructed) / np.linalg.norm(signal)
    print(f"Reconstruction error: {recon_error:.2e}")

    # ===== Plotting Test 1 =====
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle('MVMD Test 1: Simple Multi-tone Signal', fontsize=16, fontweight='bold')

    # Plot original signals
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(t[:500], signal[0, :500], linewidth=1.5, color='black')
    ax1.set_title('Original Signal - Channel 1', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(t[:500], signal[1, :500], linewidth=1.5, color='darkblue')
    ax2.set_title('Original Signal - Channel 2', fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    # Plot extracted modes
    colors = ['crimson', 'green', 'purple']
    for k in range(3):
        # Channel 1
        ax = plt.subplot(4, 2, 2*k + 3)
        ax.plot(t[:500], time_modes[k, 0, :500], color=colors[k], linewidth=1.5)
        ax.set_title(f'Mode {k+1} - Ch1 (f={center_freqs[k]:.1f} Hz)', fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        # Channel 2
        ax = plt.subplot(4, 2, 2*k + 4)
        ax.plot(t[:500], time_modes[k, 1, :500], color=colors[k], linewidth=1.5)
        ax.set_title(f'Mode {k+1} - Ch2 (f={center_freqs[k]:.1f} Hz)', fontweight='bold')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    plt.xlabel('Time (s)')
    plt.tight_layout()

    # ===== Test 2: Frequency Spectrum Analysis =====
    print("\n=== Test 2: Frequency Spectrum Analysis ===")

    # Compute FFT for verification
    freqs_fft = np.fft.fftfreq(N, 1/fs)
    freqs_pos = freqs_fft[:N//2]

    signal_fft_ch1 = np.abs(np.fft.fft(signal[0]))[:N//2]
    mode_ffts_ch1 = [np.abs(np.fft.fft(time_modes[k, 0]))[:N//2] for k in range(3)]

    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('MVMD Test 2: Frequency Spectrum Analysis', fontsize=16, fontweight='bold')

    # Original signal spectrum
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(freqs_pos, signal_fft_ch1, linewidth=1.5, color='black')
    ax1.set_title('Original Signal Spectrum (Channel 1)', fontweight='bold')
    ax1.set_ylabel('Magnitude')
    ax1.set_xlim([0, 150])
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=f1, color='crimson', linestyle='--', alpha=0.7, label=f'True: {f1} Hz')
    ax1.axvline(x=f2, color='green', linestyle='--', alpha=0.7, label=f'True: {f2} Hz')
    ax1.axvline(x=f3, color='purple', linestyle='--', alpha=0.7, label=f'True: {f3} Hz')
    ax1.legend(loc='upper right')

    # Mode spectra
    for k in range(3):
        ax = plt.subplot(4, 1, k+2)
        ax.plot(freqs_pos, mode_ffts_ch1[k], linewidth=1.5, color=colors[k])
        ax.set_title(f'Mode {k+1} Spectrum (Center: {center_freqs[k]:.1f} Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude')
        ax.set_xlim([0, 150])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=center_freqs[k], color=colors[k], linestyle='--', linewidth=2,
                   alpha=0.8, label=f'Extracted: {center_freqs[k]:.1f} Hz')
        ax.legend(loc='upper right')

    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()

    # ===== Test 3: Reconstruction Verification =====
    print("\n=== Test 3: Reconstruction Verification ===")

    fig3 = plt.figure(figsize=(14, 8))
    fig3.suptitle('MVMD Test 3: Signal Reconstruction', fontsize=16, fontweight='bold')

    # Channel 1 reconstruction
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t[:500], signal[0, :500], label='Original', linewidth=2, alpha=0.8, color='black')
    ax1.plot(t[:500], reconstructed[0, :500], label='Reconstructed',
             linestyle='--', linewidth=2, alpha=0.8, color='red')
    ax1.set_title('Channel 1: Original vs Reconstructed', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Channel 2 reconstruction
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t[:500], signal[1, :500], label='Original', linewidth=2, alpha=0.8, color='black')
    ax2.plot(t[:500], reconstructed[1, :500], label='Reconstructed',
             linestyle='--', linewidth=2, alpha=0.8, color='blue')
    ax2.set_title('Channel 2: Original vs Reconstructed', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # ===== Test 4: Mode Alignment Verification =====
    print("\n=== Test 4: Mode Alignment Verification ===")
    print("Verifying that modes share same center frequency across channels...")

    for k in range(3):
        # Compute dominant frequency for each channel
        fft_ch1 = np.abs(np.fft.fft(time_modes[k, 0]))[:N//2]
        fft_ch2 = np.abs(np.fft.fft(time_modes[k, 1]))[:N//2]

        dom_freq_ch1 = freqs_pos[np.argmax(fft_ch1)]
        dom_freq_ch2 = freqs_pos[np.argmax(fft_ch2)]

        freq_diff = abs(dom_freq_ch1 - dom_freq_ch2)

        print(f"\nMode {k+1} (Center: {center_freqs[k]:.1f} Hz):")
        print(f"  Channel 1 dominant: {dom_freq_ch1:.1f} Hz")
        print(f"  Channel 2 dominant: {dom_freq_ch2:.1f} Hz")
        print(f"  Difference: {freq_diff:.2f} Hz")

        if freq_diff < 2.0:
            print(f"  Status: ALIGNED")
        else:
            print(f"  Status: WARNING - modes may not be well aligned")

    # ===== Test 5: Complex Signal (AM + Multiple Frequencies) =====
    print("\n=== Test 5: Complex Signal with Amplitude Modulation ===")

    # Generate more complex signal
    N2 = 2000
    t2 = np.arange(N2) / fs

    # AM signal (carrier: 40 Hz, modulator: 3 Hz)
    carrier_freq = 40
    mod_freq = 3
    am_ch1 = (1 + 0.5 * np.cos(2 * np.pi * mod_freq * t2)) * np.sin(2 * np.pi * carrier_freq * t2)
    am_ch2 = (1 + 0.4 * np.cos(2 * np.pi * mod_freq * t2)) * np.sin(2 * np.pi * carrier_freq * t2 + np.pi/6)
    am_ch3 = (1 + 0.6 * np.cos(2 * np.pi * mod_freq * t2)) * np.sin(2 * np.pi * carrier_freq * t2 - np.pi/4)

    # High frequency tone
    tone_freq = 150
    tone_ch1 = 0.4 * np.sin(2 * np.pi * tone_freq * t2)
    tone_ch2 = 0.3 * np.sin(2 * np.pi * tone_freq * t2 + np.pi/3)
    tone_ch3 = 0.5 * np.sin(2 * np.pi * tone_freq * t2 - np.pi/6)

    # Low frequency component
    low_freq = 8
    low_ch1 = 0.8 * np.sin(2 * np.pi * low_freq * t2)
    low_ch2 = 0.7 * np.sin(2 * np.pi * low_freq * t2 + np.pi/8)
    low_ch3 = 0.9 * np.sin(2 * np.pi * low_freq * t2 - np.pi/5)

    # Combine
    complex_signal = np.vstack([
        am_ch1 + tone_ch1 + low_ch1,
        am_ch2 + tone_ch2 + low_ch2,
        am_ch3 + tone_ch3 + low_ch3
    ])

    print(f"Complex signal shape: {complex_signal.shape}")

    # Apply MVMD
    config2 = config.copy()
    config2['alpha'] = 3000  # Higher alpha for tighter bandwidth
    mvmd2 = MVMD(config2)
    result2 = mvmd2.decompose(complex_signal, num_modes=3)

    time_modes2 = result2['time_modes']
    center_freqs2 = result2['center_freqs'][-1, :]

    print(f"Extracted center frequencies: {center_freqs2} Hz")

    reconstructed2 = np.sum(time_modes2, axis=0)
    recon_error2 = np.linalg.norm(complex_signal - reconstructed2) / np.linalg.norm(complex_signal)
    print(f"Reconstruction error: {recon_error2:.2e}")

    # Plot complex signal results
    fig4 = plt.figure(figsize=(16, 12))
    fig4.suptitle('MVMD Test 5: Complex Signal (AM + Tones)', fontsize=16, fontweight='bold')

    # Original signals (3 channels)
    for c in range(3):
        ax = plt.subplot(4, 3, c+1)
        ax.plot(t2[:1000], complex_signal[c, :1000], linewidth=1)
        ax.set_title(f'Original - Channel {c+1}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

    # Extracted modes
    for k in range(3):
        for c in range(3):
            ax = plt.subplot(4, 3, 3*(k+1) + c + 1)
            ax.plot(t2[:1000], time_modes2[k, c, :1000], color=colors[k], linewidth=1)
            ax.set_title(f'Mode {k+1}, Ch{c+1} ({center_freqs2[k]:.1f} Hz)',
                        fontweight='bold', fontsize=9)
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)

    plt.xlabel('Time (s)')
    plt.tight_layout()

    # ===== Summary =====
    print("\n" + "="*80)
    print(" Test Summary ".center(80, "="))
    print("="*80)
    print(f"Test 1 - Simple signal reconstruction error: {recon_error:.2e}")
    print(f"Test 5 - Complex signal reconstruction error: {recon_error2:.2e}")
    print("\nAll tests completed successfully!")
    print("Close plot windows to exit...")
    print("="*80 + "\n")

    plt.show()
