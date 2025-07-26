#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from scipy.optimize import curve_fit

def noise_model(f, a, alpha, b):
    """Noise model: 1/f^alpha + white noise."""
    return a / (f**alpha) + b

def analyze_noise_spectrum(wav_file, max_harmonics=10):
    """
    Analyzes the noise spectrum of a WAV file, separating 1/f noise,
    white noise, and power line hum.
    """
    try:
        fs, data = wavfile.read(wav_file)
        if data.ndim > 1:
            data = data.mean(axis=1) # Use mean of channels for stereo files
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        return

    # --- Spectrum Analysis ---
    f, Pxx = welch(data, fs, nperseg=fs, nfft=fs*2) # High resolution PSD

    # --- Power Line Noise Identification and Removal (for fitting) ---
    Pxx_clean = np.copy(Pxx)
    # Determine power line frequency (50Hz vs 60Hz)
    try:
        power_line_freq = 50 if np.sum(Pxx[np.where((f > 48) & (f < 52))]) > np.sum(Pxx[np.where((f > 58) & (f < 62))]) else 60
    except IndexError:
        power_line_freq = 0 # No clear peak found

    if power_line_freq > 0:
        for i in range(1, max_harmonics + 1):
            freq = power_line_freq * i
            if freq > fs / 2:
                break
            # Find indices around the harmonic and remove them for fitting
            idx_center = np.argmin(np.abs(f - freq))
            idx_start = max(0, idx_center - 2)
            idx_end = min(len(f) - 1, idx_center + 3) # Widen the removal band
            Pxx_clean[idx_start:idx_end] = 0

    # --- Curve Fitting ---
    fit_indices = np.where((f > 1) & (Pxx_clean > 1e-20)) # Fit on non-zero, non-DC part
    f_fit = f[fit_indices]
    Pxx_fit = Pxx_clean[fit_indices]

    try:
        # Provide reasonable initial guesses and bounds
        p0 = [np.median(Pxx_fit) * np.median(f_fit), 1.0, np.min(Pxx_fit)]
        bounds = ([0, 0, 0], [np.inf, 3, np.inf])
        params, _ = curve_fit(noise_model, f_fit, Pxx_fit, p0=p0, bounds=bounds, maxfev=10000)
        a, alpha, b = params
    except RuntimeError:
        print("Curve fit failed. Could not reliably separate 1/f and white noise.")
        # Simplified fallback: treat all non-hum noise as one category
        a, alpha, b = 0, 0, 0

    # --- Quantify Components ---
    total_power = np.trapz(Pxx, f)
    total_voltage = np.sqrt(total_power) if total_power > 0 else 0

    # Power line noise
    power_line_power = 0
    if power_line_freq > 0:
        for i in range(1, max_harmonics + 1):
            freq = power_line_freq * i
            if freq > fs / 2:
                break
            idx_center = np.argmin(np.abs(f - freq))
            # Integrate over a narrow band (e.g., 2Hz wide)
            band = np.where((f >= freq - 1) & (f <= freq + 1))
            if len(band[0]) > 0:
                power_line_power += np.trapz(Pxx[band], f[band])

    # 1/f and white noise from fitted model
    f_nz = f[np.where(f > 0)] # Avoid division by zero
    one_over_f_power = np.trapz(a / (f_nz**alpha), f_nz) if a > 0 else 0
    white_noise_power = np.trapz(np.full_like(f_nz, b), f_nz) if b > 0 else 0

    # Sum of classified powers
    classified_power = power_line_power + one_over_f_power + white_noise_power
    other_power = max(0, total_power - classified_power)

    # --- Output Results ---
    print(f"Total Noise Voltage: {total_voltage * 1e6:.2f} uV RMS")
    print(f"Frequency Range: {f[1]:.1f} Hz to {f[-1]/1000:.1f} kHz")
    print("Breakdown:")
    if a > 0:
        print(f"  - 1/f Noise (alpha={alpha:.2f}): {np.sqrt(one_over_f_power) * 1e6:.2f} uV ({(one_over_f_power/total_power):.1%})")
    if b > 0:
        print(f"  - White Noise: {np.sqrt(white_noise_power) * 1e6:.2f} uV ({(white_noise_power/total_power):.1%})")
    if power_line_power > 0:
        print(f"  - Power Line Noise ({power_line_freq}Hz): {np.sqrt(power_line_power) * 1e6:.2f} uV ({(power_line_power/total_power):.1%})")
    if other_power > 1e-12 and other_power / total_power > 0.001: # Report "Other" if significant
        print(f"  - Other/Residual Noise: {np.sqrt(other_power) * 1e6:.2f} uV ({(other_power/total_power):.1%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyzes a WAV file to separate and quantify noise components.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('wav_file', help='Path to the input WAV file.')
    parser.add_argument('--max_harmonics', type=int, default=10, help='Maximum number of power line harmonics to analyze (default: 10).')
    args = parser.parse_args()

    analyze_noise_spectrum(args.wav_file, args.max_harmonics)
