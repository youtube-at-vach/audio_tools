import numpy as np
from scipy import signal
import math

def dbfs_to_linear(dbfs: float) -> float:
    """Converts dBFS to linear amplitude (0 dBFS = 1.0 linear)."""
    if dbfs is None: # Or handle as an error
        return 0.0
    return 10**(dbfs / 20.0)

def linear_to_dbfs(linear_amp: float, min_dbfs: float = -120.0) -> float:
    """Converts linear amplitude to dBFS, returning min_dbfs for non-positive input."""
    if linear_amp <= 0:
        return min_dbfs
    return 20 * np.log10(linear_amp)

def bandpass_filter(signal_data: np.ndarray, sampling_rate: float, lowcut: float = 20.0, highcut: float = 20000.0, order: int = 8) -> np.ndarray:
    """Applies a Butterworth bandpass filter to the signal."""
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure low and high are within (0, 1)
    low = max(0.000001, low) # prevent log(0) issues if lowcut is 0
    high = min(0.999999, high) # prevent issues if highcut is nyquist

    if low >= high:
        # This can happen if highcut is too close to or below lowcut after normalization
        # Or if sampling_rate is too low for the given cuts.
        # Returning original signal or raising an error are options.
        # For now, let's print a warning and return original data.
        print(f"Warning: Bandpass filter low cut {low} >= high cut {high}. Check frequencies and sampling rate. Returning original signal.")
        return signal_data

    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, signal_data)
    return filtered_signal

def notch_filter(signal_data: np.ndarray, sampling_rate: float, target_frequency: float, quality_factor: float = 30.0, order: int = 2) -> np.ndarray:
    """Applies a Butterworth notch filter to the signal."""
    nyquist = 0.5 * sampling_rate
    w0 = target_frequency / nyquist  # Normalized frequency

    if w0 <= 0 or w0 >= 1:
        print(f"Warning: Notch filter target frequency {target_frequency} Hz is outside the valid range (0, {nyquist} Hz). Returning original signal.")
        return signal_data
    
    # For a notch filter, 'bw' (bandwidth) is w0/Q
    bw = w0 / quality_factor
    
    # Ensure the band [w0-bw/2, w0+bw/2] is within (0,1)
    low = max(0.000001, w0 - bw / 2)
    high = min(0.999999, w0 + bw / 2)

    if low >= high:
         print(f"Warning: Notch filter bandwidth is too narrow or invalid. Low: {low}, High: {high}. Returning original signal.")
         return signal_data
    
    # For bandstop, the order for Nth order Butterworth is N.
    # scipy.signal.butter documentation for 'N': "The order of the filter."
    # For digital filters, this is the number of poles and zeros.
    # The original audiocalc.py used iirfilter with order=2, which corresponds to a 2nd order filter.
    # Here, 'order' directly refers to the Butterworth filter order.
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, signal_data)
    return filtered_signal


def perform_fft(data: np.ndarray, sample_rate: float, window_name: str = 'hann') -> tuple[np.ndarray, np.ndarray]:
    """
    Performs FFT on the data, applies a window, and returns frequencies and scaled magnitudes.
    """
    N = len(data)
    if N == 0:
        return np.array([]), np.array([])
        
    window_samples = signal.get_window(window_name, N)
    windowed_data = data * window_samples
    
    # Perform RFFT (for real inputs)
    fft_values = np.fft.rfft(windowed_data)
    fft_frequencies = np.fft.rfftfreq(N, d=1.0/sample_rate)
    
    # Scale magnitudes
    # The scaling factor 2.0 / np.sum(window_samples) correctly scales for power/amplitude.
    # For a sine wave of amplitude A, the peak in the FFT (single-sided) will be A.
    # DC component (0 Hz) and Nyquist frequency component should not be doubled.
    fft_magnitudes_scaled = np.abs(fft_values) * (2.0 / np.sum(window_samples))

    if N % 2 == 0: # Even N, Nyquist frequency is present
        fft_magnitudes_scaled[0] /= 2.0 # DC component
        fft_magnitudes_scaled[-1] /= 2.0 # Nyquist component
    else: # Odd N, Nyquist frequency is not uniquely present
        fft_magnitudes_scaled[0] /= 2.0 # DC component
        
    return fft_frequencies, fft_magnitudes_scaled


def find_peak_magnitude(fft_magnitudes: np.ndarray, fft_frequencies: np.ndarray, target_frequency: float, search_half_width_hz: float = 20.0) -> tuple[float, float]:
    """
    Finds the peak magnitude and actual frequency of a target_frequency within a specified band.
    """
    if len(fft_magnitudes) == 0 or len(fft_frequencies) == 0:
        return target_frequency, 0.0

    min_freq = target_frequency - search_half_width_hz
    max_freq = target_frequency + search_half_width_hz
    
    # Find indices within the frequency band
    relevant_indices = np.where((fft_frequencies >= min_freq) & (fft_frequencies <= max_freq))[0]
    
    if len(relevant_indices) == 0:
        # No frequencies in the search band, try to find the closest one if that's desired
        # For now, return 0 magnitude as per original logic if band is empty.
        # Or, find the absolute closest index to target_frequency:
        # closest_idx = np.argmin(np.abs(fft_frequencies - target_frequency))
        # if np.abs(fft_frequencies[closest_idx] - target_frequency) <= search_half_width_hz * 2: # Arbitrary wider check
        # relevant_indices = np.array([closest_idx])
        # else:
        return target_frequency, 0.0 

    peak_magnitude_linear = np.max(fft_magnitudes[relevant_indices])
    peak_index = relevant_indices[np.argmax(fft_magnitudes[relevant_indices])]
    actual_frequency_at_peak = fft_frequencies[peak_index]
    
    return actual_frequency_at_peak, peak_magnitude_linear

def analyze_harmonics(
    audio_data: np.ndarray,
    fundamental_freq: float,
    sample_rate: float,
    window_name: str = 'hann',
    num_harmonics: int = 9, # Excludes fundamental, so H2 to H(num_harmonics+1)
    min_dbfs_for_analysis: float = -100.0,
    min_dbr_for_harmonic: float = -120.0, # Relative to fundamental for a harmonic to be listed
    peak_search_half_width_hz: float = 20.0
) -> dict:
    """
    Performs harmonic analysis on audio data, returning details of fundamental and harmonics, and THD.
    """
    results = {
        'fundamental': {'frequency': fundamental_freq, 'amplitude_dbfs': min_dbfs_for_analysis, 'amplitude_linear': 0.0, 'phase_deg': 0.0},
        'harmonics': [],
        'thd_percent': 0.0,
        'thd_db': linear_to_dbfs(0.0) # Effectively -infinity dB
    }

    if len(audio_data) == 0:
        return results

    fft_freqs, fft_mags_lin = perform_fft(audio_data, sample_rate, window_name)

    if len(fft_freqs) == 0: # FFT failed or empty data
        return results

    # Analyze Fundamental
    fund_actual_freq, fund_amp_lin = find_peak_magnitude(fft_mags_lin, fft_freqs, fundamental_freq, peak_search_half_width_hz)
    fund_amp_dbfs = linear_to_dbfs(fund_amp_lin)

    results['fundamental']['frequency'] = fund_actual_freq
    results['fundamental']['amplitude_linear'] = fund_amp_lin
    results['fundamental']['amplitude_dbfs'] = fund_amp_dbfs
    # Phase calculation requires the complex FFT output, not just magnitudes.
    # This is a simplification for now. For phase, one would use np.angle on fft_values.
    # And then find the phase at the peak_index corresponding to fund_actual_freq.
    # results['fundamental']['phase_deg'] = np.degrees(np.angle(fft_values[peak_idx_for_fundamental]))

    if fund_amp_dbfs < min_dbfs_for_analysis:
        # Signal too weak, THD is not meaningful or will be noise-dominated
        return results 

    sum_harmonics_power_sq = 0.0

    for i in range(2, num_harmonics + 2): # H2, H3, ..., H(num_harmonics+1)
        harmonic_freq_nominal = fundamental_freq * i
        
        # Adjust search width for higher harmonics if needed, or keep it fixed.
        # For simplicity, using the same search_half_width_hz.
        actual_h_freq, h_amp_lin = find_peak_magnitude(fft_mags_lin, fft_freqs, harmonic_freq_nominal, peak_search_half_width_hz)
        
        if fund_amp_lin > 0: # Avoid division by zero if fundamental is silent
            h_amp_dbr = linear_to_dbfs(h_amp_lin / fund_amp_lin)
        else:
            h_amp_dbr = linear_to_dbfs(0) # Effectively -infinity dB

        # Phase calculation similar to fundamental - requires complex FFT values
        # harmonic_phase_deg = np.degrees(np.angle(fft_values[peak_idx_for_harmonic]))

        if h_amp_dbr >= min_dbr_for_harmonic: # Only include significant harmonics
            results['harmonics'].append({
                'order': i,
                'frequency': actual_h_freq,
                'amplitude_dbr_fundamental': h_amp_dbr,
                'amplitude_dbfs': linear_to_dbfs(h_amp_lin),
                'amplitude_linear': h_amp_lin,
                'phase_deg': 0.0 # Placeholder for phase
            })
            sum_harmonics_power_sq += h_amp_lin**2

    if fund_amp_lin > 0:
        thd_ratio = np.sqrt(sum_harmonics_power_sq) / fund_amp_lin
        results['thd_percent'] = thd_ratio * 100.0
        results['thd_db'] = linear_to_dbfs(thd_ratio) if thd_ratio > 0 else linear_to_dbfs(0.0)
    
    return results


def calculate_thdn_value(
    signal_data: np.ndarray,
    sample_rate: float,
    fundamental_freq: float,
    # window_name: str = 'hann', # Windowing is not typically used for RMS-based THD+N after notching
    min_dbfs_for_analysis: float = -100.0,
    num_harmonics_to_notch: int = 0 # Set to 0 to notch only fundamental, or >0 to notch harmonics too
                                     # For pure THD+N, only fundamental is notched.
                                     # Notching harmonics as well gives THD+N+OtherDistortion
) -> dict:
    """
    Calculates THD+N (Total Harmonic Distortion plus Noise) and SINAD.
    Notches out fundamental (and optionally harmonics) and compares RMS values.
    """
    results = {
        'thdn_percent': 100.0, # Default to worst case
        'thdn_db': 0.0,        # Default to worst case (0dB means all noise/distortion)
        'sinad_db': linear_to_dbfs(0.0) # Default to worst case
    }
    
    if len(signal_data) == 0:
        return results

    # 1. Calculate RMS of the original signal (total signal power)
    #    For a sine wave, RMS = Amplitude / sqrt(2).
    #    However, if we use FFT, the peak magnitude is the amplitude.
    #    To be consistent with audio measurements, it's often better to calculate RMS directly from time-domain data.
    rms_total_signal = np.sqrt(np.mean(signal_data**2))
    if rms_total_signal == 0: # No signal
        return results

    rms_total_signal_dbfs = linear_to_dbfs(rms_total_signal)
    if rms_total_signal_dbfs < min_dbfs_for_analysis:
        # Signal too weak to reliably calculate THD+N
        return results

    # 2. Create the notched signal
    signal_after_notching = np.copy(signal_data)

    # Notch fundamental
    # Quality factor for fundamental notch can be high to be very selective
    signal_after_notching = notch_filter(signal_after_notching, sample_rate, fundamental_freq, quality_factor=50, order=4) # Higher order for steeper notch

    if num_harmonics_to_notch > 0:
        for i in range(2, num_harmonics_to_notch + 2):
            harmonic_freq = fundamental_freq * i
            # Use a slightly lower Q for harmonics, or make it configurable
            signal_after_notching = notch_filter(signal_after_notching, sample_rate, harmonic_freq, quality_factor=20, order=2)

    # 3. Calculate RMS of the notched signal (noise + distortion)
    rms_noise_plus_distortion = np.sqrt(np.mean(signal_after_notching**2))

    # 4. Calculate THD+N ratio
    if rms_total_signal > 0: # Should be true if we passed the min_dbfs_for_analysis check
        thdn_ratio = rms_noise_plus_distortion / rms_total_signal
    else: # Should not be reached
        thdn_ratio = 1.0 

    results['thdn_percent'] = thdn_ratio * 100.0
    results['thdn_db'] = linear_to_dbfs(thdn_ratio) if thdn_ratio > 0 else linear_to_dbfs(0.0) # 0 linear is -inf dB

    # 5. Calculate SINAD (Signal to Noise and Distortion Ratio)
    # SINAD is the reciprocal of THD+N ratio (when both are linear, not percent)
    if thdn_ratio > 0:
        sinad_ratio = 1.0 / thdn_ratio
        results['sinad_db'] = linear_to_dbfs(sinad_ratio) # This is actually 20*log10(S_rms / (N+D)_rms)
                                                          # which is -thdn_db (if thdn_db is 20*log10((N+D)/S))
                                                          # So, results['sinad_db'] = -results['thdn_db'] is common.
                                                          # Let's use the direct calculation for clarity.
    else: # Noise + Distortion is zero (theoretically, or below float precision)
        results['sinad_db'] = linear_to_dbfs(1e12) # Very high SINAD, effectively infinite

    return results

# Example usage (for testing, not part of the library file usually)
if __name__ == '__main__':
    sample_rate = 48000
    duration = 1.0
    N = int(sample_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    # Test Signal: 1kHz fundamental + 0.1 amplitude 3kHz harmonic + some noise
    fundamental_freq = 1000
    amp_fundamental = dbfs_to_linear(-6.0) # -6 dBFS fundamental
    amp_h3 = dbfs_to_linear(-26.0)      # -20 dBr 3rd harmonic (-26 dBFS)
    
    signal_pure = amp_fundamental * np.sin(2 * np.pi * fundamental_freq * t)
    signal_harmonic = amp_h3 * np.sin(2 * np.pi * fundamental_freq * 3 * t)
    noise = np.random.normal(0, dbfs_to_linear(-70.0), N) # -70 dBFS RMS noise
    
    test_signal = signal_pure + signal_harmonic + noise

    print(f"Fundamental Amplitude (Linear): {amp_fundamental:.4f}, dBFS: {linear_to_dbfs(amp_fundamental):.2f}")
    print(f"3rd Harmonic Amplitude (Linear): {amp_h3:.4f}, dBFS: {linear_to_dbfs(amp_h3):.2f}")
    print(f"Noise RMS (Linear): {dbfs_to_linear(-70.0):.6f}, dBFS: {-70.0:.2f}")
    print(f"Signal RMS (approx): {linear_to_dbfs(np.sqrt(np.mean(test_signal**2))):.2f} dBFS")


    # --- Test bandpass_filter ---
    # filtered_signal_bp = bandpass_filter(test_signal, sample_rate, lowcut=800, highcut=1200, order=4)
    # print(f"\nRMS after 800-1200Hz bandpass: {linear_to_dbfs(np.sqrt(np.mean(filtered_signal_bp**2))):.2f} dBFS")

    # --- Test notch_filter ---
    # filtered_signal_notch = notch_filter(test_signal, sample_rate, fundamental_freq, quality_factor=30, order=2)
    # print(f"RMS after 1kHz notch: {linear_to_dbfs(np.sqrt(np.mean(filtered_signal_notch**2))):.2f} dBFS")


    # --- Test perform_fft & find_peak_magnitude ---
    # fft_freqs, fft_mags = perform_fft(test_signal, sample_rate)
    # peak_freq, peak_mag = find_peak_magnitude(fft_mags, fft_freqs, fundamental_freq)
    # print(f"\nFFT: Found peak at {peak_freq:.2f} Hz with magnitude {linear_to_dbfs(peak_mag):.2f} dBFS (Expected: {fundamental_freq} Hz, {linear_to_dbfs(amp_fundamental):.2f} dBFS)")
    # peak_freq_h3, peak_mag_h3 = find_peak_magnitude(fft_mags, fft_freqs, fundamental_freq * 3)
    # print(f"FFT: Found H3 peak at {peak_freq_h3:.2f} Hz with magnitude {linear_to_dbfs(peak_mag_h3):.2f} dBFS (Expected: {fundamental_freq*3} Hz, {linear_to_dbfs(amp_h3):.2f} dBFS)")

    # --- Test analyze_harmonics ---
    harmonics_results = analyze_harmonics(test_signal, fundamental_freq, sample_rate, num_harmonics=5)
    print(f"\n--- Harmonic Analysis ---")
    print(f"Fundamental: {harmonics_results['fundamental']['frequency']:.1f} Hz, {harmonics_results['fundamental']['amplitude_dbfs']:.2f} dBFS")
    for h in harmonics_results['harmonics']:
        print(f"  H{h['order']}: {h['frequency']:.1f} Hz, {h['amplitude_dbr_fundamental']:.2f} dBr, {h['amplitude_dbfs']:.2f} dBFS")
    print(f"THD: {harmonics_results['thd_percent']:.4f} %, {harmonics_results['thd_db']:.2f} dB")
    # Expected THD for this signal: (amp_h3 / amp_fundamental)
    # expected_thd_ratio = amp_h3 / amp_fundamental
    # print(f"Expected THD ratio: {expected_thd_ratio:.4f} ({linear_to_dbfs(expected_thd_ratio):.2f} dB)")


    # --- Test calculate_thdn_value ---
    thdn_results = calculate_thdn_value(test_signal, sample_rate, fundamental_freq)
    print(f"\n--- THD+N Analysis ---")
    print(f"THD+N: {thdn_results['thdn_percent']:.4f} %")
    print(f"THD+N (dB): {thdn_results['thdn_db']:.2f} dB")
    print(f"SINAD: {thdn_results['sinad_db']:.2f} dB")

    # For this specific signal, distortion is only H3. Noise is at -70dBFS.
    # Fundamental is -6dBFS. H3 is -26dBFS.
    # Power_fundamental_sq = amp_fundamental**2
    # Power_h3_sq = amp_h3**2
    # Power_noise_sq = (dbfs_to_linear(-70.0))**2 * 2 # RMS noise, so power is RMS^2. FFT peak approx RMS * sqrt(2) for noise.
                                                    # Or more simply, RMS of noise is dbfs_to_linear(-70).
                                                    # Power_noise = (dbfs_to_linear(-70.0))**2
    # THD+N ratio approx = sqrt(Power_h3_sq + Power_noise_sq) / sqrt(Power_fundamental_sq)
    #                  = sqrt(amp_h3**2 + (dbfs_to_linear(-70.0))**2) / amp_fundamental
    # expected_thdn_ratio = np.sqrt(amp_h3**2 + (dbfs_to_linear(-70.0))**2) / amp_fundamental
    # print(f"Expected THD+N ratio (approx): {expected_thdn_ratio:.4f} ({linear_to_dbfs(expected_thdn_ratio):.2f} dB)")
    # print(f"Expected SINAD (approx): {linear_to_dbfs(1.0/expected_thdn_ratio):.2f} dB")

    # --- Test with a very low signal to check min_dbfs_for_analysis ---
    # low_signal = test_signal * dbfs_to_linear(-100) # Attenuate heavily
    # print(f"\n--- Analysis on very low signal (RMS: {linear_to_dbfs(np.sqrt(np.mean(low_signal**2))):.2f} dBFS) ---")
    # harmonics_low_sig_results = analyze_harmonics(low_signal, fundamental_freq, sample_rate, min_dbfs_for_analysis=-80)
    # print(f"Low Sig Harmonics THD: {harmonics_low_sig_results['thd_db']:.2f} dB (Fund: {harmonics_low_sig_results['fundamental']['amplitude_dbfs']:.2f} dBFS)")
    # thdn_low_sig_results = calculate_thdn_value(low_signal, sample_rate, fundamental_freq, min_dbfs_for_analysis=-80)
    # print(f"Low Sig THD+N: {thdn_low_sig_results['thdn_db']:.2f} dB, SINAD: {thdn_low_sig_results['sinad_db']:.2f} dB")
    
    # --- Test bandpass safety ---
    # print("\n--- Testing bandpass safety ---")
    # bandpass_filter(test_signal, sample_rate, lowcut=0, highcut=10, order=4) # lowcut=0
    # bandpass_filter(test_signal, sample_rate, lowcut=23000, highcut=24000, order=4) # > nyquist
    # bandpass_filter(test_signal, sample_rate, lowcut=1000, highcut=500, order=4) # low > high

    # --- Test notch safety ---
    # print("\n--- Testing notch safety ---")
    # notch_filter(test_signal, sample_rate, target_frequency=0, quality_factor=1)
    # notch_filter(test_signal, sample_rate, target_frequency=sample_rate/2, quality_factor=1)
    # notch_filter(test_signal, sample_rate, target_frequency=1000, quality_factor=0.1) # very wide, low Q makes bw > w0
    # notch_filter(test_signal, sample_rate, target_frequency=1000, quality_factor=1000) # very narrow

    # --- Test empty data ---
    # print("\n--- Testing empty data ---")
    # empty_arr = np.array([])
    # perform_fft(empty_arr, sample_rate)
    # analyze_harmonics(empty_arr, 1000, sample_rate)
    # calculate_thdn_value(empty_arr, sample_rate, 1000)
    # find_peak_magnitude(empty_arr, empty_arr, 1000)
    # dbfs_to_linear(None)

```
