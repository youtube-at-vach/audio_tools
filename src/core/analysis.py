import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window
from scipy.optimize import minimize_scalar

class AudioCalc:
    """
    Shared audio calculation utilities.
    """
    @staticmethod
    def bandpass_filter(signal, sampling_rate, lowcut=20.0, highcut=20000.0):
        nyquist = 0.5 * sampling_rate
        # Ensure valid bounds
        lowcut = max(0.1, lowcut)
        highcut = min(nyquist - 1, highcut)
        if lowcut >= highcut:
            return signal
        sos = butter(8, [lowcut / nyquist, highcut / nyquist], btype='bandpass', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def notch_filter(signal, sampling_rate, target_frequency, quality_factor=30):
        nyquist = 0.5 * sampling_rate
        w0 = target_frequency / nyquist
        bandwidth = w0 / quality_factor
        sos = butter(2, [w0 - bandwidth/2, w0 + bandwidth/2], btype='bandstop', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def optimize_frequency(signal, sampling_rate, freq_guess):
        """
        Optimizes frequency estimate using Sine Fitting (minimizing residual RMS).
        """
        N = len(signal)
        t = np.arange(N) / sampling_rate
        
        def get_residual_rms(f):
            w = 2 * np.pi * f
            sin_t = np.sin(w * t)
            cos_t = np.cos(w * t)
            ones = np.ones(N)
            M = np.column_stack([sin_t, cos_t, ones])
            coeffs, _, _, _ = np.linalg.lstsq(M, signal, rcond=None)
            fitted = M @ coeffs
            residual = signal - fitted
            return np.sqrt(np.mean(residual**2))

        # Search around guess (+/- 5%)
        bounds = (freq_guess * 0.95, freq_guess * 1.05)
        res = minimize_scalar(get_residual_rms, bounds=bounds, method='bounded')
        return res.x

    @staticmethod
    def calculate_thdn_sine_fit(signal, sampling_rate, freq_guess):
        """
        Calculates THD+N using Sine Fitting method.
        Returns (thdn_db, fund_rms, noise_dist_rms)
        """
        N = len(signal)
        t = np.arange(N) / sampling_rate
        
        # 1. Optimize Frequency
        best_freq = AudioCalc.optimize_frequency(signal, sampling_rate, freq_guess)
        
        # 2. Get Final Residual
        w = 2 * np.pi * best_freq
        sin_t = np.sin(w * t)
        cos_t = np.cos(w * t)
        ones = np.ones(N)
        M = np.column_stack([sin_t, cos_t, ones])
        coeffs, _, _, _ = np.linalg.lstsq(M, signal, rcond=None)
        fitted_fund = M @ coeffs
        residual = signal - fitted_fund
        
        # 3. Bandwidth Limit Residual (20Hz - 20kHz)
        # Highpass 20Hz (Remove DC/Drift if any left)
        if sampling_rate > 40:
            sos_hp = butter(4, 20, 'hp', fs=sampling_rate, output='sos')
            residual = sosfiltfilt(sos_hp, residual)
        
        # Lowpass 20kHz
        if sampling_rate > 44100:
            sos_lp = butter(4, 20000, 'lp', fs=sampling_rate, output='sos')
            residual = sosfiltfilt(sos_lp, residual)
            
        # 4. Calculate RMS
        # Trim edges slightly to avoid filter transients from bandwidth limit
        trim = min(100, N//8)
        if N > 2*trim:
            nd_rms = np.sqrt(np.mean(residual[trim:-trim]**2))
            fund_rms = np.sqrt(np.mean(fitted_fund[trim:-trim]**2))
        else:
            nd_rms = np.sqrt(np.mean(residual**2))
            fund_rms = np.sqrt(np.mean(fitted_fund**2))
            
        if fund_rms == 0:
            return -140.0, 0.0, 0.0
            
        ratio = nd_rms / fund_rms
        thdn_db = 20 * np.log10(ratio + 1e-12)
        
        return thdn_db, fund_rms, nd_rms
    
    @staticmethod
    def analyze_harmonics(audio_data, fundamental_freq, window_name, sampling_rate, min_db=-140.0):
        window = get_window(window_name, len(audio_data))
        windowed_data = audio_data * window
        fft_result = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sampling_rate)

        # Coherent gain correction
        coherent_gain = np.sum(window) / len(window)
        
        # Amplitude spectrum (Peak)
        # rfft returns N/2+1 bins. Magnitude is |X|/N * 2 (except DC and Nyquist)
        amplitude_spectrum = (np.abs(fft_result) / len(audio_data)) * 2 / coherent_gain
        
        # Find Fundamental Peak
        # Search near expected frequency
        search_window = 0.1 * fundamental_freq # +/- 10%
        idx_min = np.searchsorted(freqs, fundamental_freq - search_window)
        idx_max = np.searchsorted(freqs, fundamental_freq + search_window)
        if idx_max <= idx_min:
            idx_max = idx_min + 1
            
        # Find max in range
        if idx_max < len(amplitude_spectrum):
            subset = amplitude_spectrum[idx_min:idx_max]
            if len(subset) > 0:
                local_max_idx = np.argmax(subset)
                peak_idx = idx_min + local_max_idx
            else:
                peak_idx = np.argmin(np.abs(freqs - fundamental_freq))
        else:
            peak_idx = np.argmin(np.abs(freqs - fundamental_freq))
            
        max_freq = freqs[peak_idx]
        max_amplitude = amplitude_spectrum[peak_idx]
        
        # Refine Frequency using Parabolic Interpolation
        if 0 < peak_idx < len(amplitude_spectrum) - 1:
            alpha = amplitude_spectrum[peak_idx-1]
            beta = amplitude_spectrum[peak_idx]
            gamma = amplitude_spectrum[peak_idx+1]
            
            denom = alpha - 2*beta + gamma
            if denom != 0:
                p = 0.5 * (alpha - gamma) / denom
                max_freq = freqs[peak_idx] + p * (freqs[1] - freqs[0])
                # Optional: Refine amplitude estimate
                # max_amplitude = beta - 0.25 * (alpha - gamma) * p
        
        amp_dbfs = 20 * np.log10(max_amplitude + 1e-12)
        
        result = {
            'frequency': max_freq,
            'amplitude_dbfs': amp_dbfs,
            'max_amplitude': max_amplitude
        }
        
        # Harmonics
        harmonic_results = []
        harmonic_amplitudes_linear = []
        
        # Up to 10th harmonic
        for i in range(2, 11): 
            harmonic_freq = max_freq * i
            if harmonic_freq >= sampling_rate / 2:
                break
                
            # Search near harmonic
            h_idx_min = np.searchsorted(freqs, harmonic_freq - search_window)
            h_idx_max = np.searchsorted(freqs, harmonic_freq + search_window)
            
            if h_idx_max < len(amplitude_spectrum) and h_idx_max > h_idx_min:
                subset = amplitude_spectrum[h_idx_min:h_idx_max]
                local_max_h = np.argmax(subset)
                h_peak_idx = h_idx_min + local_max_h
                
                h_amp = amplitude_spectrum[h_peak_idx]
                h_freq = freqs[h_peak_idx]
                
                relative_amp = h_amp / max_amplitude if max_amplitude > 0 else 0
                amp_db = 20 * np.log10(relative_amp + 1e-12)
                
                harmonic_results.append({
                    'order': i,
                    'frequency': h_freq,
                    'amplitude_dbr': amp_db,
                    'amplitude_linear': h_amp
                })
                harmonic_amplitudes_linear.append(h_amp)
            else:
                 harmonic_results.append({
                    'order': i,
                    'frequency': harmonic_freq,
                    'amplitude_dbr': min_db,
                    'amplitude_linear': 0
                })
        
        # THD Calculation
        # THD = sqrt(sum(harmonics^2)) / fundamental
        if max_amplitude > 0:
            thd_linear = np.sqrt(sum(a**2 for a in harmonic_amplitudes_linear)) / max_amplitude
            thd_percent = thd_linear * 100
            thd_db = 20 * np.log10(thd_linear + 1e-12)
        else:
            thd_percent = 0
            thd_db = min_db
            
        # THD+N Calculation (Sine Fit)
        # Use raw audio_data (no window applied yet)
        thdn_db, fund_rms, res_rms = AudioCalc.calculate_thdn_sine_fit(audio_data, sampling_rate, max_freq)
        thdn_linear = 10**(thdn_db/20)
            
        thdn_percent = thdn_linear * 100
        sinad_db = -thdn_db
        
        return {
            'basic_wave': result,
            'harmonics': harmonic_results,
            'thd_percent': thd_percent,
            'thd_db': thd_db,
            'thdn_percent': thdn_percent,
            'thdn_db': thdn_db,
            'sinad_db': sinad_db,
            # Raw components for averaging
            'raw_fund_rms': fund_rms,
            'raw_res_rms': res_rms,
            'raw_harmonics': harmonic_amplitudes_linear,
            'raw_fund_amp': max_amplitude
        }
