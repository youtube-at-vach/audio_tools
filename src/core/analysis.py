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
    def lowpass_filter(signal, sampling_rate, cutoff=20000.0):
        nyquist = 0.5 * sampling_rate
        cutoff = min(nyquist - 1, max(0.1, cutoff))
        sos = butter(8, cutoff / nyquist, btype='lowpass', output='sos')
        return sosfiltfilt(sos, signal)

    @staticmethod
    def highpass_filter(signal, sampling_rate, cutoff=20.0):
        nyquist = 0.5 * sampling_rate
        cutoff = min(nyquist - 1, max(0.1, cutoff))
        sos = butter(8, cutoff / nyquist, btype='highpass', output='sos')
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

        # Search around guess
        # The objective function has local minima spaced by approx sampling_rate/N.
        # We need to constrain the search to the main lobe, roughly +/- sampling_rate/N.
        bin_width = sampling_rate / N
        search_width = 1.5 * bin_width # Slightly more than 1 bin width to be safe
        
        if not np.isfinite(freq_guess):
            return freq_guess
            
        bounds = (freq_guess - search_width, freq_guess + search_width)
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
        
        if not np.isfinite(best_freq):
            return -140.0, 0.0, 0.0
        
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

    @staticmethod
    def _find_peak(mag, freqs, target_freq, width=20.0):
        mask = (freqs >= target_freq - width) & (freqs <= target_freq + width)
        if not np.any(mask):
            return 0.0
        return np.max(mag[mask])

    @staticmethod
    def calculate_imd_smpte(mag, freqs, f1, f2, num_sidebands=3):
        # SMPTE: f1 (low), f2 (high). IMD products at f2 +/- n*f1
        amp_f2 = AudioCalc._find_peak(mag, freqs, f2, width=max(50.0, f1*0.1))
        
        if amp_f2 < 1e-6:
            return {'imd': 0.0, 'imd_db': -100.0}
            
        sum_sq_sidebands = 0.0
        for n in range(1, num_sidebands + 1):
            sb_upper = f2 + n * f1
            sb_lower = f2 - n * f1
            
            amp_upper = AudioCalc._find_peak(mag, freqs, sb_upper)
            amp_lower = AudioCalc._find_peak(mag, freqs, sb_lower)
            
            sum_sq_sidebands += amp_upper**2 + amp_lower**2
            
        imd = np.sqrt(sum_sq_sidebands) / amp_f2
        return {
            'imd': imd * 100,
            'imd_db': 20 * np.log10(imd) if imd > 1e-9 else -100.0
        }

    @staticmethod
    def calculate_imd_ccif(mag, freqs, f1, f2):
        # CCIF: f1, f2 close (e.g. 19k, 20k). 
        # d2 = f2 - f1
        # d3 = 2f1 - f2, 2f2 - f1
        
        amp_f1 = AudioCalc._find_peak(mag, freqs, f1)
        amp_f2 = AudioCalc._find_peak(mag, freqs, f2)
        total_amp = amp_f1 + amp_f2
        
        if total_amp < 1e-6:
            return {'imd': 0.0, 'imd_db': -100.0}
            
        # d2
        d2_freq = abs(f2 - f1)
        amp_d2 = AudioCalc._find_peak(mag, freqs, d2_freq)
        
        # d3
        d3_low = 2*f1 - f2
        d3_high = 2*f2 - f1
        amp_d3_low = AudioCalc._find_peak(mag, freqs, d3_low) if d3_low > 0 else 0
        amp_d3_high = AudioCalc._find_peak(mag, freqs, d3_high)
        
        distortion_sum_sq = amp_d2**2 + amp_d3_low**2 + amp_d3_high**2
        imd = np.sqrt(distortion_sum_sq) / total_amp
        
        return {
            'imd': imd * 100,
            'imd_db': 20 * np.log10(imd) if imd > 1e-9 else -100.0
        }

    @staticmethod
    def calculate_multitone_tdn(mag, freqs, tone_freqs, window_width_pct=0.05):
        """
        Calculates Multi-tone TD+N.
        mag: Linear magnitude spectrum
        freqs: Frequency bins
        tone_freqs: List of expected tone frequencies
        """
        # Use a mask to identify bins belonging to tones
        is_tone_bin = np.zeros(len(mag), dtype=bool)
        
        for f in tone_freqs:
            # Find peak near f
            width = max(10.0, f * window_width_pct)
            mask_search = (freqs >= f - width) & (freqs <= f + width)
            
            if np.any(mask_search):
                subset_idxs = np.where(mask_search)[0]
                subset_mag = mag[subset_idxs]
                local_peak_idx_rel = np.argmax(subset_mag)
                peak_idx = subset_idxs[local_peak_idx_rel]
                
                # Mark bins around peak as tone
                # Blackman-Harris main lobe is approx +/- 4 bins
                start = max(0, peak_idx - 4)
                end = min(len(mag), peak_idx + 5)
                is_tone_bin[start:end] = True
                
        # Calculate Energies
        # We can sum squares directly
        tone_energy = np.sum(mag[is_tone_bin]**2)
        noise_energy = np.sum(mag[~is_tone_bin]**2)
        
        if tone_energy <= 1e-12:
            return {'tdn': 0.0, 'tdn_db': -100.0}
            
        tdn = np.sqrt(noise_energy / tone_energy)
        
        return {
            'tdn': tdn * 100,
            'tdn_db': 20 * np.log10(tdn) if tdn > 1e-9 else -100.0
        }

    @staticmethod
    def calculate_spdr(mag, freqs, fundamental_freq, window_width_pct=0.1):
        """
        Calculates Spurious-Free Dynamic Range (SPDR).
        SPDR is the ratio of the fundamental signal power to the power of the 
        largest spurious signal (harmonic or non-harmonic).
        """
        # Find Fundamental Peak
        width = max(10.0, fundamental_freq * window_width_pct)
        fund_mask = (freqs >= fundamental_freq - width) & (freqs <= fundamental_freq + width)
        
        if not np.any(fund_mask):
            return {'spdr_db': 0.0, 'max_spur_freq': 0.0, 'max_spur_amp': 0.0}
            
        fund_amp = np.max(mag[fund_mask])
        
        if fund_amp < 1e-9:
             return {'spdr_db': 0.0, 'max_spur_freq': 0.0, 'max_spur_amp': 0.0}
             
        # Mask out fundamental for spur search
        # We also typically mask out DC
        search_mask = (freqs > 20.0) & ~fund_mask
        
        if not np.any(search_mask):
             return {'spdr_db': 100.0, 'max_spur_freq': 0.0, 'max_spur_amp': 0.0}
             
        # Find max spur
        spur_idx_rel = np.argmax(mag[search_mask])
        spur_idxs = np.where(search_mask)[0]
        spur_idx = spur_idxs[spur_idx_rel]
        
        spur_amp = mag[spur_idx]
        spur_freq = freqs[spur_idx]
        
        if spur_amp < 1e-12:
            spdr_db = 140.0 # High dynamic range
        else:
            spdr_db = 20 * np.log10(fund_amp / spur_amp)
            
        return {
            'spdr_db': spdr_db,
            'max_spur_freq': spur_freq,
            'max_spur_amp': spur_amp
        }

    @staticmethod
    def calculate_pim(mag, freqs, f1, f2, order=3):
        """
        Calculates Passive Intermodulation (PIM) / Phase Intermodulation.
        For 2-tone test, PIM usually manifests as IMD products.
        This implementation focuses on odd-order IMD products which are typical for PIM.
        """
        # Similar to IMD CCIF/SMPTE but we look for specific PIM orders (IM3, IM5, IM7)
        # IM3: 2f1 - f2, 2f2 - f1
        # IM5: 3f1 - 2f2, 3f2 - 2f1
        # IM7: 4f1 - 3f2, 4f2 - 3f1
        
        # Find carrier amplitudes
        amp_f1 = AudioCalc._find_peak(mag, freqs, f1)
        amp_f2 = AudioCalc._find_peak(mag, freqs, f2)
        carrier_amp = (amp_f1 + amp_f2) / 2 # Average carrier power
        
        if carrier_amp < 1e-6:
            return {'pim_db': -100.0, 'products': []}
            
        products = []
        sum_sq_pim = 0.0
        
        # Calculate up to specified order (must be odd)
        for n in range(3, order + 2, 2):
            # n is order (3, 5, 7...)
            # For order n, coeffs sum to 1? No.
            # IM3: 2,-1 (sum 1). 
            # IM5: 3,-2 (sum 1).
            # General: k * f1 - (k-1) * f2
            # where 2k - 1 = n => k = (n+1)/2
            
            k = (n + 1) // 2
            m = k - 1
            
            # Lower side
            im_low = k * f1 - m * f2
            # Upper side
            im_high = k * f2 - m * f1
            
            amp_low = AudioCalc._find_peak(mag, freqs, im_low) if im_low > 0 else 0
            amp_high = AudioCalc._find_peak(mag, freqs, im_high)
            
            sum_sq_pim += amp_low**2 + amp_high**2
            
            products.append({
                'order': n,
                'freq_low': im_low,
                'amp_low': amp_low,
                'freq_high': im_high,
                'amp_high': amp_high
            })
            
        pim_rms = np.sqrt(sum_sq_pim)
        
        if pim_rms < 1e-12:
            pim_db = -140.0
        else:
            # PIM is often relative to carrier power (dBc)
            pim_db = 20 * np.log10(pim_rms / carrier_amp)
            
        return {
            'pim_db': pim_db,
            'products': products
        }

    @staticmethod
    def calculate_noise_profile(mag, freqs, sampling_rate):
        """
        Calculates noise profile including Hum, White, and 1/f noise.
        mag: Magnitude spectrum (Linear V/rtHz)
        freqs: Frequency bins
        """
        results = {}
        
        # 1. Hum Noise Detection (50Hz vs 60Hz)
        # Search for peaks at 50Hz and 60Hz
        def get_power_in_band(f_center, width=5.0):
            mask = (freqs >= f_center - width) & (freqs <= f_center + width)
            if not np.any(mask):
                return 0.0
            # Integration: Power = sum(PSD^2 * bin_width)
            # mag is V/rtHz. mag^2 is V^2/Hz.
            # bin_width = fs / N = freqs[1] - freqs[0]
            bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            power = np.sum(mag[mask]**2) * bin_width
            return power

        p50 = get_power_in_band(50.0)
        p60 = get_power_in_band(60.0)
        
        base_freq = 50.0 if p50 > p60 else 60.0
        results['hum_freq'] = base_freq
        
        # Sum harmonics
        hum_power = 0.0
        hum_components = []
        for i in range(1, 11): # Fundamental + 9 harmonics
            f_h = base_freq * i
            if f_h > sampling_rate / 2:
                break
            p_h = get_power_in_band(f_h)
            hum_power += p_h
            hum_components.append((f_h, np.sqrt(p_h)))
            
        results['hum_rms'] = np.sqrt(hum_power)
        results['hum_components'] = hum_components
        
        # 2. 1/f Noise Analysis
        # Fit log(PSD) vs log(f) in 1Hz - 100Hz
        # Avoid Hum frequencies
        fit_mask = (freqs >= 1.0) & (freqs <= 100.0)
        # Exclude hum regions
        for f_h, _ in hum_components:
            fit_mask &= ~((freqs >= f_h - 5.0) & (freqs <= f_h + 5.0))
            
        # Estimate White Noise (Median of 1k-20k)
        white_mask = (freqs >= 1000.0) & (freqs <= 20000.0)
        if np.any(white_mask):
            # Median is robust to peaks, but under-estimates RMS of Gaussian noise (Rayleigh magnitude)
            # Factor: RMS / Median = 1 / sqrt(ln(2)) ~= 1.2011
            white_density = np.median(mag[white_mask]) * 1.2011
        else:
            white_density = 1e-9 # Fallback
            
        results['white_density'] = white_density # V/rtHz
        
        # Determine Fit Upper Bound
        # Find first frequency where mag < white_density * 1.5 (approx 3.5dB margin)
        # Search in 1Hz - 1kHz range
        search_mask = (freqs >= 1.0) & (freqs <= 1000.0)
        search_freqs = freqs[search_mask]
        search_mags = mag[search_mask]
        
        # Smooth magnitudes slightly to avoid triggering on dips
        # Simple moving average of 3 bins
        if len(search_mags) > 3:
            search_mags_smooth = np.convolve(search_mags, np.ones(3)/3, mode='same')
        else:
            search_mags_smooth = search_mags
            
        # Find knee
        knee_indices = np.where(search_mags_smooth < white_density * 2.0)[0]
        if len(knee_indices) > 0:
            f_knee = search_freqs[knee_indices[0]]
        else:
            f_knee = 100.0 # Default if never drops
            
        # Clamp knee
        f_max_fit = np.clip(f_knee, 5.0, 400.0) # Minimum 5Hz range, max 400Hz
        
        # Fit 1/f
        # Range: 1Hz to f_max_fit
        # Exclude Hum regions
        mask_1f = (freqs >= 1.0) & (freqs <= f_max_fit)
        
        # Exclude hum
        for h_freq, h_amp in hum_components:
            mask_1f &= ~((freqs >= h_freq - 5.0) & (freqs <= h_freq + 5.0))
            
        if np.sum(mask_1f) > 5:
            f_log = np.log10(freqs[mask_1f])
            m_log = np.log10(mag[mask_1f] + 1e-15)
            
            # Linear regression: m_log = slope * f_log + intercept
            slope, intercept = np.polyfit(f_log, m_log, 1)
            results['flicker_slope'] = slope
            results['flicker_intercept'] = intercept
        else:
            results['flicker_slope'] = 0.0
            results['flicker_intercept'] = 0.0
        
        # Calculate Corner Frequency
        if results['flicker_slope'] != 0:
            log_white = np.log10(white_density + 1e-15)
            x_c = (log_white - results['flicker_intercept']) / results['flicker_slope']
            
            if x_c > 9:
                results['corner_freq'] = 1e9
            elif x_c < -9:
                results['corner_freq'] = 1e-9
            else:
                results['corner_freq'] = 10**x_c
        else:
            results['corner_freq'] = 0.0
            
        # Explicit 1/f Power Calculation
        # Integrate the fitted 1/f curve from 20Hz to 20kHz (or Corner Freq)
        # Power density P(f) = (10^(slope*log10(f) + intercept))^2
        # P(f) = 10^(2*intercept) * f^(2*slope)
        # Integral P(f) df = C * [ f^(2*slope + 1) / (2*slope + 1) ]
        
        if results['flicker_slope'] != 0:
            # We integrate 1/f component over the full audio bandwidth (20Hz-20kHz)
            # because physically 1/f noise exists at all frequencies, even if buried under white noise.
            f_start = 20.0
            f_end = 20000.0
            
            if f_end > f_start:
                A = 10**(results['flicker_intercept'])
                alpha = results['flicker_slope']
                # Density V(f) = A * f^alpha
                # Power Density S(f) = V(f)^2 = A^2 * f^(2*alpha)
                
                # Integral of x^k is x^(k+1)/(k+1)
                k = 2 * alpha
                C = A**2
                
                if abs(k + 1) < 1e-9: # 1/f case (slope -0.5 -> k=-1)
                    # Integral is ln(f)
                    power_flicker = C * (np.log(f_end) - np.log(f_start))
                else:
                    power_flicker = C * ((f_end**(k+1)) - (f_start**(k+1))) / (k+1)
                    
                results['flicker_rms'] = np.sqrt(max(0, power_flicker))
            else:
                results['flicker_rms'] = 0.0
        else:
            results['flicker_rms'] = 0.0

        # 4. Integrated Noise in Bands
        def integrate_band(f_start, f_end):
            mask = (freqs >= f_start) & (freqs < f_end)
            if not np.any(mask):
                return 0.0
            bin_width = freqs[1] - freqs[0]
            return np.sqrt(np.sum(mag[mask]**2) * bin_width)
            
        results['noise_rms_20k'] = integrate_band(20, 20000)
        results['noise_rms_100k'] = integrate_band(20, 100000)
        
        # Peak Detection
        # Find peak in 20Hz-20kHz, excluding Hum components
        peak_mask = (freqs >= 20.0) & (freqs <= 20000.0)
        
        # Exclude Hum regions from peak search (optional, but requested to find "Other" noise)
        # If we want the absolute peak, we shouldn't exclude hum.
        # But user asked for "Other" noise.
        # Let's find the absolute peak first.
        if np.any(peak_mask):
            peak_idx_rel = np.argmax(mag[peak_mask])
            peak_freqs = freqs[peak_mask]
            peak_mags = mag[peak_mask]
            
            results['peak_freq'] = peak_freqs[peak_idx_rel]
            results['peak_amp'] = peak_mags[peak_idx_rel]
        else:
            results['peak_freq'] = 0.0
            results['peak_amp'] = 0.0
        
        # A-weighting Integration
        # Ra(f) = (12194^2 * f^4) / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
        # Gain = 20*log10(Ra(f)) + 2.00
        # Linear Gain = Ra(f) * 10^(2.0/20) = Ra(f) * 1.2589
        
        f = freqs
        f2 = f**2
        const = 12194**2 * f**4
        denom = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
        # Avoid division by zero
        denom[denom == 0] = 1.0
        Ra = const / denom
        weighting_linear = Ra * 1.2589
        
        # Apply weighting to magnitude (V/rtHz)
        mag_a = mag * weighting_linear
        
        # Integrate A-weighted spectrum (20Hz - 20kHz)
        mask_a = (freqs >= 20) & (freqs <= 20000)
        if np.any(mask_a):
            bin_width = freqs[1] - freqs[0]
            results['noise_rms_a_weighted'] = np.sqrt(np.sum(mag_a[mask_a]**2) * bin_width)
        else:
            results['noise_rms_a_weighted'] = 0.0
        
        return results
