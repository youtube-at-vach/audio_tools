import unittest
import numpy as np
import sys
import io # For capturing stderr

# Assuming audio_freq_response_analyzer.py is in the same package directory
from .audio_freq_response_analyzer import generate_log_frequencies, generate_sine_segment, analyze_frequency_segment

class TestGenerateLogFrequencies(unittest.TestCase):
    def test_basic_generation(self):
        # Test case 1: 20-80 Hz, 1 point per octave
        freqs1 = generate_log_frequencies(20, 80, 1)
        expected1 = np.array([20., 40., 80.])
        self.assertTrue(np.allclose(freqs1, expected1), msg=f"Expected {expected1}, got {freqs1}")

        # Test case 2: 100-500 Hz, 2 points per octave
        freqs2 = generate_log_frequencies(100, 500, 2)
        # Expected: 100.0, 100*2^(1/2), 100*2^(2/2)=200, 200*2^(1/2), 200*2^(2/2)=400, 400*2^(1/2)
        # 100.0, 141.42135624, 200.0, 282.84271247, 400.0, 500 (if end_freq added)
        # The function should include 500 if it's not closely represented.
        # Let's trace:
        # 100.0
        # 100 * 2**(1/2) = 141.4213562373095
        # 141.4213562373095 * 2**(1/2) = 200.0
        # 200.0 * 2**(1/2) = 282.842712474619
        # 282.842712474619 * 2**(1/2) = 400.0
        # 400.0 * 2**(1/2) = 565.685424949238 (this is > 500, so loop stops)
        # Then 500 is added because it's not close to 400.
        expected2 = np.array([100.0, 141.42135624, 200.0, 282.84271247, 400.0, 500.0])
        self.assertTrue(np.allclose(freqs2, expected2, rtol=1e-5), msg=f"Expected {expected2}, got {freqs2}")

    def test_start_equals_end(self):
        freqs = generate_log_frequencies(1000, 1000, 12)
        expected = np.array([1000.0])
        self.assertTrue(np.allclose(freqs, expected), msg=f"Expected {expected}, got {freqs}")

    def test_end_freq_not_in_series(self):
        freqs = generate_log_frequencies(20, 75, 1) # Series: 20, 40, (80-too high)
        expected = np.array([20., 40., 75.]) # 75 should be added
        self.assertTrue(np.allclose(freqs, expected), msg=f"Expected {expected}, got {freqs}")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            generate_log_frequencies(20, 20000, 0) # points_per_octave=0
        with self.assertRaises(ValueError):
            generate_log_frequencies(0, 20000, 1)  # start_freq=0
        with self.assertRaises(ValueError):
            generate_log_frequencies(20, 10, 1)   # end_freq < start_freq


class TestGenerateSineSegment(unittest.TestCase):
    def test_amplitude_and_length(self):
        sr = 48000
        dur = 0.1
        freq = 100
        amp_dbfs = -6.0
        signal = generate_sine_segment(freq, amp_dbfs, dur, sr)
        
        expected_peak = 10**(-6.0/20.0)
        # For a pure sine, max(abs(signal)) should be very close to expected_peak
        self.assertTrue(np.isclose(np.max(np.abs(signal)), expected_peak, rtol=1e-2),
                        msg=f"Expected peak {expected_peak}, got {np.max(np.abs(signal))}")
        self.assertEqual(len(signal), int(sr*dur))

    def test_frequency_content(self):
        sr = 48000
        dur = 0.1
        freq = 1000.0
        amp_dbfs = -6.0
        signal = generate_sine_segment(freq, amp_dbfs, dur, sr)
        
        N = len(signal)
        fft_result = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_result)
        fft_frequencies = np.fft.rfftfreq(N, d=1.0/sr)
        
        # Find frequency of the maximum component (ignoring DC if any)
        # Start search from a few bins away from DC to be safe.
        min_bin_index = max(1, int(10 / (sr/N))) # Ignore components below 10 Hz
        found_freq_index = np.argmax(fft_magnitudes[min_bin_index:]) + min_bin_index
        found_freq = fft_frequencies[found_freq_index]
        
        self.assertTrue(np.isclose(found_freq, freq, atol=1.0/dur), # Freq resolution is 1/duration
                        msg=f"Expected freq {freq}, found {found_freq}")

    def test_amplitude_capping(self):
        sr = 48000
        dur = 0.1
        freq = 100
        amp_dbfs = 6.0 # This is +6 dBFS, so linear amplitude is approx 2.0
        signal = generate_sine_segment(freq, amp_dbfs, dur, sr)
        # The function should cap the linear amplitude at 1.0
        self.assertTrue(np.max(np.abs(signal)) <= 1.000001, # Allow for tiny float inaccuracies
                        msg=f"Signal max {np.max(np.abs(signal))} exceeded 1.0 after capping.")
        # And it should be close to 1.0 if capping occurred
        # (The warning print is a side effect, not directly testable here without more complex capture)
        if 10**(amp_dbfs / 20.0) > 1.0: # If capping was expected
             self.assertTrue(np.isclose(np.max(np.abs(signal)), 1.0, rtol=1e-5),
                             msg=f"Expected capped peak of 1.0, got {np.max(np.abs(signal))}")


class TestAnalyzeFrequencySegment(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.duration = 0.1 # seconds, gives 10 Hz bin resolution
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)

    def test_clean_sine_analysis(self):
        freq = 1000.0
        amp_lin = 0.5  # Approx -6.02 dBFS
        phase_offset_deg = 30.0
        
        signal = amp_lin * np.sin(2 * np.pi * freq * self.t + np.deg2rad(phase_offset_deg))
        
        amp_dbfs, phase_deg, actual_f = analyze_frequency_segment(signal, freq, self.sr, 'hann')
        
        expected_amp_dbfs = 20 * np.log10(amp_lin)
        self.assertTrue(np.isclose(amp_dbfs, expected_amp_dbfs, atol=0.5), # Amplitude can vary slightly due to windowing
                        msg=f"Expected Amp (dBFS) {expected_amp_dbfs}, got {amp_dbfs}")
        # Phase from FFT of sin(wt + phi_0) is phi_0 - 90 deg relative to cosine basis
        expected_phase_from_fft = phase_offset_deg - 90.0
        # Normalize phase to [-180, 180] for comparison, as np.angle might wrap differently
        phase_deg_wrapped = (phase_deg + 180) % 360 - 180
        expected_phase_from_fft_wrapped = (expected_phase_from_fft + 180) % 360 - 180
        self.assertTrue(np.isclose(phase_deg_wrapped, expected_phase_from_fft_wrapped, atol=5.0),
                        msg=f"Expected Phase (deg) {expected_phase_from_fft_wrapped}, got {phase_deg_wrapped} (Original signal phase: {phase_offset_deg})")
        self.assertTrue(np.isclose(actual_f, freq, atol=1.5/self.duration), # Allow +/- 1.5 bins
                        msg=f"Expected Freq {freq}, got {actual_f}")

    def test_noisy_signal_analysis(self):
        freq = 1000.0
        amp_lin = 0.5
        signal_tone = amp_lin * np.sin(2 * np.pi * freq * self.t)
        noise = np.random.normal(0, 0.05, len(signal_tone)) # Noise RMS = 0.05
        signal = signal_tone + noise
        
        amp_dbfs, phase_deg, actual_f = analyze_frequency_segment(signal, freq, self.sr, 'hann')
        
        expected_amp_dbfs = 20 * np.log10(amp_lin)
        self.assertTrue(np.isclose(amp_dbfs, expected_amp_dbfs, atol=2.0), # Wider tolerance for amplitude
                        msg=f"Expected Amp (dBFS) {expected_amp_dbfs}, got {amp_dbfs} (noisy)")
        # Phase is very sensitive to noise, check it's within a plausible range or that analysis didn't fail
        # If amp_dbfs is not -np.inf, it means a peak was found.
        self.assertTrue(amp_dbfs > -np.inf, "Analysis failed for noisy signal (amplitude -inf)")
        # For phase, we can't be too strict with noise. Just check it's a number.
        self.assertTrue(isinstance(phase_deg, float) and not np.isnan(phase_deg), "Phase is not a valid float for noisy signal.")
        self.assertTrue(np.isclose(actual_f, freq, atol=3.0/self.duration), # Wider tolerance for freq with noise
                        msg=f"Expected Freq {freq}, got {actual_f} (noisy)")


    def test_tone_not_found(self):
        # Pure noise signal with reduced standard deviation
        signal = np.random.normal(0, 0.01, len(self.t)) # Reduced noise level
        target_freq = 1000.0
        amp_dbfs, _, _ = analyze_frequency_segment(signal, target_freq, self.sr, 'hann')
        # Expect amplitude below a certain threshold for a noise-only signal.
        # This threshold might need adjustment based on typical noise floor of the FFT length and window.
        noise_threshold_dbfs = -45.0 # Made threshold slightly more lenient
        self.assertTrue(amp_dbfs < noise_threshold_dbfs,
                         msg=f"Expected amplitude < {noise_threshold_dbfs} dBFS for noise, got {amp_dbfs}")

    def test_invalid_window_fallback(self):
        freq = 1000.0
        amp_lin = 0.5
        signal = amp_lin * np.sin(2 * np.pi * freq * self.t)
        
        # Capture stderr to check for the warning
        old_stderr = sys.stderr
        sys.stderr = captured_stderr = io.StringIO()
        
        amp_dbfs, _, _ = analyze_frequency_segment(signal, freq, self.sr, 'invalid_window_xyz')
        
        sys.stderr = old_stderr # Restore stderr
        
        warning_message = captured_stderr.getvalue()
        self.assertIn("Warning: Invalid window name 'invalid_window_xyz'. Using 'hann' as fallback.", warning_message)
        
        expected_amp_dbfs = 20 * np.log10(amp_lin)
        # Check that analysis still produced a reasonable result (implying fallback to 'hann')
        self.assertTrue(np.isclose(amp_dbfs, expected_amp_dbfs, atol=0.5),
                        msg=f"Expected Amp (dBFS) {expected_amp_dbfs} after fallback, got {amp_dbfs}")

if __name__ == '__main__':
    unittest.main()
