import unittest
import numpy as np
import math
import sys

# Attempt to import functions from the main script.
# This path might need adjustment based on how tests are run (e.g., relative import if it's a package)
try:
    from audio_crosstalk_analyzer.audio_crosstalk_analyzer import (
        generate_sine_wave,
        _find_peak_amplitude_in_band,
        analyze_recorded_channels,
        channel_spec_to_index,
        dbfs_to_linear, # For test_generate_sine_wave
        linear_to_dbfs # For test_crosstalk_calculation_logic
    )
except ImportError:
    # Fallback for running directly from the directory, assuming audio_crosstalk_analyzer.py is in the same dir or PYTHONPATH is set
    # This is less ideal but common for simpler project structures.
    # For proper package structure, the above import should work.
    # If this script is in audio_crosstalk_analyzer/ and main script is also there,
    # then from .audio_crosstalk_analyzer import ... would be typical if audio_crosstalk_analyzer is a package.
    # If running tests from parent of audio_crosstalk_analyzer, then the original try should work.
    # Let's assume for now the original try is the target for a structured project.
    # Adding a simple path modification for local testing if needed:
    # import os
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add parent dir
    # from audio_crosstalk_analyzer.audio_crosstalk_analyzer import ...
    # This is complex to get right without knowing execution context.
    # For now, let the original try block handle it.
    # If it fails, the tests won't run, which is an indicator of import issues.
    print("Error: Could not import functions from audio_crosstalk_analyzer.audio_crosstalk_analyzer. Ensure PYTHONPATH is set correctly or adjust import paths.", file=sys.stderr)
    # Re-raising the error to make it clear
    raise

class TestAudioCrosstalkAnalyzer(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 48000
        self.duration = 1.0
        self.N = int(self.sample_rate * self.duration)
        self.t = np.linspace(0, self.duration, self.N, endpoint=False)

    def test_generate_sine_wave(self):
        freq = 1000.0
        amp_dbfs = -6.0
        expected_amp_linear = dbfs_to_linear(amp_dbfs) # Approx 0.501

        signal = generate_sine_wave(freq, amp_dbfs, self.duration, self.sample_rate)

        self.assertEqual(len(signal), self.N, "Signal length incorrect")
        # Check peak amplitude (can be slightly less than expected_amp_linear due to discrete time points)
        self.assertAlmostEqual(np.max(np.abs(signal)), expected_amp_linear, delta=0.01, msg="Signal peak amplitude incorrect")

        # FFT check
        fft_result = np.fft.rfft(signal * np.hanning(self.N)) # Apply window
        fft_freqs = np.fft.rfftfreq(self.N, d=1/self.sample_rate)
        peak_idx = np.argmax(np.abs(fft_result))
        detected_freq = fft_freqs[peak_idx]
        self.assertAlmostEqual(detected_freq, freq, delta=1.0, msg="Dominant frequency in generated signal is incorrect")

    def test_find_peak_amplitude_in_band(self):
        fft_frequencies = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        fft_magnitudes = np.array([0.1, 0.5, 1.0, 0.8, 0.6, 0.4, 0.2]) # Peak at 20Hz, amplitude 1.0

        # Test 1: Peak well within band
        target_freq = 20.0
        search_half_width = 5.0 # Band [15, 25]
        act_freq, amp = _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width)
        self.assertEqual(act_freq, 20.0)
        self.assertEqual(amp, 1.0)

        # Test 2: Peak at edge of band (lower edge search)
        target_freq = 17.5 # Searching around 17.5, band [12.5, 22.5]
        act_freq, amp = _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width)
        self.assertEqual(act_freq, 20.0) # Still finds 20Hz as peak
        self.assertEqual(amp, 1.0)

        # Test 3: No peak in band
        target_freq = 100.0
        search_half_width = 5.0 # Band [95, 105]
        act_freq, amp = _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width)
        self.assertEqual(act_freq, 100.0) # Returns nominal target_freq
        self.assertEqual(amp, 0.0)     # Returns 0 amplitude

        # Test 4: Exact frequency matching, narrow band
        target_freq = 30.0
        search_half_width = 1.0 # Band [29, 31]
        act_freq, amp = _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width)
        self.assertEqual(act_freq, 30.0)
        self.assertEqual(amp, 0.8)
        
        # Test 5: Empty FFT data (should not happen in practice if analyze_recorded_channels guards N > 0)
        empty_freqs = np.array([])
        empty_mags = np.array([])
        act_freq, amp = _find_peak_amplitude_in_band(empty_mags, empty_freqs, 20.0, 5.0)
        self.assertEqual(act_freq, 20.0) # Returns nominal target_freq
        self.assertEqual(amp, 0.0)     # Returns 0 amplitude


    def test_analyze_recorded_channels(self):
        freq = 1000.0
        amp_ch0_lin = 0.5
        amp_ch1_lin = 0.05

        ch0_signal = amp_ch0_lin * np.sin(2 * np.pi * freq * self.t)
        ch1_signal = amp_ch1_lin * np.sin(2 * np.pi * freq * self.t)
        
        # Add a tiny bit of noise to avoid pure zeros if signal is zero
        ch0_signal += np.random.normal(0, 1e-9, self.N)
        ch1_signal += np.random.normal(0, 1e-9, self.N)

        recorded_data = np.array([ch0_signal, ch1_signal]).T # Shape (N, 2)
        window_name = 'hann'

        analysis_results = analyze_recorded_channels(recorded_data, self.sample_rate, freq, window_name)

        self.assertEqual(len(analysis_results), 2, "Analyze_recorded_channels did not return results for 2 channels")
        
        # Channel 0
        self.assertAlmostEqual(analysis_results[0]['amplitude_linear'], amp_ch0_lin, delta=0.02, msg="Ch0 amplitude mismatch")
        self.assertAlmostEqual(analysis_results[0]['actual_freq'], freq, delta=1.0, msg="Ch0 frequency mismatch")
        
        # Channel 1
        self.assertAlmostEqual(analysis_results[1]['amplitude_linear'], amp_ch1_lin, delta=0.02, msg="Ch1 amplitude mismatch")
        self.assertAlmostEqual(analysis_results[1]['actual_freq'], freq, delta=1.0, msg="Ch1 frequency mismatch")

    def test_analyze_recorded_channels_empty_input(self):
        empty_data = np.array([]).reshape(0,2) # 0 samples, 2 channels
        results = analyze_recorded_channels(empty_data, self.sample_rate, 1000.0, 'hann')
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertEqual(res['amplitude_linear'], 0.0)

        none_data = None
        results_none = analyze_recorded_channels(none_data, self.sample_rate, 1000.0, 'hann')
        self.assertEqual(len(results_none), 0)


    def test_crosstalk_calculation_logic(self):
        # Case 1: Standard crosstalk
        amp_driven_linear = 0.5
        amp_undriven_linear = 0.005
        expected_crosstalk_db = 20 * math.log10(amp_undriven_linear / amp_driven_linear) # Should be -40 dB
        
        # Simulate what happens in main loop
        # Here we directly test the formula as it appears in main, not by calling main()
        calculated_crosstalk_db = np.nan
        if amp_driven_linear > 1e-12:
            if amp_undriven_linear > 1e-12:
                calculated_crosstalk_db = 20 * math.log10(amp_undriven_linear / amp_driven_linear)
            else:
                calculated_crosstalk_db = -np.inf
        elif amp_undriven_linear > 1e-12: # driven is zero, undriven is not
            calculated_crosstalk_db = np.inf
        
        self.assertAlmostEqual(calculated_crosstalk_db, expected_crosstalk_db)

        # Case 2: Undriven is zero (or very low)
        amp_undriven_linear_zero = 1e-15 
        expected_crosstalk_db_zero_undriven = -np.inf
        
        calculated_crosstalk_db_zero_undriven = np.nan
        if amp_driven_linear > 1e-12:
            if amp_undriven_linear_zero > 1e-12: # This will be false
                calculated_crosstalk_db_zero_undriven = 20 * math.log10(amp_undriven_linear_zero / amp_driven_linear)
            else:
                calculated_crosstalk_db_zero_undriven = -np.inf
        elif amp_undriven_linear_zero > 1e-12:
             calculated_crosstalk_db_zero_undriven = np.inf
        self.assertEqual(calculated_crosstalk_db_zero_undriven, expected_crosstalk_db_zero_undriven)

        # Case 3: Driven is zero (or very low), undriven has signal
        amp_driven_linear_zero = 1e-15
        amp_undriven_linear_signal = 0.01
        expected_crosstalk_db_zero_driven = np.inf
        
        calculated_crosstalk_db_zero_driven = np.nan
        if amp_driven_linear_zero > 1e-12: # This will be false
            # ...
            pass
        elif amp_undriven_linear_signal > 1e-12:
            calculated_crosstalk_db_zero_driven = np.inf
        self.assertEqual(calculated_crosstalk_db_zero_driven, expected_crosstalk_db_zero_driven)

        # Case 4: Both driven and undriven are zero (or very low)
        amp_driven_both_zero = 1e-15
        amp_undriven_both_zero = 1e-14
        expected_crosstalk_db_both_zero = np.nan # Remains NaN as per logic: driven <= 1e-12, and undriven <= 1e-12
        
        calculated_crosstalk_db_both_zero = np.nan
        if amp_driven_both_zero > 1e-12:
            pass
        elif amp_undriven_both_zero > 1e-12:
            calculated_crosstalk_db_both_zero = np.inf
        # else: it remains np.nan
        self.assertTrue(np.isnan(calculated_crosstalk_db_both_zero))


    def test_channel_spec_to_index(self):
        # Assuming a device with 2 channels for L/R tests
        self.assertEqual(channel_spec_to_index('L', device_channels=2, channel_type="output"), 0)
        self.assertEqual(channel_spec_to_index('R', device_channels=2, channel_type="output"), 1)
        
        # Numeric strings (0-based)
        self.assertEqual(channel_spec_to_index('0', device_channels=2, channel_type="input"), 0)
        self.assertEqual(channel_spec_to_index('1', device_channels=2, channel_type="input"), 1)
        
        # Test with more channels
        self.assertEqual(channel_spec_to_index('2', device_channels=4, channel_type="input"), 2)
        self.assertEqual(channel_spec_to_index(2, device_channels=4, channel_type="input"), 2) # Integer input

        # Invalid specifiers
        self.assertIsNone(channel_spec_to_index('X', device_channels=2, channel_type="input")) # Invalid string
        self.assertIsNone(channel_spec_to_index('2', device_channels=2, channel_type="input")) # Out of bounds (0, 1 are valid)
        self.assertIsNone(channel_spec_to_index('-1', device_channels=2, channel_type="input"))# Out of bounds
        self.assertIsNone(channel_spec_to_index('R', device_channels=1, channel_type="input")) # 'R' not valid for 1-channel device
        self.assertEqual(channel_spec_to_index('L', device_channels=1, channel_type="input"), 0) # 'L' is valid for 1-channel device

if __name__ == '__main__':
    # This allows running the tests directly from this file
    unittest.main()
