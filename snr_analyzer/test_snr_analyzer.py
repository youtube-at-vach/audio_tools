import unittest
import numpy as np
import math
import sys
import os

# Add the parent directory to sys.path to allow direct import of snr_analyzer
# This is necessary for running the test script directly or with some test runners.
# The `python -m unittest` command from the root directory usually handles this,
# but adding it here makes the script more robust.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from snr_analyzer.snr_analyzer import calculate_rms, generate_sine_wave

# For testing the SNR calculation logic, we'll replicate the core math here
# as the measure_snr function itself is more of an integration test due to audio I/O.
def calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise):
    """
    Calculates SNR in dB based on RMS values of (signal+noise) and noise.
    This function replicates the core SNR calculation logic from measure_snr
    for testing purposes.
    """
    if rms_noise < 1e-12: # Effectively zero noise
        if rms_signal_plus_noise < 1e-12: # Effectively zero signal+noise
             return 0.0 # Or some indicator of 0/0, but 0dB seems reasonable
        return float('inf') # Or a very large dB value as per implementation

    power_signal_plus_noise = rms_signal_plus_noise**2
    power_noise = rms_noise**2

    if power_signal_plus_noise < power_noise:
        # This implies signal power is negative, which is not physical.
        # The actual signal RMS is effectively zero in this measurement.
        rms_signal_only_val = 1e-12 # Effectively zero signal
    else:
        rms_signal_only_val = np.sqrt(power_signal_plus_noise - power_noise)
    
    if rms_signal_only_val < 1e-12: # Effectively zero signal
        if rms_noise < 1e-12: # Avoid log(0/0) case, though covered by initial rms_noise check
            return 0.0 # Or handle as very high SNR if signal is also non-zero
        # Return a very low SNR for signal being zero or much smaller than resolvable noise
        # Based on snr_analyzer.py, it uses 1e-12 / rms_noise in log if signal is too low
        return 20 * math.log10(1e-12 / rms_noise)
    
    snr_db = 20 * math.log10(rms_signal_only_val / rms_noise)
    return snr_db


class TestCalculateRMS(unittest.TestCase):
    def test_rms_sine_wave(self):
        amplitude = 1.0
        samplerate = 48000
        duration = 1.0
        frequency = 100.0
        # generate_sine_wave is assumed to be available and tested separately
        # For pure RMS test, we can construct a simple sine wave directly
        t = np.linspace(0, duration, int(samplerate * duration), False)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        
        expected_rms = amplitude / np.sqrt(2)
        self.assertAlmostEqual(calculate_rms(sine_wave), expected_rms, places=5)

    def test_rms_dc_signal(self):
        dc_value = 0.5
        dc_signal = np.full(1000, dc_value)
        expected_rms = dc_value
        self.assertAlmostEqual(calculate_rms(dc_signal), expected_rms, places=5)

    def test_rms_zero_signal(self):
        zero_signal = np.zeros(1000)
        # calculate_rms returns 1e-12 for zero signal to avoid log(0) issues later
        self.assertAlmostEqual(calculate_rms(zero_signal), 1e-12, places=15)

    def test_rms_empty_signal(self):
        empty_signal = np.array([])
        # calculate_rms returns 1e-12 for empty signal
        self.assertAlmostEqual(calculate_rms(empty_signal), 1e-12, places=15)
        
    def test_rms_none_signal(self):
        none_signal = None
        # calculate_rms returns 1e-12 for None signal
        self.assertAlmostEqual(calculate_rms(none_signal), 1e-12, places=15)


class TestGenerateSineWave(unittest.TestCase):
    def test_wave_properties(self):
        frequency = 100.0
        duration = 0.5
        amplitude = 0.7
        samplerate = 44100
        
        wave = generate_sine_wave(frequency, duration, amplitude, samplerate)
        
        self.assertIsInstance(wave, np.ndarray)
        self.assertEqual(wave.shape[0], int(samplerate * duration))
        self.assertEqual(wave.dtype, np.float32)
        
        # Check if all values are within [-amplitude, +amplitude]
        self.assertTrue(np.all(wave >= -amplitude - 1e-6)) # Adding tolerance for float precision
        self.assertTrue(np.all(wave <= amplitude + 1e-6))
        
        # Check if at least one peak is close to amplitude and one trough close to -amplitude
        # This is a sanity check that it's not all zeros for example
        if duration * frequency >= 1: # Ensure at least one full cycle
            self.assertTrue(np.max(wave) > amplitude * 0.95)
            self.assertTrue(np.min(wave) < -amplitude * 0.95)


class TestSNRCalculationLogic(unittest.TestCase):
    # Using the helper calculate_snr_db_from_rms which mimics the logic in measure_snr

    def test_snr_ideal_case(self):
        # Signal RMS = 1, Noise RMS = 1. Signal+Noise RMS = sqrt(1^2 + 1^2) = sqrt(2)
        rms_signal_plus_noise = np.sqrt(2.0)
        rms_noise = 1.0
        
        # Derived RMS_signal_only = sqrt( (sqrt(2))^2 - 1^2 ) = sqrt(2 - 1) = 1
        # SNR = 20 * log10(1/1) = 0 dB
        expected_snr = 0.0
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_high_snr(self):
        # Example: Signal RMS = approx 10, Noise RMS = 0.1
        # RMS_signal_only = np.sqrt(10.0**2 - 0.1**2) is not how rms_signal_plus_noise is constructed.
        # Let's define rms_signal_only and rms_noise, then calculate rms_signal_plus_noise
        rms_signal_target = 10.0 
        rms_noise = 0.1
        rms_signal_plus_noise = np.sqrt(rms_signal_target**2 + rms_noise**2) # Assuming uncorrelated
        
        # Derived RMS_signal_only should be close to rms_signal_target
        # SNR = 20 * log10(rms_signal_target / rms_noise)
        expected_snr = 20 * np.log10(rms_signal_target / rms_noise) # 20 * log10(10 / 0.1) = 20 * log10(100) = 20 * 2 = 40 dB
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_low_snr_signal_weaker_than_noise(self):
        # Signal power is less than noise power implies (RMS_signal+noise)^2 < RMS_noise^2 if signal was negative,
        # or more realistically, RMS_signal+noise is very close to RMS_noise.
        # Example: True Signal RMS = 0.1, Noise RMS = 1.0
        # Then RMS_signal+noise = sqrt(0.1^2 + 1.0^2) = sqrt(0.01 + 1.0) = sqrt(1.01)
        rms_signal_plus_noise = np.sqrt(1.01) 
        rms_noise = 1.0

        # The logic in calculate_snr_db_from_rms should find:
        # power_signal_plus_noise = 1.01
        # power_noise = 1.0
        # rms_signal_only_val = np.sqrt(1.01 - 1.0) = np.sqrt(0.01) = 0.1
        # expected_snr = 20 * math.log10(0.1 / 1.0) = 20 * math.log10(0.1) = 20 * (-1) = -20 dB
        expected_snr = -20.0
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_signal_much_weaker_than_noise_effective_zero_signal(self):
        # Case where (RMS_signal+noise)^2 is less than RMS_noise^2 (due to measurement variance or actual low signal)
        # The calculation should result in rms_signal_only being effectively 0 (1e-12).
        rms_signal_plus_noise = 0.5 
        rms_noise = 1.0
        
        # power_signal_plus_noise = 0.25
        # power_noise = 1.0
        # power_signal_plus_noise < power_noise, so rms_signal_only_val becomes 1e-12
        # expected_snr = 20 * math.log10(1e-12 / 1.0) = 20 * (-12) = -240 dB
        expected_snr = 20 * math.log10(1e-12 / rms_noise)
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_zero_noise(self):
        rms_signal_plus_noise = 1.0
        rms_noise_val = 1e-13 # Value that is < 1e-12, considered zero by the function

        # According to calculate_snr_db_from_rms, this should return float('inf')
        expected_snr = float('inf')
        self.assertEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise_val), expected_snr)

    def test_snr_zero_noise_and_zero_signal_plus_noise(self):
        rms_signal_plus_noise = 1e-13
        rms_noise_val = 1e-13

        # According to calculate_snr_db_from_rms, 0/0 case should return 0.0 dB
        expected_snr = 0.0
        self.assertEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise_val), expected_snr)
        
    def test_snr_zero_signal_only(self):
        # This happens if rms_signal_plus_noise is equal to rms_noise
        rms_signal_plus_noise = 1.0
        rms_noise = 1.0
        
        # power_signal_plus_noise = 1.0
        # power_noise = 1.0
        # rms_signal_only_val becomes sqrt(0) = 0, which is then treated as 1e-12
        # expected_snr = 20 * math.log10(1e-12 / 1.0) = -240 dB
        expected_snr = 20 * math.log10(1e-12 / rms_noise)
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)


if __name__ == '__main__':
    # This allows running the tests directly from this file: python snr_analyzer/test_snr_analyzer.py
    # However, the standard way is `python -m unittest snr_analyzer.test_snr_analyzer` from root.
    
    # If snr_analyzer is not installed and we are running this script directly,
    # we might need to adjust sys.path. The code at the top attempts to handle this.
    # Ensure that snr_analyzer.snr_analyzer can be imported.
    # For `python -m unittest`, this is typically handled correctly if run from the project root.
    
    # Verify imports are working before running tests if script is run directly
    try:
        from snr_analyzer.snr_analyzer import calculate_rms, generate_sine_wave
        print("Successfully imported from snr_analyzer.snr_analyzer")
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Please ensure that the script is run from the project root directory,")
        print("or that snr_analyzer is in the PYTHONPATH, or use `python -m unittest discover`.")
        sys.exit(1)
        
    unittest.main()
