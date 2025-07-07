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

from snr_analyzer.snr_analyzer import calculate_rms, generate_sine_wave, measure_snr
from unittest.mock import patch, MagicMock

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
        # Tests a scenario where the estimated signal power equals the noise power.
        # This is achieved by setting RMS_signal_plus_noise = sqrt(2) and RMS_noise = 1.
        # This implies an original RMS_signal_only of 1 (since sqrt(1^2 + 1^2) = sqrt(2)).
        # The expected SNR is 0 dB because RMS_signal_only / RMS_noise = 1 / 1 = 1.
        # This case is important for verifying the baseline calculation when signal and noise contribute equally.
        rms_signal_plus_noise = np.sqrt(2.0)
        rms_noise = 1.0
        
        # Derived RMS_signal_only = sqrt( (sqrt(2))^2 - 1^2 ) = sqrt(2 - 1) = 1
        # SNR = 20 * log10(1/1) = 0 dB
        expected_snr = 0.0
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_high_snr(self):
        # Tests a scenario with a strong signal relative to noise.
        # Here, RMS_signal_target = 10.0 and RMS_noise = 0.1.
        # RMS_signal_plus_noise is calculated assuming uncorrelated signal and noise.
        # Expected SNR is 40 dB (20 * log10(10 / 0.1)).
        # This verifies the calculation for typical, good signal conditions.
        rms_signal_target = 10.0 
        rms_noise = 0.1
        rms_signal_plus_noise = np.sqrt(rms_signal_target**2 + rms_noise**2) # Assuming uncorrelated
        
        # Derived RMS_signal_only should be close to rms_signal_target
        # SNR = 20 * log10(rms_signal_target / rms_noise)
        expected_snr = 20 * np.log10(rms_signal_target / rms_noise) # 20 * log10(10 / 0.1) = 20 * log10(100) = 20 * 2 = 40 dB
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_low_snr_signal_weaker_than_noise(self):
        # Tests a scenario where the true signal is weaker than the noise.
        # Example: True Signal RMS = 0.1, Noise RMS = 1.0.
        # RMS_signal_plus_noise = sqrt(0.1^2 + 1.0^2) = sqrt(1.01).
        # The calculation should correctly estimate the original signal RMS as 0.1.
        # Expected SNR = 20 * log10(0.1 / 1.0) = -20 dB.
        # This verifies behavior when signal is below noise, but still extractable.
        rms_signal_plus_noise = np.sqrt(1.01) 
        rms_noise = 1.0
        expected_snr = -20.0
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_signal_much_weaker_than_noise_effective_zero_signal(self):
        # Tests the case where measured (Signal+Noise) power is less than measured Noise power.
        # This could happen due to measurement variance or if the signal is truly negligible.
        # Example: RMS_signal_plus_noise = 0.5, RMS_noise = 1.0.
        # The logic should cap estimated signal power at zero (represented by RMS 1e-12).
        # Expected SNR = 20 * log10(1e-12 / 1.0) = -240 dB.
        # This ensures graceful handling of scenarios where signal is swamped or measurement is problematic.
        rms_signal_plus_noise = 0.5 
        rms_noise = 1.0
        expected_snr = 20 * math.log10(1e-12 / rms_noise)
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)

    def test_snr_zero_noise(self):
        # Tests the scenario where noise is effectively zero (below 1e-12 threshold).
        # Example: RMS_signal_plus_noise = 1.0, RMS_noise = 1e-13.
        # An SNR of infinity is expected, as per the implemented logic.
        # This verifies the handling of extremely low or zero noise conditions.
        rms_signal_plus_noise = 1.0
        rms_noise_val = 1e-13 # Value that is < 1e-12, considered zero by the function
        expected_snr = float('inf')
        self.assertEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise_val), expected_snr)

    def test_snr_zero_noise_and_zero_signal_plus_noise(self):
        # Tests the edge case where both (Signal+Noise) and Noise are effectively zero.
        # Example: RMS_signal_plus_noise = 1e-13, RMS_noise = 1e-13.
        # The implemented logic should return 0.0 dB for this 0/0-like condition.
        # This verifies a specific boundary condition.
        rms_signal_plus_noise = 1e-13
        rms_noise_val = 1e-13
        expected_snr = 0.0
        self.assertEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise_val), expected_snr)
        
    def test_snr_zero_signal_only(self):
        # Tests the case where the estimated signal power is zero because RMS_signal_plus_noise equals RMS_noise.
        # Example: RMS_signal_plus_noise = 1.0, RMS_noise = 1.0.
        # This implies estimated signal RMS is 0, which is treated as 1e-12.
        # Expected SNR = 20 * log10(1e-12 / 1.0) = -240 dB.
        # This verifies the handling when the subtraction (Power_total - Power_noise) results in zero.
        rms_signal_plus_noise = 1.0
        rms_noise = 1.0
        expected_snr = 20 * math.log10(1e-12 / rms_noise)
        self.assertAlmostEqual(calculate_snr_db_from_rms(rms_signal_plus_noise, rms_noise), expected_snr, places=5)


class TestMeasureSNR(unittest.TestCase):
    @patch('snr_analyzer.snr_analyzer.sd.rec')
    @patch('snr_analyzer.snr_analyzer.sd.playrec')
    @patch('snr_analyzer.snr_analyzer.sd.query_devices')
    def test_measure_snr_logic(self, mock_query_devices, mock_playrec, mock_rec):
        # Setup mock device info
        mock_device_info = {
            'name': 'mock_device',
            'max_output_channels': 2,
            'max_input_channels': 2,
            'default_samplerate': 48000
        }
        mock_query_devices.return_value = mock_device_info

        # --- Test Data ---
        samplerate = 48000
        signal_amp = 0.7
        noise_amp = 0.07 
        duration = 1.0
        
        # Create a known signal and noise
        t = np.linspace(0, duration, int(samplerate * duration), False)
        signal_wave = signal_amp * np.sin(2 * np.pi * 1000 * t)
        noise_wave = np.random.normal(0, noise_amp, len(t))

        # Mock the return values of sounddevice functions
        mock_playrec.return_value = (signal_wave + noise_wave).astype(np.float32)
        mock_rec.return_value = noise_wave.astype(np.float32)

        # --- Expected Values ---
        rms_signal_only_theoretical = signal_amp / np.sqrt(2)
        rms_noise_theoretical = np.sqrt(np.mean(noise_wave**2))
        expected_snr_db = 20 * np.log10(rms_signal_only_theoretical / rms_noise_theoretical)

        # --- Run the function ---
        snr_db, rms_signal, rms_noise = measure_snr(
            device_id=0,
            output_channel_idx=1,
            input_channel_idx=1,
            samplerate=samplerate,
            signal_freq=1000,
            signal_amp=signal_amp,
            signal_duration=duration,
            noise_duration=duration
        )

        # --- Assertions ---
        # Check if the calculated SNR is close to the theoretical value.
        # We use a larger delta because the random noise adds variability.
        self.assertAlmostEqual(snr_db, expected_snr_db, delta=1.0)
        
        # Check if the calculated RMS values are reasonable
        self.assertAlmostEqual(rms_noise, rms_noise_theoretical, delta=rms_noise_theoretical*0.1)
        
        # The calculated signal RMS is derived from (S+N) - N, so it will also have variance.
        # Let's check it against the theoretical value.
        self.assertAlmostEqual(rms_signal, rms_signal_only_theoretical, delta=rms_signal_only_theoretical*0.1)


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
