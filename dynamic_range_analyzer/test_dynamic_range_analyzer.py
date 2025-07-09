import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add the parent directory to the path to allow imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic_range_analyzer import dynamic_range_analyzer

class TestDynamicRangeMeasurement(unittest.TestCase):

    @patch('dynamic_range_analyzer.dynamic_range_analyzer.sd.query_devices')
    @patch('dynamic_range_analyzer.dynamic_range_analyzer.sd.playrec')
    def test_unweighted_dr_calculation(self, mock_playrec, mock_query_devices):
        """Test the unweighted DR calculation logic with ideal, mocked data."""
        # --- Setup Mock Data ---
        samplerate = 48000
        duration = 2
        
        # Mock device info
        mock_query_devices.return_value = {
            'name': 'mock_device',
            'max_input_channels': 2,
            'max_output_channels': 2
        }

        # Create an ideal recorded signal: -60 dBFS tone + known noise
        num_samples = int(samplerate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        # Test tone at -60 dBFS
        test_tone = (10**(-60.0 / 20.0)) * np.sin(2 * np.pi * dynamic_range_analyzer.TEST_FREQUENCY * t)
        
        # White noise scaled to a known level: -100 dBFS RMS
        noise_unscaled = np.random.normal(0, 1, num_samples)
        rms_noise_unscaled = np.sqrt(np.mean(noise_unscaled**2))
        target_rms_noise = 10**(-100.0 / 20.0)
        noise = noise_unscaled * (target_rms_noise / rms_noise_unscaled)

        # The mocked recorded signal is the sum of the tone and the noise
        mock_recorded_signal = (test_tone + noise).astype(np.float32).reshape(-1, 1)
        mock_playrec.return_value = mock_recorded_signal

        # --- Run Measurement ---
        dr_db, rms_noise_measured = dynamic_range_analyzer.measure_unweighted_dynamic_range(
            device_id=0,
            output_channel=1,
            input_channel=1,
            samplerate=samplerate,
            duration=duration
        )

        # --- Assertions ---
        # The notch filter will remove the tone. The remaining RMS should be very
        # close to the noise floor we injected.
        # The expected DR is -20*log10(RMS of noise) = -20*log10(10**-5) = 100 dB.
        self.assertIsNotNone(dr_db)
        self.assertAlmostEqual(100.0, dr_db, places=0, msg="DR should be very close to 100 dB for a -100 dBFS noise floor")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)