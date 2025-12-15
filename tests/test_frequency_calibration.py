import unittest
from unittest.mock import MagicMock
import numpy as np
import unittest.mock
from src.gui.widgets.frequency_counter import FrequencyCounter

class TestFrequencyCalibration(unittest.TestCase):
    def setUp(self):
        self.mock_audio_engine = MagicMock()
        # Mock calibration object
        self.mock_audio_engine.calibration = MagicMock()
        self.mock_audio_engine.calibration.frequency_calibration = 1.0
        
        self.counter = FrequencyCounter(self.mock_audio_engine)
        self.counter.is_running = True # Enable processing

    def test_calibration_applied(self):
        # Setup
        self.mock_audio_engine.calibration.frequency_calibration = 1.0
        
        # Mock AudioCalc.optimize_frequency to return a known value
        with unittest.mock.patch('src.core.analysis.AudioCalc.optimize_frequency') as mock_opt:
            mock_opt.return_value = 1000.0
            
            # Create a sine wave in input_buffer to pass gate and coarse check
            sr = 48000
            t = np.arange(len(self.counter.input_buffer)) / sr
            self.counter.input_buffer = np.sin(2 * np.pi * 1000 * t)
            self.counter.audio_engine.sample_rate = sr
            
            # Case 1: Factor 1.0
            self.mock_audio_engine.calibration.frequency_calibration = 1.0
            mock_opt.return_value = 1000.0
            
            freq = self.counter.process()
            self.assertAlmostEqual(freq, 1000.0)
            
            # Case 2: Factor 1.000001 (1ppm offset)
            self.mock_audio_engine.calibration.frequency_calibration = 1.000001
            
            freq = self.counter.process()
            self.assertAlmostEqual(freq, 1000.001)

if __name__ == '__main__':
    unittest.main()
