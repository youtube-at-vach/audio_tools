import unittest
import numpy as np
from unittest.mock import MagicMock
from src.gui.widgets.frequency_counter import FrequencyCounter

class TestFrequencyCounterLogic(unittest.TestCase):
    def setUp(self):
        self.mock_audio_engine = MagicMock()
        self.mock_audio_engine.sample_rate = 48000
        self.counter = FrequencyCounter(self.mock_audio_engine)
        self.counter.is_running = True # Simulate running

    def test_process_silence(self):
        # Create silence buffer
        self.counter.input_buffer = np.zeros(self.counter.buffer_size)
        
        # Process
        freq = self.counter.process()
        
        # Should return None because of gate
        self.assertIsNone(freq)
        self.assertLess(self.counter.current_amp_db, self.counter.gate_threshold_db)

    def test_process_sine_wave(self):
        # Create sine wave 1kHz
        sr = 48000
        t = np.arange(self.counter.buffer_size) / sr
        target_freq = 1000.0
        signal = 0.5 * np.sin(2 * np.pi * target_freq * t) # -6dBFS
        
        self.counter.input_buffer = signal
        
        # Process
        freq = self.counter.process()
        
        # Should detect frequency
        self.assertIsNotNone(freq)
        self.assertAlmostEqual(freq, target_freq, delta=1.0) # Allow small error
        self.assertGreater(self.counter.current_amp_db, self.counter.gate_threshold_db)

    def test_buffer_management(self):
        # Test that buffer rolls correctly (mocking the callback logic partially)
        # Manually manipulating buffer to simulate callback effect
        
        # Fill with 1s
        self.counter.input_buffer = np.ones(self.counter.buffer_size)
        
        # New data (0s)
        new_data = np.zeros(100)
        
        # Roll logic from callback
        self.counter.input_buffer = np.roll(self.counter.input_buffer, -len(new_data))
        self.counter.input_buffer[-len(new_data):] = new_data
        
        # Check end is 0
        self.assertTrue(np.all(self.counter.input_buffer[-100:] == 0))
        # Check start is 1
        self.assertTrue(np.all(self.counter.input_buffer[:-100] == 1))

if __name__ == '__main__':
    unittest.main()
