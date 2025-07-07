
import unittest
from unittest.mock import patch
import numpy as np
from rt60_analyzer import calculate_rt60, record_impulse_response

class TestRT60Analyzer(unittest.TestCase):

    def test_calculate_rt60_with_generated_decay(self):
        samplerate = 48000
        duration = 2
        # Create a signal with a known exponential decay
        # RT60 = 0.5 seconds means the level drops by 60 dB in 0.5s
        # The decay factor alpha can be derived from: exp(-alpha * 0.5) = 10^(-60/20)
        rt60_target = 0.5
        alpha = (60 / 20) * np.log(10) / rt60_target
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        # Using a decaying sine wave with a noise floor for a more realistic signal
        carrier_freq = 1000.0
        decay_component = np.exp(-alpha * t) * np.sin(2 * np.pi * carrier_freq * t)
        noise_component = (np.random.randn(len(t))) * (10**(-70/20)) # -70dB noise floor
        decay_signal = decay_component + noise_component

        rt60, _, _ = calculate_rt60(decay_signal, samplerate)

        self.assertIsNotNone(rt60)
        self.assertAlmostEqual(rt60, rt60_target, delta=0.05)

    @patch('sounddevice.rec')
    @patch('sounddevice.wait')
    def test_record_impulse_response(self, mock_wait, mock_rec):
        # Configure the mock to return a predictable numpy array
        samplerate = 48000
        duration = 3
        expected_output = np.random.randn(duration * samplerate, 1).astype(np.float32)
        mock_rec.return_value = expected_output

        # Call the function
        recording = record_impulse_response(duration=duration, samplerate=samplerate)

        # Assertions
        mock_rec.assert_called_once_with(int(duration * samplerate), samplerate=samplerate, channels=1, blocking=True, device=None)
        mock_wait.assert_called_once()
        np.testing.assert_array_equal(recording, expected_output.flatten())

if __name__ == '__main__':
    unittest.main()
