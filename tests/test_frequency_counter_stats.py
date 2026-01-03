import unittest
from collections import deque
from unittest.mock import MagicMock

import numpy as np

from src.gui.widgets.frequency_counter import FrequencyCounter


class TestFrequencyCounterStats(unittest.TestCase):
    def setUp(self):
        self.mock_audio_engine = MagicMock()
        self.counter = FrequencyCounter(self.mock_audio_engine)
        self.counter.freq_history = deque(maxlen=100)

    def test_stats_constant(self):
        # Constant frequency
        self.counter.freq_history.extend([1000.0] * 10)
        self.counter.calculate_stats()

        self.assertEqual(self.counter.std_dev, 0.0)
        self.assertEqual(self.counter.allan_deviation, 0.0)

    def test_stats_linear_drift(self):
        # Linear drift: 0, 1, 2, 3, 4
        data = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.counter.freq_history.extend(data)
        self.counter.calculate_stats()

        # Std Dev
        # Mean = 2.0
        # Variance (ddof=1) = ((4+1+0+1+4) / 4) = 2.5
        expected_std = np.sqrt(2.5)
        self.assertAlmostEqual(self.counter.std_dev, expected_std, places=5)

        # Allan Dev
        # Diffs = [1, 1, 1, 1]
        # Mean(Diffs^2) = 1
        # Sigma = sqrt(0.5 * 1) = 0.70710678
        expected_allan = np.sqrt(0.5)
        self.assertAlmostEqual(self.counter.allan_deviation, expected_allan, places=5)

    def test_stats_alternating(self):
        # Alternating: 1000, 1002, 1000, 1002
        data = [1000.0, 1002.0, 1000.0, 1002.0]
        self.counter.freq_history.extend(data)
        self.counter.calculate_stats()

        # Std Dev
        # Mean = 1001
        # Variance = ((1+1+1+1)/3) = 4/3 = 1.333...
        expected_std = np.sqrt(4/3)
        self.assertAlmostEqual(self.counter.std_dev, expected_std, places=5)

        # Allan Dev
        # Diffs = [2, -2, 2]
        # Diffs^2 = [4, 4, 4]
        # Mean(Diffs^2) = 4
        # Sigma = sqrt(0.5 * 4) = sqrt(2) = 1.4142...
        expected_allan = np.sqrt(2)
        self.assertAlmostEqual(self.counter.allan_deviation, expected_allan, places=5)

if __name__ == '__main__':
    unittest.main()
