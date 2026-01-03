import unittest
from collections import deque
from unittest.mock import MagicMock

import numpy as np

from src.gui.widgets.frequency_counter import FrequencyCounter


class TestFrequencyCounterAllanPlot(unittest.TestCase):
    def setUp(self):
        self.mock_audio_engine = MagicMock()
        self.counter = FrequencyCounter(self.mock_audio_engine)
        self.counter.update_interval_ms = 100
        self.counter.freq_history = deque(maxlen=2000)

    def test_allan_plot_white_noise(self):
        # White Noise FM: Slope should be -1/2 in log-log (sigma ~ tau^-0.5)
        # Generate random noise
        np.random.seed(42)
        n = 1000
        noise = np.random.normal(1000, 1.0, n)
        self.counter.freq_history.extend(noise)

        taus, devs = self.counter.calculate_allan_plot_data()

        self.assertGreater(len(taus), 5)
        self.assertGreater(len(devs), 5)

        # Check slope roughly
        # log(sigma) = -0.5 * log(tau) + C
        log_taus = np.log10(taus)
        log_devs = np.log10(devs)

        slope, intercept = np.polyfit(log_taus, log_devs, 1)

        # For white noise FM, slope is -0.5
        # Allow some margin due to limited sample size
        self.assertAlmostEqual(slope, -0.5, delta=0.2)

    def test_allan_plot_random_walk(self):
        # Random Walk FM: Slope should be +1/2 (sigma ~ tau^0.5)
        np.random.seed(42)
        n = 1000
        steps = np.random.normal(0, 0.1, n)
        walk = 1000 + np.cumsum(steps)
        self.counter.freq_history.extend(walk)

        taus, devs = self.counter.calculate_allan_plot_data()

        log_taus = np.log10(taus)
        log_devs = np.log10(devs)

        slope, intercept = np.polyfit(log_taus, log_devs, 1)

        # For random walk FM, slope is +0.5
        self.assertAlmostEqual(slope, 0.5, delta=0.2)

if __name__ == '__main__':
    unittest.main()
