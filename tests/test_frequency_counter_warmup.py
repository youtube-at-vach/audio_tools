import unittest
from unittest.mock import MagicMock

from src.gui.widgets.frequency_counter import FrequencyCounter


class TestFrequencyCounterWarmup(unittest.TestCase):
    def setUp(self):
        self.mock_audio_engine = MagicMock()
        self.mock_audio_engine.sample_rate = 48000

    def test_warmup_samples_are_not_added_to_history(self):
        counter = FrequencyCounter(self.mock_audio_engine)
        counter.warmup_discard_points = 2
        counter._warmup_remaining = 2

        # First two valid readings should be discarded from history
        recorded = counter.record_frequency_measurement(1000.0, now_t=100.0)
        self.assertFalse(recorded)
        self.assertEqual(len(counter.freq_history), 0)

        recorded = counter.record_frequency_measurement(1000.0, now_t=100.1)
        self.assertFalse(recorded)
        self.assertEqual(len(counter.freq_history), 0)

        # Third one should be accepted
        recorded = counter.record_frequency_measurement(1000.0, now_t=100.2)
        self.assertTrue(recorded)
        self.assertEqual(len(counter.freq_history), 1)
        self.assertAlmostEqual(counter.freq_history[-1], 1000.0, places=6)


if __name__ == '__main__':
    unittest.main()
