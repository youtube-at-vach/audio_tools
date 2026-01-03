import sys
import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication


# Mock AudioEngine
class MockCalibration:
    def __init__(self):
        self.lockin_gain_offset = 0.0
        self.output_gain = 1.0
        self.input_sensitivity = 1.0 # 1.0 Vpeak = 0dBFS

    def get_frequency_correction(self, freq):
        return 0.0, 0.0

class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.calibration = MockCalibration()

    def register_callback(self, cb):
        return 1

    def unregister_callback(self, id):
        pass

# Import module
from src.gui.widgets.lock_in_amplifier import LockInAmplifier, LockInAmplifierWidget


class TestLockInStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.engine = MockAudioEngine()
        self.module = LockInAmplifier(self.engine)
        self.widget = LockInAmplifierWidget(self.module)

        # Setup module
        self.module.gen_frequency = 1000
        self.module.buffer_size = 1000 # Short buffer for test
        self.module.input_data = np.zeros((1000, 2))
        self.module.ref_channel = 1
        self.module.signal_channel = 0
        self.module.external_mode = False # Internal ref

    def test_stats_calculation(self):
        # Simulate clean signal first
        self.module.averaging_count = 10
        self.module.history.clear()

        # Add identical results to history
        complex_val = 0.5 + 0.0j # mag 0.5, phase 0
        for _ in range(10):
            self.module.history.append(complex_val)

        # Process (just to trigger stats logic? No, process_data DOES calculation)
        # But process_data calculates one result and appends it.
        # We need to manually trigger the stat calculation part or run process_data with mock input.

        # Running process_data with mock input is better.
        # Generate input data: perfect sine
        t = np.arange(1000) / 48000
        sig = 0.5 * np.cos(2*np.pi*1000*t)
        ref = 0.5 * np.cos(2*np.pi*1000*t)
        self.module.input_data[:, 0] = sig
        self.module.input_data[:, 1] = ref

        # We need to run process_data 10 times
        for _ in range(10):
            self.module.process_data()

        # Check stats
        # Should be near 0 because perfect sine
        self.assertAlmostEqual(self.module.current_magnitude_std, 0.0, places=6)

        # Now add noise to input
        # Note: process_data re-reads input_data every time.
        # We should vary input_data between calls to simulate noise.

        stds = []
        for _ in range(10):
            noise = np.random.normal(0, 0.001, 1000) # Small noise
            self.module.input_data[:, 0] = sig + noise
            self.module.process_data()
            stds.append(self.module.current_magnitude_std)

        print(f"Magnitude Std: {self.module.current_magnitude_std}")
        self.assertGreater(self.module.current_magnitude_std, 0.0)

    def test_decimal_places(self):
        # Test get_decimal_places logic

        # Linear Std -> Places
        self.assertEqual(self.widget.get_decimal_places(0.01), 2)   # 0.01 -> 1e-2 -> 2
        self.assertEqual(self.widget.get_decimal_places(0.005), 3)  # 0.005 -> 10^-2.3 -> 3 (Show first uncertain digit)
        self.assertEqual(self.widget.get_decimal_places(0.001), 3)  # 1e-3 -> 3
        self.assertEqual(self.widget.get_decimal_places(1e-6), 6)
        self.assertEqual(self.widget.get_decimal_places(1e-9), 6) # Max 6

        # dB Std Logic
        # if input is linear std, convert to dB precision
        # mag=1.0, std=0.01. dB uncertainty approx 8.686 * 0.01 = 0.08 dB.
        # 0.08 -> 10^-1.xxx -> floor(-2). Places=2.
        self.assertEqual(self.widget.get_decimal_places(0.01, val_abs=1.0, is_db=True), 2)

        # mag=1.0, std=0.001. dB uncertainty approx 0.008 dB.
        # 0.008 -> 10^-2.xxx -> floor(-3). Places=3.
        self.assertEqual(self.widget.get_decimal_places(0.001, val_abs=1.0, is_db=True), 3)

        # mag=0.001 (-60dB), std=0.00001 (1% noise).
        # dB uncertainty = 8.686 * (0.01) = 0.08 dB. -> 2 places.
        self.assertEqual(self.widget.get_decimal_places(0.00001, val_abs=0.001, is_db=True), 2)

if __name__ == '__main__':
    unittest.main()
