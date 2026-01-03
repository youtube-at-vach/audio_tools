import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui.widgets.boxcar_averager import BoxcarAverager


class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000

    def register_callback(self, cb):
        return 1

    def unregister_callback(self, id):
        pass


def _feed(boxcar: BoxcarAverager, frames: int, start_value: int):
    left = np.arange(start_value, start_value + frames, dtype=float)
    right = np.zeros(frames, dtype=float)
    indata = np.column_stack((left, right))
    outdata = np.zeros_like(indata)
    boxcar._callback(indata, outdata, frames, 0, None)
    boxcar.process()


def test_internal_gate_accumulates_only_within_window():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)

    boxcar.mode = "Internal Pulse"
    boxcar.input_channel = "Left"
    boxcar.period_samples = 10

    boxcar.gate_enabled = True
    boxcar.gate_start_samples = 0
    boxcar.gate_length_samples = 3  # bins 0,1,2

    boxcar.start_analysis()

    # Two periods worth of samples: 0..19
    _feed(boxcar, frames=20, start_value=0)

    assert boxcar.count == 2

    # Only bins 0..2 should have data.
    acc = boxcar.accumulator[:, 0]

    # For bin k: values were k (period0) and 10+k (period1)
    expected_acc = np.zeros(10, dtype=float)
    expected_acc[0:3] = np.array([0 + 10, 1 + 11, 2 + 12], dtype=float)

    np.testing.assert_allclose(acc, expected_acc, rtol=0, atol=1e-12)


def test_internal_gate_wraps_across_period_end():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)

    boxcar.mode = "Internal Pulse"
    boxcar.input_channel = "Left"
    boxcar.period_samples = 10

    boxcar.gate_enabled = True
    boxcar.gate_start_samples = 8
    boxcar.gate_length_samples = 4  # bins 8,9,0,1

    boxcar.start_analysis()

    _feed(boxcar, frames=20, start_value=0)

    assert boxcar.count == 2

    acc = boxcar.accumulator[:, 0]

    expected_acc = np.zeros(10, dtype=float)
    # bin8: 8 and 18
    expected_acc[8] = 8 + 18
    # bin9: 9 and 19
    expected_acc[9] = 9 + 19
    # bin0: 0 and 10
    expected_acc[0] = 0 + 10
    # bin1: 1 and 11
    expected_acc[1] = 1 + 11

    np.testing.assert_allclose(acc, expected_acc, rtol=0, atol=1e-12)
