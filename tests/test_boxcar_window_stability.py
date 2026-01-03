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
    # Provide deterministic input; values track absolute sample index.
    left = np.arange(start_value, start_value + frames, dtype=float)
    right = np.zeros(frames, dtype=float)
    indata = np.column_stack((left, right))
    outdata = np.zeros_like(indata)
    boxcar._callback(indata, outdata, frames, 0, None)
    boxcar.process()


def test_internal_reset_keeps_window_aligned_to_absolute_period_boundary():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)

    boxcar.mode = "Internal Pulse"
    boxcar.input_channel = "Left"
    boxcar.period_samples = 10

    boxcar.start_analysis()

    # Feed 7 samples: indices 0..6
    _feed(boxcar, frames=7, start_value=0)

    # Reset mid-stream. The next accumulation should start at the next period boundary
    # (absolute index 10), not at the next processed chunk boundary.
    boxcar.reset_average()

    # Feed indices 7..13 then 14..19. The reset logic should skip 7..9,
    # and the first full accumulated period should be exactly 10..19.
    _feed(boxcar, frames=7, start_value=7)
    _feed(boxcar, frames=6, start_value=14)

    assert boxcar.count == 1

    expected = np.arange(10, 20, dtype=float)
    got = boxcar.accumulator[:, 0] / boxcar.count

    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
