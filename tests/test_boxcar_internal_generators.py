import sys
import os
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


def _mls16_expected(n: int, seed: int = 0xACE1) -> np.ndarray:
    reg = seed & 0xFFFF
    if reg == 0:
        reg = 1
    out = np.empty((n,), dtype=np.float64)
    for i in range(n):
        lsb = reg & 1
        out[i] = 1.0 if lsb else -1.0
        reg >>= 1
        if lsb:
            reg ^= 0xB400
    return out


def test_internal_impulse_is_one_sample_per_period():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)
    boxcar.mode = "Internal Impulse"
    boxcar.input_channel = "Left"
    boxcar.period_samples = 8

    boxcar.start_analysis()

    frames = 8
    indata = np.zeros((frames, 2), dtype=float)

    out = np.zeros_like(indata)
    boxcar._callback(indata, out, frames, 0, None)

    expected = np.zeros(frames, dtype=float)
    expected[0] = 0.5
    np.testing.assert_allclose(out[:, 0], expected, rtol=0, atol=1e-12)

    # Next period should repeat impulse at first sample again
    out2 = np.zeros_like(indata)
    boxcar._callback(indata, out2, frames, 0, None)
    np.testing.assert_allclose(out2[:, 0], expected, rtol=0, atol=1e-12)


def test_internal_prbs_mls_repeats_per_period_deterministically():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)
    boxcar.mode = "Internal PRBS/MLS"
    boxcar.input_channel = "Left"
    boxcar.period_samples = 8

    boxcar.start_analysis()

    frames = 8
    indata = np.zeros((frames, 2), dtype=float)

    out = np.zeros_like(indata)
    boxcar._callback(indata, out, frames, 0, None)

    mls = _mls16_expected(8)
    expected = 0.5 * mls
    np.testing.assert_allclose(out[:, 0], expected, rtol=0, atol=1e-12)

    # Next period should repeat the same cached sequence
    out2 = np.zeros_like(indata)
    boxcar._callback(indata, out2, frames, 0, None)
    np.testing.assert_allclose(out2[:, 0], expected, rtol=0, atol=1e-12)
