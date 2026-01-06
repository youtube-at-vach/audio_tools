import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.signal_generator import SignalGenerator, SignalParameters


class MockEngine:
    def __init__(self, sr=48000):
        self.sample_rate = sr
        self.register_callback = lambda cb: 1
        self.unregister_callback = lambda _id: None
        class _Cal:
            output_gain = 1.0
        self.calibration = _Cal()


def test_windowed_burst_starts_and_ends_near_zero():
    sg = SignalGenerator(MockEngine())
    p = SignalParameters()
    p.waveform = 'burst'
    p.frequency = 2500.0
    p.burst_on_cycles = 100
    p.burst_off_cycles = 100
    p.burst_windowed = True

    buf = sg._generate_burst(p, 48000)

    assert buf.size > 0

    # Start at (or extremely close to) zero.
    assert abs(float(buf[0])) < 1e-9

    # End of ON segment should be tapered to ~0 (avoid click at ON->OFF).
    on_samples = int((p.burst_on_cycles / p.frequency) * 48000)
    on_samples = min(max(on_samples, 0), buf.size)
    if on_samples >= 2:
        assert abs(float(buf[on_samples - 1])) < 1e-3

    # Tail of buffer (OFF portion) should be exactly zero.
    assert np.allclose(buf[-10:], 0.0, atol=0.0)
