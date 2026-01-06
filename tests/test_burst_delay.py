import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.signal_generator import SignalGenerator


def test_burst_delay_ms_integer_samples_aligns_channels():
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.calibration.output_gain = 1.0

    sg = SignalGenerator(mock_engine)

    # Use a long "burst" with no off time so it behaves like a continuous sine
    # within the tested window (avoids envelope edge effects).
    for p in (sg.params_L, sg.params_R):
        p.waveform = 'burst'
        p.frequency = 2500.0
        p.amplitude = 1.0
        p.burst_on_cycles = 1000
        p.burst_off_cycles = 0

    # Apply +1.0ms delay to Right channel (48 samples @ 48kHz)
    sg.params_L.delay_ms = 0.0
    sg.params_R.delay_ms = 1.0

    sg.start_generation()

    args, _ = mock_engine.register_callback.call_args
    callback = args[0]

    frames = 480  # 10ms
    outdata = np.zeros((frames, 2), dtype=float)

    callback(None, outdata, frames, None, None)

    l = outdata[:, 0]
    r = outdata[:, 1]

    shift = 48

    # Right is delayed: r[n] ~= l[n-shift] (for n>=shift)
    # So r[shift:] should match l[:-shift].
    assert np.allclose(r[shift:], l[:-shift], atol=1e-4)
