import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.bnim_meter import BNIMMeter


def _first_idx_above(x: np.ndarray, thresh: float) -> int:
    idx = np.flatnonzero(np.abs(x) > thresh)
    return int(idx[0]) if idx.size else -1


def test_bnim_click_play_build_and_callback_delay():
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.register_callback.return_value = 1

    m = BNIMMeter(mock_engine)
    m.start_analysis()

    # Trigger a 1kHz burst with +0.8ms ITD (left delayed; should localize to the right)
    m.trigger_click_test_playback(freq_hz=1000.0, itd_ms=0.8, on_cycles=10, off_cycles=900, ild_atten_db=0.0)

    # Pull the registered callback
    args, _ = mock_engine.register_callback.call_args
    cb = args[0]

    frames = 480  # 10 cycles @ 1kHz, 48kHz
    indata = np.zeros((frames, 2), dtype=np.float32)
    outdata = np.zeros((frames, 2), dtype=np.float32)

    cb(indata, outdata, frames, None, None)

    # Should output something (not all zeros)
    assert np.any(np.abs(outdata) > 1e-6)

    # Left should start later than right by ~0.8ms (~38.4 samples)
    l = outdata[:, 0]
    r = outdata[:, 1]

    i_l = _first_idx_above(l, 1e-3)
    i_r = _first_idx_above(r, 1e-3)

    assert i_l >= 0 and i_r >= 0

    expected = 0.8e-3 * 48000
    # Allow a couple samples slack due to thresholding and Hann onset
    assert (i_l - i_r) >= int(expected) - 3


def test_bnim_click_play_ild_ratio():
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.register_callback.return_value = 1

    m = BNIMMeter(mock_engine)
    m.start_analysis()

    # Playback ILD attenuation applies to the ITD-delayed ear.
    # For +ITD, left ear is delayed, so left should be attenuated.
    buf = m.build_click_test_burst(freq_hz=1000.0, itd_ms=0.8, on_cycles=10, off_cycles=0, ild_atten_db=20.0)

    # Choose a region where both channels are active (after the ITD delay).
    delay_samples = int(np.round(0.8e-3 * 48000))
    start = delay_samples + 20
    end = start + 200
    l = buf[start:end, 0]
    r = buf[start:end, 1]

    l_rms = float(np.sqrt(np.mean(l * l)))
    r_rms = float(np.sqrt(np.mean(r * r)))

    assert l_rms > 0
    assert r_rms > 0

    ratio = l_rms / r_rms
    # ~ -20 dB => about 0.1
    assert ratio > 0.06
    assert ratio < 0.16


def test_bnim_click_play_loop_wraps_within_block():
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.register_callback.return_value = 1

    m = BNIMMeter(mock_engine)
    m.start_analysis()
    m.play_loop = True

    # Small buffer: 10 cycles @ 1kHz => ~480 samples (+ small pad)
    m.trigger_click_test_playback(freq_hz=1000.0, itd_ms=0.0, on_cycles=10, off_cycles=0, ild_atten_db=0.0)

    args, _ = mock_engine.register_callback.call_args
    cb = args[0]

    frames = 2000  # larger than buffer length; should wrap
    indata = np.zeros((frames, 2), dtype=np.float32)
    outdata = np.zeros((frames, 2), dtype=np.float32)

    cb(indata, outdata, frames, None, None)

    # In loop mode, we should keep outputting after wrap.
    # The end of the buffer should not be all zeros.
    tail = outdata[-200:, 0]
    assert np.any(np.abs(tail) > 1e-6)


def test_bnim_click_play_loop_live_update_rebuilds_buffer():
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.register_callback.return_value = 1

    m = BNIMMeter(mock_engine)
    m.start_analysis()
    m.play_enable_click = True
    m.play_loop = True

    m.play_on_cycles = 10
    m.play_off_cycles = 0
    m.play_ild_atten_db = 0.0

    m.trigger_click_test_playback(freq_hz=1000.0, itd_ms=0.8, on_cycles=m.play_on_cycles, off_cycles=m.play_off_cycles, ild_atten_db=m.play_ild_atten_db)

    with m._play_lock:
        n0 = len(m._play_buffer)

    # Change off_cycles and refresh; buffer should get longer
    m.play_off_cycles = 900
    m.refresh_click_test_playback_if_looping()

    with m._play_lock:
        n1 = len(m._play_buffer)

    assert n1 > n0
