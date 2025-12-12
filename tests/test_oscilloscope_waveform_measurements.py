import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.gui.widgets.oscilloscope import Oscilloscope


def test_estimate_frequency_hz_sine():
    sr = 48000
    f = 1000.0
    t = np.arange(sr // 10) / sr  # 0.1s
    y = 0.8 * np.sin(2 * np.pi * f * t)
    est = Oscilloscope.estimate_frequency_hz(t, y)
    assert est is not None
    assert abs(est - f) < 5.0


def test_estimate_rise_fall_times_step_like_square():
    sr = 48000
    t = np.arange(sr // 200) / sr  # 5ms

    # Create a single rising edge then falling edge within the window.
    y = np.full_like(t, -1.0)
    y[t >= 0.001] = 1.0
    y[t >= 0.003] = -1.0

    rise_s, fall_s, low, high = Oscilloscope.estimate_rise_fall_times_s(t, y)

    # Ideal step has ~0 rise/fall; estimator may return None (too sharp) or a tiny positive.
    assert low is not None and high is not None
    assert high > low
    if rise_s is not None:
        assert 0 <= rise_s < 1e-3
    if fall_s is not None:
        assert 0 <= fall_s < 1e-3


def test_estimate_rise_fall_times_ramp_has_expected_10_90_time():
    # Create a clean low->high ramp of 25us so 10-90% should be 20us.
    dt = 1e-6
    t = np.arange(0.0, 200e-6, dt)
    y = np.full_like(t, -1.0)

    t0 = 50e-6
    ramp = 25e-6
    t1 = t0 + ramp
    # Rising ramp
    idx = (t >= t0) & (t <= t1)
    y[idx] = -1.0 + 2.0 * ((t[idx] - t0) / ramp)
    y[t > t1] = 1.0

    # Falling ramp later
    t2 = 120e-6
    t3 = t2 + ramp
    idx2 = (t >= t2) & (t <= t3)
    y[idx2] = 1.0 - 2.0 * ((t[idx2] - t2) / ramp)
    y[t > t3] = -1.0

    rise_s, fall_s, low, high = Oscilloscope.estimate_rise_fall_times_s(t, y)
    assert low is not None and high is not None
    assert rise_s is not None
    assert fall_s is not None

    assert abs(rise_s - 20e-6) < 2e-6
    assert abs(fall_s - 20e-6) < 2e-6
