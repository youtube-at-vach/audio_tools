import numpy as np

from src.core.dsp import FractionalDelayLine


def test_delayline_impulse_integer_delay():
    dl = FractionalDelayLine(initial_capacity=64)
    sr = 48000
    delay_samples = 10.0
    alpha = 0.25

    x = np.zeros(64, dtype=np.float32)
    x[0] = 1.0

    delayed = dl.process(x, delay_samples)
    y = x + alpha * delayed

    # Impulse at n=0 plus delayed impulse at n=delay.
    assert np.isclose(y[0], 1.0)
    assert np.isclose(y[int(delay_samples)], alpha)
    # Everything else should be ~0.
    mask = np.ones_like(y, dtype=bool)
    mask[0] = False
    mask[int(delay_samples)] = False
    assert np.allclose(y[mask], 0.0)


def test_delayline_fractional_delay_linear_interp_impulse():
    dl = FractionalDelayLine(initial_capacity=64)
    delay_samples = 2.5
    alpha = 1.0

    x = np.zeros(16, dtype=np.float32)
    x[0] = 1.0

    delayed = dl.process(x, delay_samples)
    y = x + alpha * delayed

    # With linear interpolation, the delayed impulse splits across 2 samples.
    # x[n-2.5] at n=2 uses x[-0.5] ~ 0, at n=3 uses x[0.5] = 0.5.
    # More directly, the non-zero delayed values should appear at indices 2 and 3:
    # at n=2: 0.5, at n=3: 0.5 (depending on convention). Let's assert sum.
    assert np.isclose(np.sum(delayed), 1.0, atol=1e-6)
    # And delayed energy is around the expected location.
    nz = np.where(np.abs(delayed) > 1e-6)[0]
    assert nz.min() >= 2 and nz.max() <= 3


def test_delayline_stateful_across_blocks():
    dl = FractionalDelayLine(initial_capacity=64)
    delay_samples = 5.0

    x1 = np.zeros(8, dtype=np.float32)
    x1[0] = 1.0
    d1 = dl.process(x1, delay_samples)

    x2 = np.zeros(8, dtype=np.float32)
    d2 = dl.process(x2, delay_samples)

    # The delayed impulse at overall index 5 is within the first block.
    assert np.isclose(d1[5], 1.0)
    # Second block should have no delayed impulse for this delay.
    assert np.allclose(d2, 0.0)


def test_two_channels_independent_delays():
    dl_l = FractionalDelayLine(initial_capacity=64)
    dl_r = FractionalDelayLine(initial_capacity=64)

    x = np.zeros(32, dtype=np.float32)
    x[0] = 1.0

    d_l = dl_l.process(x, 3.0)
    d_r = dl_r.process(x, 7.0)

    assert np.isclose(d_l[3], 1.0)
    assert np.isclose(d_r[7], 1.0)
