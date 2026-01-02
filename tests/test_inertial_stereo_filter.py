import numpy as np

from src.core.stereo_angle_dynamics import (
    InertialAttractorState,
    process_stereo_inertial_attractor_block,
    PhaseSpaceInertiaState,
    process_stereo_phase_space_inertia_block,
    prefilter_low_correlation_stereo_block,
    wrap_angle_rad,
)


def test_wrap_angle_rad_range():
    vals = np.array([
        -10 * np.pi,
        -3.5 * np.pi,
        -np.pi,
        -0.1,
        0.0,
        0.1,
        np.pi - 1e-9,
        np.pi,
        3.5 * np.pi,
        10 * np.pi,
    ])
    wrapped = wrap_angle_rad(vals)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_inertial_filter_smooths_step_pan():
    sr = 48000.0
    n = 4000

    theta_left = -np.pi / 3
    theta_right = np.pi / 3

    theta = np.full(n, theta_left, dtype=np.float64)
    theta[n // 2 :] = theta_right

    l = np.cos(theta).astype(np.float32)
    r = np.sin(theta).astype(np.float32)
    x = np.column_stack([l, r])

    # Small alpha => noticeable inertia. Disable gravity to isolate behavior.
    out, _state = process_stereo_inertial_attractor_block(
        x,
        sample_rate=sr,
        alpha=0.01,
        beta=0.0,
        tau_seconds=0.0,
        state=InertialAttractorState(),
    )

    theta_out = np.arctan2(out[:, 1], out[:, 0]).astype(np.float64)

    # Starts near the initial angle.
    assert abs(float(wrap_angle_rad(theta_out[0] - theta_left))) < 1e-3

    # Immediately after the step, output should not jump to target.
    idx = n // 2
    step_err0 = abs(float(wrap_angle_rad(theta_out[idx] - theta_right)))
    assert step_err0 > 1.0

    # Error should decrease over time (approach the new angle).
    tail = theta_out[idx : idx + 500]
    err = np.abs(wrap_angle_rad(theta_right - tail))
    assert float(err[-1]) < float(err[0])


def test_phase_space_inertia_smooths_step_pan():
    sr = 48000.0
    n = 4000

    theta_left = -np.pi / 3
    theta_right = np.pi / 3

    theta = np.full(n, theta_left, dtype=np.float64)
    theta[n // 2 :] = theta_right

    l = np.cos(theta).astype(np.float32)
    r = np.sin(theta).astype(np.float32)
    x = np.column_stack([l, r])

    out, _state = process_stereo_phase_space_inertia_block(
        x,
        sample_rate=sr,
        alpha=0.01,
        beta=0.0,
        tau_seconds=0.0,
        state=PhaseSpaceInertiaState(),
    )

    theta_out = np.arctan2(out[:, 1], out[:, 0]).astype(np.float64)

    assert abs(float(wrap_angle_rad(theta_out[0] - theta_left))) < 1e-3

    idx = n // 2
    step_err0 = abs(float(wrap_angle_rad(theta_out[idx] - theta_right)))
    assert step_err0 > 1.0

    tail = theta_out[idx : idx + 500]
    err = np.abs(wrap_angle_rad(theta_right - tail))
    assert float(err[-1]) < float(err[0])


def test_phase_space_inertia_smooths_magnitude_step():
    sr = 48000.0
    n = 4000
    idx = n // 2

    # Keep angle at 0 rad (hard-left axis), only change magnitude.
    rmag = np.full(n, 0.05, dtype=np.float32)
    rmag[idx:] = 1.0

    l = rmag
    r = np.zeros_like(rmag)
    x = np.column_stack([l, r])

    out, _state = process_stereo_phase_space_inertia_block(
        x,
        sample_rate=sr,
        alpha=0.01,
        beta=0.0,
        tau_seconds=0.0,
        state=PhaseSpaceInertiaState(),
    )

    out_mag = np.hypot(out[:, 0], out[:, 1]).astype(np.float64)

    # Immediately after the magnitude step, it shouldn't jump to 1.0.
    assert float(out_mag[idx]) < 0.7

    # It should approach the target over time.
    assert float(out_mag[idx + 500]) > float(out_mag[idx])


def test_corr_prefilter_collapses_uncorrelated_to_mono():
    rng = np.random.default_rng(0)
    n = 8192

    # Independent signals => correlation ~ 0.
    l = rng.standard_normal(n).astype(np.float32)
    r = rng.standard_normal(n).astype(np.float32)
    x = np.column_stack([l, r])

    y = prefilter_low_correlation_stereo_block(x, threshold=0.5, window_frames=1024)

    # After prefiltering, windows should become mono (L == R), so overall corr is high.
    yl = y[:, 0].astype(np.float64)
    yr = y[:, 1].astype(np.float64)
    corr = float(np.corrcoef(yl, yr)[0, 1])
    assert corr > 0.90
