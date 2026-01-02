import numpy as np

from src.core.spatial_stereo_dynamics import (
    DiffuseSide3DState,
    NeuralPrecedence3DState,
    process_stereo_diffuse_side_3d_block,
    process_stereo_neural_precedence_3d_block,
)


def test_diffuse_side_3d_leaves_mono_unchanged_and_is_finite():
    sr = 48000.0
    n = 4096

    t = np.arange(n, dtype=np.float64) / sr
    mono = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    x = np.column_stack([mono, mono])

    y, _st = process_stereo_diffuse_side_3d_block(
        x,
        sample_rate=sr,
        alpha=0.8,
        beta=0.8,
        tau_seconds=0.010,
        state=DiffuseSide3DState(),
    )

    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y[:, 0] - mono))) < 1e-6
    assert float(np.max(np.abs(y[:, 1] - mono))) < 1e-6


def test_neural_precedence_3d_leaves_mono_unchanged_and_is_finite():
    sr = 48000.0
    n = 4096

    t = np.arange(n, dtype=np.float64) / sr
    mono = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    x = np.column_stack([mono, mono])

    y, _st = process_stereo_neural_precedence_3d_block(
        x,
        sample_rate=sr,
        alpha=0.8,
        beta=0.8,
        tau_seconds=0.010,
        state=NeuralPrecedence3DState(),
    )

    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y[:, 0] - mono))) < 1e-6
    assert float(np.max(np.abs(y[:, 1] - mono))) < 1e-6
