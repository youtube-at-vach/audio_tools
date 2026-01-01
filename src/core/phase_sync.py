from __future__ import annotations

import numpy as np


def estimate_ref_phase_rad(
    ref: np.ndarray,
    *,
    sample_rate: float,
    ref_frequency_hz: float,
) -> float:
    """Estimate reference phase (radians) of a near-sinusoid at ref_frequency_hz.

    Uses a windowed coherent projection (single-bin DFT). The returned phase is
    relative to a cosine reference (i.e., cos(2πft) corresponds to phase 0).

    Returns 0.0 if ref_frequency_hz is invalid or ref is too short.
    """
    if ref is None:
        return 0.0
    ref = np.asarray(ref, dtype=np.float64)
    n = int(ref.size)
    if n < 8:
        return 0.0

    f = float(ref_frequency_hz)
    sr = float(sample_rate)
    if not np.isfinite(f) or f <= 0.0 or not np.isfinite(sr) or sr <= 0.0:
        return 0.0

    t = np.arange(n, dtype=np.float64) / sr
    w = np.hanning(n).astype(np.float64, copy=False)
    w_mean = float(np.mean(w))
    if not np.isfinite(w_mean) or w_mean <= 0.0:
        w_mean = 1.0

    osc = np.exp(-1j * 2.0 * np.pi * f * t)
    c = 2.0 * np.mean(ref * w * osc) / w_mean
    if not np.isfinite(c.real) or not np.isfinite(c.imag):
        return 0.0

    return float(np.angle(c))


def estimate_ref_phase_offset_rad(
    ref: np.ndarray,
    *,
    abs_start_sample: int,
    sample_rate: float,
    ref_frequency_hz: float,
) -> float:
    """Estimate phase offset (radians) relative to absolute sample index.

    If the underlying signal is approximately:
      ref[k] = A*cos(2π f * (abs_start_sample + k)/sr + theta)

    then this returns theta (wrapped to [-π, π]).

    This is the quantity you want when converting to an integer sample shift
    for folding using absolute sample indices.
    """
    phi_win = estimate_ref_phase_rad(ref, sample_rate=sample_rate, ref_frequency_hz=ref_frequency_hz)

    f = float(ref_frequency_hz)
    sr = float(sample_rate)
    if not np.isfinite(f) or f <= 0.0 or not np.isfinite(sr) or sr <= 0.0:
        return 0.0

    # Phase advance contributed by the absolute start index.
    phi_abs = (2.0 * np.pi * f * (float(abs_start_sample) / sr))
    theta = float(phi_win - phi_abs)

    # Wrap to [-pi, pi]
    theta = float((theta + np.pi) % (2.0 * np.pi) - np.pi)
    return theta


def phase_to_sample_shift(phase_rad: float, period_samples: int) -> int:
    """Convert phase (rad) to an integer sample shift modulo period_samples.

    For a signal cos(2πk/N + phase), a positive returned shift means:
    aligned_index = (abs_sample + shift) % N.
    """
    n = int(period_samples)
    if n <= 0:
        return 0

    phi = float(phase_rad)
    if not np.isfinite(phi):
        return 0

    # If ref[k] ≈ A*cos(2π*k/N + phase_rad), then a fold index of
    #   aligned_index = (abs_sample + shift) % N
    # should satisfy 2π*shift/N ≈ phase_rad, i.e. shift ≈ phase_rad * N / (2π).
    shift = int(np.round((phi / (2.0 * np.pi)) * n)) % n
    return int(shift)


def fold_one_period_latest(
    signal: np.ndarray,
    abs_sample_indices: np.ndarray,
    *,
    period_samples: int,
    shift_samples: int = 0,
) -> np.ndarray:
    """Fold an arbitrary-length signal into one period using absolute sample indices.

    The folding index is: idx = (abs_sample_indices + shift_samples) % period_samples

    For each idx, the most recent sample (largest abs_sample_indices) wins.
    Unfilled bins are set to NaN.
    """
    x = np.asarray(signal, dtype=np.float64)
    sidx = np.asarray(abs_sample_indices, dtype=np.int64)

    n = int(period_samples)
    if n <= 0:
        return np.empty((0,), dtype=np.float64)

    if x.size != sidx.size:
        raise ValueError("signal and abs_sample_indices must be same length")

    if x.size == 0:
        out = np.empty((n,), dtype=np.float64)
        out.fill(np.nan)
        return out

    shift = int(shift_samples) % n

    out = np.empty((n,), dtype=np.float64)
    out.fill(np.nan)

    filled = 0
    # Iterate newest->oldest so newest sample per bin is kept.
    for val, abs_i in zip(x[::-1], sidx[::-1]):
        idx = int((int(abs_i) + shift) % n)
        if np.isnan(out[idx]):
            out[idx] = float(val)
            filled += 1
            if filled >= n:
                break

    return out
