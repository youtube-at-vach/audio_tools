import sys
import os
import numpy as np

# Add repo root to path (tests pattern in this repo)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.phase_sync import estimate_ref_phase_offset_rad, phase_to_sample_shift, fold_one_period_latest


def test_phase_to_sample_shift_basic():
    n = 100
    assert phase_to_sample_shift(0.0, n) == 0
    # +90 deg (pi/2) should correspond to +N/4 shift
    assert phase_to_sample_shift(np.pi / 2.0, n) == (n // 4)


def test_fold_one_period_latest_alignment_with_known_delay():
    sr = 48000.0
    n = 480  # 100 Hz period
    f = sr / n

    # Create a cosine that is delayed by d samples (mod N).
    # REF and DUTIN share the same delay (REF is used to estimate the shift applied to DUTIN).
    d = 37

    total = 5000
    abs_idx = np.arange(total, dtype=np.int64)
    phase = 2.0 * np.pi * f * (abs_idx / sr)

    delayed = np.cos(phase - 2.0 * np.pi * (d / n))
    ref = delayed
    dut = delayed

    # Estimate phase offset on a recent window (absolute-index aware)
    ref_win = ref[-2000:]
    idx_win = abs_idx[-2000:]
    phi = estimate_ref_phase_offset_rad(ref_win, abs_start_sample=int(idx_win[0]), sample_rate=sr, ref_frequency_hz=f)
    shift = phase_to_sample_shift(phi, n)

    # Fold DUT using the same abs indices window
    dut_win = dut[-2000:]
    aligned = fold_one_period_latest(dut_win, idx_win, period_samples=n, shift_samples=shift)

    # Compare against ideal memory map (cos over one period)
    k = np.arange(n)
    mem = np.cos(2.0 * np.pi * k / n)

    # Because of quantization and windowing, allow small error.
    # Also, bins might be NaN if not all filled (should be filled with 2000 samples).
    assert np.all(np.isfinite(aligned))

    corr = float(np.dot(mem, aligned) / (np.linalg.norm(mem) * np.linalg.norm(aligned)))
    assert corr > 0.99
