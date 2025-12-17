#!/usr/bin/env python3
"""Lock-in amplifier dynamic reserve test (simulation).

This script estimates "dynamic reserve" by sweeping an interfering component
(amplitude) and checking when the lock-in magnitude/phase measurement deviates
beyond a tolerance.

It uses the project's existing lock-in DSP implementation:
`src.gui.widgets.lock_in_amplifier.LockInAmplifier.process_data()`.

Notes
-----
- Units are full-scale normalized audio samples (float, typically -1..+1).
- If `--clip` is enabled, inputs are hard-clipped to [-1, 1] to emulate ADC
  overload.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# Add project root to path (so `import src...` works when run from scripts/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui.widgets.lock_in_amplifier import LockInAmplifier


def _wrap_deg_180(angle_deg: float) -> float:
    # map to (-180, 180]
    x = (angle_deg + 180.0) % 360.0 - 180.0
    # keep +180 instead of -180 for readability
    if x <= -180.0:
        x += 360.0
    return x


@dataclass
class _FakeCalibration:
    lockin_gain_offset: float = 0.0

    def get_frequency_correction(self, _freq_hz: float):
        return 0.0, 0.0


@dataclass
class _FakeAudioEngine:
    sample_rate: float
    calibration: _FakeCalibration


@dataclass
class _SignalState:
    phi_ref: float = 0.0
    phi_sig: float = 0.0
    phi_int: float = 0.0


def _gen_cos_block(
    n: int,
    fs: float,
    freq_hz: float,
    amplitude: float,
    phase_rad: float,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return amplitude * np.cos(2.0 * np.pi * freq_hz * t + phase_rad)


def _advance_phase(phase_rad: float, fs: float, freq_hz: float, n: int) -> float:
    phase_rad = (phase_rad + 2.0 * np.pi * freq_hz * (n / fs)) % (2.0 * np.pi)
    return phase_rad


def _get_window(name: str, n: int) -> np.ndarray:
    """Return a real-valued window of length n.

    Window is returned as float64.
    """

    name = (name or "rect").lower()
    if name in ("rect", "rectangular", "none"):
        return np.ones(n, dtype=np.float64)
    if name in ("hann", "hanning"):
        return np.hanning(n).astype(np.float64)
    if name == "blackman":
        return np.blackman(n).astype(np.float64)
    raise ValueError(f"Unknown window: {name}")


def run_dynamic_reserve_sweep(
    *,
    fs: float,
    f_ref: float,
    a_sig: float,
    a_ref: float,
    f_int: float,
    a_int_values: np.ndarray,
    noise_rms: float,
    buffer_size: int,
    averaging_count: int,
    settle_buffers: int,
    harmonic: int,
    estimate_ref: bool,
    window: str,
    window_scope: str,
    window_normalize: bool,
    clip: bool,
    mag_tol_db: float,
    phase_tol_deg: float | None,
    seed: int | None,
) -> dict:
    rng = np.random.default_rng(seed)

    engine = _FakeAudioEngine(sample_rate=fs, calibration=_FakeCalibration())
    lockin = LockInAmplifier(engine)  # type: ignore[arg-type]

    lockin.buffer_size = buffer_size
    lockin.input_data = np.zeros((buffer_size, 2), dtype=np.float64)
    lockin.averaging_count = averaging_count
    lockin.harmonic_order = harmonic
    # If external_mode is False, LockInAmplifier forces ref_freq=gen_frequency and coherence=1.0.
    # That is useful for testing the coherent projection/demod path, but it bypasses
    # the reference frequency estimator.
    lockin.external_mode = bool(estimate_ref)
    lockin.apply_calibration = False
    lockin.signal_channel = 0
    lockin.ref_channel = 1
    lockin.gen_frequency = f_ref

    state = _SignalState(
        phi_ref=rng.uniform(0, 2 * np.pi),
        phi_sig=rng.uniform(0, 2 * np.pi),
        phi_int=rng.uniform(0, 2 * np.pi),
    )

    # Define "expected" result as the signal magnitude, with phase relative to ref.
    # Because the detector removes reference phase, the expected phase is simply
    # phi_sig - phi_ref (wrapped).
    expected_mag = float(a_sig)

    # We compute expected relative phase from the *initial* phases; since we
    # advance both phases using their frequencies, the difference stays constant
    # when f_sig == f_ref (which is our model).
    expected_phase_deg = _wrap_deg_180(math.degrees(state.phi_sig - state.phi_ref))

    win = _get_window(window, buffer_size)
    win_mean = float(np.mean(win))
    win_rms = float(np.sqrt(np.mean(win * win)))
    if window_normalize:
        if win_mean <= 0 or win_rms <= 0:
            raise ValueError("window gain is non-positive")
    window_scope = (window_scope or "signal").lower()
    if window_scope not in ("signal", "both"):
        raise ValueError("window_scope must be 'signal' or 'both'")

    rows = []
    last_pass = None
    limited_by_sweep = True

    for a_int in a_int_values:
        # Reset averaging history for each point
        lockin.history.clear()

        # Run a short settle period then collect enough buffers to fill the averager
        total_buffers = int(settle_buffers + averaging_count)

        for _ in range(total_buffers):
            sig = _gen_cos_block(buffer_size, fs, f_ref, a_sig, state.phi_sig)
            ref = _gen_cos_block(buffer_size, fs, f_ref, a_ref, state.phi_ref)

            if a_int != 0.0:
                sig = sig + _gen_cos_block(buffer_size, fs, f_int, a_int, state.phi_int)

            if noise_rms > 0:
                sig = sig + rng.normal(0.0, noise_rms, size=buffer_size)

            # Windowing strategy:
            # - scope='signal': window only the signal channel to reduce leakage while
            #   keeping the reference/scalloping correction assumptions unchanged.
            #   Normalization uses mean(win) so a coherent tone preserves amplitude.
            # - scope='both': window both channels. To keep the algorithm's internal
            #   correction (ref_rms*sqrt(2) ~= A_ref) valid under windowing, normalize
            #   by RMS gain (sqrt(mean(win^2))).
            if window_scope == "signal":
                if window_normalize:
                    sig = (sig * win) / win_mean
                else:
                    sig = sig * win
            else:  # both
                if window_normalize:
                    sig = (sig * win) / win_rms
                    ref = (ref * win) / win_rms
                else:
                    sig = sig * win
                    ref = ref * win

            if clip:
                sig = np.clip(sig, -1.0, 1.0)
                ref = np.clip(ref, -1.0, 1.0)

            lockin.input_data[:, 0] = sig
            lockin.input_data[:, 1] = ref

            lockin.process_data()

            state.phi_ref = _advance_phase(state.phi_ref, fs, f_ref, buffer_size)
            state.phi_sig = _advance_phase(state.phi_sig, fs, f_ref, buffer_size)
            state.phi_int = _advance_phase(state.phi_int, fs, f_int, buffer_size)

        mag = float(lockin.current_magnitude)
        phase = float(lockin.current_phase)
        phase_err = _wrap_deg_180(phase - expected_phase_deg)

        mag_err_db = (
            20.0 * math.log10(mag / expected_mag) if (mag > 1e-15 and expected_mag > 0) else float("inf")
        )

        passes = abs(mag_err_db) <= mag_tol_db
        if phase_tol_deg is not None:
            passes = passes and (abs(phase_err) <= phase_tol_deg)

        rows.append(
            {
                "a_int": float(a_int),
                "mag": mag,
                "phase_deg": phase,
                "mag_err_db": mag_err_db,
                "phase_err_deg": phase_err,
                "pass": bool(passes),
            }
        )

        if passes:
            last_pass = float(a_int)
        elif last_pass is not None:
            # first fail after at least one pass
            limited_by_sweep = False
            break

    if last_pass is None:
        dyn_reserve_db = float("nan")
    else:
        dyn_reserve_db = 20.0 * math.log10(last_pass / a_sig) if a_sig > 0 else float("inf")

    return {
        "expected_mag": expected_mag,
        "expected_phase_deg": expected_phase_deg,
        "rows": rows,
        "dynamic_reserve_db": dyn_reserve_db,
        "a_int_max_pass": last_pass,
        "limited_by_sweep": limited_by_sweep,
    }


def _a_int_values_for_diagnose(a_sig: float, a_int_stop: float, points: int) -> np.ndarray:
    """Log sweep amplitudes for diagnosis.

    Always includes 0, then log-spaced positive values.
    """

    points = int(points)
    if points < 4:
        points = 4

    start = max(a_sig * 1e-6, 1e-12)
    stop = max(float(a_int_stop), start)
    values = np.logspace(np.log10(start), np.log10(stop), points - 1)
    return np.concatenate(([0.0], values))


def _estimate_dr_single(
    *,
    fs: float,
    buffer_size: int,
    harmonic: int,
    f_ref: float,
    f_int: float,
    a_sig: float,
    a_ref: float,
    a_int_stop: float,
    noise_dbfs: float,
    averaging_count: int,
    settle_buffers: int,
    estimate_ref: bool,
    window: str,
    window_scope: str,
    window_normalize: bool,
    clip: bool,
    mag_tol_db: float,
    phase_tol_deg: float | None,
    seed: int | None,
    points: int,
) -> dict:
    noise_rms = 0.0
    if noise_dbfs > -199:
        noise_rms = 10.0 ** (noise_dbfs / 20.0)

    a_int_values = _a_int_values_for_diagnose(a_sig, a_int_stop, points)

    return run_dynamic_reserve_sweep(
        fs=fs,
        f_ref=f_ref,
        a_sig=a_sig,
        a_ref=a_ref,
        f_int=f_int,
        a_int_values=a_int_values,
        noise_rms=noise_rms,
        buffer_size=buffer_size,
        averaging_count=averaging_count,
        settle_buffers=settle_buffers,
        harmonic=harmonic,
        estimate_ref=estimate_ref,
        window=window,
        window_scope=window_scope,
        window_normalize=window_normalize,
        clip=clip,
        mag_tol_db=mag_tol_db,
        phase_tol_deg=phase_tol_deg,
        seed=seed,
    )


def _format_dr_line(label: str, result: dict) -> str:
    dr = result.get("dynamic_reserve_db")
    amax = result.get("a_int_max_pass")
    limited = bool(result.get("limited_by_sweep", False))

    if amax is None or dr is None or not math.isfinite(float(dr)):
        return f"{label:<24} DR: n/a"
    if limited:
        return f"{label:<24} DR: >= {float(dr):7.2f} dB  (amax={float(amax):.3g})"
    return f"{label:<24} DR:  {float(dr):7.2f} dB  (amax={float(amax):.3g})"


def run_diagnose(args: argparse.Namespace) -> int:
    """Run a small set of comparative sweeps and print a short diagnosis."""

    fs = float(args.fs)
    buffer_size = int(args.buffer)
    df = fs / buffer_size

    phase_tol = None if args.phase_tol_deg < 0 else float(args.phase_tol_deg)

    # Keep this reasonably quick.
    diag_points = 18
    diag_settle = max(1, int(args.settle))

    base_kwargs = dict(
        fs=fs,
        buffer_size=buffer_size,
        harmonic=int(args.harmonic),
        f_ref=float(args.f_ref),
        f_int=float(args.f_int),
        a_sig=float(args.a_sig),
        a_ref=float(args.a_ref),
        a_int_stop=float(args.a_int_stop),
        noise_dbfs=float(args.noise_dbfs),
        averaging_count=int(args.avg),
        settle_buffers=diag_settle,
        estimate_ref=bool(args.estimate_ref),
        window=str(args.window),
        window_scope=str(args.window_scope),
        window_normalize=bool(args.window_normalize),
        clip=bool(args.clip),
        mag_tol_db=float(args.mag_tol_db),
        phase_tol_deg=phase_tol,
        seed=int(args.seed) if args.seed is not None else None,
        points=diag_points,
    )

    print("DR diagnosis (simulation)")
    print(f"fs={fs} Hz, buffer={buffer_size} (df={df:.6f} Hz), harmonic={int(args.harmonic)}")
    print(f"f_ref={float(args.f_ref)} Hz, f_int={float(args.f_int)} Hz")
    print(f"a_sig={float(args.a_sig)} (peak), a_ref={float(args.a_ref)} (peak), a_int_stop={float(args.a_int_stop)}")
    print(f"avg={int(args.avg)}, estimate_ref={bool(args.estimate_ref)}, clip={bool(args.clip)}")
    print(f"tol: |mag_err| <= {float(args.mag_tol_db)} dB" + ("" if phase_tol is None else f", |phase_err| <= {phase_tol} deg"))
    print("")

    # 1) Baseline
    r_base = _estimate_dr_single(**base_kwargs)
    print(_format_dr_line("baseline", r_base))

    # 2) Reference estimator impact
    r_ref_off = _estimate_dr_single(**{**base_kwargs, "estimate_ref": False})
    r_ref_on = _estimate_dr_single(**{**base_kwargs, "estimate_ref": True})
    print(_format_dr_line("ref_estimator=OFF", r_ref_off))
    print(_format_dr_line("ref_estimator=ON", r_ref_on))

    # 3) Averaging / time constant impact (hold other parameters)
    avg_list = [1, 4, 16]
    print("")
    print("Averaging impact")
    for a in avg_list:
        r = _estimate_dr_single(**{**base_kwargs, "averaging_count": a})
        print(_format_dr_line(f"avg={a}", r))

    # 4) Interferer frequency sensitivity (window leakage)
    # Coherent interferer at another bin (best-case) and non-coherent half-bin offsets.
    print("")
    print("Interferer frequency sensitivity")

    def coherent_freq(cycles: int) -> float:
        return cycles * fs / buffer_size

    f_ref_coh = coherent_freq(85)
    f_int_coh = coherent_freq(171)
    r_coh = _estimate_dr_single(**{**base_kwargs, "f_ref": f_ref_coh, "f_int": f_int_coh, "estimate_ref": False})
    print(_format_dr_line("coherent (best-case)", r_coh))

    offsets_bins = [0.5, 10.5, 50.5, 300.5]
    for ob in offsets_bins:
        f_int = float(base_kwargs["f_ref"]) + ob * df
        r = _estimate_dr_single(**{**base_kwargs, "f_int": f_int})
        print(_format_dr_line(f"half-bin @ {ob:g} bins", r))

    # 4b) Windowing sensitivity. Windowing mainly helps in the sidelobe region;
    # very near-in interferers (e.g. 0.5 bin) are inside the main lobe and often
    # do not improve.
    print("")
    print("Windowing sensitivity")
    for ob in [0.5, 10.5, 50.5]:
        f_int = float(base_kwargs["f_ref"]) + ob * df
        print(f"offset={ob:g} bins")
        for w in ["rect", "hann", "blackman"]:
            r = _estimate_dr_single(
                **{**base_kwargs, "f_int": f_int, "window": w, "window_scope": "both", "window_normalize": True}
            )
            print(_format_dr_line(f"  window={w}", r))

    # 5) Clipping sensitivity
    print("")
    print("Clipping sensitivity")
    r_clip_off = _estimate_dr_single(**{**base_kwargs, "clip": False})
    r_clip_on = _estimate_dr_single(**{**base_kwargs, "clip": True})
    print(_format_dr_line("clip=OFF", r_clip_off))
    print(_format_dr_line("clip=ON", r_clip_on))

    # Heuristic diagnosis
    print("")
    likely = []

    def _dr_value(res: dict) -> float | None:
        v = res.get("dynamic_reserve_db")
        if v is None or not math.isfinite(float(v)):
            return None
        return float(v)

    dr_coh = _dr_value(r_coh)
    dr_near = _dr_value(_estimate_dr_single(**{**base_kwargs, "f_int": float(base_kwargs["f_ref"]) + 0.5 * df}))
    # Window improvement heuristic: compare in a sidelobe-ish case (10.5 bins).
    f_int_side = float(base_kwargs["f_ref"]) + 10.5 * df
    dr_win_rect = _dr_value(_estimate_dr_single(**{**base_kwargs, "f_int": f_int_side, "window": "rect", "window_scope": "both", "window_normalize": True}))
    dr_win_hann = _dr_value(_estimate_dr_single(**{**base_kwargs, "f_int": f_int_side, "window": "hann", "window_scope": "both", "window_normalize": True}))
    dr_win_blackman = _dr_value(_estimate_dr_single(**{**base_kwargs, "f_int": f_int_side, "window": "blackman", "window_scope": "both", "window_normalize": True}))
    dr_ref_off = _dr_value(r_ref_off)
    dr_ref_on = _dr_value(r_ref_on)
    dr_avg1 = _dr_value(_estimate_dr_single(**{**base_kwargs, "averaging_count": 1}))
    dr_avg16 = _dr_value(_estimate_dr_single(**{**base_kwargs, "averaging_count": 16}))
    dr_clip_off = _dr_value(r_clip_off)
    dr_clip_on = _dr_value(r_clip_on)

    if dr_coh is not None and dr_near is not None and (dr_coh - dr_near) > 20:
        likely.append("Window leakage / finite integration is dominating (non-coherent interferer).")
        if dr_win_hann is not None and dr_win_rect is not None and (dr_win_hann - dr_win_rect) > 6:
            likely.append("Windowing (Hann/Blackman) improves DR in the hard case; consider applying a window in demod.")
        elif dr_win_blackman is not None and dr_win_rect is not None and (dr_win_blackman - dr_win_rect) > 6:
            likely.append("Windowing (Blackman) improves DR in the hard case; consider applying a window in demod.")
    if dr_ref_off is not None and dr_ref_on is not None and (dr_ref_off - dr_ref_on) > 10:
        likely.append("Reference frequency estimation is degrading DR (use fixed ref or stabilize estimator).")
    if dr_avg1 is not None and dr_avg16 is not None and (dr_avg16 - dr_avg1) > 10:
        likely.append("Time constant (averaging) is too short for desired DR.")
    if dr_clip_off is not None and dr_clip_on is not None and (dr_clip_off - dr_clip_on) > 10:
        likely.append("Overload/clipping is dominating (nonlinear front-end effect).")

    if not likely:
        likely.append("No single dominant factor detected with current thresholds; inspect the tables above.")

    print("Likely dominant factor(s)")
    for s in likely:
        print(f"- {s}")

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate lock-in dynamic reserve by interferer sweep (simulation).")

    preset = p.add_mutually_exclusive_group()
    preset.add_argument(
        "--dr",
        action="store_true",
        help="Use recommended settings for dynamic reserve (algorithm-only, short time constant stress-case).",
    )
    preset.add_argument(
        "--dr-est",
        action="store_true",
        help="Like --dr, but also tests reference frequency estimation.",
    )
    preset.add_argument(
        "--dr-clip",
        action="store_true",
        help="Like --dr, but enables hard clipping to emulate ADC overload.",
    )
    preset.add_argument(
        "--dr100",
        action="store_true",
        help="Best-case DR: coherent reference + coherent interferer; sweeps interferer up to --dr100-stop (default 1e5, no clip).",
    )

    p.add_argument(
        "--dr100-stop",
        type=float,
        default=None,
        help="Override --dr100 interferer stop amplitude (peak). Useful for probing numeric limits (e.g. 1e8).",
    )

    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Run a small comparative sweep set to identify what dominates DR (leakage vs estimator vs averaging vs clipping).",
    )

    p.add_argument("--fs", type=float, default=48000.0, help="Sample rate (Hz)")
    p.add_argument("--buffer", type=int, default=4096, help="Buffer size (samples)")
    p.add_argument("--avg", type=int, default=10, help="Averaging count (buffers)")
    p.add_argument("--settle", type=int, default=2, help="Settle buffers before averaging")
    p.add_argument("--harmonic", type=int, default=1, help="Demod harmonic (1..)")
    p.add_argument(
        "--estimate-ref",
        action="store_true",
        help="Enable reference frequency estimation (sets external_mode=True).",
    )

    p.add_argument(
        "--window",
        type=str,
        default="rect",
        choices=["rect", "hann", "blackman"],
        help="Apply a window to both signal/ref before processing (affects leakage).",
    )
    p.add_argument(
        "--window-scope",
        type=str,
        default="signal",
        choices=["signal", "both"],
        help="Where to apply the window: only the signal channel (safe with current correction) or both channels.",
    )
    p.add_argument(
        "--window-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize window coherent gain so the target tone amplitude is preserved.",
    )

    p.add_argument("--f-ref", type=float, default=1000.0, help="Reference frequency (Hz)")
    p.add_argument("--a-sig", type=float, default=1e-3, help="Signal amplitude (peak, 0..1)")
    p.add_argument("--a-ref", type=float, default=0.5, help="Reference amplitude (peak, 0..1)")

    p.add_argument("--f-int", type=float, default=2000.0, help="Interferer frequency (Hz)")
    p.add_argument("--a-int-start", type=float, default=0.0, help="Interferer start amplitude (peak)")
    p.add_argument("--a-int-stop", type=float, default=1.0, help="Interferer stop amplitude (peak)")
    p.add_argument("--points", type=int, default=41, help="Sweep points")
    p.add_argument("--log", action="store_true", help="Log sweep for interferer amplitude")

    p.add_argument(
        "--noise-dbfs",
        type=float,
        default=-120.0,
        help="Additive white noise level as RMS dBFS (e.g. -100). Use very low (e.g. -200) to disable.",
    )

    p.add_argument("--clip", action="store_true", help="Hard clip inputs to [-1, 1]")

    p.add_argument("--mag-tol-db", type=float, default=0.5, help="Magnitude tolerance (dB)")
    p.add_argument(
        "--phase-tol-deg",
        type=float,
        default=10.0,
        help="Phase tolerance (deg). Use negative to ignore phase.",
    )

    p.add_argument("--seed", type=int, default=1, help="RNG seed (for noise / initial phase)")

    return p.parse_args()


def _apply_dr_preset(args: argparse.Namespace) -> None:
    """Apply recommended settings for DR testing.

    Goal: provide a stable, reproducible indicator for the lock-in demod algorithm.

    Strategy:
    - Pick a coherent reference (integer cycles per buffer) so the desired tone is
      not limited by window leakage.
    - Use a non-coherent interferer with a half-bin offset at a large bin separation.
      This makes the test produce a finite, repeatable DR for a short time constant.
    - Use log sweep since DR is reported in dB.

    Note: Because the interferer offset is expressed in *bins*, increasing --buffer
    keeps (Δf * T) approximately constant, so DR will not improve just by increasing
    the buffer length for this preset.
    """

    fs = float(args.fs)
    buffer_size = int(args.buffer)

    def coherent_freq(cycles: int) -> float:
        return cycles * fs / buffer_size

    args.f_ref = coherent_freq(85)
    df = fs / buffer_size
    bin_offset = 300.5
    args.f_int = args.f_ref + bin_offset * df

    args.a_sig = 1e-3
    args.a_ref = 0.5

    args.a_int_start = 0.0
    args.a_int_stop = 1.0
    args.points = 61
    args.log = True

    # Keep averaging intentionally low so the non-coherent interferer does not
    # average out across many blocks (this makes DR a useful algorithm indicator).
    args.avg = 1
    args.settle = 1

    args.noise_dbfs = -140.0

    args.mag_tol_db = 0.5
    args.phase_tol_deg = 20.0


def _apply_dr100_preset(args: argparse.Namespace) -> None:
    """Apply a best-case preset to demonstrate that 100 dB DR is possible.

    This uses coherent tones (integer cycles in the buffer) so the interferer is
    orthogonal to the demod bin in a rectangular window, i.e. rejection is limited
    mainly by floating-point precision.

    Note: This is not a worst-case spec like many commercial instruments quote.
    It's a sanity/regression indicator for the mathematical demod path.
    """

    fs = float(args.fs)
    buffer_size = int(args.buffer)

    def coherent_freq(cycles: int) -> float:
        return cycles * fs / buffer_size

    args.f_ref = coherent_freq(85)
    args.f_int = coherent_freq(171)

    args.a_sig = 1e-3
    args.a_ref = 0.5
    args.a_int_start = 0.0
    # Go very high to probe numerical limits (a_sig=1e-3 => 1e5 gives >=160 dB).
    # This is intentionally non-physical (exceeds FS) but useful for evaluating
    # floating-point leakage/rounding in the demod math.
    args.a_int_stop = 1e5
    args.points = 18
    args.log = True

    args.avg = 1
    args.settle = 1
    args.noise_dbfs = -200.0
    args.clip = False

    args.mag_tol_db = 0.5
    args.phase_tol_deg = 20.0


def main() -> int:
    args = _parse_args()

    if bool(args.dr100):
        _apply_dr100_preset(args)
        if args.dr100_stop is not None:
            if float(args.dr100_stop) <= 0:
                raise SystemExit("--dr100-stop must be > 0")
            args.a_int_stop = float(args.dr100_stop)

    if bool(args.dr) or bool(args.dr_est) or bool(args.dr_clip):
        _apply_dr_preset(args)
        # Mode tweaks after preset
        if bool(args.dr_est):
            args.estimate_ref = True
        if bool(args.dr_clip):
            args.clip = True

    if bool(args.diagnose):
        return run_diagnose(args)

    if args.a_sig <= 0:
        raise SystemExit("--a-sig must be > 0")
    if args.a_ref <= 0:
        raise SystemExit("--a-ref must be > 0")
    if args.points < 2:
        raise SystemExit("--points must be >= 2")

    noise_rms = 0.0
    if args.noise_dbfs > -199:
        noise_rms = 10.0 ** (args.noise_dbfs / 20.0)

    phase_tol = None if args.phase_tol_deg < 0 else float(args.phase_tol_deg)

    if args.log:
        start = max(float(args.a_int_start), 1e-12)
        stop = max(float(args.a_int_stop), start)
        a_int_values = np.logspace(np.log10(start), np.log10(stop), int(args.points))
        if float(args.a_int_start) == 0.0:
            # include exact 0 as first point
            a_int_values = np.concatenate(([0.0], a_int_values))
    else:
        a_int_values = np.linspace(float(args.a_int_start), float(args.a_int_stop), int(args.points))

    result = run_dynamic_reserve_sweep(
        fs=float(args.fs),
        f_ref=float(args.f_ref),
        a_sig=float(args.a_sig),
        a_ref=float(args.a_ref),
        f_int=float(args.f_int),
        a_int_values=a_int_values,
        noise_rms=float(noise_rms),
        buffer_size=int(args.buffer),
        averaging_count=int(args.avg),
        settle_buffers=int(args.settle),
        harmonic=int(args.harmonic),
        estimate_ref=bool(args.estimate_ref),
        window=str(args.window),
        window_scope=str(args.window_scope),
        window_normalize=bool(args.window_normalize),
        clip=bool(args.clip),
        mag_tol_db=float(args.mag_tol_db),
        phase_tol_deg=phase_tol,
        seed=int(args.seed) if args.seed is not None else None,
    )

    print("Lock-in dynamic reserve sweep (simulation)")
    print(f"f_ref={args.f_ref} Hz, f_int={args.f_int} Hz, fs={args.fs} Hz")
    print(f"a_sig={args.a_sig} (peak), a_ref={args.a_ref} (peak), clip={args.clip}")
    print(f"estimate_ref={bool(args.estimate_ref)}")
    if bool(args.dr) or bool(args.dr_est) or bool(args.dr_clip):
        mode = "dr"
        if bool(args.dr_est):
            mode = "dr-est"
        if bool(args.dr_clip):
            mode = "dr-clip"
        print(f"preset={mode}")
    if bool(args.dr100):
        print("preset=dr100")
    print(f"tol: |mag_err| <= {args.mag_tol_db} dB" + ("" if phase_tol is None else f", |phase_err| <= {phase_tol} deg"))
    print("")

    hdr = f"{'a_int':>10} | {'mag':>10} | {'mag_err(dB)':>11} | {'phase_err(deg)':>13} | {'pass':>5}"
    print(hdr)
    print("-" * len(hdr))

    for r in result["rows"]:
        print(
            f"{r['a_int']:10.3e} | {r['mag']:10.3e} | {r['mag_err_db']:11.3f} | {r['phase_err_deg']:13.3f} | {str(r['pass']):>5}"
        )

    print("")
    if result["a_int_max_pass"] is None or not math.isfinite(result["dynamic_reserve_db"]):
        print("Dynamic reserve: could not be determined (no passing point).")
        return 2

    if bool(result.get("limited_by_sweep", False)):
        print(f"Max passing interferer amplitude: {result['a_int_max_pass']:.6f} (peak) (no failure within sweep)")
        print(f"Dynamic reserve: >= {result['dynamic_reserve_db']:.2f} dB")
    else:
        print(f"Max passing interferer amplitude: {result['a_int_max_pass']:.6f} (peak)")
        print(f"Dynamic reserve: {result['dynamic_reserve_db']:.2f} dB")
    print(f"Dynamic reserve (ratio): {result['a_int_max_pass'] / float(args.a_sig):.3g} x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
