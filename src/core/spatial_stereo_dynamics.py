from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _one_pole_highpass_step(x: float, *, a: float, x1: float, y1: float) -> tuple[float, float, float]:
    # y[n] = a*(y[n-1] + x[n] - x[n-1])
    y = float(a) * (float(y1) + float(x) - float(x1))
    return y, float(x), y


@dataclass
class DiffuseSide3DState:
    """State for an experimental 'diffuse side' stereo widening model."""

    # Ring buffer for the side signal.
    buf: np.ndarray | None = None
    idx: int = 0
    max_delay: int = 0

    # High-pass filter memory for side (to avoid low-frequency smear).
    hp_x1: float = 0.0
    hp_y1: float = 0.0


def process_stereo_diffuse_side_3d_block(
    data: np.ndarray,
    *,
    sample_rate: float,
    alpha: float,
    beta: float,
    tau_seconds: float,
    state: DiffuseSide3DState | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, DiffuseSide3DState]:
    """Experimental stereo '3D' model.

    Idea:
    - Convert to mid/side.
    - High-pass the side channel (reduce LF smear).
    - Create a diffused side by mixing a few short delayed taps of side.
      This reduces interaural correlation and can increase perceived width/out-of-head.
    - Recombine with controllable width.

    Parameter mapping (reuses the widget controls):
    - alpha: diffusion depth in [0,1]
    - beta : width amount in [0,1]
    - tau_seconds: base delay time (clamped to a sane range)
    """
    if state is None:
        state = DiffuseSide3DState()

    x = np.asarray(data)
    if x.ndim == 1:
        l = x.astype(np.float32, copy=False)
        r = l
    else:
        if x.shape[1] == 1:
            l = x[:, 0].astype(np.float32, copy=False)
            r = l
        elif x.shape[1] == 2:
            l = x[:, 0].astype(np.float32, copy=False)
            r = x[:, 1].astype(np.float32, copy=False)
        else:
            raise ValueError("Only mono or stereo input is supported.")

    n = int(l.shape[0])
    out = np.zeros((n, 2), dtype=np.float32)

    depth = float(np.clip(float(alpha), 0.0, 1.0))
    width = float(np.clip(1.0 + 2.0 * float(beta), 0.0, 3.0))

    tau_ms = float(tau_seconds) * 1000.0
    base_ms = float(np.clip(8.0 if tau_ms <= 0.0 else tau_ms, 3.0, 25.0))

    # Choose a few short delay taps (ms). Non-integer-ish ratios to avoid obvious combing.
    d1 = int(max(1, round((base_ms * 0.55) * float(sample_rate) / 1000.0)))
    d2 = int(max(1, round((base_ms * 0.90 + 2.0) * float(sample_rate) / 1000.0)))
    d3 = int(max(1, round((base_ms * 1.35 + 5.0) * float(sample_rate) / 1000.0)))
    max_delay = int(max(d1, d2, d3))

    if state.buf is None or int(state.max_delay) != max_delay:
        state.buf = np.zeros(max_delay + 1, dtype=np.float32)
        state.idx = 0
        state.max_delay = int(max_delay)

    fc = 150.0
    a = float(np.exp(-2.0 * np.pi * float(fc) / float(sample_rate)))

    g1 = 0.35 * depth
    g2 = 0.22 * depth
    g3 = 0.14 * depth
    dry = 1.0 - 0.55 * depth

    buf = state.buf
    idx = int(state.idx)
    x1 = float(state.hp_x1)
    y1 = float(state.hp_y1)

    for i in range(n):
        li = float(l[i])
        ri = float(r[i])
        m = 0.5 * (li + ri)
        s = 0.5 * (li - ri)

        if abs(li) + abs(ri) <= eps:
            buf[idx] = 0.0
            idx = (idx + 1) % (max_delay + 1)
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            continue

        s_hp, x1, y1 = _one_pole_highpass_step(s, a=a, x1=x1, y1=y1)

        buf[idx] = float(s_hp)

        j1 = (idx - d1) % (max_delay + 1)
        j2 = (idx - d2) % (max_delay + 1)
        j3 = (idx - d3) % (max_delay + 1)

        s_diff = dry * float(s_hp) + g1 * float(buf[j1]) + g2 * float(buf[j2]) + g3 * float(buf[j3])

        lo = float(m + width * s_diff)
        ro = float(m - width * s_diff)

        out[i, 0] = np.float32(lo)
        out[i, 1] = np.float32(ro)

        idx = (idx + 1) % (max_delay + 1)

    state.idx = int(idx)
    state.hp_x1 = float(x1)
    state.hp_y1 = float(y1)
    return out, state


@dataclass
class NeuralPrecedence3DState:
    """State for a neuro-inspired stereo spatialization experiment."""

    buf: np.ndarray | None = None
    idx: int = 0
    max_delay: int = 0

    env: float = 0.0
    env_prev: float = 0.0
    spacious: float = 0.0  # 0..1 ramp
    hold: int = 0  # samples remaining in precedence hold

    hp_x1: float = 0.0
    hp_y1: float = 0.0


def process_stereo_neural_precedence_3d_block(
    data: np.ndarray,
    *,
    sample_rate: float,
    alpha: float,
    beta: float,
    tau_seconds: float,
    state: NeuralPrecedence3DState | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, NeuralPrecedence3DState]:
    """Neuro-inspired spatialization experiment.

    Intended to reflect a few perceptual heuristics:
    - High transient sensitivity (onset detection)
    - Precedence/leading-wavefront dominance (briefly suppress spaciousness)
    - Adaptation during sustained sound (reduce effect over time)

    Parameter mapping:
    - alpha: effect strength (0..1)
    - beta : width amount (0..1)
    - tau_seconds: time scale for delays/adaptation
    """
    if state is None:
        state = NeuralPrecedence3DState()

    x = np.asarray(data)
    if x.ndim == 1:
        l = x.astype(np.float32, copy=False)
        r = l
    else:
        if x.shape[1] == 1:
            l = x[:, 0].astype(np.float32, copy=False)
            r = l
        elif x.shape[1] == 2:
            l = x[:, 0].astype(np.float32, copy=False)
            r = x[:, 1].astype(np.float32, copy=False)
        else:
            raise ValueError("Only mono or stereo input is supported.")

    n = int(l.shape[0])
    out = np.zeros((n, 2), dtype=np.float32)

    strength = float(np.clip(float(alpha), 0.0, 1.0))
    width_max = float(np.clip(1.0 + 2.0 * float(beta), 0.0, 3.0))

    tau_ms = float(tau_seconds) * 1000.0
    base_ms = float(np.clip(10.0 if tau_ms <= 0.0 else tau_ms, 3.0, 30.0))

    d1 = int(max(1, round((base_ms * 0.45) * float(sample_rate) / 1000.0)))
    d2 = int(max(1, round((base_ms * 0.80 + 1.7) * float(sample_rate) / 1000.0)))
    d3 = int(max(1, round((base_ms * 1.25 + 4.3) * float(sample_rate) / 1000.0)))
    max_delay = int(max(d1, d2, d3))

    if state.buf is None or int(state.max_delay) != max_delay:
        state.buf = np.zeros(max_delay + 1, dtype=np.float32)
        state.idx = 0
        state.max_delay = int(max_delay)

    buf = state.buf
    idx = int(state.idx)

    att = float(np.exp(-1.0 / (float(sample_rate) * 0.003)))
    rel = float(np.exp(-1.0 / (float(sample_rate) * 0.080)))

    hold_samples = int(round(0.008 * float(sample_rate)))

    ramp_tau = float(np.clip(0.020 + 0.004 * (base_ms / 10.0), 0.010, 0.080))
    ramp_g = float(1.0 - np.exp(-1.0 / (float(sample_rate) * ramp_tau)))

    adapt_tau = float(np.clip(0.250 + 0.020 * (base_ms / 10.0), 0.150, 0.600))
    adapt_g = float(1.0 - np.exp(-1.0 / (float(sample_rate) * adapt_tau)))

    fc = 180.0
    a = float(np.exp(-2.0 * np.pi * float(fc) / float(sample_rate)))

    hp_x1 = float(state.hp_x1)
    hp_y1 = float(state.hp_y1)

    env = float(state.env)
    env_prev = float(state.env_prev)
    spacious = float(state.spacious)
    hold = int(state.hold)

    for i in range(n):
        li = float(l[i])
        ri = float(r[i])

        if abs(li) + abs(ri) <= eps:
            buf[idx] = 0.0
            idx = (idx + 1) % (max_delay + 1)
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            env *= 0.999
            spacious *= 0.999
            hold = max(0, hold - 1)
            continue

        m = 0.5 * (li + ri)
        s = 0.5 * (li - ri)

        xenv = abs(m) + 0.35 * abs(s)
        if xenv > env:
            env = att * env + (1.0 - att) * xenv
        else:
            env = rel * env + (1.0 - rel) * xenv

        d_env = max(0.0, env - env_prev)
        env_prev = env

        onset = d_env > (0.010 + 0.15 * env)
        if onset:
            hold = hold_samples

        if hold > 0:
            hold -= 1
            spacious = (1.0 - ramp_g) * spacious
        else:
            spacious = float(np.clip(spacious + ramp_g * (1.0 - spacious), 0.0, 1.0))

        steady = float(np.clip(1.0 - 10.0 * d_env, 0.0, 1.0))
        spacious = float(np.clip(spacious - adapt_g * steady * 0.15, 0.0, 1.0))

        s_hp, hp_x1, hp_y1 = _one_pole_highpass_step(s, a=a, x1=hp_x1, y1=hp_y1)
        buf[idx] = float(s_hp)

        j1 = (idx - d1) % (max_delay + 1)
        j2 = (idx - d2) % (max_delay + 1)
        j3 = (idx - d3) % (max_delay + 1)

        mix = strength * spacious
        dry = 1.0 - 0.60 * mix
        g1 = 0.32 * mix
        g2 = 0.20 * mix
        g3 = 0.12 * mix
        s_diff = dry * float(s_hp) + g1 * float(buf[j1]) + g2 * float(buf[j2]) + g3 * float(buf[j3])

        width = float(np.clip(1.0 + (width_max - 1.0) * (0.25 + 0.75 * spacious), 0.0, 3.0))
        lo = float(m + width * s_diff)
        ro = float(m - width * s_diff)

        out[i, 0] = np.float32(lo)
        out[i, 1] = np.float32(ro)

        idx = (idx + 1) % (max_delay + 1)

    state.idx = int(idx)
    state.env = float(env)
    state.env_prev = float(env_prev)
    state.spacious = float(spacious)
    state.hold = int(hold)
    state.hp_x1 = float(hp_x1)
    state.hp_y1 = float(hp_y1)

    return out, state
