from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


def wrap_angle_rad(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi). Supports scalars and numpy arrays."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ema_angle_rad(prev: float, current: float, gamma: float) -> float:
    """Exponential moving average for circular quantities (angle in radians).

    Equivalent to: prev + gamma * wrap(current - prev)
    where gamma in [0, 1].
    """
    if gamma <= 0.0:
        return prev
    if gamma >= 1.0:
        return current
    return float(prev + gamma * wrap_angle_rad(current - prev))


@dataclass
class InertialAttractorState:
    theta_out: float = 0.0
    theta_m: float = 0.0
    initialized: bool = False


@dataclass
class PhaseSpaceInertiaState:
    """State for a simple polar phase-space inertial model.

    We keep both position and velocity for:
    - theta (angle)
    - r     (radius, i.e. instantaneous stereo magnitude)

    This allows inertia not only along the circular direction, but also in
    radial magnitude (perceived loudness/energy in this representation).
    """

    theta: float = 0.0
    theta_v: float = 0.0
    r: float = 0.0
    r_v: float = 0.0

    theta_m: float = 0.0
    r_m: float = 0.0
    initialized: bool = False


def _gamma_from_tau(tau_seconds: float, sample_rate: float) -> float:
    if tau_seconds <= 0.0:
        return 1.0
    # gamma = 1 - exp(-dt/tau)
    dt = 1.0 / float(sample_rate)
    return float(1.0 - np.exp(-dt / float(tau_seconds)))


def _damping_from_alpha(alpha: float) -> float:
    """Map alpha in [0, 1] to a stable damping factor in (0, 1].

    Small alpha should still have some damping to avoid runaway velocity.
    """
    a = float(alpha)
    if a <= 0.0:
        return 0.05
    if a >= 1.0:
        return 1.0
    return float(np.clip(0.05 + 0.95 * np.sqrt(a), 0.05, 1.0))


def inertial_attractor_step(
    theta_in: float,
    state: InertialAttractorState,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> InertialAttractorState:
    """One-sample update for the inertial+gravity model.

    theta_out(t+1) = theta_out(t)
      + alpha * wrap(theta_in - theta_out(t))
      + beta  * wrap(theta_m(t) - theta_out(t))

    theta_m(t) is circular EMA of theta_in.
    """
    if not state.initialized:
        state.theta_out = float(theta_in)
        state.theta_m = float(theta_in)
        state.initialized = True
        return state

    state.theta_m = ema_angle_rad(state.theta_m, theta_in, gamma)

    d_in = wrap_angle_rad(theta_in - state.theta_out)
    d_m = wrap_angle_rad(state.theta_m - state.theta_out)

    state.theta_out = float(state.theta_out + float(alpha) * float(d_in) + float(beta) * float(d_m))
    # Keep state bounded to avoid numerical drift on long files.
    state.theta_out = float(wrap_angle_rad(state.theta_out))
    return state


def process_stereo_inertial_attractor_block(
    data: np.ndarray,
    *,
    sample_rate: float,
    alpha: float,
    beta: float,
    tau_seconds: float,
    state: InertialAttractorState | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, InertialAttractorState]:
    """Process a block of mono/stereo samples and return stereo output.

    - Accepts shape (N,) mono or (N, C) where C in {1,2}.
    - Returns shape (N, 2) float32.
    - Maintains state across blocks.
    """
    if state is None:
        state = InertialAttractorState()

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

    rmag = np.hypot(l, r)
    theta = np.arctan2(r, l)

    gamma = _gamma_from_tau(tau_seconds, sample_rate)

    for i in range(n):
        if float(rmag[i]) <= eps:
            # Silence: keep state as-is, output zeros.
            continue

        state = inertial_attractor_step(
            float(theta[i]),
            state,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )

        out[i, 0] = float(rmag[i]) * np.cos(state.theta_out)
        out[i, 1] = float(rmag[i]) * np.sin(state.theta_out)

    return out, state


def phase_space_inertia_step(
    theta_in: float,
    r_in: float,
    state: PhaseSpaceInertiaState,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> PhaseSpaceInertiaState:
    """One-sample update for a polar phase-space inertial model.

    The model is an intuitive velocity-state smoother:

    v <- (1-d)*v + (alpha * error_to_input + beta * error_to_mass_center)
    x <- x + v

    Applied to both theta (with wrap-around error) and r (linear error).
    """
    if not state.initialized:
        state.theta = float(theta_in)
        state.theta_m = float(theta_in)
        state.r = float(r_in)
        state.r_m = float(r_in)
        state.theta_v = 0.0
        state.r_v = 0.0
        state.initialized = True
        return state

    # Mass center tracking (EMA). Angle needs circular EMA.
    state.theta_m = ema_angle_rad(state.theta_m, float(theta_in), float(gamma))
    state.r_m = float((1.0 - float(gamma)) * float(state.r_m) + float(gamma) * float(r_in))

    damp = _damping_from_alpha(float(alpha))
    keep = 1.0 - float(damp)

    # Angular dynamics
    e_theta_in = wrap_angle_rad(float(theta_in) - float(state.theta))
    e_theta_m = wrap_angle_rad(float(state.theta_m) - float(state.theta))
    acc_theta = float(alpha) * float(e_theta_in) + float(beta) * float(e_theta_m)
    state.theta_v = float(keep * float(state.theta_v) + acc_theta)
    state.theta = float(wrap_angle_rad(float(state.theta) + float(state.theta_v)))

    # Radial dynamics (clamp to avoid negative magnitude)
    e_r_in = float(r_in) - float(state.r)
    e_r_m = float(state.r_m) - float(state.r)
    acc_r = float(alpha) * float(e_r_in) + float(beta) * float(e_r_m)
    state.r_v = float(keep * float(state.r_v) + acc_r)
    state.r = float(max(0.0, float(state.r) + float(state.r_v)))

    return state


def process_stereo_phase_space_inertia_block(
    data: np.ndarray,
    *,
    sample_rate: float,
    alpha: float,
    beta: float,
    tau_seconds: float,
    state: PhaseSpaceInertiaState | None = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, PhaseSpaceInertiaState]:
    """Process a block with polar phase-space inertia (theta + radius inertia).

    - Accepts shape (N,) mono or (N, C) where C in {1,2}.
    - Returns shape (N, 2) float32.
    - Maintains state across blocks.

    Notes:
    - When input magnitude is near zero, we still decay the internal r state
      toward 0, but we always output zeros to avoid synthesizing sound.
    """
    if state is None:
        state = PhaseSpaceInertiaState()

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

    rmag = np.hypot(l, r)
    theta = np.arctan2(r, l)

    gamma = _gamma_from_tau(tau_seconds, sample_rate)

    for i in range(n):
        r_in = float(rmag[i])

        if r_in <= eps:
            # Treat as silence: decay r-state toward 0 but do not output.
            # Keep theta dynamics effectively frozen.
            state = phase_space_inertia_step(
                float(state.theta_m if state.initialized else 0.0),
                0.0,
                state,
                alpha=float(alpha),
                beta=float(beta),
                gamma=float(gamma),
            )
            continue

        state = phase_space_inertia_step(
            float(theta[i]),
            r_in,
            state,
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
        )

        out[i, 0] = float(state.r) * float(np.cos(state.theta))
        out[i, 1] = float(state.r) * float(np.sin(state.theta))

    return out, state


# Extensible model registry (GUI can populate from this).
StereoOfflineModelFn = Callable[[np.ndarray], np.ndarray]


MODEL_KEYS: Dict[str, str] = {
    # key -> user-visible label
    "inertial_attractor": "Inertial + Gravity (alpha/beta)",
    "phase_space_inertia": "Phase-Space Inertia (theta + radius)",
}
