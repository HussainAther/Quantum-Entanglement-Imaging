"""Time-dependent loading histories for tensegrity experiments.

The functions here are intentionally unit-agnostic. Experiments may use
physical SI units or normalized quantities, as long as they are consistent.
"""

from __future__ import annotations


def linear_ramp(time: float, final_value: float, ramp_time: float) -> float:
    """Return a linear ramp from zero to ``final_value``.

    The load reaches ``final_value`` at ``ramp_time`` and is then held there.
    """
    if ramp_time <= 0.0:
        raise ValueError("ramp_time must be positive")
    if time <= 0.0:
        return 0.0
    if time >= ramp_time:
        return float(final_value)
    return float(final_value * time / ramp_time)


def linear_ramp_rate(time: float, final_value: float, ramp_time: float) -> float:
    """Return d(load)/dt for :func:`linear_ramp`."""
    if ramp_time <= 0.0:
        raise ValueError("ramp_time must be positive")
    if 0.0 < time < ramp_time:
        return float(final_value / ramp_time)
    return 0.0


def smoothstep_ramp(time: float, final_value: float, ramp_time: float) -> float:
    """Return a C1-smooth cubic ramp followed by a hold.

    Uses ``s(s) = 3 s^2 - 2 s^3`` for normalized time ``s`` in [0, 1].
    """
    if ramp_time <= 0.0:
        raise ValueError("ramp_time must be positive")
    if time <= 0.0:
        return 0.0
    if time >= ramp_time:
        return float(final_value)
    s = time / ramp_time
    return float(final_value * (3.0 * s * s - 2.0 * s * s * s))


def smoothstep_ramp_rate(time: float, final_value: float, ramp_time: float) -> float:
    """Return d(load)/dt for :func:`smoothstep_ramp`."""
    if ramp_time <= 0.0:
        raise ValueError("ramp_time must be positive")
    if not (0.0 < time < ramp_time):
        return 0.0
    s = time / ramp_time
    return float(final_value * (6.0 * s - 6.0 * s * s) / ramp_time)
