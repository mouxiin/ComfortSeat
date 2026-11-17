from __future__ import annotations

from pathlib import Path

import numpy as np

from .signals import DT, SHAKER_FORCE_LIMIT_N, scale_to_limit, smooth_start_stop


DEFAULT_FIXED_DISTURBANCE_SEED = 20260516
DEFAULT_FIXED_DISTURBANCE_DURATION_S = 9.0
TARGET_PEAK_FRACTION = 0.92


def _unit_chirp(time_s: np.ndarray, f0: float, f1: float, phase0: float) -> np.ndarray:
    total_time = max(float(time_s[-1] - time_s[0]), 1.0e-6)
    k = (float(f1) - float(f0)) / total_time
    phase = 2.0 * np.pi * (float(f0) * time_s + 0.5 * k * time_s * time_s) + float(phase0)
    return np.sin(phase)


def fixed_shaker_disturbance(
    duration_s: float = DEFAULT_FIXED_DISTURBANCE_DURATION_S,
    dt: float = DT,
    seed: int = DEFAULT_FIXED_DISTURBANCE_SEED,
    limit_n: float = SHAKER_FORCE_LIMIT_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed, reusable complex shaker-side force disturbance.

    See ``utils/fixed_disturbance.md`` for the single reference description of
    the waveform, including component weights, ramping, and limits.
    """

    steps = int(round(float(duration_s) / float(dt)))
    time_s = np.arange(steps + 1, dtype=np.float64) * float(dt)
    rng = np.random.default_rng(int(seed))
    tone_freqs = np.asarray([0.75, 1.25, 1.85, 2.65, 3.55, 4.65], dtype=np.float64)
    tone_weights = np.asarray([0.35, 0.50, 0.65, 0.60, 0.45, 0.30], dtype=np.float64)
    tone_phases = rng.uniform(0.0, 2.0 * np.pi, size=tone_freqs.size)

    multisine_part = np.zeros(steps + 1, dtype=np.float64)
    for weight, freq, phase in zip(tone_weights, tone_freqs, tone_phases):
        multisine_part += float(weight) * np.sin(2.0 * np.pi * float(freq) * time_s + float(phase))

    up_chirp = _unit_chirp(time_s, 0.70, 5.00, rng.uniform(0.0, 2.0 * np.pi))
    down_chirp = _unit_chirp(time_s, 5.00, 0.70, rng.uniform(0.0, 2.0 * np.pi))
    amplitude_modulation = 1.0 + 0.18 * np.sin(2.0 * np.pi * 0.35 * time_s + rng.uniform(0.0, 2.0 * np.pi))
    signal = amplitude_modulation * (0.58 * multisine_part + 0.28 * up_chirp + 0.22 * down_chirp)

    # Smooth start/stop so the hardware command begins at exactly zero force.
    signal = smooth_start_stop(signal, dt=dt, ramp_s=min(0.6, 0.12 * float(duration_s))).astype(np.float64)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs > 0.0:
        signal = signal * (TARGET_PEAK_FRACTION * float(limit_n) / max_abs)
    signal = scale_to_limit(signal, float(limit_n)).astype(np.float64)
    signal[0] = 0.0
    signal[-1] = 0.0
    return time_s, signal


def save_fixed_shaker_disturbance_csv(path: Path, time_s: np.ndarray, force_n: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("sample,time,shaker_force_n\n")
        for idx, (t, d) in enumerate(zip(time_s, force_n)):
            f.write(f"{idx},{float(t)},{float(d)}\n")
