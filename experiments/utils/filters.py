from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CUTOFF_HZ = 8.0


def lowpass_alpha(dt: float, cutoff_hz: float = CUTOFF_HZ) -> float:
    dt = max(float(dt), 1.0e-9)
    tau = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    return dt / (tau + dt)


@dataclass
class FirstOrderLowPass:
    cutoff_hz: float = CUTOFF_HZ
    value: np.ndarray | None = None

    def reset(self, value) -> np.ndarray:
        self.value = np.asarray(value, dtype=np.float64).copy()
        return self.value.copy()

    def update(self, sample, dt: float) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float64)
        if self.value is None:
            return self.reset(sample)
        alpha = lowpass_alpha(dt, self.cutoff_hz)
        self.value = self.value + alpha * (sample - self.value)
        return self.value.copy()


class OnlineKinematicsFilter:
    """Causal displacement, velocity, and acceleration reconstruction."""

    def __init__(self, cutoff_hz: float = CUTOFF_HZ, dimensions: int = 2):
        self.cutoff_hz = float(cutoff_hz)
        self.dimensions = int(dimensions)
        self.x_filter = FirstOrderLowPass(cutoff_hz)
        self.v_filter = FirstOrderLowPass(cutoff_hz)
        self.a_filter = FirstOrderLowPass(cutoff_hz)
        self.prev_t: float | None = None
        self.prev_x = np.zeros(self.dimensions, dtype=np.float64)
        self.prev_v = np.zeros(self.dimensions, dtype=np.float64)

    def reset(self, displacement=None, timestamp: float | None = None) -> None:
        if displacement is None:
            displacement = np.zeros(self.dimensions, dtype=np.float64)
        x = np.asarray(displacement, dtype=np.float64).reshape(self.dimensions)
        self.x_filter.reset(x)
        self.v_filter.reset(np.zeros(self.dimensions, dtype=np.float64))
        self.a_filter.reset(np.zeros(self.dimensions, dtype=np.float64))
        self.prev_t = timestamp
        self.prev_x = x.copy()
        self.prev_v = np.zeros(self.dimensions, dtype=np.float64)

    def update(self, displacement, timestamp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        x_raw = np.asarray(displacement, dtype=np.float64).reshape(self.dimensions)
        t = float(timestamp)
        if self.prev_t is None:
            self.reset(x_raw, t)
            return self.prev_x.copy(), self.prev_v.copy(), np.zeros(self.dimensions), 0.0

        dt = max(t - self.prev_t, 1.0e-6)
        x_f = self.x_filter.update(x_raw, dt)
        v_raw = (x_f - self.prev_x) / dt
        v_f = self.v_filter.update(v_raw, dt)
        a_raw = (v_f - self.prev_v) / dt
        a_f = self.a_filter.update(a_raw, dt)

        self.prev_t = t
        self.prev_x = x_f.copy()
        self.prev_v = v_f.copy()
        return x_f, v_f, a_f, dt


def filter_displacement_series(time_s, displacement, cutoff_hz: float = CUTOFF_HZ):
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.ndim == 1:
        displacement = displacement[:, None]
    filt = OnlineKinematicsFilter(cutoff_hz=cutoff_hz, dimensions=displacement.shape[1])
    x_out = np.zeros_like(displacement, dtype=np.float64)
    v_out = np.zeros_like(displacement, dtype=np.float64)
    a_out = np.zeros_like(displacement, dtype=np.float64)
    for idx, (t, x) in enumerate(zip(time_s, displacement)):
        if idx == 0:
            filt.reset(x, float(t))
            x_f = np.asarray(x, dtype=np.float64)
            v_f = np.zeros(displacement.shape[1], dtype=np.float64)
            a_f = np.zeros(displacement.shape[1], dtype=np.float64)
        else:
            x_f, v_f, a_f, _ = filt.update(x, float(t))
        x_out[idx] = x_f
        v_out[idx] = v_f
        a_out[idx] = a_f
    return x_out.squeeze(), v_out.squeeze(), a_out.squeeze()
