from __future__ import annotations

import os
import sys
import csv
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.filters import filter_displacement_series


DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

M1 = 0.67
M2 = 0.85
FREQ_HZ = 5.0
OMEGA = 2.0 * np.pi * FREQ_HZ
MOTOR_TIME_CONSTANT = 0.010
ACTUATOR_FORCE_LIMIT = 2.0
NONLINEARITY_RATIO = 0.05
NONLINEARITY_DEFLECTION = 5.0e-3

K1 = 3000.0
K2 = 135.0
C1 = 0.0
C2 = 6.4
K1_CUBIC = NONLINEARITY_RATIO * K1 / (NONLINEARITY_DEFLECTION**2)
K2_CUBIC = NONLINEARITY_RATIO * K2 / (NONLINEARITY_DEFLECTION**2)

DT = 0.03
PLANT_INTEGRATION_MAX_STEP = 0.003
SIM_TIME = 14.0
DIST_AMP_1 = 30.0
DIST_AMP_2 = 0.0
DIST_PHASE_1 = 0.0
DIST_PHASE_2 = np.pi / 3.0

MPC_HORIZON = 30
Q_STATE = np.diag([1.0e3, 1.0e1, 1.0e10, 1.0e7])
R_INPUT = np.array([[1.0e-8]])


def matrix_exponential(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    norm = float(np.linalg.norm(a, ord=np.inf))
    scale = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    a_scaled = a / (2**scale)
    result = np.eye(a.shape[0], dtype=np.float64)
    term = np.eye(a.shape[0], dtype=np.float64)
    for k in range(1, 40):
        term = term @ a_scaled / float(k)
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1.0e-14:
            break
    for _ in range(scale):
        result = result @ result
    return result


def disturbance_forces(t: float) -> np.ndarray:
    d1 = DIST_AMP_1 * np.sin(OMEGA * t + DIST_PHASE_1)
    d2 = DIST_AMP_2 * np.sin(OMEGA * t + DIST_PHASE_2)
    return np.array([d1, d2], dtype=np.float64)


def spring_force(deflection, k_linear, k_cubic, enable_nonlinearity):
    force = k_linear * deflection
    if enable_nonlinearity:
        force += k_cubic * deflection**3
    return force


def nominal_matrices():
    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(K1 + K2) / M1, -(C1 + C2) / M1, K2 / M1, C2 / M1],
            [0.0, 0.0, 0.0, 1.0],
            [K2 / M2, C2 / M2, -K2 / M2, -C2 / M2],
        ],
        dtype=np.float64,
    )
    b = np.array([[0.0], [0.0], [0.0], [1.0 / M2]], dtype=np.float64)
    e = np.array(
        [[0.0, 0.0], [1.0 / M1, 0.0], [0.0, 0.0], [0.0, 1.0 / M2]],
        dtype=np.float64,
    )
    continuous_input = np.hstack((b, e))
    block = np.zeros((a.shape[0] + continuous_input.shape[1], a.shape[1] + continuous_input.shape[1]))
    block[: a.shape[0], : a.shape[1]] = a
    block[: a.shape[0], a.shape[1] :] = continuous_input
    discretized = matrix_exponential(block * DT)
    ad = discretized[: a.shape[0], : a.shape[1]]
    bd = discretized[: a.shape[0], a.shape[1] : a.shape[1] + b.shape[1]]
    ed = discretized[: a.shape[0], a.shape[1] + b.shape[1] :]
    return ad, bd, ed


def build_prediction_matrices(ad, bd, ed, horizon):
    n_state = ad.shape[0]
    n_input = bd.shape[1]
    n_dist = ed.shape[1]
    phi = np.zeros((horizon * n_state, n_state), dtype=np.float64)
    theta = np.zeros((horizon * n_state, horizon * n_input), dtype=np.float64)
    gamma = np.zeros((horizon * n_state, horizon * n_dist), dtype=np.float64)
    ad_powers = [np.eye(n_state, dtype=np.float64)]
    for _ in range(horizon):
        ad_powers.append(ad_powers[-1] @ ad)
    for i in range(horizon):
        phi[i * n_state : (i + 1) * n_state] = ad_powers[i + 1]
        for j in range(i + 1):
            theta[i * n_state : (i + 1) * n_state, j * n_input : (j + 1) * n_input] = ad_powers[i - j] @ bd
            gamma[i * n_state : (i + 1) * n_state, j * n_dist : (j + 1) * n_dist] = ad_powers[i - j] @ ed
    qbar = np.kron(np.eye(horizon), Q_STATE)
    rbar = np.kron(np.eye(horizon), R_INPUT)
    h_mat = theta.T @ qbar @ theta + rbar
    return phi, theta, gamma, qbar, h_mat


class NominalMPC:
    def __init__(self, horizon=MPC_HORIZON):
        self.ad, self.bd, self.ed = nominal_matrices()
        self.horizon = int(horizon)
        self.phi, self.theta, self.gamma, self.qbar, self.h_mat = build_prediction_matrices(
            self.ad, self.bd, self.ed, self.horizon
        )
        self.control_bounds = [(-ACTUATOR_FORCE_LIMIT, ACTUATOR_FORCE_LIMIT)] * self.horizon
        self.warm_start = np.zeros(self.horizon, dtype=np.float64)
        h_norm = float(np.linalg.norm(self.h_mat, ord=2))
        self.step_size = 1.0 / max(h_norm, 1.0e-9)

    def solve_box_qp(self, linear_term):
        u = np.concatenate((self.warm_start[1:], self.warm_start[-1:]))
        u = np.clip(u, -ACTUATOR_FORCE_LIMIT, ACTUATOR_FORCE_LIMIT)
        for _ in range(250):
            grad = self.h_mat @ u + linear_term
            next_u = np.clip(u - self.step_size * grad, -ACTUATOR_FORCE_LIMIT, ACTUATOR_FORCE_LIMIT)
            if np.linalg.norm(next_u - u, ord=np.inf) < 1.0e-8:
                u = next_u
                break
            u = next_u
        return u

    def compute_control(self, state, disturbance_preview):
        d_stack = disturbance_preview.reshape(-1, 1)
        x0 = state.reshape(-1, 1)
        stacked_prediction = self.phi @ x0 + self.gamma @ d_stack
        linear_term = (self.theta.T @ self.qbar @ stacked_prediction).ravel()
        self.warm_start = self.solve_box_qp(linear_term)
        return float(self.warm_start[0])


def plant_dynamics(state, u_cmd, disturbance, *, include_nonlinearity, include_motor):
    x1, v1, x2, v2, u_act = state
    d1, d2 = disturbance
    if include_motor:
        du_act = (u_cmd - u_act) / MOTOR_TIME_CONSTANT
        u_eff = u_act
    else:
        du_act = 0.0
        u_eff = u_cmd
    rel_x = x2 - x1
    rel_v = v2 - v1
    force_k1 = spring_force(x1, K1, K1_CUBIC, include_nonlinearity)
    force_k2 = spring_force(rel_x, K2, K2_CUBIC, include_nonlinearity)
    damping_1 = C1 * v1
    damping_2 = C2 * rel_v
    a1 = (d1 - force_k1 - damping_1 + force_k2 + damping_2) / M1
    a2 = (d2 + u_eff - force_k2 - damping_2) / M2
    return np.array([v1, a1, v2, a2, du_act], dtype=np.float64)


def rk4_step(state, u_cmd, disturbance, dt, **kwargs):
    substeps = max(1, int(np.ceil(dt / PLANT_INTEGRATION_MAX_STEP)))
    step_dt = dt / substeps
    next_state = state.copy()
    for _ in range(substeps):
        k1 = plant_dynamics(next_state, u_cmd, disturbance, **kwargs)
        k2 = plant_dynamics(next_state + 0.5 * step_dt * k1, u_cmd, disturbance, **kwargs)
        k3 = plant_dynamics(next_state + 0.5 * step_dt * k2, u_cmd, disturbance, **kwargs)
        k4 = plant_dynamics(next_state + step_dt * k3, u_cmd, disturbance, **kwargs)
        next_state = next_state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state


def simulate_closed_loop(name, controller, *, include_nonlinearity, include_drive_delay, include_motor):
    time_s = np.arange(0.0, SIM_TIME + DT, DT)
    states = np.zeros((time_s.size, 5), dtype=np.float64)
    controls = np.zeros(time_s.size, dtype=np.float64)
    disturbances = np.array([disturbance_forces(t) for t in time_s], dtype=np.float64)
    for i in range(time_s.size - 1):
        horizon_indices = np.minimum(np.arange(i, i + controller.horizon), time_s.size - 1)
        controls[i] = np.clip(
            controller.compute_control(states[i, :4], disturbances[horizon_indices]),
            -ACTUATOR_FORCE_LIMIT,
            ACTUATOR_FORCE_LIMIT,
        )
        plant_control = controls[i - 1] if include_drive_delay and i > 0 else controls[i]
        states[i + 1] = rk4_step(
            states[i],
            plant_control,
            disturbances[i],
            DT,
            include_nonlinearity=include_nonlinearity,
            include_motor=include_motor,
        )
    controls[-1] = controls[-2]
    x2_f, v2_f, a2_f = filter_displacement_series(time_s, states[:, 2], cutoff_hz=8.0)
    return {
        "name": name,
        "time": time_s,
        "states": states[:, :4],
        "u": controls,
        "disturbance": disturbances,
        "x2_filtered": x2_f,
        "v2_filtered": v2_f,
        "a2_filtered": a2_f,
    }


def ideal_reference_case(time_s):
    zeros = np.zeros_like(time_s)
    return {
        "name": "Ideal plant",
        "time": time_s,
        "states": np.zeros((time_s.size, 4), dtype=np.float64),
        "u": zeros.copy(),
        "disturbance": np.zeros((time_s.size, 2), dtype=np.float64),
        "x2_filtered": zeros.copy(),
        "v2_filtered": zeros.copy(),
        "a2_filtered": zeros.copy(),
    }


def main() -> Path:
    cases = [
        simulate_closed_loop(
            "Variant I",
            NominalMPC(),
            include_nonlinearity=True,
            include_drive_delay=False,
            include_motor=False,
        ),
        simulate_closed_loop(
            "Variant II",
            NominalMPC(),
            include_nonlinearity=False,
            include_drive_delay=True,
            include_motor=False,
        ),
        simulate_closed_loop(
            "Variant III",
            NominalMPC(),
            include_nonlinearity=False,
            include_drive_delay=False,
            include_motor=True,
        ),
        simulate_closed_loop(
            "Variant IV",
            NominalMPC(),
            include_nonlinearity=True,
            include_drive_delay=True,
            include_motor=True,
        ),
    ]
    cases.insert(0, ideal_reference_case(cases[0]["time"]))

    out_path = DATA_DIR / "mpc_degradation.csv"
    fields = [
        "case",
        "sample",
        "time",
        "x1",
        "v1_raw",
        "x2_raw",
        "v2_raw",
        "u",
        "disturbance_1",
        "disturbance_2",
        "x2_filtered",
        "v2_filtered",
        "a2_filtered",
        "cutoff_hz",
        "dt_nominal",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for sample, t in enumerate(case["time"]):
                writer.writerow(
                    {
                        "case": case["name"],
                        "sample": sample,
                        "time": float(t),
                        "x1": float(case["states"][sample, 0]),
                        "v1_raw": float(case["states"][sample, 1]),
                        "x2_raw": float(case["states"][sample, 2]),
                        "v2_raw": float(case["states"][sample, 3]),
                        "u": float(case["u"][sample]),
                        "disturbance_1": float(case["disturbance"][sample, 0]),
                        "disturbance_2": float(case["disturbance"][sample, 1]),
                        "x2_filtered": float(case["x2_filtered"][sample]),
                        "v2_filtered": float(case["v2_filtered"][sample]),
                        "a2_filtered": float(case["a2_filtered"][sample]),
                        "cutoff_hz": 8.0,
                        "dt_nominal": DT,
                    }
                )
    print(f"Saved data: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    main()
