from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.fixed_disturbance import fixed_shaker_disturbance, save_fixed_shaker_disturbance_csv


DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
COLLECTED_DATA_DIR = PROJECT_DIR / "hardware_stratified_protocol" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.03
M1 = 0.67
M2 = 0.85
K1 = 3000.0
K2 = 135.0
C1 = 0.0
C2 = 6.4
U_MAX = 2.0
PLANT_INTEGRATION_MAX_STEP = 0.003
B0_ADRC = 1.0 / M2

OBJECTIVE_WEIGHTS = {
    "x2": 180.0,
    "v2": 6.0,
    "a2": 2.0,
    "u": 0.04,
    "du": 0.08,
}


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


def clip_force(value: float) -> tuple[float, bool]:
    raw = float(value) if np.isfinite(float(value)) else 0.0
    sat = float(np.clip(raw, -U_MAX, U_MAX))
    return sat, bool(abs(raw - sat) > 1.0e-8)


def plant_dynamics(state: np.ndarray, u_cmd: float, disturbance_n: float) -> np.ndarray:
    x1, v1, x2, v2 = np.asarray(state, dtype=np.float64).reshape(4)
    rel_x = x2 - x1
    rel_v = v2 - v1
    force_k1 = K1 * x1
    force_k2 = K2 * rel_x
    damping_1 = C1 * v1
    damping_2 = C2 * rel_v
    a1 = (float(disturbance_n) - float(u_cmd) - force_k1 - damping_1 + force_k2 + damping_2) / M1
    a2 = (float(u_cmd) - force_k2 - damping_2) / M2
    return np.asarray([v1, a1, v2, a2], dtype=np.float64)


def rk4_step(state, u_cmd, disturbance_n, dt=DT):
    substeps = max(1, int(np.ceil(float(dt) / PLANT_INTEGRATION_MAX_STEP)))
    h = float(dt) / substeps
    x = np.asarray(state, dtype=np.float64).copy()
    for _ in range(substeps):
        k1 = plant_dynamics(x, u_cmd, disturbance_n)
        k2 = plant_dynamics(x + 0.5 * h * k1, u_cmd, disturbance_n)
        k3 = plant_dynamics(x + 0.5 * h * k2, u_cmd, disturbance_n)
        k4 = plant_dynamics(x + h * k3, u_cmd, disturbance_n)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def nominal_discrete_model(dt=DT):
    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-(K1 + K2) / M1, -(C1 + C2) / M1, K2 / M1, C2 / M1],
            [0.0, 0.0, 0.0, 1.0],
            [K2 / M2, C2 / M2, -K2 / M2, -C2 / M2],
        ],
        dtype=np.float64,
    )
    b_u = np.array([[0.0], [-1.0 / M1], [0.0], [1.0 / M2]], dtype=np.float64)
    b_d = np.array([[0.0], [1.0 / M1], [0.0], [0.0]], dtype=np.float64)
    inputs = np.hstack([b_u, b_d])
    block = np.zeros((a.shape[0] + inputs.shape[1], a.shape[1] + inputs.shape[1]), dtype=np.float64)
    block[: a.shape[0], : a.shape[1]] = a
    block[: a.shape[0], a.shape[1] :] = inputs
    disc = matrix_exponential(block * float(dt))
    return disc[:4, :4], disc[:4, 4:5], disc[:4, 5:6]


def build_prediction_matrices(ad, bd, ed, horizon):
    n = ad.shape[0]
    phi = np.zeros((horizon * n, n), dtype=np.float64)
    theta = np.zeros((horizon * n, horizon), dtype=np.float64)
    gamma = np.zeros((horizon * n, horizon), dtype=np.float64)
    powers = [np.eye(n, dtype=np.float64)]
    for _ in range(horizon):
        powers.append(powers[-1] @ ad)
    for i in range(horizon):
        phi[i * n : (i + 1) * n] = powers[i + 1]
        for j in range(i + 1):
            theta[i * n : (i + 1) * n, j : j + 1] = powers[i - j] @ bd
            gamma[i * n : (i + 1) * n, j : j + 1] = powers[i - j] @ ed
    return phi, theta, gamma


def build_delta_matrix(horizon):
    delta = np.eye(int(horizon), dtype=np.float64)
    for i in range(1, int(horizon)):
        delta[i, i - 1] = -1.0
    return delta


class BaseController:
    name = "base"

    def reset(self):
        pass

    def fallback_action(self, obs):
        return stabilizing_fallback_force(obs["state"])

    def act(self, obs):
        raise NotImplementedError


def stabilizing_fallback_force(state):
    x1, v1, x2, v2 = np.asarray(state, dtype=np.float64).reshape(4)
    rel_x = x1 - x2
    rel_v = v1 - v2
    u_raw = -135.0 * rel_x - C2 * rel_v - 18.0 * x2 - 0.6 * v2
    return clip_force(u_raw)[0]


class PassiveController(BaseController):
    name = "passive"

    def act(self, obs):
        return 0.0, {"u_raw": 0.0, "u_sat": 0.0}


class ADRCController(BaseController):
    name = "adrc"

    def __init__(self, omega_o, kp, kd, b0_scale=1.0):
        self.b0 = B0_ADRC * float(b0_scale)
        self.omega_o = float(omega_o)
        self.kp = float(kp)
        self.kd = float(kd)
        self.z = np.zeros(3, dtype=np.float64)
        self.initialized = False

    def reset(self):
        self.z[:] = 0.0
        self.initialized = False

    def act(self, obs):
        if abs(self.b0) < 1.0e-8 or not np.isfinite(self.b0):
            u = self.fallback_action(obs)
            return u, {"u_raw": u, "u_sat": u, "fallback_flag": 1}
        state = np.asarray(obs["state"], dtype=np.float64).reshape(4)
        x1, v1, x2, v2 = state
        if not self.initialized:
            self.z[:] = [x2, v2, 0.0]
            self.initialized = True
        f0 = (K2 * (x1 - x2) + C2 * (v1 - v2)) / M2
        beta1 = 3.0 * self.omega_o
        beta2 = 3.0 * self.omega_o**2
        beta3 = self.omega_o**3
        e = x2 - self.z[0]
        self.z += float(obs.get("dt", DT)) * np.asarray(
            [
                self.z[1] + beta1 * e,
                f0 + self.z[2] + self.b0 * float(obs["u_prev"]) + beta2 * e,
                beta3 * e,
            ],
            dtype=np.float64,
        )
        desired_acc = -self.kp * self.z[0] - self.kd * self.z[1]
        u_raw = (desired_acc - f0 - self.z[2]) / self.b0
        u_sat, saturation = clip_force(u_raw)
        return u_sat, {"u_raw": u_raw, "u_sat": u_sat, "saturation_flag": int(saturation)}


class MPCController(BaseController):
    name = "mpc"

    def __init__(self, horizon, q_x2, q_v2, r_u, r_du, ki_x2=0.0):
        self.horizon = int(horizon)
        self.q_x2 = float(q_x2)
        self.q_v2 = float(q_v2)
        self.r_u = float(r_u)
        self.r_du = float(r_du)
        self.ki_x2 = float(ki_x2)
        self.integral = 0.0
        self.integral_limit = 0.03
        self.ad, self.bd, self.ed = nominal_discrete_model(DT)
        self.phi, self.theta, self.gamma = build_prediction_matrices(self.ad, self.bd, self.ed, self.horizon)
        self.qbar = np.kron(np.eye(self.horizon), np.diag([0.0, 0.0, self.q_x2, self.q_v2]))
        self.delta = build_delta_matrix(self.horizon)
        self.prev_solution = np.zeros(self.horizon, dtype=np.float64)
        self.h_mat = self.theta.T @ self.qbar @ self.theta + self.r_u * np.eye(self.horizon) + self.r_du * (self.delta.T @ self.delta)
        self.h_mat = 0.5 * (self.h_mat + self.h_mat.T) + 1.0e-9 * np.eye(self.horizon)
        self.step_size = 1.0 / max(float(np.linalg.norm(self.h_mat, ord=2)), 1.0e-8)

    def reset(self):
        self.prev_solution[:] = 0.0
        self.integral = 0.0

    def act(self, obs):
        state = np.asarray(obs["state"], dtype=np.float64).reshape(4, 1)
        self.integral = float(np.clip(self.integral + DT * float(state[2, 0]), -self.integral_limit, self.integral_limit))
        d_forecast = np.full((self.horizon, 1), float(obs["disturbance"]), dtype=np.float64)
        x_free = self.phi @ state + self.gamma @ d_forecast
        first_delta_offset = np.zeros(self.horizon, dtype=np.float64)
        first_delta_offset[0] = -float(obs["u_prev"])
        f_vec = (self.theta.T @ self.qbar @ x_free).reshape(-1) + self.r_du * (self.delta.T @ first_delta_offset)
        u = np.concatenate([self.prev_solution[1:], self.prev_solution[-1:]])
        u = np.clip(u, -U_MAX, U_MAX)
        start = time.perf_counter()
        for _ in range(120):
            next_u = np.clip(u - self.step_size * (self.h_mat @ u + f_vec), -U_MAX, U_MAX)
            if np.linalg.norm(next_u - u, ord=np.inf) < 1.0e-8:
                u = next_u
                break
            u = next_u
        solve_ms = (time.perf_counter() - start) * 1000.0
        self.prev_solution = u
        u_raw = float(u[0] - self.ki_x2 * self.integral)
        u_sat, saturation = clip_force(u_raw)
        return u_sat, {"u_raw": u_raw, "u_sat": u_sat, "saturation_flag": int(saturation), "solve_time_ms": solve_ms}


def hankel_matrix(data, rows):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n, dim = data.shape
    cols = n - int(rows) + 1
    if cols <= 0:
        raise ValueError("Not enough samples for Hankel matrix.")
    out = np.zeros((int(rows) * dim, cols), dtype=np.float64)
    for col in range(cols):
        out[:, col] = data[col : col + int(rows)].reshape(-1)
    return out


def load_csv_deepc_hankel(data_dir, t_ini, n_f, max_columns):
    rows = int(t_ini) + int(n_f)
    u_blocks, d_blocks, y_blocks = [], [], []
    paths = sorted(Path(data_dir).glob("trajectory_*.csv"))
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        if len(records) < rows:
            continue
        y = np.asarray([[float(r["x1_filtered"]), float(r["v1_filtered"]), float(r["x2_filtered"]), float(r["v2_filtered"])] for r in records], dtype=np.float64)
        u = np.asarray([[float(r["voice_force_n"])] for r in records], dtype=np.float64)
        d = np.asarray([[float(r["shaker_force_n"])] for r in records], dtype=np.float64)
        u_blocks.append(hankel_matrix(u, rows))
        d_blocks.append(hankel_matrix(d, rows))
        y_blocks.append(hankel_matrix(y, rows))
    if not u_blocks:
        raise FileNotFoundError(f"No valid CSV DeePC trajectories found in {Path(data_dir).resolve()}")
    u_h = np.concatenate(u_blocks, axis=1)
    d_h = np.concatenate(d_blocks, axis=1)
    y_h = np.concatenate(y_blocks, axis=1)
    cols = min(int(max_columns), u_h.shape[1])
    idx = np.linspace(0, u_h.shape[1] - 1, cols).round().astype(int)
    return u_h[:, idx], d_h[:, idx], y_h[:, idx]


def generate_nominal_deepc_hankel(t_ini, n_f, max_columns, seed=20260516):
    rng = np.random.default_rng(int(seed))
    rows = int(t_ini) + int(n_f)
    total_steps = int(max_columns) + rows + 220
    t = np.arange(total_steps, dtype=np.float64) * DT
    u = np.zeros(total_steps)
    d = np.zeros(total_steps)
    for freq in (0.35, 0.8, 1.7, 2.9, 4.5):
        u += rng.uniform(0.15, 0.75) * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
    for freq in (0.25, 0.9, 2.1, 3.8):
        d += rng.uniform(1.0, 4.5) * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
    u = np.clip(u + rng.normal(0.0, 0.12, size=total_steps), -U_MAX, U_MAX)
    d = np.clip(d + rng.normal(0.0, 0.25, size=total_steps), -9.0, 9.0)
    states = np.zeros((total_steps, 4), dtype=np.float64)
    for k in range(total_steps - 1):
        states[k + 1] = rk4_step(states[k], u[k], d[k], DT)
    u_h = hankel_matrix(u[:, None], rows)
    d_h = hankel_matrix(d[:, None], rows)
    y_h = hankel_matrix(states, rows)
    cols = min(int(max_columns), u_h.shape[1])
    idx = np.linspace(0, u_h.shape[1] - 1, cols).round().astype(int)
    return u_h[:, idx], d_h[:, idx], y_h[:, idx]


class DeePCController(BaseController):
    name = "deepc"

    def __init__(self, params, data_dir=COLLECTED_DATA_DIR):
        self.params = dict(params)
        self.available = False
        self.init_error = ""
        try:
            if str(params.get("dictionary_source", "measured_csv")) == "measured_csv":
                self.u_h, self.d_h, self.y_h = load_csv_deepc_hankel(data_dir, params["t_ini"], params["n_f"], params["max_columns"])
                self.dictionary_source = "measured_csv"
            else:
                self.u_h, self.d_h, self.y_h = generate_nominal_deepc_hankel(params["t_ini"], params["n_f"], params["max_columns"])
                self.dictionary_source = "nominal_model"
            t_ini = int(params["t_ini"])
            n_f = int(params["n_f"])
            self.u_p = self.u_h[:t_ini]
            self.u_f = self.u_h[t_ini : t_ini + n_f]
            self.d_p = self.d_h[:t_ini]
            self.d_f = self.d_h[t_ini : t_ini + n_f]
            self.y_p = self.y_h[: 4 * t_ini]
            self.y_f = self.y_h[4 * t_ini : 4 * (t_ini + n_f)]
            self.cols = self.u_h.shape[1]
            self.available = True
        except Exception as exc:
            self.init_error = f"{type(exc).__name__}: {exc}"
            self.cols = 0
        self.reset()

    def reset(self):
        t_ini = int(self.params["t_ini"])
        self.u_history = [0.0] * t_ini
        self.d_history = [0.0] * t_ini
        self.y_history = [np.zeros(4, dtype=np.float64) for _ in range(t_ini)]

    def act(self, obs):
        self.u_history.append(float(obs["u_prev"]))
        self.d_history.append(float(obs["disturbance"]))
        self.y_history.append(np.asarray(obs["state"], dtype=np.float64).reshape(4))
        t_ini = int(self.params["t_ini"])
        self.u_history = self.u_history[-t_ini:]
        self.d_history = self.d_history[-t_ini:]
        self.y_history = self.y_history[-t_ini:]
        if not self.available:
            u = self.fallback_action(obs)
            return u, {"u_raw": u, "u_sat": u, "fallback_flag": 1}
        start = time.perf_counter()
        try:
            u_seq = self.solve(obs)
            alpha = float(self.params.get("action_alpha", 0.65))
            u_raw = alpha * float(u_seq[0]) + (1.0 - alpha) * float(obs["u_prev"])
            fallback = 0
        except Exception:
            u_raw = self.fallback_action(obs)
            fallback = 1
        solve_ms = (time.perf_counter() - start) * 1000.0
        u_sat, saturation = clip_force(u_raw)
        return u_sat, {"u_raw": u_raw, "u_sat": u_sat, "fallback_flag": fallback, "saturation_flag": int(saturation), "solve_time_ms": solve_ms}

    def solve(self, obs):
        n_f = int(self.params["n_f"])
        h = float(self.params["lambda_g"]) * np.eye(self.cols)
        b = np.zeros(self.cols)

        def add(matrix, target, weight):
            nonlocal h, b
            matrix = np.asarray(matrix, dtype=np.float64)
            target = np.asarray(target, dtype=np.float64).reshape(-1)
            h += float(weight) * (matrix.T @ matrix)
            b += float(weight) * (matrix.T @ target)

        add(self.y_f[2::4], np.zeros(n_f), self.params["q_x2"])
        add(self.y_f[3::4], np.zeros(n_f), self.params["q_v2"])
        add(self.u_f, np.zeros(n_f), self.params["r_u"])
        delta = build_delta_matrix(n_f)
        du_target = np.zeros(n_f)
        du_target[0] = float(obs["u_prev"])
        add(delta @ self.u_f, du_target, self.params["r_du"])
        add(self.u_p, np.asarray(self.u_history), self.params["lambda_uini"])
        add(self.d_p, np.asarray(self.d_history), self.params["lambda_dini"])
        add(self.y_p, np.asarray(self.y_history).reshape(-1), self.params["lambda_y"])
        add(self.d_f, np.full(n_f, float(obs["disturbance"])), self.params["lambda_df"])
        h = 0.5 * (h + h.T) + 1.0e-8 * np.eye(self.cols)
        try:
            g = np.linalg.solve(h, b)
        except np.linalg.LinAlgError:
            g = np.linalg.lstsq(h, b, rcond=None)[0]
        return np.clip((self.u_f @ g).reshape(-1), -U_MAX, U_MAX)


def make_controller(params):
    method = params["method"]
    if method == "passive":
        return PassiveController()
    if method == "adrc":
        return ADRCController(params["omega_o"], params["kp"], params["kd"], params.get("b0_scale", 1.0))
    if method == "mpc":
        return MPCController(params["horizon"], params["q_x2"], params["q_v2"], params["r_u"], params["r_du"], params.get("ki_x2", 0.0))
    if method == "deepc":
        return DeePCController(params)
    raise ValueError(method)


def candidate_variants(seed=20260516):
    rng = np.random.default_rng(seed)
    variants = [{"method": "passive", "variant": "passive"}]
    for omega_o in (3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
        for kp, kd in (
            (3.5, 0.6),
            (4.0, 0.8),
            (6.0, 1.2),
            (8.0, 1.8),
            (10.0, 2.2),
            (12.0, 2.5),
            (16.0, 3.5),
            (20.0, 4.2),
            (24.0, 5.0),
            (30.0, 6.0),
        ):
            for b0_scale in (0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5):
                variants.append({"method": "adrc", "variant": f"adrc_o{omega_o:g}_kp{kp:g}_kd{kd:g}_b{b0_scale:g}", "omega_o": omega_o, "kp": kp, "kd": kd, "b0_scale": b0_scale})
    for idx in range(72):
        variants.append(
            {
                "method": "mpc",
                "variant": f"mpc_rand_{idx + 1:02d}",
                "horizon": int(rng.choice([3, 4, 5, 6, 8, 10, 12, 14])),
                "q_x2": float(10.0 ** rng.uniform(np.log10(2.5e4), np.log10(8.0e5))),
                "q_v2": float(10.0 ** rng.uniform(np.log10(4.0e2), np.log10(2.0e4))),
                "r_u": float(10.0 ** rng.uniform(np.log10(2.0e-3), np.log10(1.5e-1))),
                "r_du": float(10.0 ** rng.uniform(np.log10(1.0e-2), np.log10(1.2))),
                "ki_x2": 0.0,
            }
        )
    for idx in range(54):
        variants.append(
            {
                "method": "deepc",
                "variant": f"deepc_rand_{idx + 1:02d}",
                "dictionary_source": "measured_csv",
                "t_ini": int(rng.choice([3, 4, 5, 6, 7, 8])),
                "n_f": int(rng.choice([3, 4, 5, 6, 8, 10])),
                "max_columns": int(rng.choice([32, 48, 64, 80, 100, 120])),
                "lambda_g": float(10.0 ** rng.uniform(-3.2, -0.35)),
                "lambda_y": float(10.0 ** rng.uniform(3.6, 5.7)),
                "lambda_uini": float(10.0 ** rng.uniform(3.6, 5.6)),
                "lambda_dini": float(10.0 ** rng.uniform(2.8, 4.8)),
                "lambda_df": float(10.0 ** rng.uniform(2.8, 4.8)),
                "q_x2": float(10.0 ** rng.uniform(4.3, 5.8)),
                "q_v2": float(10.0 ** rng.uniform(2.6, 4.3)),
                "r_u": float(10.0 ** rng.uniform(-2.8, -0.65)),
                "r_du": float(10.0 ** rng.uniform(-1.8, -0.05)),
                "action_alpha": float(rng.uniform(0.30, 0.95)),
            }
        )
    # Include hardware-comparison-like hand candidates explicitly.
    variants.extend(
        [
            {"method": "adrc", "variant": "adrc_hardware_like_o12_kp24_kd5_b1", "omega_o": 12.0, "kp": 24.0, "kd": 5.0, "b0_scale": 1.0},
            {"method": "deepc", "variant": "deepc_hardware_like_t5_n5_c60", "dictionary_source": "measured_csv", "t_ini": 5, "n_f": 5, "max_columns": 60, "lambda_g": 2.0e-2, "lambda_y": 8.0e4, "lambda_uini": 3.0e4, "lambda_dini": 8.0e3, "lambda_df": 8.0e3, "q_x2": 1.5e5, "q_v2": 4.0e3, "r_u": 2.0e-2, "r_du": 2.5e-1, "action_alpha": 0.60},
        ]
    )
    return variants


def simulate(controller, time_s, disturbance):
    states = np.zeros((len(time_s), 4), dtype=np.float64)
    u = np.zeros(len(time_s), dtype=np.float64)
    fallback = np.zeros(len(time_s), dtype=np.int32)
    saturation = np.zeros(len(time_s), dtype=np.int32)
    compute_ms = np.zeros(len(time_s), dtype=np.float64)
    controller.reset()
    prev_u = 0.0
    for i in range(len(time_s) - 1):
        obs = {"state": states[i].copy(), "dt": DT, "u_prev": prev_u, "disturbance": float(disturbance[i]), "time": float(time_s[i]), "step_idx": i}
        u_i, info = controller.act(obs)
        u_i = float(np.clip(u_i, -U_MAX, U_MAX))
        u[i] = u_i
        fallback[i] = int(info.get("fallback_flag", 0))
        saturation[i] = int(info.get("saturation_flag", 0))
        compute_ms[i] = float(info.get("solve_time_ms", 0.0))
        states[i + 1] = rk4_step(states[i], u_i, disturbance[i], DT)
        prev_u = u_i
    u[-1] = u[-2] if len(u) > 1 else 0.0
    return {"time": time_s, "states": states, "u": u, "disturbance": disturbance, "fallback": fallback, "saturation": saturation, "compute_ms": compute_ms}


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def metrics(rollout):
    states = rollout["states"]
    u = rollout["u"]
    a2 = np.zeros(states.shape[0], dtype=np.float64)
    a2[1:] = np.diff(states[:, 3]) / DT
    du = np.zeros_like(u)
    du[1:] = np.diff(u)
    cost = (
        OBJECTIVE_WEIGHTS["x2"] * states[:, 2] ** 2
        + OBJECTIVE_WEIGHTS["v2"] * states[:, 3] ** 2
        + OBJECTIVE_WEIGHTS["a2"] * a2**2
        + OBJECTIVE_WEIGHTS["u"] * u**2
        + OBJECTIVE_WEIGHTS["du"] * du**2
    )
    return {
        "objective": float(np.mean(cost)),
        "x2_rms": rms(states[:, 2]),
        "v2_rms": rms(states[:, 3]),
        "a2_rms": rms(a2),
        "u_rms": rms(u),
        "u_max_abs": float(np.max(np.abs(u))),
        "fallback_count": int(np.sum(rollout["fallback"] > 0)),
        "saturation_count": int(np.sum(rollout["saturation"] > 0)),
        "mean_compute_ms": float(np.mean(rollout["compute_ms"])),
        "max_compute_ms": float(np.max(rollout["compute_ms"])),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_by_method(rows):
    best = {}
    for row in rows:
        method = row["method"]
        if method not in best or float(row["objective"]) < float(best[method]["objective"]):
            best[method] = row
    return [best[key] for key in ("passive", "adrc", "mpc", "deepc") if key in best]


def rollout_rows(method, variant, rollout):
    states = rollout["states"]
    u = rollout["u"]
    d = rollout["disturbance"]
    time_s = rollout["time"]
    a2 = np.zeros(states.shape[0], dtype=np.float64)
    a2[1:] = np.diff(states[:, 3]) / DT
    rows = []
    for idx, t in enumerate(time_s):
        rows.append(
            {
                "method": method,
                "variant": variant,
                "sample": idx,
                "time": float(t),
                "x1": float(states[idx, 0]),
                "v1": float(states[idx, 1]),
                "x2": float(states[idx, 2]),
                "v2": float(states[idx, 3]),
                "a2": float(a2[idx]),
                "u": float(u[idx]),
                "shaker_force_n": float(d[idx]),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Tune baseline controller parameters on the ideal comfort-seat plant.")
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-variants", type=int, default=0, help="Optional cap for quick smoke tests; 0 means all candidates.")
    return parser.parse_args()


def main():
    args = parse_args()
    time_s, disturbance = fixed_shaker_disturbance(duration_s=args.duration, dt=DT, seed=args.seed)
    save_fixed_shaker_disturbance_csv(DATA_DIR / "fixed_shaker_disturbance.csv", time_s, disturbance)
    variants = candidate_variants(args.seed)
    if args.max_variants > 0:
        variants = variants[: args.max_variants]
    rows = []
    rollouts = {}
    for idx, params in enumerate(variants, start=1):
        print(f"[{idx:03d}/{len(variants):03d}] {params['method']}::{params['variant']}")
        controller = make_controller(params)
        rollout = simulate(controller, time_s, disturbance)
        row = {"method": params["method"], "variant": params["variant"], **metrics(rollout)}
        row.update({f"param_{key}": value for key, value in params.items() if key not in {"method", "variant"}})
        rows.append(row)
        rollouts[params["variant"]] = rollout
    rows.sort(key=lambda item: float(item["objective"]))
    best_rows = best_by_method(rows)
    write_csv(DATA_DIR / "tuning_summary.csv", rows)
    write_csv(DATA_DIR / "best_by_method.csv", best_rows)
    comparison = []
    for row in best_rows:
        comparison.extend(rollout_rows(row["method"], row["variant"], rollouts[row["variant"]]))
    write_csv(DATA_DIR / "comparison_rollouts.csv", comparison)
    print("Best variants:")
    for row in best_rows:
        print(f"  {row['method']:>7s}: {row['variant']} objective={float(row['objective']):.6g}")
    print(f"Saved CSV results to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
