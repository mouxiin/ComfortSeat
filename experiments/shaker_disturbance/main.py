from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.filters import CUTOFF_HZ, OnlineKinematicsFilter
from utils.fixed_disturbance import fixed_shaker_disturbance, save_fixed_shaker_disturbance_csv
from utils.hardware_io import (
    close_devices,
    finish_hardware_trial,
    import_hardware_module,
    prepare_hardware_for_trial,
    quiet_put_voice_coil_force,
    shaker_force_to_input_voltage,
    stop_outputs,
    wait_until_still,
)


DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
TUNING_CSV = PROJECT_DIR / "tune_parameter" / "data" / "best_by_method.csv"
FIXED_DISTURBANCE_CSV = PROJECT_DIR / "tune_parameter" / "data" / "fixed_shaker_disturbance.csv"
COLLECTED_DATA_DIR = PROJECT_DIR / "hardware_stratified_protocol" / "data"
AR_PPO_CHECKPOINT = PROJECT_DIR / "control_learning_compare" / "models" / "adaptive_residual_seed_666.pt"

for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = 0.03
M1 = 0.67
M2 = 0.85
K1 = 3000.0
K2 = 135.0
C1 = 0.0
C2 = 6.4
U_MAX = 2.0
SHAKER_FORCE_LIMIT_N = 9.0
B0_ADRC = 1.0 / M2
OBS_SCALE = np.asarray([0.02, 0.6, 0.02, 0.6, U_MAX, SHAKER_FORCE_LIMIT_N], dtype=np.float32)
STD_INIT = 0.35
STD_MIN = 0.03

DEFAULT_CONTROLLERS = ["passive", "adrc", "mpc", "deepc", "ar_ppo"]
CONTROLLER_LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
}


class RolloutAbort(RuntimeError):
    pass


def read_csv_dicts(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def load_best_params(path: Path = TUNING_CSV) -> dict[str, dict]:
    params: dict[str, dict] = {}
    for row in read_csv_dicts(path):
        method = row["method"]
        if method == "adrc":
            params[method] = {
                "omega_o": as_float(row.get("param_omega_o"), 16.0),
                "kp": as_float(row.get("param_kp"), 30.0),
                "kd": as_float(row.get("param_kd"), 6.0),
                "b0_scale": as_float(row.get("param_b0_scale"), 1.0),
            }
        elif method == "mpc":
            params[method] = {
                "horizon": int(as_float(row.get("param_horizon"), 3)),
                "q_x2": as_float(row.get("param_q_x2"), 4.0e5),
                "q_v2": as_float(row.get("param_q_v2"), 6000.0),
                "r_u": as_float(row.get("param_r_u"), 0.01),
                "r_du": as_float(row.get("param_r_du"), 0.02),
                "ki_x2": as_float(row.get("param_ki_x2"), 0.0),
            }
        elif method == "deepc":
            params[method] = {
                "dictionary_source": row.get("param_dictionary_source") or "measured_csv",
                "t_ini": int(as_float(row.get("param_t_ini"), 8)),
                "n_f": int(as_float(row.get("param_n_f"), 3)),
                "max_columns": int(as_float(row.get("param_max_columns"), 80)),
                "q_x2": as_float(row.get("param_q_x2"), 3.0e4),
                "q_v2": as_float(row.get("param_q_v2"), 5000.0),
                "r_u": as_float(row.get("param_r_u"), 0.01),
                "r_du": as_float(row.get("param_r_du"), 0.12),
                "lambda_g": as_float(row.get("param_lambda_g"), 1.0e-3),
                "lambda_y": as_float(row.get("param_lambda_y"), 4.0e5),
                "lambda_uini": as_float(row.get("param_lambda_uini"), 1.0e4),
                "lambda_dini": as_float(row.get("param_lambda_dini"), 1.0e4),
                "lambda_df": as_float(row.get("param_lambda_df"), 900.0),
                "action_alpha": as_float(row.get("param_action_alpha"), 0.7),
            }
    return params


def load_fixed_disturbance() -> tuple[np.ndarray, np.ndarray]:
    if FIXED_DISTURBANCE_CSV.exists():
        rows = read_csv_dicts(FIXED_DISTURBANCE_CSV)
        t = np.asarray([float(r["time"]) for r in rows], dtype=np.float64)
        d = np.asarray([float(r["shaker_force_n"]) for r in rows], dtype=np.float64)
    else:
        t, d = fixed_shaker_disturbance()
    save_fixed_shaker_disturbance_csv(DATA_DIR / "fixed_test_disturbance.csv", t, d)
    return t, d


def clip_force(value: float) -> tuple[float, bool]:
    raw = float(value) if np.isfinite(float(value)) else 0.0
    sat = float(np.clip(raw, -U_MAX, U_MAX))
    return sat, bool(abs(raw - sat) > 1.0e-8)


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


def fallback_force(state):
    x1, v1, x2, v2 = np.asarray(state, dtype=np.float64).reshape(4)
    u_raw = -135.0 * (x1 - x2) - C2 * (v1 - v2) - 18.0 * x2 - 0.6 * v2
    return clip_force(u_raw)[0]


class BaseController:
    name = "base"

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> tuple[float, dict]:
        raise NotImplementedError


class PassiveController(BaseController):
    name = "passive"

    def act(self, obs: dict) -> tuple[float, dict]:
        return 0.0, {"u_raw": 0.0, "u_sat": 0.0}


class ADRCController(BaseController):
    name = "adrc"

    def __init__(self, omega_o: float, kp: float, kd: float, b0_scale: float = 1.0):
        self.omega_o = float(omega_o)
        self.kp = float(kp)
        self.kd = float(kd)
        self.b0 = B0_ADRC * float(b0_scale)
        self.z = np.zeros(3, dtype=np.float64)
        self.initialized = False

    def reset(self) -> None:
        self.z[:] = 0.0
        self.initialized = False

    def act(self, obs: dict) -> tuple[float, dict]:
        if abs(self.b0) < 1.0e-8 or not np.isfinite(self.b0):
            u = fallback_force(obs["state"])
            return u, {"u_raw": u, "u_sat": u, "fallback_flag": 1}
        x1, v1, x2, v2 = np.asarray(obs["state"], dtype=np.float64).reshape(4)
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

    def reset(self) -> None:
        self.prev_solution[:] = 0.0
        self.integral = 0.0

    def act(self, obs: dict) -> tuple[float, dict]:
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
    for path in sorted(Path(data_dir).glob("trajectory_*.csv")):
        records = read_csv_dicts(path)
        if len(records) < rows:
            continue
        y = np.asarray(
            [[float(r["x1_filtered"]), float(r["v1_filtered"]), float(r["x2_filtered"]), float(r["v2_filtered"])] for r in records],
            dtype=np.float64,
        )
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


class DeePCController(BaseController):
    name = "deepc"

    def __init__(self, params: dict, data_dir: Path = COLLECTED_DATA_DIR):
        self.params = dict(params)
        self.available = False
        self.init_error = ""
        try:
            self.u_h, self.d_h, self.y_h = load_csv_deepc_hankel(
                data_dir,
                self.params["t_ini"],
                self.params["n_f"],
                self.params["max_columns"],
            )
            t_ini = int(self.params["t_ini"])
            n_f = int(self.params["n_f"])
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

    def reset(self) -> None:
        t_ini = int(self.params["t_ini"])
        self.u_history = [0.0] * t_ini
        self.d_history = [0.0] * t_ini
        self.y_history = [np.zeros(4, dtype=np.float64) for _ in range(t_ini)]

    def act(self, obs: dict) -> tuple[float, dict]:
        self.u_history.append(float(obs["u_prev"]))
        self.d_history.append(float(obs["disturbance"]))
        self.y_history.append(np.asarray(obs["state"], dtype=np.float64).reshape(4))
        t_ini = int(self.params["t_ini"])
        self.u_history = self.u_history[-t_ini:]
        self.d_history = self.d_history[-t_ini:]
        self.y_history = self.y_history[-t_ini:]
        if not self.available:
            u = fallback_force(obs["state"])
            return u, {"u_raw": u, "u_sat": u, "fallback_flag": 1, "init_error": self.init_error}
        start = time.perf_counter()
        try:
            u_seq = self.solve(obs)
            alpha = float(self.params.get("action_alpha", 0.7))
            u_raw = alpha * float(u_seq[0]) + (1.0 - alpha) * float(obs["u_prev"])
            fallback = 0
        except Exception:
            u_raw = fallback_force(obs["state"])
            fallback = 1
        solve_ms = (time.perf_counter() - start) * 1000.0
        u_sat, saturation = clip_force(u_raw)
        return u_sat, {"u_raw": u_raw, "u_sat": u_sat, "fallback_flag": fallback, "saturation_flag": int(saturation), "solve_time_ms": solve_ms}

    def solve(self, obs: dict) -> np.ndarray:
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


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.log(torch.tensor([STD_INIT], dtype=torch.float32)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        log_std = torch.clamp(self.log_std, math.log(STD_MIN), math.log(1.0))
        return mean, torch.exp(log_std).expand_as(mean)


class ARPPOController(BaseController):
    name = "ar_ppo"

    def __init__(self, checkpoint_path: Path, adrc_params: dict):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        self.actor = Actor(6, hidden_dim)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()
        self.residual_u_max = float(ckpt.get("residual_u_max", U_MAX))
        self.obs_scale = np.asarray(ckpt.get("obs_scale", OBS_SCALE), dtype=np.float32)
        base_params = dict(ckpt.get("adrc_params", adrc_params))
        self.base = ADRCController(
            base_params.get("omega_o", adrc_params["omega_o"]),
            base_params.get("kp", adrc_params["kp"]),
            base_params.get("kd", adrc_params["kd"]),
            base_params.get("b0_scale", adrc_params.get("b0_scale", 1.0)),
        )

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def act(self, obs: dict) -> tuple[float, dict]:
        start = time.perf_counter()
        base_u, base_info = self.base.act(obs)
        physical = np.asarray(
            [
                obs["state"][0],
                obs["state"][1],
                obs["state"][2],
                obs["state"][3],
                obs["u_prev"],
                obs["disturbance"],
            ],
            dtype=np.float32,
        )
        obs_t = torch.as_tensor((physical / self.obs_scale).reshape(1, -1), dtype=torch.float32)
        mean, _ = self.actor(obs_t)
        residual = float(self.residual_u_max * torch.tanh(mean).squeeze().cpu().item())
        u_raw = float(base_u + residual)
        u_sat, saturation = clip_force(u_raw)
        solve_ms = (time.perf_counter() - start) * 1000.0
        return u_sat, {
            "u_raw": u_raw,
            "u_sat": u_sat,
            "base_u": base_u,
            "ppo_residual": residual,
            "saturation_flag": int(saturation),
            "solve_time_ms": solve_ms,
            "base_u_raw": base_info.get("u_raw", base_u),
        }


def make_controller(name: str, best_params: dict[str, dict]) -> BaseController:
    if name == "passive":
        return PassiveController()
    if name == "adrc":
        return ADRCController(**best_params["adrc"])
    if name == "mpc":
        return MPCController(**best_params["mpc"])
    if name == "deepc":
        return DeePCController(best_params["deepc"])
    if name == "ar_ppo":
        return ARPPOController(AR_PPO_CHECKPOINT, best_params["adrc"])
    raise ValueError(f"Unknown controller: {name}")


def rollout_path(controller: str, repeat: int) -> Path:
    return ROLLOUT_DIR / f"{controller}_repeat_{int(repeat):02d}.csv"


def valid_rollout(path: Path, min_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv_dicts(path)
        return len(rows) == int(min_rows) and "x2" in rows[0] and "computation_time_ms" in rows[0]
    except Exception:
        return False


def save_rollout(path: Path, rows: list[dict]) -> None:
    fields = [
        "controller",
        "controller_label",
        "repeat",
        "sample",
        "time",
        "timestamp_abs",
        "dt_sample",
        "cutoff_hz",
        "x1_raw",
        "x2_raw",
        "x1",
        "v1",
        "a1",
        "x2",
        "v2",
        "a2",
        "x1_filtered",
        "v1_filtered",
        "a1_filtered",
        "x2_filtered",
        "v2_filtered",
        "a2_filtered",
        "shaker_force_n",
        "shaker_voltage_v",
        "u_raw",
        "u_applied",
        "voice_force_n",
        "base_u",
        "ppo_residual",
        "saturation_flag",
        "fallback_flag",
        "computation_time_ms",
        "solve_time_ms",
    ]
    write_csv(path, rows, fields)


def collect_rollout(
    controller_name: str,
    repeat: int,
    controller: BaseController,
    disturbance: np.ndarray,
    hardware,
    x_task,
    x_ser,
    y_ser,
    arduino_task,
    *,
    jump_abort_m: float,
    settle_time_s: float,
    post_wait_s: float,
    abs_displacement_limit_m: float,
) -> Path:
    steps = len(disturbance)
    controller.reset()
    prepare_hardware_for_trial(x_task, x_ser, y_ser, arduino_task)
    offset = wait_until_still(hardware.get_xy_by_serial, x_ser, y_ser, duration_s=settle_time_s, sample_dt=DT)
    filt = OnlineKinematicsFilter(cutoff_hz=CUTOFF_HZ, dimensions=2)
    rows: list[dict] = []
    prev_raw = None
    u_prev = 0.0

    try:
        x_task.start()
    except Exception:
        pass

    start_time = time.perf_counter()
    filt.reset(np.zeros(2, dtype=np.float64), start_time)

    try:
        for step in range(steps):
            target = start_time + step * DT
            while True:
                remaining = target - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(remaining, 0.002))

            now = time.perf_counter()
            raw_xy = np.asarray(hardware.get_xy_by_serial(x_ser, y_ser), dtype=np.float64) - offset
            if prev_raw is not None and np.any(np.abs(raw_xy - prev_raw) > float(jump_abort_m)):
                raise RolloutAbort(f"sudden displacement jump at sample {step}: {raw_xy - prev_raw}")
            if np.any(np.abs(raw_xy) > float(abs_displacement_limit_m)):
                raise RolloutAbort(f"absolute displacement limit exceeded at sample {step}: {raw_xy}")

            x_f, v_f, a_f, dt_sample = filt.update(raw_xy, now)
            shaker_force = float(np.clip(disturbance[step], -SHAKER_FORCE_LIMIT_N, SHAKER_FORCE_LIMIT_N))
            obs = {
                "state": np.asarray([x_f[0], v_f[0], x_f[1], v_f[1]], dtype=np.float64),
                "disturbance": shaker_force,
                "u_prev": u_prev,
                "dt": float(dt_sample),
            }
            control_start = time.perf_counter()
            u_cmd, info = controller.act(obs)
            computation_ms = (time.perf_counter() - control_start) * 1000.0
            u_applied = quiet_put_voice_coil_force(arduino_task, u_cmd)
            shaker_voltage = shaker_force_to_input_voltage(shaker_force)
            x_task.write(shaker_voltage, auto_start=False)

            rows.append(
                {
                    "controller": controller_name,
                    "controller_label": CONTROLLER_LABELS[controller_name],
                    "repeat": int(repeat),
                    "sample": int(step),
                    "time": float(now - start_time),
                    "timestamp_abs": float(now),
                    "dt_sample": float(dt_sample),
                    "cutoff_hz": CUTOFF_HZ,
                    "x1_raw": float(raw_xy[0]),
                    "x2_raw": float(raw_xy[1]),
                    "x1": float(x_f[0]),
                    "v1": float(v_f[0]),
                    "a1": float(a_f[0]),
                    "x2": float(x_f[1]),
                    "v2": float(v_f[1]),
                    "a2": float(a_f[1]),
                    "x1_filtered": float(x_f[0]),
                    "v1_filtered": float(v_f[0]),
                    "a1_filtered": float(a_f[0]),
                    "x2_filtered": float(x_f[1]),
                    "v2_filtered": float(v_f[1]),
                    "a2_filtered": float(a_f[1]),
                    "shaker_force_n": shaker_force,
                    "shaker_voltage_v": float(shaker_voltage),
                    "u_raw": float(info.get("u_raw", u_cmd)),
                    "u_applied": float(u_applied),
                    "voice_force_n": float(u_applied),
                    "base_u": float(info.get("base_u", 0.0)),
                    "ppo_residual": float(info.get("ppo_residual", 0.0)),
                    "saturation_flag": int(info.get("saturation_flag", 0)),
                    "fallback_flag": int(info.get("fallback_flag", 0)),
                    "computation_time_ms": float(computation_ms),
                    "solve_time_ms": float(info.get("solve_time_ms", computation_ms)),
                }
            )
            prev_raw = raw_xy.copy()
            u_prev = float(u_applied)
    finally:
        finish_hardware_trial(x_task, x_ser, y_ser, arduino_task, post_wait_s=post_wait_s)

    if len(rows) != steps:
        raise RuntimeError(f"Expected {steps} rows, collected {len(rows)} rows.")
    path = rollout_path(controller_name, repeat)
    save_rollout(path, rows)
    return path


def rollout_metrics(path: Path) -> dict:
    rows = read_csv_dicts(path)
    controller = rows[0]["controller"]
    repeat = int(float(rows[0]["repeat"]))
    out = {"controller": controller, "controller_label": CONTROLLER_LABELS.get(controller, controller), "repeat": repeat, "path": str(path)}
    for key in ("x2", "v2", "a2"):
        arr = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_rms"] = float(np.sqrt(np.mean(arr**2)))
        out[f"{key}_mean_abs"] = float(np.mean(np.abs(arr)))
        out[f"{key}_peak_abs"] = float(np.max(np.abs(arr)))
    compute = np.asarray([float(r["computation_time_ms"]) for r in rows], dtype=np.float64)
    out["compute_time_mean_ms"] = float(np.mean(compute))
    out["compute_time_std_ms"] = float(np.std(compute, ddof=1)) if len(compute) > 1 else 0.0
    out["compute_time_max_ms"] = float(np.max(compute))
    out["saturation_count"] = int(sum(int(float(r.get("saturation_flag", 0))) for r in rows))
    out["fallback_count"] = int(sum(int(float(r.get("fallback_flag", 0))) for r in rows))
    return out


def refresh_metric_tables(controllers: list[str], repeats: int) -> None:
    repeat_rows = []
    for controller in controllers:
        for repeat in range(1, int(repeats) + 1):
            path = rollout_path(controller, repeat)
            if path.exists():
                repeat_rows.append(rollout_metrics(path))
    metric_fields = [
        "controller",
        "controller_label",
        "repeat",
        "path",
        "x2_rms",
        "x2_mean_abs",
        "x2_peak_abs",
        "v2_rms",
        "v2_mean_abs",
        "v2_peak_abs",
        "a2_rms",
        "a2_mean_abs",
        "a2_peak_abs",
        "compute_time_mean_ms",
        "compute_time_std_ms",
        "compute_time_max_ms",
        "saturation_count",
        "fallback_count",
    ]
    write_csv(DATA_DIR / "repeat_metrics.csv", repeat_rows, metric_fields)

    stat_rows = []
    metrics = [field for field in metric_fields if field not in {"controller", "controller_label", "repeat", "path"}]
    for controller in controllers:
        group = [r for r in repeat_rows if r["controller"] == controller]
        if not group:
            continue
        row = {"controller": controller, "controller_label": CONTROLLER_LABELS.get(controller, controller), "n_repeats": len(group)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        stat_rows.append(row)
    stat_fields = []
    for row in stat_rows:
        for key in row:
            if key not in stat_fields:
                stat_fields.append(key)
    write_csv(DATA_DIR / "aggregate_statistics.csv", stat_rows, stat_fields)


def collect_all(args: argparse.Namespace) -> None:
    time_s, disturbance = load_fixed_disturbance()
    steps = len(disturbance)
    best_params = load_best_params()
    controllers = [str(c) for c in args.controllers]
    missing = [c for c in ("adrc", "mpc", "deepc") if c in controllers and c not in best_params]
    if missing:
        raise FileNotFoundError(f"Missing tuned parameters for: {missing}. Expected {TUNING_CSV.resolve()}")
    if "ar_ppo" in controllers and not AR_PPO_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing AR-PPO checkpoint: {AR_PPO_CHECKPOINT.resolve()}")

    total_jobs = len(controllers) * int(args.repeats)
    completed = sum(valid_rollout(rollout_path(c, r), steps) for r in range(1, int(args.repeats) + 1) for c in controllers)
    print(f"Fixed disturbance: {DATA_DIR / 'fixed_test_disturbance.csv'} ({time_s[-1]:.2f} s, {steps} samples)")
    print(f"Completed rollouts: {completed}/{total_jobs}")

    hardware = import_hardware_module()
    x_task = x_ser = y_ser = arduino_task = None
    try:
        x_task, x_ser, y_ser, arduino_task = hardware.all_devices_setup_on_demand()
        stop_outputs(x_task, arduino_task)
        job_index = 0
        for repeat in range(1, int(args.repeats) + 1):
            for controller_name in controllers:
                job_index += 1
                path = rollout_path(controller_name, repeat)
                if valid_rollout(path, steps) and not args.overwrite:
                    print(f"Skipping {controller_name} repeat {repeat}/{args.repeats}: already saved")
                    continue
                controller = make_controller(controller_name, best_params)
                attempt = 0
                while True:
                    attempt += 1
                    print(
                        f"Collecting {CONTROLLER_LABELS[controller_name]} repeat {repeat}/{args.repeats} "
                        f"(job {job_index}/{total_jobs}) attempt={attempt}"
                    )
                    try:
                        saved = collect_rollout(
                            controller_name,
                            repeat,
                            controller,
                            disturbance,
                            hardware,
                            x_task,
                            x_ser,
                            y_ser,
                            arduino_task,
                            jump_abort_m=args.jump_abort_m,
                            settle_time_s=args.settle_time_s,
                            post_wait_s=args.post_wait_s,
                            abs_displacement_limit_m=args.abs_displacement_limit_m,
                        )
                        print(f"Saved {CONTROLLER_LABELS[controller_name]} repeat {repeat}/{args.repeats}: {saved.name}")
                        refresh_metric_tables(controllers, args.repeats)
                        break
                    except RolloutAbort as exc:
                        print(f"Restarting {controller_name} repeat {repeat}/{args.repeats}: {exc}")
                        if attempt >= int(args.max_retries):
                            raise RuntimeError(f"{controller_name} repeat {repeat} exceeded max_retries={args.max_retries}") from exc
    finally:
        close_devices(x_task, x_ser, y_ser, arduino_task)
    refresh_metric_tables(controllers, args.repeats)
    print(f"Saved metrics: {DATA_DIR / 'repeat_metrics.csv'}")
    print(f"Saved statistics: {DATA_DIR / 'aggregate_statistics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-repeat hardware controller comparison on the fixed shaker disturbance.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=DEFAULT_CONTROLLERS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--post-wait-s", type=float, default=2.0)
    parser.add_argument("--jump-abort-m", type=float, default=0.008)
    parser.add_argument("--abs-displacement-limit-m", type=float, default=0.04)
    return parser.parse_args()


if __name__ == "__main__":
    collect_all(parse_args())
