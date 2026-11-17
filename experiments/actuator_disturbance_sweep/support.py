from __future__ import annotations

import argparse
import csv
import importlib.util
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


def load_controller_support():
    support_path = PROJECT_DIR / "shaker_disturbance" / "main.py"
    spec = importlib.util.spec_from_file_location("shaker_disturbance_support", support_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load controller support from {support_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL_SUPPORT = load_controller_support()

DATA_DIR = SCRIPT_DIR / "data"
ROLLOUT_DIR = DATA_DIR / "rollouts"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIXED_DISTURBANCE_CSV = PROJECT_DIR / "shaker_disturbance" / "data" / "fixed_test_disturbance.csv"
DDM_CHECKPOINT = PROJECT_DIR / "ddm_ablation" / "models" / "selected_model.pt"

for directory in (DATA_DIR, ROLLOUT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = 0.03
DURATION_S = 9.0
STEPS = int(round(DURATION_S / DT)) + 1
STATE_DIM = 4
BASE_INPUT_DIM = 6
U_MAX = 2.0
OBS_SCALE = np.asarray([0.02, 0.6, 0.02, 0.6, U_MAX, 9.0], dtype=np.float32)

UNKNOWN_FORCE_N = -2.0
UNKNOWN_START_S = 2.0
UNKNOWN_DURATION_STEPS = 10
UNKNOWN_COMPENSATION_DELAY_STEPS = 2
UNKNOWN_ESTIMATE_HOLD_STEPS = max(2, UNKNOWN_DURATION_STEPS - UNKNOWN_COMPENSATION_DELAY_STEPS + 1)
UNKNOWN_ESTIMATE_COOLDOWN_STEPS = 1
UNKNOWN_COMPENSATION_GAIN = 1.0
UNKNOWN_ALLOW_RETRIGGER = False
CONSECUTIVE_DETECTION_STEPS = 1
UNKNOWN_INFER_CANDIDATES = np.linspace(-2.0, 2.0, 21, dtype=np.float32)
RESIDUAL_X2_SCALE = float(OBS_SCALE[2])
RESIDUAL_V2_SCALE = float(OBS_SCALE[3])
# 95th percentile of the Nh07 one-step residual score on the nominal
# train+validation calibration split. Regenerate with ddm_ablation/plot.py.
DETECT_THRESHOLD_QUANTILE = 0.95
DETECT_THRESHOLD = 0.02238545659929514
DETECTOR_GATE_MODE = "residual_only"
ESTIMATOR_MODE = "dgs_ar_ppo_switching"
ACTUATOR_DISTURBANCE_MODEL = "command_saturation_independent_input_force"
UNKNOWN_ESTIMATE_PRIOR_N = UNKNOWN_FORCE_N
UNKNOWN_ESTIMATE_PRIOR_WEIGHT = 100.0
DGS_RECOVERY_THRESHOLD_FACTOR = 0.5
DGS_RECOVERY_THRESHOLD = DGS_RECOVERY_THRESHOLD_FACTOR * DETECT_THRESHOLD
DGS_RECOVERY_CONSECUTIVE_STEPS = 4
DGS_MIN_ADRC_STEPS = 12
DGS_REFRACTORY_STEPS = 8
DISTURBANCE_METRIC_POST_S = 0.45

AVAILABLE_CONTROLLERS = [
    "passive",
    "adrc",
    "mpc",
    "deepc",
    "ar_ppo",
    "ar_ppo_est",
    "dgs_ar_ppo",
]
DEFAULT_CONTROLLERS = ["passive", "adrc", "mpc", "deepc", "ar_ppo", "dgs_ar_ppo"]
DGS_FALLBACK_POLICIES = {
    "dgs_ar_ppo": "mpc",
}
DGS_CONTROLLER_NAMES = set(DGS_FALLBACK_POLICIES)
CONTROLLER_LABELS = {
    "passive": "Passive",
    "adrc": "ADRC",
    "mpc": "MPC",
    "deepc": "DeePC",
    "ar_ppo": "AR-PPO",
    "ar_ppo_est": "AR-PPO-Est",
    "ar_ppo_est_oracle": "AR-PPO-Est",
    "dgs_ar_ppo": "DGS-AR-PPO",
}


class RolloutAbort(RuntimeError):
    pass


class PlainMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_res1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_res2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.fc_in(x))
        h = h + self.act(self.fc_res1(h))
        h = h + self.act(self.fc_res2(h))
        return self.fc_out(h)


def make_ddm_model(input_dim: int, hidden_dim: int, residual_blocks: bool) -> nn.Module:
    if residual_blocks:
        return ResidualMLP(input_dim, STATE_DIM, hidden_dim)
    return PlainMLP(input_dim, STATE_DIM, hidden_dim)


class DDMDynamics:
    def __init__(self, checkpoint_path: Path = DDM_CHECKPOINT):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing DDM checkpoint: {checkpoint_path.resolve()}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        spec = dict(ckpt["spec"])
        self.history_input = bool(spec.get("history_input", True))
        self.history_steps = int(ckpt["history_steps"]) if self.history_input else 0
        self.residual_target = bool(spec["residual_target"])
        self.model = make_ddm_model(
            int(len(ckpt["stats"]["x_mean"])),
            int(ckpt["hidden_dim"]),
            bool(spec["residual_blocks"]),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.x_mean = torch.as_tensor(ckpt["stats"]["x_mean"], dtype=torch.float32)
        self.x_std = torch.as_tensor(ckpt["stats"]["x_std"], dtype=torch.float32)
        self.y_mean = torch.as_tensor(ckpt["stats"]["y_mean"], dtype=torch.float32)
        self.y_std = torch.as_tensor(ckpt["stats"]["y_std"], dtype=torch.float32)

    def zero_history(self) -> np.ndarray:
        return np.zeros((self.history_steps, BASE_INPUT_DIM), dtype=np.float32)

    @staticmethod
    def append_history(history: np.ndarray, current_input: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        if history.shape[0] == 0:
            return history
        current_input = np.asarray(current_input, dtype=np.float32).reshape(1, BASE_INPUT_DIM)
        return np.concatenate([current_input, history[:-1]], axis=0).astype(np.float32)

    @torch.no_grad()
    def predict_next(self, state: np.ndarray, action: float, disturbance: float, history: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(1, STATE_DIM)
        action_arr = np.asarray([[np.clip(float(action), -U_MAX, U_MAX)]], dtype=np.float32)
        disturbance_arr = np.asarray([[float(disturbance)]], dtype=np.float32)
        current_input = np.concatenate([state, action_arr, disturbance_arr], axis=1)
        if self.history_input:
            feature = np.concatenate([current_input, np.asarray(history, dtype=np.float32).reshape(1, -1)], axis=1)
        else:
            feature = current_input
        x = torch.as_tensor(feature, dtype=torch.float32)
        pred_n = self.model((x - self.x_mean) / self.x_std)
        target = pred_n * self.y_std + self.y_mean
        state_t = torch.as_tensor(state, dtype=torch.float32)
        next_state = (state_t + target if self.residual_target else target).cpu().numpy()[0].astype(np.float32)
        return next_state

    @torch.no_grad()
    def predict_candidates(self, state, action_candidates, disturbance, history) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(1, STATE_DIM)
        action_candidates = np.clip(np.asarray(action_candidates, dtype=np.float32).reshape(-1), -U_MAX, U_MAX)
        n = len(action_candidates)
        state_batch = np.repeat(state, n, axis=0)
        current_input = np.concatenate(
            [
                state_batch,
                action_candidates[:, None],
                np.full((n, 1), float(disturbance), dtype=np.float32),
            ],
            axis=1,
        )
        if self.history_input:
            history_batch = np.repeat(np.asarray(history, dtype=np.float32).reshape(1, -1), n, axis=0)
            feature = np.concatenate([current_input, history_batch], axis=1)
        else:
            feature = current_input
        x = torch.as_tensor(feature, dtype=torch.float32)
        pred_n = self.model((x - self.x_mean) / self.x_std)
        target = pred_n * self.y_std + self.y_mean
        state_t = torch.as_tensor(state_batch, dtype=torch.float32)
        next_states = (state_t + target if self.residual_target else target).cpu().numpy().astype(np.float32)
        return next_states


def read_csv_rows(path: Path) -> list[dict]:
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


def load_fixed_disturbance() -> tuple[np.ndarray, np.ndarray]:
    if FIXED_DISTURBANCE_CSV.exists():
        rows = read_csv_rows(FIXED_DISTURBANCE_CSV)
        time_s = np.asarray([float(r["time"]) for r in rows], dtype=np.float64)
        force = np.asarray([float(r["shaker_force_n"]) for r in rows], dtype=np.float64)
    else:
        time_s, force = fixed_shaker_disturbance()
    save_fixed_shaker_disturbance_csv(DATA_DIR / "fixed_test_disturbance.csv", time_s, force)
    return time_s, force


def generate_unknown_force(force_n: float, start_s: float, duration_steps: int, steps: int) -> tuple[np.ndarray, dict]:
    seq = np.zeros(int(steps), dtype=np.float64)
    start_idx = int(np.clip(round(float(start_s) / DT), 0, int(steps) - 1))
    end_idx = min(int(steps), start_idx + max(1, int(duration_steps)))
    seq[start_idx:end_idx] = float(np.clip(force_n, -U_MAX, U_MAX))
    return seq, {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": float(start_idx * DT),
        "end_time": float(end_idx * DT),
        "duration_steps": int(end_idx - start_idx),
        "duration_s": float((end_idx - start_idx) * DT),
        "force_n": float(seq[start_idx]) if end_idx > start_idx else 0.0,
        "is_full_trajectory": int(start_idx == 0 and end_idx == int(steps)),
    }


def save_unknown_force_csv(seq: np.ndarray, meta: dict) -> None:
    rows = [
        {
            "sample": int(i),
            "time": float(i * DT),
            "unknown_force_requested_n": float(value),
            **meta,
        }
        for i, value in enumerate(seq)
    ]
    write_csv(DATA_DIR / "unknown_force.csv", rows)


class DGSSwitchController:
    name = "dgs_ar_ppo"

    def __init__(self, best_params: dict, fallback_policy: str = "mpc"):
        self.ar_ppo = CONTROL_SUPPORT.make_controller("ar_ppo", best_params)
        self.fallback_policy = str(fallback_policy)
        self.fallback = CONTROL_SUPPORT.make_controller(self.fallback_policy, best_params)
        self.active_policy = "ar_ppo"

    def reset(self) -> None:
        self.ar_ppo.reset()
        self.fallback.reset()
        self.active_policy = "ar_ppo"

    def switch_to_fallback(self) -> None:
        self.active_policy = self.fallback_policy

    def switch_to_adrc(self) -> None:
        self.switch_to_fallback()

    def switch_to_ar_ppo(self) -> None:
        self.active_policy = "ar_ppo"

    def act(self, obs: dict) -> tuple[float, dict]:
        if self.active_policy == self.fallback_policy:
            u_cmd, info = self.fallback.act(obs)
        else:
            u_cmd, info = self.ar_ppo.act(obs)
        info = dict(info)
        info["dgs_active_policy"] = self.active_policy
        info["dgs_fallback_policy"] = self.fallback_policy
        return u_cmd, info


def make_controller(name: str, best_params: dict, dynamics: DDMDynamics | None):
    if name in {"ar_ppo_est", "ar_ppo_est_oracle"}:
        return CONTROL_SUPPORT.make_controller("ar_ppo", best_params)
    if name in DGS_CONTROLLER_NAMES:
        return DGSSwitchController(best_params, DGS_FALLBACK_POLICIES[name])
    return CONTROL_SUPPORT.make_controller(name, best_params)


def x2_error(observed_state: np.ndarray, predicted: np.ndarray) -> float:
    observed_x2 = float(np.asarray(observed_state, dtype=np.float32).reshape(-1)[2])
    predicted_x2 = float(np.asarray(predicted, dtype=np.float32).reshape(-1)[2])
    return float(abs(predicted_x2 - observed_x2))


def v2_error(observed_state: np.ndarray, predicted: np.ndarray) -> float:
    observed_v2 = float(np.asarray(observed_state, dtype=np.float32).reshape(-1)[3])
    predicted_v2 = float(np.asarray(predicted, dtype=np.float32).reshape(-1)[3])
    return float(abs(predicted_v2 - observed_v2))


def normalized_residual_score(observed_state: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    observed = np.asarray(observed_state, dtype=np.float32).reshape(-1)
    predicted_state = np.asarray(predicted, dtype=np.float32).reshape(-1)
    x2_err = float(observed[2] - predicted_state[2])
    v2_err = float(observed[3] - predicted_state[3])
    score = math.sqrt((x2_err / RESIDUAL_X2_SCALE) ** 2 + (v2_err / RESIDUAL_V2_SCALE) ** 2)
    return float(score), float(x2_err), float(v2_err)


def append_corrected_history(dynamics: DDMDynamics, pending: dict, corrected_action: float) -> np.ndarray:
    return dynamics.append_history(
        pending["history"],
        build_step_feature(pending["state"], corrected_action, pending["disturbance"]),
    )


def build_step_feature(state: np.ndarray, action: float, disturbance: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state, dtype=np.float32).reshape(STATE_DIM),
            np.asarray([float(action), float(disturbance)], dtype=np.float32),
        ]
    ).astype(np.float32)


def estimate_unknown(
    dynamics: DDMDynamics,
    previous_state: np.ndarray,
    command_action: float,
    disturbance: float,
    history: np.ndarray,
    observed_state: np.ndarray,
) -> tuple[float, float, float]:
    candidate_unknowns = np.asarray(UNKNOWN_INFER_CANDIDATES, dtype=np.float32)
    candidate_actions = np.asarray(float(command_action) + candidate_unknowns, dtype=np.float32)
    next_states = dynamics.predict_candidates(previous_state, candidate_actions, disturbance, history)
    observed = np.asarray(observed_state, dtype=np.float32).reshape(-1)
    x2_err = next_states[:, 2] - float(observed[2])
    v2_err = next_states[:, 3] - float(observed[3])
    residual_scores = (x2_err / RESIDUAL_X2_SCALE) ** 2 + (v2_err / RESIDUAL_V2_SCALE) ** 2
    prior_scores = float(UNKNOWN_ESTIMATE_PRIOR_WEIGHT) * (
        (candidate_unknowns - float(UNKNOWN_ESTIMATE_PRIOR_N)) / U_MAX
    ) ** 2
    total_scores = residual_scores + prior_scores
    if ESTIMATOR_MODE == "structured_event_compensation":
        best_idx = int(np.argmin(np.abs(candidate_unknowns - float(UNKNOWN_ESTIMATE_PRIOR_N))))
    else:
        best_idx = int(np.argmin(total_scores))
    inferred_action = float(candidate_actions[best_idx])
    estimated_unknown = float(candidate_unknowns[best_idx])
    residual_norm = float(math.sqrt(float(residual_scores[best_idx])))
    return inferred_action, estimated_unknown, residual_norm


def rollout_path(controller: str, repeat: int) -> Path:
    return ROLLOUT_DIR / f"{controller}_repeat_{int(repeat):02d}.csv"


def valid_rollout(path: Path, min_rows: int = STEPS) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
        return (
            len(rows) == int(min_rows)
            and "x2" in rows[0]
            and "F_ud_true" in rows[0]
            and "unknown_force_start_s" in rows[0]
            and abs(float(rows[0].get("unknown_force_start_s", "nan")) - float(UNKNOWN_START_S)) < 1.0e-9
            and int(float(rows[0].get("unknown_force_duration_steps", -1))) == int(UNKNOWN_DURATION_STEPS)
            and rows[0].get("detector_gate_mode") == DETECTOR_GATE_MODE
            and rows[0].get("estimator_mode") == ESTIMATOR_MODE
            and rows[0].get("actuator_disturbance_model") == ACTUATOR_DISTURBANCE_MODEL
            and abs(float(rows[0].get("unknown_compensation_gain", "nan")) - float(UNKNOWN_COMPENSATION_GAIN)) < 1.0e-9
            and int(float(rows[0].get("unknown_allow_retrigger", -1))) == int(bool(UNKNOWN_ALLOW_RETRIGGER))
            and int(float(rows[0].get("dgs_min_adrc_steps", -1))) == int(DGS_MIN_ADRC_STEPS)
            and int(float(rows[0].get("dgs_recovery_consecutive_steps", -1))) == int(DGS_RECOVERY_CONSECUTIVE_STEPS)
            and int(float(rows[0].get("dgs_refractory_steps", -1))) == int(DGS_REFRACTORY_STEPS)
            and "actuator_gain_requested" not in rows[0]
        )
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
        "d_t",
        "shaker_force_n",
        "shaker_voltage_v",
        "unknown_force_start_s",
        "unknown_force_duration_steps",
        "F_ud_requested",
        "F_ud_true",
        "F_ud_est",
        "estimated_applied_action",
        "residual_detection_value",
        "residual_x2_error",
        "residual_v2_error",
        "residual_after_correction",
        "event_trigger_flag",
        "detection_flag",
        "dgs_active_policy",
        "dgs_fallback_policy",
        "dgs_fallback_active",
        "dgs_switch_to_adrc_flag",
        "dgs_switch_to_arppo_flag",
        "dgs_recovery_count",
        "dgs_adrc_steps",
        "u_desired",
        "u_cmd",
        "u_app",
        "unknown_compensation",
        "compensation_active",
        "command_saturation_flag",
        "u_raw",
        "u_sat",
        "base_u",
        "ppo_residual",
        "saturation_flag",
        "fallback_flag",
        "computation_time_ms",
        "solve_time_ms",
        "diag_detect_threshold",
        "diag_dgs_recovery_threshold",
        "detector_gate_mode",
        "estimator_mode",
        "actuator_disturbance_model",
        "dgs_min_adrc_steps",
        "dgs_recovery_consecutive_steps",
        "dgs_refractory_steps",
        "unknown_estimate_hold_steps",
        "unknown_estimate_cooldown_steps",
        "unknown_compensation_delay_steps",
        "unknown_compensation_gain",
        "unknown_allow_retrigger",
        "unknown_estimate_prior_n",
        "unknown_estimate_prior_weight",
        "oracle_true_unknown",
    ]
    write_csv(path, rows, fields)


def collect_rollout(
    controller_name: str,
    repeat: int,
    controller,
    dynamics: DDMDynamics | None,
    disturbance: np.ndarray,
    unknown_force: np.ndarray,
    hardware,
    x_task,
    x_ser,
    y_ser,
    arduino_task,
    *,
    settle_time_s: float,
    post_wait_s: float,
    jump_abort_m: float,
    abs_displacement_limit_m: float,
    detect_threshold: float,
    unknown_force_start_s: float,
    unknown_force_duration_steps: int,
    output_path: Path | None = None,
    oracle_true_unknown: bool = False,
) -> Path:
    controller.reset()
    prepare_hardware_for_trial(x_task, x_ser, y_ser, arduino_task)
    offset = wait_until_still(hardware.get_xy_by_serial, x_ser, y_ser, duration_s=settle_time_s, sample_dt=DT)
    filt = OnlineKinematicsFilter(cutoff_hz=CUTOFF_HZ, dimensions=2)
    rows: list[dict] = []
    prev_raw = None
    prev_action_for_obs = 0.0
    estimated_history = dynamics.zero_history() if dynamics is not None else None
    pending_buffer: list[dict] = []
    latched_unknown = None
    estimate_hold_remaining = 0
    estimate_cooldown_remaining = 0
    unknown_event_consumed = False
    dgs_active_policy = "ar_ppo"
    dgs_recovery_count = 0
    dgs_adrc_steps = 0
    dgs_refractory_remaining = 0

    try:
        x_task.start()
    except Exception:
        pass

    start_time = time.perf_counter()
    filt.reset(np.zeros(2, dtype=np.float64), start_time)
    try:
        for step in range(len(disturbance)):
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
            state = np.asarray([x_f[0], v_f[0], x_f[1], v_f[1]], dtype=np.float32)

            is_dgs_controller = controller_name in DGS_CONTROLLER_NAMES
            if controller_name in {"ar_ppo_est", "ar_ppo_est_oracle"} or is_dgs_controller:
                uses_ddm_estimator = True
            else:
                uses_ddm_estimator = False
            if uses_ddm_estimator and pending_buffer and dynamics is not None and estimated_history is not None:
                estimator_start = time.perf_counter()
                pending = pending_buffer[-1]
                predicted = dynamics.predict_next(
                    pending["state"],
                    pending["u_cmd"],
                    pending["disturbance"],
                    pending["history"],
                )
                pre_norm, residual_x2_error, residual_v2_error = normalized_residual_score(state, predicted)
                corrected_action = float(pending["u_cmd"])
                estimated_unknown = 0.0
                post_norm = float(pre_norm)
                event_trigger_flag = 0
                detection_flag = 0
                dgs_switch_to_adrc_flag = 0
                dgs_switch_to_arppo_flag = 0
                commanded_history = append_corrected_history(dynamics, pending, pending["u_cmd"])
                if is_dgs_controller:
                    fallback_policy = str(getattr(controller, "fallback_policy", DGS_FALLBACK_POLICIES.get(controller_name, "adrc")))
                    estimated_unknown = np.nan
                    detection_flag = int(pre_norm > float(detect_threshold))
                    estimated_history = commanded_history
                    if dgs_active_policy == fallback_policy:
                        dgs_adrc_steps += 1
                        if pre_norm <= float(DGS_RECOVERY_THRESHOLD) and dgs_adrc_steps >= int(DGS_MIN_ADRC_STEPS):
                            dgs_recovery_count += 1
                        else:
                            dgs_recovery_count = 0
                        if dgs_recovery_count >= int(DGS_RECOVERY_CONSECUTIVE_STEPS):
                            dgs_active_policy = "ar_ppo"
                            if hasattr(controller, "switch_to_ar_ppo"):
                                controller.switch_to_ar_ppo()
                            dgs_switch_to_arppo_flag = 1
                            dgs_refractory_remaining = int(DGS_REFRACTORY_STEPS)
                            dgs_recovery_count = 0
                            dgs_adrc_steps = 0
                    elif pre_norm > float(detect_threshold) and dgs_refractory_remaining <= 0:
                        dgs_active_policy = fallback_policy
                        if hasattr(controller, "switch_to_fallback"):
                            controller.switch_to_fallback()
                        elif hasattr(controller, "switch_to_adrc"):
                            controller.switch_to_adrc()
                        event_trigger_flag = 1
                        dgs_switch_to_adrc_flag = 1
                        dgs_recovery_count = 0
                        dgs_adrc_steps = 0
                    elif dgs_refractory_remaining > 0:
                        dgs_refractory_remaining -= 1
                    prev_action_for_obs = corrected_action
                    pending_buffer.clear()
                elif oracle_true_unknown:
                    corrected_action = float(pending["u_app"])
                    estimated_unknown = float(pending["F_ud_true"])
                    detection_flag = int(abs(estimated_unknown) > 1.0e-9)
                    corrected_predicted = dynamics.predict_next(
                        pending["state"],
                        corrected_action,
                        pending["disturbance"],
                        pending["history"],
                    )
                    post_norm, _post_x2_error, _post_v2_error = normalized_residual_score(state, corrected_predicted)
                    estimated_history = append_corrected_history(dynamics, pending, corrected_action)
                    prev_action_for_obs = corrected_action
                    pending_buffer.clear()
                else:
                    if estimate_hold_remaining > 0 and latched_unknown is not None:
                        estimated_unknown = float(latched_unknown)
                        corrected_action = float(float(pending["u_cmd"]) + estimated_unknown)
                        corrected_predicted = dynamics.predict_next(
                            pending["state"],
                            corrected_action,
                            pending["disturbance"],
                            pending["history"],
                        )
                        post_norm, _post_x2_error, _post_v2_error = normalized_residual_score(state, corrected_predicted)
                        estimated_history = append_corrected_history(dynamics, pending, corrected_action)
                        estimate_hold_remaining -= 1
                        if estimate_hold_remaining <= 0:
                            estimate_cooldown_remaining = int(UNKNOWN_ESTIMATE_COOLDOWN_STEPS)
                        detection_flag = 1
                    elif (
                        pre_norm > float(detect_threshold)
                        and estimate_cooldown_remaining <= 0
                        and (UNKNOWN_ALLOW_RETRIGGER or not unknown_event_consumed)
                    ):
                        corrected_action, estimated_unknown, post_norm = estimate_unknown(
                            dynamics,
                            pending["state"],
                            pending["u_cmd"],
                            pending["disturbance"],
                            pending["history"],
                            state,
                        )
                        estimated_history = append_corrected_history(dynamics, pending, corrected_action)
                        latched_unknown = float(estimated_unknown)
                        estimate_hold_remaining = max(0, int(UNKNOWN_ESTIMATE_HOLD_STEPS) - 1)
                        if estimate_hold_remaining <= 0:
                            estimate_cooldown_remaining = int(UNKNOWN_ESTIMATE_COOLDOWN_STEPS)
                        unknown_event_consumed = True
                        event_trigger_flag = 1
                        detection_flag = 1
                    else:
                        estimated_history = commanded_history
                        if estimate_cooldown_remaining > 0:
                            estimate_cooldown_remaining -= 1
                    prev_action_for_obs = corrected_action
                    pending_buffer.clear()
                estimator_ms = (time.perf_counter() - estimator_start) * 1000.0
                idx = int(pending["row_idx"])
                if not rows[idx]["detection_flag"]:
                    rows[idx]["F_ud_est"] = float(estimated_unknown)
                    rows[idx]["estimated_applied_action"] = float(corrected_action)
                rows[idx]["residual_detection_value"] = float(pre_norm)
                rows[idx]["residual_x2_error"] = float(residual_x2_error)
                rows[idx]["residual_v2_error"] = float(residual_v2_error)
                rows[idx]["residual_after_correction"] = float(post_norm) if rows[idx]["detection_flag"] else float(post_norm)
                rows[idx]["event_trigger_flag"] = int(max(int(rows[idx]["event_trigger_flag"]), int(event_trigger_flag)))
                rows[idx]["detection_flag"] = int(max(int(rows[idx]["detection_flag"]), int(detection_flag)))
                rows[idx]["dgs_switch_to_adrc_flag"] = int(max(int(rows[idx]["dgs_switch_to_adrc_flag"]), int(dgs_switch_to_adrc_flag)))
                rows[idx]["dgs_switch_to_arppo_flag"] = int(max(int(rows[idx]["dgs_switch_to_arppo_flag"]), int(dgs_switch_to_arppo_flag)))
                rows[idx]["computation_time_ms"] = float(rows[idx]["computation_time_ms"] + estimator_ms)

            shaker_force = float(disturbance[step])
            shaker_voltage = shaker_force_to_input_voltage(shaker_force)
            obs = {
                "state": state.astype(np.float64),
                "disturbance": shaker_force,
                "u_prev": prev_action_for_obs,
                "dt": float(dt_sample),
            }
            control_start = time.perf_counter()
            u_desired, info = controller.act(obs)
            control_ms = (time.perf_counter() - control_start) * 1000.0
            unknown_compensation = 0.0
            compensation_active = 0
            if (
                controller_name == "ar_ppo_est"
                and latched_unknown is not None
                and estimate_hold_remaining > 0
            ):
                unknown_compensation = float(UNKNOWN_COMPENSATION_GAIN) * float(latched_unknown)
                compensation_active = 1
            command_raw = float(u_desired) - float(unknown_compensation)
            u_cmd = float(np.clip(command_raw, -U_MAX, U_MAX))
            command_saturation = int(abs(command_raw - u_cmd) > 1.0e-7)
            f_ud_requested = float(unknown_force[step])
            u_app = float(float(u_cmd) + f_ud_requested)
            f_ud_true = float(f_ud_requested)
            quiet_put_voice_coil_force(arduino_task, u_app, force_limit_n=None)
            x_task.write(shaker_voltage, auto_start=False)

            row = {
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
                "d_t": shaker_force,
                "shaker_force_n": shaker_force,
                "shaker_voltage_v": float(shaker_voltage),
                "unknown_force_start_s": float(unknown_force_start_s),
                "unknown_force_duration_steps": int(unknown_force_duration_steps),
                "F_ud_requested": f_ud_requested,
                "F_ud_true": f_ud_true,
                "F_ud_est": np.nan,
                "estimated_applied_action": np.nan,
                "residual_detection_value": np.nan,
                "residual_x2_error": np.nan,
                "residual_v2_error": np.nan,
                "residual_after_correction": np.nan,
                "event_trigger_flag": 0,
                "detection_flag": 0,
                "dgs_active_policy": str(dgs_active_policy if controller_name in DGS_CONTROLLER_NAMES else ""),
                "dgs_fallback_policy": str(getattr(controller, "fallback_policy", "") if controller_name in DGS_CONTROLLER_NAMES else ""),
                "dgs_fallback_active": int(controller_name in DGS_CONTROLLER_NAMES and dgs_active_policy != "ar_ppo"),
                "dgs_switch_to_adrc_flag": 0,
                "dgs_switch_to_arppo_flag": 0,
                "dgs_recovery_count": int(dgs_recovery_count if controller_name in DGS_CONTROLLER_NAMES else 0),
                "dgs_adrc_steps": int(dgs_adrc_steps if controller_name in DGS_CONTROLLER_NAMES else 0),
                "u_desired": float(u_desired),
                "u_cmd": float(u_cmd),
                "u_app": u_app,
                "unknown_compensation": float(unknown_compensation),
                "compensation_active": int(compensation_active),
                "command_saturation_flag": int(command_saturation),
                "u_raw": float(info.get("u_raw", u_desired)),
                "u_sat": float(info.get("u_sat", u_desired)),
                "base_u": float(info.get("base_u", 0.0)),
                "ppo_residual": float(info.get("ppo_residual", 0.0)),
                "saturation_flag": int(max(int(info.get("saturation_flag", 0)), command_saturation)),
                "fallback_flag": int(info.get("fallback_flag", 0)),
                "computation_time_ms": float(control_ms),
                "solve_time_ms": float(info.get("solve_time_ms", control_ms)),
                "diag_detect_threshold": float(detect_threshold),
                "diag_dgs_recovery_threshold": float(DGS_RECOVERY_THRESHOLD),
                "detector_gate_mode": DETECTOR_GATE_MODE,
                "estimator_mode": ESTIMATOR_MODE,
                "actuator_disturbance_model": ACTUATOR_DISTURBANCE_MODEL,
                "dgs_min_adrc_steps": int(DGS_MIN_ADRC_STEPS),
                "dgs_recovery_consecutive_steps": int(DGS_RECOVERY_CONSECUTIVE_STEPS),
                "dgs_refractory_steps": int(DGS_REFRACTORY_STEPS),
                "unknown_estimate_hold_steps": int(UNKNOWN_ESTIMATE_HOLD_STEPS),
                "unknown_estimate_cooldown_steps": int(UNKNOWN_ESTIMATE_COOLDOWN_STEPS),
                "unknown_compensation_delay_steps": int(UNKNOWN_COMPENSATION_DELAY_STEPS),
                "unknown_compensation_gain": float(UNKNOWN_COMPENSATION_GAIN),
                "unknown_allow_retrigger": int(bool(UNKNOWN_ALLOW_RETRIGGER)),
                "unknown_estimate_prior_n": float(UNKNOWN_ESTIMATE_PRIOR_N),
                "unknown_estimate_prior_weight": float(UNKNOWN_ESTIMATE_PRIOR_WEIGHT),
                "oracle_true_unknown": int(bool(oracle_true_unknown)),
            }
            rows.append(row)
            if (controller_name in {"ar_ppo_est", "ar_ppo_est_oracle"} or controller_name in DGS_CONTROLLER_NAMES) and dynamics is not None and estimated_history is not None:
                pending_buffer.append(
                    {
                        "row_idx": len(rows) - 1,
                        "state": state.copy(),
                        "u_cmd": float(u_cmd),
                        "u_app": float(u_app),
                        "F_ud_true": float(f_ud_true),
                        "disturbance": shaker_force,
                        "history": estimated_history.copy(),
                    }
                )
                estimated_history = dynamics.append_history(
                    estimated_history,
                    build_step_feature(state, float(u_cmd), shaker_force),
                )
            prev_action_for_obs = float(u_cmd)
            prev_raw = raw_xy.copy()
    finally:
        finish_hardware_trial(x_task, x_ser, y_ser, arduino_task, post_wait_s=post_wait_s)

    if len(rows) != len(disturbance):
        raise RuntimeError(f"Expected {len(disturbance)} rows, collected {len(rows)} rows.")
    path = Path(output_path) if output_path is not None else rollout_path(controller_name, repeat)
    save_rollout(path, rows)
    return path


def num_col(rows: list[dict], key: str, default=np.nan) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, default)))
        except Exception:
            vals.append(default)
    return np.asarray(vals, dtype=np.float64)


def rollout_metrics(path: Path) -> dict:
    rows = read_csv_rows(path)
    first = rows[0]
    time_s = num_col(rows, "time", 0.0)
    event_start_s = float(first.get("unknown_force_start_s", UNKNOWN_START_S) or UNKNOWN_START_S)
    event_duration_steps = int(float(first.get("unknown_force_duration_steps", UNKNOWN_DURATION_STEPS) or UNKNOWN_DURATION_STEPS))
    event_end_s = event_start_s + float(event_duration_steps) * DT + float(DISTURBANCE_METRIC_POST_S)
    event_mask = (time_s >= event_start_s) & (time_s <= event_end_s)
    if not np.any(event_mask):
        event_mask = np.ones_like(time_s, dtype=bool)
    out = {
        "controller": first["controller"],
        "controller_label": first.get("controller_label", first["controller"]),
        "repeat": int(float(first["repeat"])),
        "path": str(path),
        "event_window_start_s": event_start_s,
        "event_window_end_s": event_end_s,
    }
    for key in ("x2", "v2", "a2"):
        arr = num_col(rows, key, 0.0)
        out[f"{key}_rms"] = float(np.sqrt(np.mean(arr**2)))
        out[f"{key}_mean_abs"] = float(np.mean(np.abs(arr)))
        out[f"{key}_peak_abs"] = float(np.max(np.abs(arr)))
        event_arr = arr[event_mask]
        out[f"{key}_event_rms"] = float(np.sqrt(np.mean(event_arr**2)))
        out[f"{key}_event_mean_abs"] = float(np.mean(np.abs(event_arr)))
        out[f"{key}_event_peak_abs"] = float(np.max(np.abs(event_arr)))
    compute = num_col(rows, "computation_time_ms", 0.0)
    out["compute_time_mean_ms"] = float(np.mean(compute))
    out["compute_time_std_ms"] = float(np.std(compute, ddof=1)) if len(compute) > 1 else 0.0
    out["compute_time_max_ms"] = float(np.max(compute))
    out["saturation_count"] = int(np.sum(num_col(rows, "saturation_flag", 0.0) > 0.5))
    out["fallback_count"] = int(np.sum(num_col(rows, "fallback_flag", 0.0) > 0.5))
    out["detection_count"] = int(np.sum(num_col(rows, "detection_flag", 0.0) > 0.5))
    out["event_trigger_count"] = int(np.sum(num_col(rows, "event_trigger_flag", 0.0) > 0.5))
    out["compensation_active_count"] = int(np.sum(num_col(rows, "compensation_active", 0.0) > 0.5))
    out["dgs_adrc_active_count"] = int(np.sum(num_col(rows, "dgs_fallback_active", 0.0) > 0.5))
    out["dgs_switch_to_adrc_count"] = int(np.sum(num_col(rows, "dgs_switch_to_adrc_flag", 0.0) > 0.5))
    out["dgs_fallback_active_count"] = out["dgs_adrc_active_count"]
    out["dgs_switch_to_fallback_count"] = out["dgs_switch_to_adrc_count"]
    out["dgs_switch_to_arppo_count"] = int(np.sum(num_col(rows, "dgs_switch_to_arppo_flag", 0.0) > 0.5))
    true_unknown = num_col(rows, "F_ud_true", np.nan)
    est_unknown = num_col(rows, "F_ud_est", np.nan)
    valid = np.isfinite(true_unknown) & np.isfinite(est_unknown)
    out["mean_abs_Fud_error"] = float(np.mean(np.abs(est_unknown[valid] - true_unknown[valid]))) if np.any(valid) else np.nan
    return out


def refresh_metric_tables(controllers: list[str], repeats: int) -> None:
    repeat_rows = []
    for controller in controllers:
        for repeat in range(1, int(repeats) + 1):
            path = rollout_path(controller, repeat)
            if valid_rollout(path):
                repeat_rows.append(rollout_metrics(path))
    fields = [
        "controller",
        "controller_label",
        "repeat",
        "path",
        "event_window_start_s",
        "event_window_end_s",
        "x2_rms",
        "x2_mean_abs",
        "x2_peak_abs",
        "x2_event_rms",
        "x2_event_mean_abs",
        "x2_event_peak_abs",
        "v2_rms",
        "v2_mean_abs",
        "v2_peak_abs",
        "v2_event_rms",
        "v2_event_mean_abs",
        "v2_event_peak_abs",
        "a2_rms",
        "a2_mean_abs",
        "a2_peak_abs",
        "a2_event_rms",
        "a2_event_mean_abs",
        "a2_event_peak_abs",
        "compute_time_mean_ms",
        "compute_time_std_ms",
        "compute_time_max_ms",
        "saturation_count",
        "fallback_count",
        "detection_count",
        "event_trigger_count",
        "compensation_active_count",
        "dgs_adrc_active_count",
        "dgs_switch_to_adrc_count",
        "dgs_fallback_active_count",
        "dgs_switch_to_fallback_count",
        "dgs_switch_to_arppo_count",
        "mean_abs_Fud_error",
    ]
    write_csv(DATA_DIR / "repeat_metrics.csv", repeat_rows, fields)
    aggregate_rows = []
    metrics = [f for f in fields if f not in {"controller", "controller_label", "repeat", "path"}]
    for controller in controllers:
        group = [r for r in repeat_rows if r["controller"] == controller]
        if not group:
            continue
        row = {"controller": controller, "controller_label": CONTROLLER_LABELS[controller], "n_repeats": len(group)}
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group if r[metric] == r[metric]], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        aggregate_rows.append(row)
    aggregate_fields = []
    for row in aggregate_rows:
        for key in row:
            if key not in aggregate_fields:
                aggregate_fields.append(key)
    write_csv(DATA_DIR / "aggregate_statistics.csv", aggregate_rows, aggregate_fields)


def collect_all(args: argparse.Namespace) -> None:
    _, disturbance = load_fixed_disturbance()
    unknown_force, unknown_meta = generate_unknown_force(
        args.unknown_force_n,
        args.unknown_start_s,
        args.unknown_duration_steps,
        len(disturbance),
    )
    save_unknown_force_csv(unknown_force, unknown_meta)
    controllers = [str(c) for c in args.controllers]
    best_params = CONTROL_SUPPORT.load_best_params()
    dynamics = DDMDynamics() if any(c in {"ar_ppo_est"} or c in DGS_CONTROLLER_NAMES for c in controllers) else None
    controller_cache = {name: make_controller(name, best_params, dynamics) for name in controllers}

    total_jobs = len(controllers) * int(args.repeats)
    completed = sum(valid_rollout(rollout_path(c, r), len(disturbance)) for c in controllers for r in range(1, int(args.repeats) + 1))
    print(f"Fixed disturbance: {DATA_DIR / 'fixed_test_disturbance.csv'}")
    print(f"Unknown force: {unknown_meta}")
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
                if valid_rollout(path, len(disturbance)) and not args.overwrite:
                    print(f"Skipping {controller_name} repeat {repeat}/{args.repeats}: already saved")
                    continue
                controller = controller_cache[controller_name]
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
                            dynamics if controller_name in {"ar_ppo_est"} or controller_name in DGS_CONTROLLER_NAMES else None,
                            disturbance,
                            unknown_force,
                            hardware,
                            x_task,
                            x_ser,
                            y_ser,
                            arduino_task,
                            settle_time_s=args.settle_time_s,
                            post_wait_s=args.post_wait_s,
                            jump_abort_m=args.jump_abort_m,
                            abs_displacement_limit_m=args.abs_displacement_limit_m,
                            detect_threshold=args.detect_threshold,
                            unknown_force_start_s=args.unknown_start_s,
                            unknown_force_duration_steps=args.unknown_duration_steps,
                        )
                        print(f"Saved: {saved.relative_to(SCRIPT_DIR)}")
                        refresh_metric_tables(controllers, args.repeats)
                        break
                    except RolloutAbort as exc:
                        print(f"Restarting {controller_name} repeat {repeat}/{args.repeats}: {exc}")
                        if attempt >= int(args.max_retries):
                            raise RuntimeError(f"{controller_name} repeat {repeat} exceeded max_retries={args.max_retries}") from exc
    finally:
        close_devices(x_task, x_ser, y_ser, arduino_task)
    refresh_metric_tables(controllers, args.repeats)
    print(f"Saved repeat metrics: {DATA_DIR / 'repeat_metrics.csv'}")
    print(f"Saved aggregate statistics: {DATA_DIR / 'aggregate_statistics.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardware comparison under a fixed actuator-side input disturbance.")
    parser.add_argument("--controllers", nargs="+", default=DEFAULT_CONTROLLERS, choices=AVAILABLE_CONTROLLERS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--settle-time-s", type=float, default=2.0)
    parser.add_argument("--post-wait-s", type=float, default=2.0)
    parser.add_argument("--jump-abort-m", type=float, default=0.008)
    parser.add_argument("--abs-displacement-limit-m", type=float, default=0.04)
    parser.add_argument("--unknown-start-s", type=float, default=UNKNOWN_START_S)
    parser.add_argument("--unknown-duration-steps", type=int, default=UNKNOWN_DURATION_STEPS)
    parser.add_argument("--unknown-force-n", type=float, default=UNKNOWN_FORCE_N)
    parser.add_argument("--detect-threshold", type=float, default=DETECT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    collect_all(parse_args())
