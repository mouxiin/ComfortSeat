from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.filters import CUTOFF_HZ, lowpass_alpha
from utils.signals import DT, SHAKER_FORCE_LIMIT_N, build_channel_signal, scale_to_limit


DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR = SCRIPT_DIR / "models"
DDM_CHECKPOINT = PROJECT_DIR / "ddm_ablation" / "models" / "selected_model.pt"
ADRC_TUNING_CSV = PROJECT_DIR / "tune_parameter" / "data" / "best_by_method.csv"

for directory in (DATA_DIR, FIGURES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STATE_DIM = 4
BASE_INPUT_DIM = 6
OBS_DIM = 6
U_MAX = 2.0
M2 = 0.85
K2 = 135.0
C2 = 6.4
B0_ADRC = 1.0 / M2

OBS_SCALE = np.asarray([0.02, 0.6, 0.02, 0.6, U_MAX, SHAKER_FORCE_LIMIT_N], dtype=np.float32)
REWARD_SCALE = 0.1
W_X = 180.0
W_V = 6.0
W_A = 2.0
W_U = 0.04
W_DU = 0.08

GAMMA = 0.995
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
TARGET_KL = 0.03
ENTROPY_COEF = 0.01
STD_INIT = 0.35
STD_MIN = 0.03

RESIDUAL_U_MAX_INIT = 0.8
RESIDUAL_U_MAX_MAX = U_MAX
RESIDUAL_U_MAX_GROWTH = 1.35
STABILITY_WINDOW = 8
STABILITY_WARMUP = 18
STABILITY_STD_ABS = 0.02
STABILITY_STD_RATIO = 0.06
STABILITY_SLOPE_ABS = 6.0e-4
RESIDUAL_UPDATE_COOLDOWN = 8

METHODS = {
    "non_residual": {
        "label": "NR-PPO",
        "mode": "non_residual",
        "adaptive": False,
        "initial_residual_u_max": RESIDUAL_U_MAX_MAX,
    },
    "fixed_residual": {
        "label": "FR-PPO",
        "mode": "residual",
        "adaptive": False,
        "initial_residual_u_max": RESIDUAL_U_MAX_MAX,
    },
    "adaptive_residual": {
        "label": "AR-PPO",
        "mode": "residual",
        "adaptive": True,
        "initial_residual_u_max": RESIDUAL_U_MAX_INIT,
    },
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return torch.device(name)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
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
    def __init__(self, checkpoint_path: Path, device: torch.device):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing DDM checkpoint: {checkpoint_path.resolve()}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        spec = dict(ckpt["spec"])
        self.history_input = bool(spec.get("history_input", True))
        self.history_steps = int(ckpt["history_steps"]) if self.history_input else 0
        self.residual_target = bool(spec["residual_target"])
        self.device = device
        self.model = make_ddm_model(
            int(len(ckpt["stats"]["x_mean"])),
            int(ckpt["hidden_dim"]),
            bool(spec["residual_blocks"]),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.x_mean = torch.as_tensor(ckpt["stats"]["x_mean"], dtype=torch.float32, device=device)
        self.x_std = torch.as_tensor(ckpt["stats"]["x_std"], dtype=torch.float32, device=device)
        self.y_mean = torch.as_tensor(ckpt["stats"]["y_mean"], dtype=torch.float32, device=device)
        self.y_std = torch.as_tensor(ckpt["stats"]["y_std"], dtype=torch.float32, device=device)
        self.checkpoint_path = str(checkpoint_path.resolve())

    def zero_history(self, batch_size: int) -> np.ndarray:
        return np.zeros((int(batch_size), self.history_steps, BASE_INPUT_DIM), dtype=np.float32)

    @staticmethod
    def append_history(history: np.ndarray, current_input: np.ndarray) -> np.ndarray:
        if history.shape[1] == 0:
            return history
        return np.concatenate([current_input[:, None, :], history[:, :-1, :]], axis=1).astype(np.float32)

    @torch.no_grad()
    def step_batch(
        self,
        states: np.ndarray,
        command_force: np.ndarray,
        disturbance_force: np.ndarray,
        history: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        u = np.clip(np.asarray(command_force, dtype=np.float32), -U_MAX, U_MAX)
        d = np.asarray(disturbance_force, dtype=np.float32)
        current_input = np.concatenate([states, u[:, None], d[:, None]], axis=1).astype(np.float32)
        if self.history_input:
            feature = np.concatenate([current_input, history.reshape(states.shape[0], -1)], axis=1).astype(np.float32)
        else:
            feature = current_input
        x = torch.as_tensor(feature, dtype=torch.float32, device=self.device)
        pred_n = self.model((x - self.x_mean) / self.x_std)
        target = pred_n * self.y_std + self.y_mean
        state_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = (state_t + target if self.residual_target else target).cpu().numpy().astype(np.float32)
        a2_raw = ((next_states[:, 3] - states[:, 3]) / DT).astype(np.float32)
        next_history = self.append_history(history, current_input)
        return next_states, u.astype(np.float32), a2_raw, next_history


def build_observation_batch(states: np.ndarray, prev_action: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
    obs = np.concatenate(
        [
            np.asarray(states, dtype=np.float32),
            np.asarray(prev_action, dtype=np.float32)[:, None],
            np.asarray(disturbance, dtype=np.float32)[:, None],
        ],
        axis=1,
    )
    return (obs / OBS_SCALE.reshape(1, -1)).astype(np.float32)


def physical_from_observation(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32) * OBS_SCALE


def reward_from_step(next_states: np.ndarray, a2: np.ndarray, action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    x2 = next_states[:, 2]
    v2 = next_states[:, 3]
    u = np.asarray(action, dtype=np.float32)
    du = u - np.asarray(prev_action, dtype=np.float32)
    cost = W_X * x2**2 + W_V * v2**2 + W_A * np.asarray(a2) ** 2 + W_U * u**2 + W_DU * du**2
    return (-REWARD_SCALE * cost).astype(np.float32)


class DDMSurrogateVectorEnv:
    def __init__(self, dynamics: DDMDynamics):
        self.dynamics = dynamics

    def reset(self, disturbances: np.ndarray) -> np.ndarray:
        self.disturbances = np.asarray(disturbances, dtype=np.float32)
        self.num_envs = int(self.disturbances.shape[0])
        self.states = np.zeros((self.num_envs, STATE_DIM), dtype=np.float32)
        self.prev_action = np.zeros(self.num_envs, dtype=np.float32)
        self.a2_filtered = np.zeros(self.num_envs, dtype=np.float32)
        self.history = self.dynamics.zero_history(self.num_envs)
        self.step_index = 0
        return self.current_observation()

    def current_observation(self) -> np.ndarray:
        d_t = self.disturbances[:, min(self.step_index, self.disturbances.shape[1] - 1)]
        return build_observation_batch(self.states, self.prev_action, d_t)

    def step(self, command_force: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        d_t = self.disturbances[:, self.step_index]
        prev_action = self.prev_action.copy()
        next_states, applied, a2_raw, next_history = self.dynamics.step_batch(
            self.states, command_force, d_t, self.history
        )
        alpha = lowpass_alpha(DT, CUTOFF_HZ)
        self.a2_filtered = (self.a2_filtered + alpha * (a2_raw - self.a2_filtered)).astype(np.float32)
        a2 = self.a2_filtered.copy()
        reward = reward_from_step(next_states, a2, applied, prev_action)
        self.states = next_states
        self.prev_action = applied
        self.history = next_history
        self.step_index += 1
        done = self.step_index >= self.disturbances.shape[1] - 1
        info = {"applied_force": applied, "a2": a2, "state": next_states}
        return self.current_observation(), reward, done, info


def load_adrc_params(path: Path = ADRC_TUNING_CSV) -> dict:
    defaults = {"omega_o": 16.0, "kp": 30.0, "kd": 6.0, "b0_scale": 1.0, "variant": "fallback_adrc"}
    path = Path(path)
    if not path.exists():
        return defaults
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "adrc":
                out = dict(defaults)
                out.update(
                    {
                        "omega_o": float(row.get("param_omega_o") or defaults["omega_o"]),
                        "kp": float(row.get("param_kp") or defaults["kp"]),
                        "kd": float(row.get("param_kd") or defaults["kd"]),
                        "b0_scale": float(row.get("param_b0_scale") or defaults["b0_scale"]),
                        "variant": row.get("variant") or defaults["variant"],
                    }
                )
                return out
    return defaults


class PassiveController:
    name = "passive"

    def reset(self, num_envs: int = 1) -> None:
        pass

    def act_batch(self, obs_batch: np.ndarray, deterministic: bool = True):
        zeros = np.zeros(len(obs_batch), dtype=np.float32)
        return zeros, zeros, zeros

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        return np.zeros(len(obs_batch), dtype=np.float32)


class ADRCController:
    name = "adrc"

    def __init__(self, omega_o: float, kp: float, kd: float, b0_scale: float = 1.0):
        self.omega_o = float(omega_o)
        self.kp = float(kp)
        self.kd = float(kd)
        self.b0 = B0_ADRC * float(b0_scale)
        self.z = np.zeros((0, 3), dtype=np.float32)
        self.initialized = np.zeros(0, dtype=bool)

    def reset(self, num_envs: int = 1) -> None:
        self.z = np.zeros((int(num_envs), 3), dtype=np.float32)
        self.initialized = np.zeros(int(num_envs), dtype=bool)

    def act_batch(self, obs_batch: np.ndarray, deterministic: bool = True):
        physical = physical_from_observation(obs_batch)
        states = physical[:, :4]
        u_prev = physical[:, 4]
        n = states.shape[0]
        if self.z.shape[0] != n:
            self.reset(n)
        x1, v1, x2, v2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        not_init = ~self.initialized
        if np.any(not_init):
            self.z[not_init, 0] = x2[not_init]
            self.z[not_init, 1] = v2[not_init]
            self.z[not_init, 2] = 0.0
            self.initialized[not_init] = True
        if abs(self.b0) < 1.0e-8:
            force = np.zeros(n, dtype=np.float32)
            return force, force, np.zeros(n, dtype=np.float32)
        f0 = (K2 * (x1 - x2) + C2 * (v1 - v2)) / M2
        beta1 = 3.0 * self.omega_o
        beta2 = 3.0 * self.omega_o**2
        beta3 = self.omega_o**3
        e = x2 - self.z[:, 0]
        self.z[:, 0] += DT * (self.z[:, 1] + beta1 * e)
        self.z[:, 1] += DT * (f0 + self.z[:, 2] + self.b0 * u_prev + beta2 * e)
        self.z[:, 2] += DT * (beta3 * e)
        desired_acc = -self.kp * self.z[:, 0] - self.kd * self.z[:, 1]
        u_raw = (desired_acc - f0 - self.z[:, 2]) / self.b0
        force = np.clip(u_raw, -U_MAX, U_MAX).astype(np.float32)
        pre_tanh = np.arctanh(np.clip(force / U_MAX, -0.999999, 0.999999)).astype(np.float32)
        return force, pre_tanh, np.zeros(n, dtype=np.float32)

    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        return np.zeros(len(obs_batch), dtype=np.float32)


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


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def squashed_log_prob(dist: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
    return (dist.log_prob(pre_tanh) - torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1.0e-6)).sum(dim=1, keepdim=True)


def compute_advantage(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, dones: np.ndarray) -> np.ndarray:
    deltas = rewards + GAMMA * next_values * (1.0 - dones) - values
    advantages = np.zeros_like(deltas, dtype=np.float32)
    gae = 0.0
    for idx in range(len(deltas) - 1, -1, -1):
        gae = deltas[idx] + GAMMA * LAMBDA_GAE * (1.0 - dones[idx]) * gae
        advantages[idx] = gae
    return advantages


class PPOAgent:
    def __init__(
        self,
        mode: str,
        hidden_dim: int,
        device: torch.device,
        residual_u_max: float,
        total_episodes: int,
        base_controller: ADRCController | None = None,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 1.0e-3,
    ):
        self.mode = str(mode)
        self.device = device
        self.actor = Actor(OBS_DIM, hidden_dim).to(device)
        self.critic = Critic(OBS_DIM, hidden_dim).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optim, start_factor=1.0, end_factor=0.2, total_iters=max(int(total_episodes), 1)
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.critic_optim, start_factor=1.0, end_factor=0.2, total_iters=max(int(total_episodes), 1)
        )
        self.residual_u_max = float(residual_u_max)
        self.base_controller = base_controller

    def reset(self, num_envs: int = 1) -> None:
        if self.base_controller is not None:
            self.base_controller.reset(num_envs)

    def actor_std(self) -> float:
        return float(torch.exp(self.actor.log_std.detach()).cpu().item())

    def rescale_actor_output(self, old_u_max: float, new_u_max: float) -> None:
        if old_u_max <= 0.0 or new_u_max <= 0.0:
            return
        scale = float(old_u_max / new_u_max)
        with torch.no_grad():
            last = self.actor.net[-1]
            last.weight.mul_(scale)
            last.bias.mul_(scale)
            self.actor.log_std.add_(math.log(scale))

    def action_from_pre_tanh_batch(self, obs_batch: np.ndarray, pre_tanh: torch.Tensor) -> torch.Tensor:
        if self.mode == "residual":
            if self.base_controller is None:
                raise RuntimeError("Residual PPO requires a base controller.")
            base_force = self.base_controller.act_batch(obs_batch, deterministic=True)[0]
            base_t = torch.as_tensor(base_force, dtype=torch.float32, device=self.device).view(-1, 1)
            residual = self.residual_u_max * torch.tanh(pre_tanh)
            return torch.clamp(base_t + residual, -U_MAX, U_MAX)
        return U_MAX * torch.tanh(pre_tanh)

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        mean, std = self.actor(obs_t)
        dist = Normal(mean, std)
        pre_tanh = mean if deterministic else dist.rsample()
        action = self.action_from_pre_tanh_batch(obs_batch, pre_tanh)
        log_prob = squashed_log_prob(dist, pre_tanh)
        return (
            action.squeeze(1).cpu().numpy().astype(np.float32),
            pre_tanh.squeeze(1).cpu().numpy().astype(np.float32),
            log_prob.squeeze(1).cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def value_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        return self.critic(obs_t).squeeze(1).cpu().numpy().astype(np.float32)

    def update(self, traj: dict[str, np.ndarray], epochs: int, minibatch_size: int) -> dict:
        self.actor.train()
        self.critic.train()
        obs = torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device)
        pre_tanh = torch.as_tensor(traj["pre_tanh"], dtype=torch.float32, device=self.device).view(-1, 1)
        old_log_prob = torch.as_tensor(traj["log_prob"], dtype=torch.float32, device=self.device).view(-1, 1)
        returns = torch.as_tensor(traj["returns"], dtype=torch.float32, device=self.device).view(-1, 1)
        advantages = torch.as_tensor(traj["advantages"], dtype=torch.float32, device=self.device).view(-1, 1)
        advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(), min=1.0e-8)
        num_samples = obs.shape[0]
        approx_kl = 0.0
        actor_loss_v = 0.0
        critic_loss_v = 0.0
        for _ in range(int(epochs)):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, int(minibatch_size)):
                idx = indices[start : start + int(minibatch_size)]
                mean, std = self.actor(obs[idx])
                dist = Normal(mean, std)
                new_log_prob = squashed_log_prob(dist, pre_tanh[idx])
                ratio = torch.exp(new_log_prob - old_log_prob[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages[idx]
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy
                critic_loss = torch.nn.functional.mse_loss(self.critic(obs[idx]), returns[idx])
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                actor_loss_v = float(actor_loss.detach().cpu().item())
                critic_loss_v = float(critic_loss.detach().cpu().item())
            with torch.no_grad():
                mean, std = self.actor(obs)
                approx_kl = float((old_log_prob - squashed_log_prob(Normal(mean, std), pre_tanh)).mean().cpu().item())
            if approx_kl > TARGET_KL:
                break
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return {"actor_loss": actor_loss_v, "critic_loss": critic_loss_v, "approx_kl": approx_kl}

    def save(self, path: Path, episode: int, seed: int, method: str, adrc_params: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "mode": self.mode,
                "residual_u_max": self.residual_u_max,
                "episode": int(episode),
                "seed": int(seed),
                "method": method,
                "obs_scale": OBS_SCALE,
                "adrc_params": adrc_params,
                "ddm_checkpoint": str(DDM_CHECKPOINT.resolve()),
            },
            path,
        )


def generate_disturbance_suite(count: int, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    suite = []
    for idx in range(int(count)):
        local = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        kind = str(local.choice(["multisine", "chirp", "mixed"]))
        direction = "down" if idx % 2 else "up"
        signal = build_channel_signal(
            local,
            kind,
            int(steps) + 1,
            DT,
            SHAKER_FORCE_LIMIT_N,
            chirp_direction=direction,
            allow_prbs=False,
            smooth_edges=True,
        )
        ramp_steps = max(1, int(round(0.5 / DT)))
        window = np.ones_like(signal, dtype=np.float32)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp_steps))).astype(np.float32)
        window[:ramp_steps] = ramp
        window[-ramp_steps:] = ramp[::-1]
        suite.append(scale_to_limit(signal * window, SHAKER_FORCE_LIMIT_N))
    return np.stack(suite, axis=0).astype(np.float32)


def save_fixed_eval_disturbances(path: Path, disturbances: np.ndarray) -> None:
    rows = []
    for case_idx, signal in enumerate(np.asarray(disturbances), start=1):
        for sample, value in enumerate(signal):
            rows.append(
                {
                    "case": f"eval_{case_idx:02d}",
                    "case_index": case_idx,
                    "sample": sample,
                    "time": sample * DT,
                    "shaker_force_n": float(value),
                }
            )
    write_csv(path, rows, ["case", "case_index", "sample", "time", "shaker_force_n"])


def rollout_controller_batch(controller, disturbances: np.ndarray, dynamics: DDMDynamics, deterministic: bool = True) -> dict:
    disturbances = np.asarray(disturbances, dtype=np.float32)
    num_envs = disturbances.shape[0]
    num_steps = disturbances.shape[1] - 1
    if hasattr(controller, "reset"):
        controller.reset(num_envs)
    env = DDMSurrogateVectorEnv(dynamics)
    obs = env.reset(disturbances)
    obs_traj = np.zeros((num_steps, num_envs, OBS_DIM), dtype=np.float32)
    pre_tanh_traj = np.zeros((num_steps, num_envs), dtype=np.float32)
    log_prob_traj = np.zeros((num_steps, num_envs), dtype=np.float32)
    reward_traj = np.zeros((num_steps, num_envs), dtype=np.float32)
    value_traj = np.zeros((num_steps, num_envs), dtype=np.float32)
    done_traj = np.zeros((num_steps, num_envs), dtype=np.float32)

    for step in range(num_steps):
        action, pre_tanh, log_prob = controller.act_batch(obs, deterministic=deterministic)
        value = controller.value_batch(obs)
        next_obs, reward, done, _info = env.step(action)
        obs_traj[step] = obs
        pre_tanh_traj[step] = pre_tanh
        log_prob_traj[step] = log_prob
        reward_traj[step] = reward
        value_traj[step] = value
        done_traj[step] = 1.0 if done else 0.0
        obs = next_obs
    if num_steps:
        done_traj[-1] = 1.0
    next_values = np.zeros_like(value_traj)
    if num_steps > 1:
        next_values[:-1] = value_traj[1:]
    advantages = np.zeros_like(reward_traj)
    returns = np.zeros_like(reward_traj)
    for env_idx in range(num_envs):
        adv = compute_advantage(reward_traj[:, env_idx], value_traj[:, env_idx], next_values[:, env_idx], done_traj[:, env_idx])
        advantages[:, env_idx] = adv
        returns[:, env_idx] = adv + value_traj[:, env_idx]
    episode_returns = reward_traj.sum(axis=0).astype(np.float32)
    return {
        "traj": {
            "obs": obs_traj.reshape(-1, OBS_DIM),
            "pre_tanh": pre_tanh_traj.reshape(-1),
            "log_prob": log_prob_traj.reshape(-1),
            "returns": returns.reshape(-1),
            "advantages": advantages.reshape(-1),
        },
        "case_returns": episode_returns,
        "average_return": float(episode_returns.mean()),
        "return_std": float(episode_returns.std()),
    }


def evaluate_controller(controller, fixed_eval_disturbances: np.ndarray, dynamics: DDMDynamics) -> dict:
    return rollout_controller_batch(controller, fixed_eval_disturbances, dynamics, deterministic=True)


def record_eval(records: list[dict], episode: int, method: str, seed: int, result: dict, agent: PPOAgent | None = None) -> None:
    records.append(
        {
            "episode": int(episode),
            "method": method,
            "seed": int(seed),
            "average_return": float(result["average_return"]),
            "return_std": float(result["return_std"]),
            "residual_u_max": "" if agent is None else float(agent.residual_u_max),
            "actor_std": "" if agent is None else float(agent.actor_std()),
        }
    )


def is_return_stable(eval_returns: list[float], episode_steps: int) -> bool:
    if len(eval_returns) < STABILITY_WINDOW:
        return False
    window = np.asarray(eval_returns[-STABILITY_WINDOW:], dtype=np.float32) / max(int(episode_steps), 1)
    mean = float(window.mean())
    std = float(window.std())
    slope = float((window[-1] - window[0]) / max(len(window) - 1, 1))
    std_threshold = max(STABILITY_STD_ABS, STABILITY_STD_RATIO * max(abs(mean), 1.0e-6))
    return std <= std_threshold and abs(slope) <= STABILITY_SLOPE_ABS


def make_adrc(params: dict) -> ADRCController:
    return ADRCController(params["omega_o"], params["kp"], params["kd"], params["b0_scale"])


def train_method(method_key: str, method: dict, args, dynamics: DDMDynamics, fixed_eval: np.ndarray, adrc_params: dict) -> list[dict]:
    method_seed_offsets = {"non_residual": 101, "fixed_residual": 202, "adaptive_residual": 303}
    set_seed(int(args.seed) + method_seed_offsets.get(method_key, 0))
    base = make_adrc(adrc_params) if method["mode"] == "residual" else None
    agent = PPOAgent(
        mode=method["mode"],
        hidden_dim=int(args.hidden_dim),
        device=dynamics.device,
        residual_u_max=float(method["initial_residual_u_max"]),
        total_episodes=int(args.episodes),
        base_controller=base,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
    )
    records: list[dict] = []
    eval0 = evaluate_controller(agent, fixed_eval, dynamics)
    record_eval(records, 0, method_key, args.seed, eval0, agent)
    eval_history = [float(eval0["average_return"])]
    last_residual_update = -RESIDUAL_UPDATE_COOLDOWN
    returns_path = DATA_DIR / f"training_returns_{method_key}.csv"
    write_csv(returns_path, records)

    start_time = time.time()
    for episode in range(1, int(args.episodes) + 1):
        disturbances = generate_disturbance_suite(int(args.num_envs), int(args.steps), int(args.seed) * 1000 + episode)
        train_result = rollout_controller_batch(agent, disturbances, dynamics, deterministic=False)
        update_info = agent.update(train_result["traj"], int(args.ppo_epochs), int(args.minibatch_size))
        if episode % int(args.eval_interval) == 0 or episode == int(args.episodes):
            eval_result = evaluate_controller(agent, fixed_eval, dynamics)
            eval_history.append(float(eval_result["average_return"]))
            if (
                bool(method["adaptive"])
                and episode >= STABILITY_WARMUP
                and (episode - last_residual_update) >= RESIDUAL_UPDATE_COOLDOWN
                and agent.residual_u_max < RESIDUAL_U_MAX_MAX
                and is_return_stable(eval_history, int(args.steps))
            ):
                old = agent.residual_u_max
                new = min(old * RESIDUAL_U_MAX_GROWTH, RESIDUAL_U_MAX_MAX)
                agent.rescale_actor_output(old, new)
                agent.residual_u_max = new
                last_residual_update = episode
            record_eval(records, episode, method_key, args.seed, eval_result, agent)
            write_csv(returns_path, records)
            agent.save(MODELS_DIR / f"{method_key}_seed_{int(args.seed)}.pt", episode, args.seed, method_key, adrc_params)
        print(
            f"{method_key} episode {episode:03d}/{int(args.episodes):03d} "
            f"train={train_result['average_return']:.5g} "
            f"eval={eval_history[-1]:.5g} "
            f"std={agent.actor_std():.3f} "
            f"res={agent.residual_u_max:.2f} "
            f"kl={update_info['approx_kl']:.3g}"
        )
    agent.save(MODELS_DIR / f"{method_key}_seed_{int(args.seed)}.pt", int(args.episodes), args.seed, method_key, adrc_params)
    records[-1]["training_duration_s"] = float(time.time() - start_time)
    write_csv(returns_path, records)
    return records


def final_summary_rows(records: list[dict]) -> list[dict]:
    rows = []
    for method in ("passive", "adrc", "non_residual", "fixed_residual", "adaptive_residual"):
        sub = [row for row in records if row["method"] == method]
        if not sub:
            continue
        best = max(sub, key=lambda row: float(row["average_return"]))
        final = sorted(sub, key=lambda row: int(row["episode"]))[-1]
        rows.append(
            {
                "method": method,
                "best_episode": int(best["episode"]),
                "best_average_return": float(best["average_return"]),
                "final_episode": int(final["episode"]),
                "final_average_return": float(final["average_return"]),
                "final_return_std": float(final["return_std"]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO controllers on the selected DDM surrogate.")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=1.0e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--methods", nargs="+", default=["non_residual", "fixed_residual", "adaptive_residual"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.steps = int(round(float(args.duration) / DT))
    set_seed(args.seed)
    device = resolve_device(args.device)
    dynamics = DDMDynamics(DDM_CHECKPOINT, device)
    adrc_params = load_adrc_params()
    fixed_eval = generate_disturbance_suite(int(args.eval_count), int(args.steps), int(args.seed) + 20260413)
    save_fixed_eval_disturbances(DATA_DIR / "fixed_eval_disturbances.csv", fixed_eval)

    records: list[dict] = []
    passive = PassiveController()
    adrc = make_adrc(adrc_params)
    record_eval(records, 0, "passive", args.seed, evaluate_controller(passive, fixed_eval, dynamics))
    record_eval(records, 0, "adrc", args.seed, evaluate_controller(adrc, fixed_eval, dynamics))
    for method_key in args.methods:
        if method_key not in METHODS:
            raise KeyError(f"Unknown method {method_key}. Available: {sorted(METHODS)}")
        records.extend(train_method(method_key, METHODS[method_key], args, dynamics, fixed_eval, adrc_params))

    write_csv(DATA_DIR / "average_return_comparison.csv", records)
    write_csv(DATA_DIR / "final_summary.csv", final_summary_rows(records))
    print(f"DDM checkpoint: {dynamics.checkpoint_path}")
    print(f"ADRC variant: {adrc_params['variant']}")
    print(f"Saved CSV results to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
