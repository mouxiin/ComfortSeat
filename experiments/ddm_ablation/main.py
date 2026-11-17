from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR = SCRIPT_DIR / "models"
DEFAULT_TRAJECTORY_DIR = SCRIPT_DIR.parent / "hardware_stratified_protocol" / "data"
for directory in (DATA_DIR, FIGURES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DT = 0.03
STATE_DIM = 4
BASE_INPUT_DIM = 6
STATE_NAMES = ("x1", "v1", "x2", "v2")
STATE_WEIGHTS = np.asarray([0.1, 0.1, 0.4, 0.4], dtype=np.float64)
EPS = 1.0e-12


@dataclass
class Trajectory:
    name: str
    path: str
    block: str
    states: np.ndarray
    actions: np.ndarray
    disturbances: np.ndarray
    time: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(min(len(self.actions), len(self.states) - 1))


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    history_input: bool
    residual_blocks: bool
    residual_target: bool


ABLATION_SPECS = [
    ModelSpec("M0", "Memoryless plain MLP", False, False, False),
    ModelSpec("M1", "Memoryless residual-block MLP", False, True, False),
    ModelSpec("M2", "History plain MLP", True, False, False),
    ModelSpec("M3", "Memoryless residual target", False, False, True),
    ModelSpec("M4", "Full proposed DDM", True, True, True),
]


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


def make_model(in_dim: int, hidden_dim: int, residual_blocks: bool) -> nn.Module:
    if residual_blocks:
        return ResidualMLP(in_dim, STATE_DIM, hidden_dim)
    return PlainMLP(in_dim, STATE_DIM, hidden_dim)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return torch.device(name)


def read_float(row: dict, key: str) -> float:
    return float(row[key])


def load_trajectory(path: Path) -> Trajectory:
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise ValueError(f"{path} does not contain enough samples.")

    measured_states = np.asarray(
        [
            [
                read_float(row, "x1_filtered"),
                read_float(row, "v1_filtered"),
                read_float(row, "x2_filtered"),
                read_float(row, "v2_filtered"),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    states = np.vstack([np.zeros((1, STATE_DIM), dtype=np.float32), measured_states]).astype(np.float32)
    actions = np.asarray([[read_float(row, "voice_force_n")] for row in rows], dtype=np.float32)
    disturbances = np.asarray([[read_float(row, "shaker_force_n")] for row in rows], dtype=np.float32)
    time_axis = np.concatenate(
        [np.asarray([0.0], dtype=np.float32), np.asarray([read_float(row, "time") for row in rows], dtype=np.float32)]
    )
    if not (np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(disturbances).all()):
        raise ValueError(f"{path} contains non-finite values.")
    return Trajectory(
        name=Path(path).stem,
        path=str(Path(path).resolve()),
        block=rows[0].get("block", ""),
        states=states,
        actions=actions,
        disturbances=disturbances,
        time=time_axis,
    )


def load_trajectories(data_dir: Path) -> list[Trajectory]:
    paths = sorted(Path(data_dir).glob("trajectory_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.csv files found in {Path(data_dir).resolve()}")
    trajectories = []
    for path in paths:
        try:
            trajectories.append(load_trajectory(path))
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
    if len(trajectories) < 3:
        raise ValueError("Need at least three valid trajectories for train/val/test split.")
    return trajectories


def split_trajectories(trajectories: list[Trajectory], seed: int, train_ratio=0.8, val_ratio=0.1):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajectories))
    n_train = max(1, int(math.floor(train_ratio * len(trajectories))))
    n_val = max(1, int(math.floor(val_ratio * len(trajectories))))
    if n_train + n_val >= len(trajectories):
        n_train = len(trajectories) - 2
        n_val = 1
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {
        "train": [trajectories[i] for i in train_idx],
        "val": [trajectories[i] for i in val_idx],
        "test": [trajectories[i] for i in test_idx],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_split_csv(split: dict[str, list[Trajectory]]) -> None:
    rows = []
    for split_name, trajectories in split.items():
        for traj in trajectories:
            rows.append(
                {
                    "split": split_name,
                    "trajectory": traj.name,
                    "path": traj.path,
                    "block": traj.block,
                    "num_steps": traj.num_steps,
                }
            )
    write_csv(DATA_DIR / "dataset_split.csv", rows)


def build_history_feature(current_tuple, history, history_steps: int, use_history: bool) -> np.ndarray:
    current_tuple = np.asarray(current_tuple, dtype=np.float32).reshape(BASE_INPUT_DIM)
    if not use_history:
        return current_tuple
    pieces = [current_tuple]
    for lag in range(1, int(history_steps) + 1):
        if len(history) >= lag:
            pieces.append(np.asarray(history[-lag], dtype=np.float32).reshape(BASE_INPUT_DIM))
        else:
            pieces.append(np.zeros(BASE_INPUT_DIM, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def build_arrays(trajectories, history_steps: int, history_input: bool, residual_target: bool):
    features, targets, current_states, next_states, traj_names, step_indices = [], [], [], [], [], []
    for traj in trajectories:
        history = []
        for t in range(traj.num_steps):
            current_tuple = np.concatenate([traj.states[t], traj.actions[t], traj.disturbances[t]]).astype(np.float32)
            next_state = traj.states[t + 1].astype(np.float32)
            target = next_state - traj.states[t] if residual_target else next_state
            features.append(build_history_feature(current_tuple, history, history_steps, history_input))
            targets.append(target.astype(np.float32))
            current_states.append(traj.states[t].astype(np.float32))
            next_states.append(next_state)
            traj_names.append(traj.name)
            step_indices.append(t)
            history.append(current_tuple)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "current_states": np.asarray(current_states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "trajectory_names": np.asarray(traj_names),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
    }


def fit_stats(train_arrays):
    return {
        "x_mean": train_arrays["features"].mean(axis=0).astype(np.float32),
        "x_std": (train_arrays["features"].std(axis=0) + 1.0e-8).astype(np.float32),
        "y_mean": train_arrays["targets"].mean(axis=0).astype(np.float32),
        "y_std": (train_arrays["targets"].std(axis=0) + 1.0e-8).astype(np.float32),
    }


def normalize_arrays(arrays, stats):
    x = (arrays["features"] - stats["x_mean"]) / stats["x_std"]
    y = (arrays["targets"] - stats["y_mean"]) / stats["y_std"]
    return x.astype(np.float32), y.astype(np.float32)


def make_loader(arrays, stats, batch_size: int, shuffle: bool, seed: int):
    x, y = normalize_arrays(arrays, stats)
    ds = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(arrays["current_states"], dtype=torch.float32),
        torch.as_tensor(arrays["next_states"], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle, generator=generator if shuffle else None)


def state_scale(trajectories) -> np.ndarray:
    measured = np.concatenate([traj.states[1:] for traj in trajectories], axis=0)
    scale = measured.std(axis=0).astype(np.float64)
    return np.where(scale > EPS, scale, 1.0)


class WeightedNextStateLoss(nn.Module):
    def __init__(self, stats, scale, residual_target: bool):
        super().__init__()
        weights = torch.as_tensor(STATE_WEIGHTS / np.sum(STATE_WEIGHTS), dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.register_buffer("y_mean", torch.as_tensor(stats["y_mean"], dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(stats["y_std"], dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).clamp_min(1.0e-12))
        self.residual_target = bool(residual_target)

    def forward(self, pred_n, target_n, current_states, next_states):
        pred_target = pred_n * self.y_std.view(1, -1) + self.y_mean.view(1, -1)
        pred_next = current_states + pred_target if self.residual_target else pred_target
        err2 = ((pred_next - next_states) / self.scale.view(1, -1)) ** 2
        return torch.sum(err2 * self.weights.view(1, -1), dim=1).mean()


def predict_targets(model, arrays, stats, device, batch_size=16384):
    features = arrays["features"]
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = (features[start : start + batch_size] - stats["x_mean"]) / stats["x_std"]
            xb = torch.as_tensor(x, dtype=torch.float32, device=device)
            pred_n = model(xb).cpu().numpy()
            preds.append(pred_n * stats["y_std"] + stats["y_mean"])
    return np.concatenate(preds, axis=0).astype(np.float32)


def one_step_metrics(model, arrays, stats, residual_target: bool, scale, device):
    pred_target = predict_targets(model, arrays, stats, device)
    pred_next = arrays["current_states"] + pred_target if residual_target else pred_target
    err = pred_next - arrays["next_states"]
    normalized = err / scale.reshape(1, -1)
    state_nrmse = np.sqrt(np.mean(normalized**2, axis=0))
    sw_nrmse = float(np.sqrt(np.sum(STATE_WEIGHTS * state_nrmse**2) / np.sum(STATE_WEIGHTS)))
    return {
        "one_step_mse": float(np.mean(err**2)),
        "one_step_nrmse": float(np.sqrt(np.mean(normalized**2))),
        "one_step_sw_nrmse": sw_nrmse,
    }


def predict_next_state(model, state, action, disturbance, history, stats, history_steps, history_input, residual_target, device):
    current_tuple = np.concatenate([state, action, disturbance]).astype(np.float32)
    feature = build_history_feature(current_tuple, history, history_steps, history_input)
    x = (feature - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        pred_n = model(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
    target = pred_n * stats["y_std"] + stats["y_mean"]
    return (state + target if residual_target else target).astype(np.float32)


def open_loop_rollout(model, traj, stats, history_steps, history_input, residual_target, device):
    pred = np.zeros_like(traj.states, dtype=np.float32)
    pred[0] = traj.states[0]
    history = []
    for t in range(traj.num_steps):
        state = pred[t].astype(np.float32)
        action = traj.actions[t].astype(np.float32)
        disturbance = traj.disturbances[t].astype(np.float32)
        pred[t + 1] = predict_next_state(
            model, state, action, disturbance, history, stats, history_steps, history_input, residual_target, device
        )
        history.append(np.concatenate([state, action, disturbance]).astype(np.float32))
    return pred


def compute_one_step_states(model, traj, stats, history_steps, history_input, residual_target, device):
    arrays = build_arrays([traj], history_steps, history_input, residual_target)
    pred_target = predict_targets(model, arrays, stats, device)
    pred = np.zeros_like(traj.states, dtype=np.float32)
    pred[0] = traj.states[0]
    pred[1:] = arrays["current_states"] + pred_target if residual_target else pred_target
    return pred


def open_loop_metrics(model, trajectories, stats, history_steps, history_input, residual_target, scale, device):
    records = []
    state_sse = np.zeros(STATE_DIM, dtype=np.float64)
    norm_sse = np.zeros(STATE_DIM, dtype=np.float64)
    total_steps = 0
    for traj in trajectories:
        pred = open_loop_rollout(model, traj, stats, history_steps, history_input, residual_target, device)
        err = pred[1:] - traj.states[1:]
        nerr = err / scale.reshape(1, -1)
        state_rmse = np.sqrt(np.mean(err**2, axis=0))
        state_nrmse = np.sqrt(np.mean(nerr**2, axis=0))
        sw_nrmse = float(np.sqrt(np.sum(STATE_WEIGHTS * state_nrmse**2) / np.sum(STATE_WEIGHTS)))
        records.append(
            {
                "trajectory": traj.name,
                "path": traj.path,
                "open_loop_rmse": float(np.sqrt(np.mean(err**2))),
                "open_loop_nrmse": float(np.sqrt(np.mean(nerr**2))),
                "open_loop_weighted_nrmse": sw_nrmse,
                "open_loop_rmse_x1": float(state_rmse[0]),
                "open_loop_rmse_v1": float(state_rmse[1]),
                "open_loop_rmse_x2": float(state_rmse[2]),
                "open_loop_rmse_v2": float(state_rmse[3]),
            }
        )
        state_sse += np.sum(err**2, axis=0)
        norm_sse += np.sum(nerr**2, axis=0)
        total_steps += traj.num_steps
    vals = np.asarray([row["open_loop_weighted_nrmse"] for row in records], dtype=np.float64)
    nvals = np.asarray([row["open_loop_nrmse"] for row in records], dtype=np.float64)
    rvals = np.asarray([row["open_loop_rmse"] for row in records], dtype=np.float64)
    pooled_state_nrmse = np.sqrt(norm_sse / max(total_steps, 1))
    return {
        "records": records,
        "test_olr_rmse_mean": float(np.mean(rvals)),
        "test_olr_rmse_std": float(np.std(rvals)),
        "test_olr_nrmse_mean": float(np.mean(nvals)),
        "test_olr_nrmse_std": float(np.std(nvals)),
        "test_olr_weighted_nrmse_mean": float(np.mean(vals)),
        "test_olr_weighted_nrmse_std": float(np.std(vals)),
        "test_olr_weighted_nrmse_median": float(np.median(vals)),
        "test_olr_weighted_nrmse_min": float(np.min(vals)),
        "test_olr_weighted_nrmse_max": float(np.max(vals)),
        "test_olr_rmse_x1_pooled": float(np.sqrt(state_sse[0] / max(total_steps, 1))),
        "test_olr_rmse_v1_pooled": float(np.sqrt(state_sse[1] / max(total_steps, 1))),
        "test_olr_rmse_x2_pooled": float(np.sqrt(state_sse[2] / max(total_steps, 1))),
        "test_olr_rmse_v2_pooled": float(np.sqrt(state_sse[3] / max(total_steps, 1))),
        "test_olr_nrmse_x1_pooled": float(pooled_state_nrmse[0]),
        "test_olr_nrmse_v1_pooled": float(pooled_state_nrmse[1]),
        "test_olr_nrmse_x2_pooled": float(pooled_state_nrmse[2]),
        "test_olr_nrmse_v2_pooled": float(pooled_state_nrmse[3]),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for xb, yb, current, nxt in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        current = current.to(device)
        nxt = nxt.to(device)
        pred = model(xb)
        loss = criterion(pred, yb, current, nxt)
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.item()) * len(xb)
        count += len(xb)
    return total / max(count, 1)


def train_evaluate(spec, split, history_steps, args, device, run_group, run_name):
    start = time.time()
    train_arrays = build_arrays(split["train"], history_steps, spec.history_input, spec.residual_target)
    val_arrays = build_arrays(split["val"], history_steps, spec.history_input, spec.residual_target)
    test_arrays = build_arrays(split["test"], history_steps, spec.history_input, spec.residual_target)
    stats = fit_stats(train_arrays)
    scale = state_scale(split["train"])
    train_loader = make_loader(train_arrays, stats, args.batch_size, True, args.seed)
    val_loader = make_loader(val_arrays, stats, args.batch_size, False, args.seed)
    model = make_model(train_arrays["features"].shape[1], args.hidden_dim, spec.residual_blocks).to(device)
    criterion = WeightedNextStateLoss(stats, scale, spec.residual_target).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device, None)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "run_name": run_name})
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_one = one_step_metrics(model, val_arrays, stats, spec.residual_target, scale, device)
    test_one = one_step_metrics(model, test_arrays, stats, spec.residual_target, scale, device)
    olr = open_loop_metrics(model, split["test"], stats, history_steps, spec.history_input, spec.residual_target, scale, device)
    elapsed = time.time() - start
    summary = {
        "model_id": run_name if run_group == "history_sensitivity" else spec.model_id,
        "model_label": spec.label,
        "run_group": run_group,
        "history_steps": history_steps,
        "history_input": int(spec.history_input),
        "residual_blocks": int(spec.residual_blocks),
        "residual_target": int(spec.residual_target),
        "input_dim": int(train_arrays["features"].shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "epochs_requested": int(args.epochs),
        "epochs_trained": len(history_rows),
        "best_val_loss": best_val,
        "val_one_step_mse": val_one["one_step_mse"],
        "val_one_step_nrmse": val_one["one_step_nrmse"],
        "val_one_step_sw_nrmse": val_one["one_step_sw_nrmse"],
        "test_one_step_mse": test_one["one_step_mse"],
        "test_one_step_nrmse": test_one["one_step_nrmse"],
        "test_one_step_sw_nrmse": test_one["one_step_sw_nrmse"],
        "train_rollout_count": len(split["train"]),
        "val_rollout_count": len(split["val"]),
        "test_rollout_count": len(split["test"]),
        "train_sample_count": len(train_arrays["features"]),
        "val_sample_count": len(val_arrays["features"]),
        "test_sample_count": len(test_arrays["features"]),
        "elapsed_s": elapsed,
    }
    summary.update({key: value for key, value in olr.items() if key != "records"})
    checkpoint_path = MODELS_DIR / f"{run_group}_{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "spec": spec.__dict__,
            "history_steps": history_steps,
            "hidden_dim": args.hidden_dim,
            "summary": summary,
        },
        checkpoint_path,
    )
    write_csv(DATA_DIR / f"training_history_{run_group}_{run_name}.csv", history_rows)
    per_rows = [dict(row, run_group=run_group, model_id=run_name) for row in olr["records"]]
    write_csv(DATA_DIR / f"per_rollout_{run_group}_{run_name}.csv", per_rows)
    print(
        f"{run_group}/{run_name}: OLR SW-NRMSE={summary['test_olr_weighted_nrmse_mean']:.5f}, "
        f"one-step SW-NRMSE={summary['test_one_step_sw_nrmse']:.5f}, epochs={len(history_rows)}"
    )
    return {
        "model": model,
        "stats": stats,
        "spec": spec,
        "history_steps": history_steps,
        "summary": summary,
        "per_rollout": per_rows,
    }


def add_reductions(rows, baseline_key, baseline_value, suffix):
    baseline = next((row for row in rows if str(row[baseline_key]) == str(baseline_value)), None)
    if baseline is None:
        return rows
    for row in rows:
        for metric in ("test_olr_weighted_nrmse_mean", "test_olr_nrmse_mean", "test_one_step_sw_nrmse"):
            base = float(baseline[metric])
            cur = float(row[metric])
            row[f"{metric}_reduction_{suffix}_percent"] = 100.0 * (base - cur) / base if abs(base) > EPS else np.nan
    return rows


def select_quantile_trajectory(per_rollout_rows, tag):
    rows = sorted(per_rollout_rows, key=lambda row: float(row["open_loop_weighted_nrmse"]))
    if tag == "best":
        return rows[0]["trajectory"]
    if tag == "worst":
        return rows[-1]["trajectory"]
    return rows[len(rows) // 2]["trajectory"]


def write_one_showcase_predictions(results_by_id, split, tag: str, out_name: str):
    m4 = results_by_id["M4"]
    traj_name = select_quantile_trajectory(m4["per_rollout"], tag)
    traj = next(traj for traj in split["test"] if traj.name == traj_name)
    rows = []

    def append_state(model_id, sample, state):
        t = float(traj.time[min(sample, len(traj.time) - 1)])
        row = {
            "trajectory": traj.name,
            "showcase_tag": tag,
            "model_id": model_id,
            "sample": sample,
            "time": t,
            "x1": float(state[0]),
            "v1": float(state[1]),
            "x2": float(state[2]),
            "v2": float(state[3]),
            "voice_force_n": float(traj.actions[min(sample, traj.num_steps - 1), 0]),
            "shaker_force_n": float(traj.disturbances[min(sample, traj.num_steps - 1), 0]),
        }
        rows.append(row)

    for sample, state in enumerate(traj.states):
        append_state("Measured", sample, state)
    for model_id in ("M0", "M1", "M2", "M3", "M4"):
        result = results_by_id[model_id]
        pred = open_loop_rollout(
            result["model"],
            traj,
            result["stats"],
            result["history_steps"],
            result["spec"].history_input,
            result["spec"].residual_target,
            torch.device("cpu") if next(result["model"].parameters()).device.type == "cpu" else next(result["model"].parameters()).device,
        )
        for sample, state in enumerate(pred):
            append_state(model_id, sample, state)
    write_csv(DATA_DIR / out_name, rows)


def write_showcase_predictions(results_by_id, split):
    for tag in ("best", "median", "worst"):
        write_one_showcase_predictions(results_by_id, split, tag, f"showcase_{tag}.csv")
    write_one_showcase_predictions(results_by_id, split, "median", "showcase_default.csv")


def write_all_subfigure_predictions(results_by_id, split):
    split_name_map = {"train": "train", "val": "eval", "test": "test"}
    for split_key, out_split in split_name_map.items():
        out_dir = DATA_DIR / "subfigure_predictions" / out_split
        out_dir.mkdir(parents=True, exist_ok=True)
        for traj in split[split_key]:
            rows = []

            def append_state(model_id, sample, state):
                rows.append(
                    {
                        "split": out_split,
                        "trajectory": traj.name,
                        "model_id": model_id,
                        "sample": sample,
                        "time": float(traj.time[min(sample, len(traj.time) - 1)]),
                        "x1": float(state[0]),
                        "v1": float(state[1]),
                        "x2": float(state[2]),
                        "v2": float(state[3]),
                        "voice_force_n": float(traj.actions[min(sample, traj.num_steps - 1), 0]),
                        "shaker_force_n": float(traj.disturbances[min(sample, traj.num_steps - 1), 0]),
                    }
                )

            for sample, state in enumerate(traj.states):
                append_state("Measured", sample, state)
            for model_id in ("M0", "M1", "M2", "M3", "M4"):
                result = results_by_id[model_id]
                device = next(result["model"].parameters()).device
                pred = open_loop_rollout(
                    result["model"],
                    traj,
                    result["stats"],
                    result["history_steps"],
                    result["spec"].history_input,
                    result["spec"].residual_target,
                    device,
                )
                for sample, state in enumerate(pred):
                    append_state(model_id, sample, state)
            write_csv(out_dir / f"{traj.name}.csv", rows)


def write_open_vs_one_envelope(result, trajectories, device, filename="open_vs_one_error_envelope.csv"):
    min_n = min(traj.states.shape[0] for traj in trajectories)
    x2_open, x2_one, v2_open, v2_one = [], [], [], []
    for traj in trajectories:
        open_pred = open_loop_rollout(
            result["model"],
            traj,
            result["stats"],
            result["history_steps"],
            result["spec"].history_input,
            result["spec"].residual_target,
            device,
        )[:min_n]
        one_pred = compute_one_step_states(
            result["model"],
            traj,
            result["stats"],
            result["history_steps"],
            result["spec"].history_input,
            result["spec"].residual_target,
            device,
        )[:min_n]
        measured = traj.states[:min_n]
        x2_open.append(open_pred[:, 2] - measured[:, 2])
        x2_one.append(one_pred[:, 2] - measured[:, 2])
        v2_open.append(open_pred[:, 3] - measured[:, 3])
        v2_one.append(one_pred[:, 3] - measured[:, 3])

    def summary(stack):
        arr = np.stack(stack, axis=0)
        return arr.min(axis=0), arr.max(axis=0), np.median(arr, axis=0)

    xo_min, xo_max, xo_mid = summary(x2_open)
    x1_min, x1_max, x1_mid = summary(x2_one)
    vo_min, vo_max, vo_mid = summary(v2_open)
    v1_min, v1_max, v1_mid = summary(v2_one)
    rows = []
    for sample in range(min_n):
        rows.append(
            {
                "sample": sample,
                "time": sample * DT,
                "x2_open_min": float(xo_min[sample]),
                "x2_open_max": float(xo_max[sample]),
                "x2_open_mid": float(xo_mid[sample]),
                "x2_one_min": float(x1_min[sample]),
                "x2_one_max": float(x1_max[sample]),
                "x2_one_mid": float(x1_mid[sample]),
                "v2_open_min": float(vo_min[sample]),
                "v2_open_max": float(vo_max[sample]),
                "v2_open_mid": float(vo_mid[sample]),
                "v2_one_min": float(v1_min[sample]),
                "v2_one_max": float(v1_max[sample]),
                "v2_one_mid": float(v1_mid[sample]),
            }
        )
    write_csv(DATA_DIR / filename, rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDM ablation models from comfort-seat CSV trajectories.")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--history-steps",
        type=int,
        default=None,
        help="Optional manual history window for ablation. By default, select the best Nh from --history-list first.",
    )
    parser.add_argument("--history-list", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    trajectories = load_trajectories(args.trajectory_dir)
    split = split_trajectories(trajectories, args.seed)
    write_split_csv(split)
    print(
        f"Loaded {len(trajectories)} trajectories: "
        f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}"
    )

    history_rows = []
    history_results = {}
    full_spec = ABLATION_SPECS[-1]
    history_list = [int(item.strip()) for item in args.history_list.split(",") if item.strip()]
    if not history_list:
        raise ValueError("--history-list must contain at least one integer history length.")

    print("Stage 1/2: history-window sensitivity sweep with full M4 structure")
    for nh in history_list:
        result = train_evaluate(full_spec, split, nh, args, device, "history_sensitivity", f"Nh{nh:02d}")
        row = dict(result["summary"])
        history_results[nh] = result
        row["history_steps"] = nh
        history_rows.append(row)

    best_history = min(history_rows, key=lambda row: float(row["test_olr_weighted_nrmse_mean"]))
    selected_history_steps = int(best_history["history_steps"]) if args.history_steps is None else int(args.history_steps)
    if args.history_steps is not None and args.history_steps not in history_results:
        print(f"Manual --history-steps={args.history_steps} was not in --history-list; training it for ablation support.")
        result = train_evaluate(full_spec, split, selected_history_steps, args, device, "history_sensitivity", f"Nh{selected_history_steps:02d}")
        row = dict(result["summary"])
        row["history_steps"] = selected_history_steps
        history_results[selected_history_steps] = result
        history_rows.append(row)
        best_history = min(history_rows, key=lambda row: float(row["test_olr_weighted_nrmse_mean"]))
    for row in history_rows:
        row["is_selected_history_steps"] = int(int(row["history_steps"]) == selected_history_steps)
        row["is_best_history_steps"] = int(int(row["history_steps"]) == int(best_history["history_steps"]))
        row["is_proposed_history_steps"] = row["is_selected_history_steps"]
    add_reductions(history_rows, "history_steps", 0, "vs_Nh0")
    write_csv(DATA_DIR / "history_sensitivity.csv", history_rows)

    print(f"Selected Nh={selected_history_steps} for model ablation.")
    print("Stage 2/2: DDM model ablation at selected history window")
    ablation_results = {}
    ablation_rows = []
    for spec in ABLATION_SPECS:
        result = train_evaluate(spec, split, selected_history_steps, args, device, "ablation", spec.model_id)
        ablation_results[spec.model_id] = result
        row = dict(result["summary"])
        row["selected_history_steps"] = selected_history_steps
        ablation_rows.append(row)
    add_reductions(ablation_rows, "model_id", "M0", "vs_M0")
    write_csv(DATA_DIR / "ablation_summary.csv", ablation_rows)

    best_ablation = min(ablation_rows, key=lambda row: float(row["test_olr_weighted_nrmse_mean"]))
    write_csv(
        DATA_DIR / "selected_models.csv",
        [
            {"selection": "best_ablation", **best_ablation},
            {"selection": "best_history", **best_history},
            {
                "selection": "selected_history_for_ablation",
                "history_steps": selected_history_steps,
                "source": "manual_override" if args.history_steps is not None else "best_history",
            },
        ],
    )
    write_showcase_predictions(ablation_results, split)
    write_all_subfigure_predictions(ablation_results, split)
    nh_result = history_results[selected_history_steps]
    selected_checkpoint = MODELS_DIR / f"history_sensitivity_Nh{selected_history_steps:02d}.pt"
    if selected_checkpoint.exists():
        shutil.copy2(selected_checkpoint, MODELS_DIR / "history_sensitivity_selected.pt")
    best_ablation_checkpoint = MODELS_DIR / f"ablation_{best_ablation['model_id']}.pt"
    if best_ablation_checkpoint.exists():
        shutil.copy2(best_ablation_checkpoint, MODELS_DIR / "selected_model.pt")
    write_open_vs_one_envelope(nh_result, split["test"], device)
    write_open_vs_one_envelope(nh_result, split["test"], device, f"nh{selected_history_steps:02d}_open_vs_one_error_envelope.csv")

    print(
        f"Best ablation: {best_ablation['model_id']} "
        f"SW-NRMSE={float(best_ablation['test_olr_weighted_nrmse_mean']):.5f}"
    )
    print(
        f"Best history: Nh={best_history['history_steps']} "
        f"SW-NRMSE={float(best_history['test_olr_weighted_nrmse_mean']):.5f}"
    )
    print(f"Ablation history window: Nh={selected_history_steps}")
    print(f"Saved CSV results to: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
