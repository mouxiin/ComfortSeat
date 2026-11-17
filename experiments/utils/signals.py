from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


DT = 0.03
DURATION_S = 6.0
STEPS = int(round(DURATION_S / DT))
SHAKER_FORCE_LIMIT_N = 9.0
VOICE_FORCE_LIMIT_N = 2.0
FREQ_MIN_HZ = 0.2
FREQ_MAX_HZ = 5.0


def smooth_start_stop(signal, dt: float = DT, ramp_s: float = 0.5, limit: float | None = None) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).copy()
    if signal.size == 0:
        return signal.astype(np.float32)
    ramp_steps = min(signal.size, max(1, int(round(float(ramp_s) / float(dt)))))
    window = np.ones_like(signal, dtype=np.float64)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp_steps)))
    window[:ramp_steps] = ramp
    window[-ramp_steps:] = np.minimum(window[-ramp_steps:], ramp[::-1])
    signal = signal * window
    signal[0] = 0.0
    signal[-1] = 0.0
    if limit is not None:
        return scale_to_limit(signal, float(limit))
    return signal.astype(np.float32)


def scale_to_limit(signal, limit: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs > float(limit) > 0.0:
        signal = signal * (float(limit) / max_abs)
    return np.clip(signal, -float(limit), float(limit)).astype(np.float32)


def multisine(rng: np.random.Generator, steps: int, dt: float, limit: float) -> np.ndarray:
    t = np.arange(int(steps), dtype=np.float64) * float(dt)
    signal = np.zeros_like(t)
    tone_count = int(rng.integers(1, 5))
    for _ in range(tone_count):
        amp = float(rng.uniform(0.1, 1.0) * limit)
        freq = float(rng.uniform(FREQ_MIN_HZ, FREQ_MAX_HZ))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        signal += amp * np.sin(2.0 * np.pi * freq * t + phase)
    return scale_to_limit(signal, limit)


def chirp(
    rng: np.random.Generator,
    steps: int,
    dt: float,
    limit: float,
    direction: str = "up",
) -> np.ndarray:
    t = np.arange(int(steps), dtype=np.float64) * float(dt)
    total_time = max(float(steps - 1) * float(dt), 1.0e-6)
    if direction == "down":
        f0, f1 = FREQ_MAX_HZ, FREQ_MIN_HZ
    else:
        f0, f1 = FREQ_MIN_HZ, FREQ_MAX_HZ
    amp = float(rng.uniform(0.1, 1.0) * limit)
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    k = (f1 - f0) / total_time
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + phase0
    return scale_to_limit(amp * np.sin(phase), limit)


def prbs(rng: np.random.Generator, steps: int, limit: float) -> np.ndarray:
    levels = np.linspace(-float(limit), float(limit), 20)
    signal = np.zeros(int(steps), dtype=np.float64)
    idx = 0
    while idx < int(steps):
        hold = int(rng.integers(1, 21))
        value = float(rng.choice(levels))
        signal[idx : idx + hold] = value
        idx += hold
    return scale_to_limit(signal, limit)


def mixed(
    rng: np.random.Generator,
    steps: int,
    dt: float,
    limit: float,
    components: tuple[str, ...],
    chirp_direction: str = "up",
) -> np.ndarray:
    signal = np.zeros(int(steps), dtype=np.float64)
    weights = rng.uniform(0.3, 1.0, size=len(components))
    for weight, component in zip(weights, components):
        if component == "multisine":
            signal += float(weight) * multisine(rng, steps, dt, limit)
        elif component == "chirp":
            signal += float(weight) * chirp(rng, steps, dt, limit, chirp_direction)
        elif component == "prbs":
            signal += float(weight) * prbs(rng, steps, limit)
        else:
            raise ValueError(f"Unknown mixed component: {component}")
    return scale_to_limit(signal, limit)


def build_channel_signal(
    rng: np.random.Generator,
    signal_type: str,
    steps: int,
    dt: float,
    limit: float,
    chirp_direction: str = "up",
    allow_prbs: bool = True,
    smooth_edges: bool = False,
    ramp_s: float = 0.5,
) -> np.ndarray:
    if signal_type == "zero":
        return np.zeros(int(steps), dtype=np.float32)
    if signal_type == "multisine":
        signal = multisine(rng, steps, dt, limit)
    elif signal_type == "chirp":
        signal = chirp(rng, steps, dt, limit, chirp_direction)
    elif signal_type == "prbs" and allow_prbs:
        signal = prbs(rng, steps, limit)
    elif signal_type == "mixed":
        components = ("multisine", "chirp", "prbs") if allow_prbs else ("multisine", "chirp")
        signal = mixed(rng, steps, dt, limit, components, chirp_direction)
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
    if smooth_edges:
        return smooth_start_stop(signal, dt=dt, ramp_s=ramp_s, limit=limit)
    return signal


def _balanced_labels(labels: tuple[str, ...], count: int, rng: np.random.Generator) -> list[str]:
    repeated = list(labels) * int(np.ceil(float(count) / len(labels)))
    repeated = repeated[: int(count)]
    rng.shuffle(repeated)
    return repeated


def build_protocol_plan(seed: int = 20260516) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    plan: list[dict] = []

    b1_types = _balanced_labels(("multisine", "chirp", "mixed"), 75, rng)
    b2_types = _balanced_labels(("multisine", "chirp", "prbs", "mixed"), 75, rng)
    b3_voice = _balanced_labels(("multisine", "chirp", "prbs", "mixed"), 150, rng)
    b3_shaker = _balanced_labels(("multisine", "chirp", "mixed"), 150, rng)

    chirp_counter = 0

    def next_chirp_direction() -> str:
        nonlocal chirp_counter
        direction = "up" if chirp_counter % 2 == 0 else "down"
        chirp_counter += 1
        return direction

    for label in b1_types:
        direction = next_chirp_direction() if label in {"chirp", "mixed"} else "none"
        plan.append(
            {
                "block": "B1",
                "purpose": "shaker_disturbance_channel",
                "shaker_type": label,
                "voice_type": "zero",
                "chirp_direction": direction,
            }
        )

    for label in b2_types:
        direction = next_chirp_direction() if label in {"chirp", "mixed"} else "none"
        plan.append(
            {
                "block": "B2",
                "purpose": "voice_coil_actuation_path",
                "shaker_type": "zero",
                "voice_type": label,
                "chirp_direction": direction,
            }
        )

    for voice_label, shaker_label in zip(b3_voice, b3_shaker):
        needs_chirp = voice_label in {"chirp", "mixed"} or shaker_label in {"chirp", "mixed"}
        direction = next_chirp_direction() if needs_chirp else "none"
        plan.append(
            {
                "block": "B3",
                "purpose": "coupled_two_input_dynamics",
                "shaker_type": shaker_label,
                "voice_type": voice_label,
                "chirp_direction": direction,
            }
        )

    for idx, item in enumerate(plan, start=1):
        item["index"] = idx
        item["total"] = len(plan)
        item["duration_s"] = DURATION_S
        item["dt"] = DT
        item["steps"] = STEPS
        item["seed"] = int(seed + idx * 1009)
    return plan


PLAN_FIELDS = [
    "index",
    "total",
    "block",
    "purpose",
    "shaker_type",
    "voice_type",
    "chirp_direction",
    "duration_s",
    "dt",
    "steps",
    "seed",
]


def save_plan(path: Path, plan: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan)


def _coerce_plan_row(row: dict) -> dict:
    out = dict(row)
    for key in ("index", "total", "steps", "seed"):
        out[key] = int(float(out[key]))
    for key in ("duration_s", "dt"):
        out[key] = float(out[key])
    return out


def load_or_create_plan(path: Path, seed: int = 20260516) -> list[dict]:
    path = Path(path)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            plan = [_coerce_plan_row(row) for row in csv.DictReader(f)]
        if plan and all(item["duration_s"] == DURATION_S and item["steps"] == STEPS for item in plan):
            return plan
    plan = build_protocol_plan(seed=seed)
    save_plan(path, plan)
    return plan
