# Control Learning Compare Experiment

## Purpose

This experiment trains PPO controllers on the selected data-driven dynamics model from `ddm_ablation`. It answers two questions:

1. Can PPO improve the DDM-simulated comfort-seat control objective beyond Passive and tuned ADRC baselines?
2. Is residual learning around ADRC more stable or more effective than directly learning the full voice-coil force?

The experiment is a surrogate-model control-learning comparison. It does not collect new hardware data. All rollouts are performed through the selected DDM checkpoint:

```text
../ddm_ablation/models/selected_model.pt
```

## DDM Surrogate

The current selected DDM checkpoint is:

```text
model_id = M4
architecture = history-augmented residual-block MLP
target = residual state increment
history_steps = 7
input_dim = 48
hidden_dim = 64
```

The DDM state is:

```text
s_t = [x1, v1, x2, v2]
```

The one-step base tuple is:

```text
z_t = [x1, v1, x2, v2, u_t, d_t]
```

where:

- `u_t` is the clipped voice-coil force command in newtons;
- `d_t` is the current shaker disturbance force in newtons.

Because `history_steps = 7`, the DDM feature contains the current tuple plus seven lagged tuples:

```text
phi_t = [z_t, z_{t-1}, ..., z_{t-7}]
feature dimension = (7 + 1) * 6 = 48
```

The checkpoint stores the normalization statistics and residual-target setting. If the model predicts `delta_s_t`, the rollout reconstructs:

```text
s_{t+1} = s_t + delta_s_t
```

## Environment

The vectorized surrogate environment is `DDMSurrogateVectorEnv` in `main.py`.

Default rollout settings:

```text
dt = 0.03 s
episode duration = 6.0 s
steps = 200
state dimension = 4
observation dimension = 6
voice-coil force limit = +/- 2.0 N
shaker force limit = +/- 9.0 N
```

The controller observation is:

```text
o_t = [x1, v1, x2, v2, u_{t-1}, d_t]
```

The observation is normalized by:

```text
[0.02, 0.6, 0.02, 0.6, 2.0, 9.0]
```

Controllers see the current shaker disturbance, but they do not receive future disturbance preview.

## Acceleration And Reward

The DDM predicts displacement and velocity states. Raw payload acceleration is reconstructed from velocity:

```text
a2_raw(t) = (v2(t+1) - v2(t)) / dt
```

The reward uses the same causal first-order low-pass filtering idea as the hardware pipeline. `a2_raw` is filtered online with the `8 Hz` cutoff from `utils.filters`.

The per-step reward is:

```text
r_t = -0.1 * (180*x2^2 + 6*v2^2 + 2*a2^2 + 0.04*u^2 + 0.08*du^2)
```

where:

- `x2` and `v2` are the predicted payload displacement and velocity at `t+1`;
- `a2` is the filtered payload acceleration;
- `u` is the applied clipped voice-coil force;
- `du = u_t - u_{t-1}`.

The reported average return is the mean episodic reward sum over the fixed evaluation disturbance suite.

## Baseline Controllers

The comparison includes two non-learning baselines:

| Method | Description |
|---|---|
| Passive | zero voice-coil force |
| ADRC | tuned active disturbance rejection controller from `tune_parameter` |

ADRC parameters are loaded from:

```text
../tune_parameter/data/best_by_method.csv
```

Current fallback defaults if the tuning file is missing:

```text
omega_o = 16.0
kp = 30.0
kd = 6.0
b0_scale = 1.0
m2 = 0.85 kg
b0 = 1 / m2
k2 = 135.0 N/m
c2 = 6.4 N s/m
```

The ADRC observer is reset at the start of every rollout batch.

## PPO Controllers

Three PPO methods are trained:

| Method | Label | Policy type |
|---|---|---|
| `non_residual` | NR-PPO | directly learns full voice-coil force |
| `fixed_residual` | FR-PPO | learns residual around ADRC with fixed residual range |
| `adaptive_residual` | AR-PPO | learns residual around ADRC with adaptive residual range |

For NR-PPO:

```text
u_t = 2.0 * tanh(a_t)
```

For residual PPO:

```text
u_t = clip(u_ADRC,t + u_res,t, -2.0, 2.0)
u_res,t = residual_u_max * tanh(a_t)
```

FR-PPO uses:

```text
residual_u_max = 2.0 N
```

AR-PPO starts with a smaller residual range and expands it when evaluation returns become stable:

```text
initial residual_u_max = 0.8 N
maximum residual_u_max = 2.0 N
growth factor = 1.35
warmup = 18 episodes
stability window = 8 evaluation points
cooldown = 8 episodes
std absolute threshold = 0.02
std relative threshold = 0.06
slope threshold = 6e-4
```

When AR-PPO expands the residual range, the final actor layer and log standard deviation are rescaled so the physical residual output changes continuously.

## Disturbances

Training disturbances are regenerated every PPO episode. They use the same signal vocabulary as the hardware identification protocol:

```text
multisine
chirp
mixed
```

PRBS is not used for the shaker disturbance channel. Each disturbance is bounded by `+/- 9 N` and uses smooth start/stop edges.

Evaluation uses a fixed disturbance suite:

```text
eval_count = 10
seed = experiment_seed + 20260413
default experiment_seed = 666
default evaluation seed = 20261079
```

The fixed evaluation suite is saved to:

```text
data/fixed_eval_disturbances.csv
```

The file contains `10` cases and `201` samples per case. The environment uses `200` control steps from each `201`-sample disturbance sequence.

## PPO Training

Default training settings:

```text
seed = 666
episodes = 300
num_envs = 64
eval_interval = 5
hidden_dim = 128
ppo_epochs = 4
minibatch_size = 4096
actor_lr = 3e-4
critic_lr = 1e-3
gamma = 0.995
lambda_GAE = 0.95
clip_eps = 0.2
target_KL = 0.03
entropy_coef = 0.01
initial_actor_std = 0.35
minimum_actor_std = 0.03
device = auto
```

The actor and critic are two-layer `tanh` MLPs with hidden width `128`. Actor outputs are sampled as Gaussian pre-`tanh` actions during training and evaluated deterministically using the actor mean.

For each PPO method, the script:

1. evaluates the untrained policy at episode `0`;
2. trains on fresh random disturbance batches;
3. evaluates on the fixed disturbance suite every `5` episodes and at the final episode;
4. writes `data/training_returns_{method}.csv` during training;
5. saves a checkpoint under `models/{method}_seed_{seed}.pt`.

Passive and ADRC are evaluated once at episode `0`.

## Outputs

Generated data:

- `data/fixed_eval_disturbances.csv`
- `data/training_returns_non_residual.csv`
- `data/training_returns_fixed_residual.csv`
- `data/training_returns_adaptive_residual.csv`
- `data/average_return_comparison.csv`
- `data/final_summary.csv`

Model checkpoints:

- `models/non_residual_seed_666.pt`
- `models/fixed_residual_seed_666.pt`
- `models/adaptive_residual_seed_666.pt`

Figure:

- `figures/average_return_comparison.pdf`

`plot.py` is standalone: it reads `data/average_return_comparison.csv` and writes the average-return PDF without retraining policies.

## Commands

Train the default comparison:

```bash
python main.py
```

Run a short smoke test:

```bash
python main.py --episodes 1 --num-envs 2 --eval-interval 1 --methods non_residual
```

Train only AR-PPO:

```bash
python main.py --methods adaptive_residual
```

Regenerate the average-return figure:

```bash
python plot.py
```
