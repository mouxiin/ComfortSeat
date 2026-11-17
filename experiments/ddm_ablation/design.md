# DDM Ablation Experiment

## Purpose

This experiment identifies a data-driven dynamics model (DDM) suitable for two downstream roles:

1. nominal one-step dynamics prediction for residual-based event detection;
2. open-loop rollout prediction for evaluating whether the learned dynamics remain stable when recursively simulated.

The key design question is not only whether the DDM fits local one-step transitions, but whether the architecture can preserve physically plausible trajectories over a multi-second hardware rollout. This is why the primary model-selection metric is open-loop rollout weighted NRMSE rather than one-step MSE alone.

The final selected architecture is the full proposed DDM:

```text
M4 = history-augmented residual-block MLP with residual state-increment target
selected history length: N_h = 7
stable downstream checkpoint: models/selected_model.pt
```

`models/selected_model.pt` is a copy of `models/ablation_M4.pt`. The `N_h = 7` threshold calibration file is generated from the corresponding selected history model, `models/history_sensitivity_Nh07.pt`, which uses the same architecture and history length.

## Input Data

The DDM is trained from hardware trajectories collected by:

```text
../hardware_stratified_protocol/data/trajectory_*.csv
```

The experiment uses filtered displacement and velocity channels:

```text
state s_t = [x1_filtered, v1_filtered, x2_filtered, v2_filtered]
input u_t = voice_force_n
disturbance d_t = shaker_force_n
```

A one-step transition is:

```text
[s_t, u_t, d_t] -> s_{t+1}
```

The filtered states come from the same causal online kinematic filtering pipeline used in the hardware protocol. Acceleration channels are not used by the DDM input or target; they are reserved for controller evaluation and plotting.

Important dataset parameters:

```text
dt = 0.03 s
state dimension = 4
base input dimension = 6 = [x1, v1, x2, v2, u, d]
number of trajectories = 300
measured rows per trajectory = 200
train trajectories = 240
validation trajectories = 30
test trajectories = 30
train samples = 48000
validation samples = 6000
test samples = 6000
split seed = 666
```

The split is trajectory-wise rather than sample-wise. This prevents adjacent samples from the same hardware rollout appearing in different splits.

## Feature Construction

For memoryless models, the feature is:

```text
phi_t = [s_t, u_t, d_t]
```

For history-augmented models, the feature includes a finite input-state history:

```text
phi_t = [[s_t, u_t, d_t],
         [s_{t-1}, u_{t-1}, d_{t-1}],
         ...,
         [s_{t-N_h}, u_{t-N_h}, d_{t-N_h}]]
```

Missing history at the beginning of each trajectory is zero padded. The history sweep evaluates:

```text
N_h = 0, 1, ..., 15
```

The selected history length is the one with the lowest held-out open-loop rollout weighted NRMSE.

## Model Variants

The ablation isolates three modeling choices:

- whether the DDM is memoryless or history-augmented;
- whether the MLP is a plain feedforward network or uses residual hidden blocks;
- whether the target is the absolute next state or the residual state increment.

The five ablation variants are:

| Model | History | Hidden structure | Target |
|---|---|---|---|
| M0 | no | plain MLP | absolute next state |
| M1 | no | residual-block MLP | absolute next state |
| M2 | yes | plain MLP | absolute next state |
| M3 | no | plain MLP | state increment |
| M4 | yes | residual-block MLP | state increment |

The plain MLP uses three hidden layers with `tanh` activation. The residual-block MLP uses an input projection, two residual hidden updates, and an output layer. The hidden width is `64` for all variants.

## Training

Features and targets are standardized using training-split statistics only. The loss is a state-weighted normalized next-state error:

```text
state weights [x1, v1, x2, v2] = [0.1, 0.1, 0.4, 0.4]
```

The payload states `x2` and `v2` receive higher weight because they are the comfort-relevant channels and are also used by the downstream residual detector.

Training configuration:

```text
optimizer = AdamW
learning rate = 1e-3
weight decay = 1e-4
batch size = 4096
maximum epochs = 160
early stopping patience = 35
gradient clipping norm = 1.0
random seed = 666
device = auto
```

Each checkpoint stores the model specification, normalization statistics, learned weights, and history length so that plotting and downstream control experiments can load the model without retraining.

## Selection Procedure

The experiment is run in two stages:

1. Train the full M4 structure for all history lengths `N_h = 0..15`.
2. Select the best history length using held-out test open-loop rollout weighted NRMSE.
3. Retrain the five ablation variants M0-M4 using the selected history length.
4. Select the best ablation model using the same open-loop rollout metric.
5. Save the best ablation checkpoint to `models/selected_model.pt`.

The selected history length is:

```text
N_h = 7
```

The selected ablation model is:

```text
M4
```

## Evaluation Metrics

Two evaluation modes are reported.

One-step prediction:

```text
hat{s}_{t+1} = f_DDM(s_t, u_t, d_t, history_t)
```

Every prediction starts from the measured current state. This measures local transition fitting and is used by the downstream residual detector.

Open-loop rollout:

```text
hat{s}_{t+1} = f_DDM(hat{s}_t, u_t, d_t, history_t)
```

Only the initial measured state is given. The model then recursively feeds back its own predicted state while receiving the measured input and disturbance sequence. This is the primary metric because it exposes compounding model error and rollout instability.

The main scalar metric is:

```text
test_olr_weighted_nrmse_mean
```

with standard deviation computed across held-out test trajectories. The weighted NRMSE uses the same state weights as training:

```text
[x1, v1, x2, v2] = [0.1, 0.1, 0.4, 0.4]
```

## Residual Threshold For DGS

The selected DDM is also used to construct the event detector in the actuator-side disturbance sweep. The detector is calibrated from nominal one-step residuals, not from actuator-fault data.

For each nominal transition, the selected `N_h = 7` DDM predicts the next state. The detector score uses only the payload channels:

```text
e_x2(t) = x2(t) - hat{x2}(t)
e_v2(t) = v2(t) - hat{v2}(t)

r(t) = sqrt((e_x2(t) / 0.02)^2 + (e_v2(t) / 0.6)^2)
```

The threshold is the `95%` empirical quantile of the train+validation nominal score distribution:

```text
calibration split = train + validation
calibration samples = 8911
tau_DGS = 0.02238545659929514
nominal calibration exceedance = 5.0%
```

The corresponding single-channel magnitudes are:

```text
|e_x2| = 4.477e-4 m
|e_v2| = 1.343e-2 m/s
```

This threshold is intentionally sensitive. In the downstream control task, missed detection of an actuator-side input fault is more harmful than a short fallback activation. Switching chatter is handled downstream by recovery hysteresis, a minimum fallback dwell time, and a refractory interval.

The calibration statistics are saved in:

```text
data/nh07_one_step_residual_threshold_stats.csv
```

## Output Files

Core data:

- `data/dataset_split.csv`
- `data/history_sensitivity.csv`
- `data/ablation_summary.csv`
- `data/selected_models.csv`
- `data/per_rollout_history_sensitivity_Nh*.csv`
- `data/per_rollout_ablation_M*.csv`
- `data/nh07_one_step_residual_threshold_stats.csv`
- `data/nh07_open_vs_one_error_envelope.csv`

Model checkpoints:

- `models/history_sensitivity_NhXX.pt`
- `models/history_sensitivity_selected.pt`
- `models/ablation_M0.pt` through `models/ablation_M4.pt`
- `models/selected_model.pt`

Figures:

- `figures/history_sensitivity_olr.pdf`
- `figures/ablation_bar_olr.pdf`
- `figures/nh07_open_vs_one_error_envelope.pdf`
- `figures/nh07_open_vs_one_error_envelope_6s.pdf`
- `figures/subfigures/{train,eval,test}/*.pdf`

## Commands

Train the history sweep and ablation models:

```bash
python main.py
```

Regenerate figures and residual-threshold statistics from saved checkpoints:

```bash
python plot.py
```

Force CPU training:

```bash
python main.py --device cpu
```
