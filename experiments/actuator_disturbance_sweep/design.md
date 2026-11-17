# Actuator Disturbance Sweep Hardware Experiment

## Purpose

This experiment evaluates hardware robustness to actuator-side input disturbances that are not available to the controller. The main question is whether `DGS-AR-PPO` can preserve the nominal performance of `AR-PPO` while avoiding the large transient errors caused by unmodeled actuator force pulses.

The experiment is a robustness test, not a tuning pass. The fixed shaker disturbance, controller parameters, DDM checkpoint, residual threshold, and DGS switching constants are held fixed across the full force-duration grid.

## Compared Controllers

The default controller set is:

```text
Passive
ADRC
MPC
DeePC
AR-PPO
DGS-AR-PPO
```

`DGS-AR-PPO` uses the same AR-PPO policy as the standalone `AR-PPO` baseline. The added component is a DDM residual gate that switches from AR-PPO to an MPC fallback when an actuator-side event is detected:

```text
primary policy = AR-PPO
fallback policy = MPC
```

Some internal CSV fields retain the historical suffix `adrc`, such as `dgs_adrc_active_count` and `dgs_switch_to_adrc_count`. In this experiment those fields mean "DGS fallback active" and "DGS switch to fallback"; the actual fallback policy is MPC.

## Disturbance Grid

The actuator-side unknown force grid is:

```text
unknown forces = [-1.0, -1.5, -2.0] N
duration steps = [10, 20, 30]
unknown start time = 2.0 s
dt = 0.03 s
rollout duration = 9.0 s
samples per rollout = 301
```

The full default sweep contains:

```text
9 disturbance cases = 3 force levels * 3 durations
270 hardware rollouts = 9 cases * 6 controllers * 5 repeats
```

Because `2.0 s / 0.03 s = 66.67`, the discrete event starts at:

```text
start_idx = round(2.0 / 0.03) = 67
sample time = 2.01 s
```

Case identifiers encode the absolute force and pulse duration:

| Case | Unknown force | Duration |
|---|---:|---:|
| `fm10_d10` | -1.0 N | 10 samples = 0.30 s |
| `fm10_d20` | -1.0 N | 20 samples = 0.60 s |
| `fm10_d30` | -1.0 N | 30 samples = 0.90 s |
| `fm15_d10` | -1.5 N | 10 samples = 0.30 s |
| `fm15_d20` | -1.5 N | 20 samples = 0.60 s |
| `fm15_d30` | -1.5 N | 30 samples = 0.90 s |
| `fm20_d10` | -2.0 N | 10 samples = 0.30 s |
| `fm20_d20` | -2.0 N | 20 samples = 0.60 s |
| `fm20_d30` | -2.0 N | 30 samples = 0.90 s |

The generated plan is saved to:

```text
data/sweep_plan.csv
data/unknown_forces/{case}.csv
```

## Known Shaker Disturbance

Every rollout also uses the same known shaker-side test disturbance:

```text
source = ../shaker_disturbance/data/fixed_test_disturbance.csv
local copy = data/fixed_test_disturbance.csv
duration = 9.0 s
dt = 0.03 s
samples = 301
force range = -7.663 N to 8.280 N
```

This shaker disturbance is provided to the controller observation and saved as `d_t` and `shaker_force_n`. The actuator-side pulse is not provided to the controller.

## Actuator Disturbance Model

The unknown actuator force is injected after the controller produces its desired command:

```text
command_raw(k) = u_desired(k) - unknown_compensation(k)
u_cmd(k)      = clip(command_raw(k), -2, 2)
u_app(k)      = u_cmd(k) + F_ud(k)
```

For the default controllers:

```text
unknown_compensation(k) = 0
```

The code sends `u_app` to the voice-coil output with `force_limit_n = None`. This makes the tested fault an independent additive actuator-side input force, not a shrinkage of the controller saturation bound.

For each sweep case:

```text
F_ud(k) = force_n,  start_idx <= k < start_idx + duration_steps
F_ud(k) = 0,        otherwise
```

The actuator disturbance model is recorded in every valid rollout as:

```text
actuator_disturbance_model = command_saturation_independent_input_force
```

## DDM Residual Detector

The DGS supervisor uses the selected history-augmented DDM from `ddm_ablation`:

```text
checkpoint = ../ddm_ablation/models/selected_model.pt
selected model = M4, Nh = 7
state = [x1, v1, x2, v2]
input per step = [state, actuator_action, shaker_disturbance]
```

At each step, the previous measured state, previous saturated actuator command `u_cmd`, known shaker disturbance, and DDM history are used to predict the next state. After the next measurement arrives, the detector computes payload-channel residuals:

```text
e_x2(t) = x2(t) - x2_hat(t)
e_v2(t) = v2(t) - v2_hat(t)
```

The normalized residual score is:

```text
r(t) = sqrt((e_x2(t) / 0.02)^2 + (e_v2(t) / 0.6)^2)
```

The detection threshold is:

```text
tau_DGS = 0.02238545659929514
```

This value is the `95%` empirical quantile of the nominal train+validation residual-score distribution from the selected `Nh07` DDM. The calibration set contains `8911` nominal one-step residual samples:

```text
source = ../ddm_ablation/data/nh07_one_step_residual_threshold_stats.csv
split = calibration_train_val
checkpoint = history_sensitivity_Nh07.pt
history_steps = 7
samples = 8911
threshold_quantile = 0.95
```

The nominal calibration exceedance rate is approximately `5.0%`. The gate is therefore intentionally sensitive: the goal is to detect actuator-side events quickly, while short false activations are constrained by recovery hysteresis, minimum fallback dwell, and refractory logic.

Equivalent single-channel magnitudes are:

```text
|e_x2| = 4.477e-4 m
|e_v2| = 1.343e-2 m/s
```

The detector is force-agnostic and duration-agnostic. It does not use `F_ud`, the force grid, the tested pulse duration, or the known event window.

## DGS Switching Logic

`DGS-AR-PPO` is implemented by `DGSSwitchController` in `support.py`. The active policy starts as AR-PPO at the beginning of every rollout.

The online state machine is:

1. Run AR-PPO while `r(t) <= tau_DGS`.
2. If `r(t) > tau_DGS` and the refractory counter is zero, switch to MPC.
3. Keep MPC active for at least `12` samples.
4. After the minimum dwell, count recovery samples where `r(t) <= tau_recovery`.
5. Return to AR-PPO after `4` consecutive recovery samples.
6. After returning, block immediate retriggering for `8` samples.

The recovery threshold is:

```text
tau_recovery = 0.5 * tau_DGS = 0.01119272829964757
```

The switching constants are:

```text
DGS_MIN_ADRC_STEPS = 12
DGS_RECOVERY_CONSECUTIVE_STEPS = 4
DGS_REFRACTORY_STEPS = 8
```

The constant name `DGS_MIN_ADRC_STEPS` is historical; in this experiment it means minimum fallback dwell and the fallback is MPC.

## Hardware Collection Protocol

Each rollout follows the hardware sequence in `support.collect_rollout`:

1. Reset the controller.
2. Prepare DAQ, serial sensors, and actuator outputs.
3. Wait for the hardware to settle and estimate the displacement offset.
4. Initialize the online kinematics filter.
5. Run a `301`-sample loop at `DT = 0.03 s`.
6. Read raw `x/y` displacement, subtract the offset, and filter to obtain `x1`, `v1`, `a1`, `x2`, `v2`, `a2`.
7. Evaluate the controller with `{state, disturbance, u_prev, dt}`.
8. Apply command saturation to `u_cmd`.
9. Add the actuator-side unknown force to obtain `u_app`.
10. Write `u_app` to the voice coil and the known shaker voltage to the shaker output.
11. Save raw signals, filtered states, commands, detector diagnostics, DGS state, and timing diagnostics.

Safety aborts:

```text
jump_abort_m = 0.008
abs_displacement_limit_m = 0.04
max_retries = 10
```

Existing valid rollout files are skipped unless `--overwrite` is passed.

## Metrics

For each rollout, `main.rollout_metrics()` computes full-trajectory and event-window metrics for:

```text
x2
v2
a2
```

For each channel:

```text
full RMS, mean absolute value, peak absolute value
event RMS, event mean absolute value, event peak absolute value
```

The event window is:

```text
event_start = unknown_force_start_s
event_end = unknown_force_start_s + duration_steps * DT + 0.45 s
```

The `0.45 s` extension captures immediate recovery after the actuator pulse ends.

Diagnostic metrics include:

```text
compute_time_mean_ms
compute_time_std_ms
compute_time_max_ms
saturation_count
fallback_count
detection_count
event_trigger_count
dgs_fallback_active_count
dgs_switch_to_fallback_count
dgs_switch_to_arppo_count
mean_abs_Fud_error
```

For DGS switching analysis, use `dgs_fallback_active_count`, `dgs_switch_to_fallback_count`, and `dgs_switch_to_arppo_count`. `mean_abs_Fud_error` is not used by the default `DGS-AR-PPO` controller because this variant detects events but does not estimate the disturbance magnitude for compensation.

## Outputs

Generated data:

```text
data/sweep_plan.csv
data/fixed_test_disturbance.csv
data/unknown_forces/{case}.csv
data/rollouts/{case}/{controller}_repeat_{repeat}.csv
data/repeat_metrics.csv
data/aggregate_statistics.csv
```

Generated figures:

```text
figures/actuator_disturbance_pulse_sweep_rms_all_bar.pdf
figures/actuator_disturbance_pulse_sweep_a2_v2_contours_all.pdf
figures/time_response_cases/actuator_disturbance_pulse_sweep_{case}_time_response_comparison.pdf
```

## Commands

Collect or resume the full sweep:

```bash
python main.py
```

Run a small subset:

```bash
python main.py --forces -1 -2 --durations 10 --repeats 1 --controllers ar_ppo dgs_ar_ppo
```

Redo saved rollouts:

```bash
python main.py --overwrite
```

Plot after collection:

```bash
python plot.py
```

Plot selected time-response cases:

```bash
python plot.py --skip-summary --cases fm10_d10 fm20_d30
```

## Role In The Full Study

This sweep should be presented as a hardware robustness experiment under structured actuator-side pulse faults. The central methodological point is that DGS is not tuned on the force-duration labels. It detects actuator-side events from nominal DDM one-step residuals and temporarily delegates to a model-based fallback. The result therefore tests whether a learned nominal controller can be made fault-aware through residual-gated supervision.
