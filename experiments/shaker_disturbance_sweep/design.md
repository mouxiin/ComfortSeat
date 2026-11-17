# Shaker Disturbance Sweep Hardware Experiment

## Purpose

This experiment evaluates hardware controller robustness across a grid of sinusoidal shaker-side disturbances. It extends the single fixed-trajectory `shaker_disturbance` experiment by varying both frequency and disturbance amplitude.

The tested controllers are:

```text
Passive
ADRC
MPC
DeePC
AR-PPO
```

The experiment answers three questions:

1. How do the controllers behave across frequency and shaker amplitude, rather than on one fixed broadband trajectory?
2. Which controller gives the lowest payload displacement, velocity, and acceleration over the sweep grid?
3. Do the learned and optimization-based controllers remain computationally feasible and free from fallback or saturation events on hardware?

## Sweep Grid

The default sweep uses five frequencies and five shaker voltage amplitudes:

```text
frequencies = [1, 2, 3, 4, 5] Hz
voltage amplitudes = [0.6, 1.2, 1.8, 2.4, 3.0] V
```

The shaker force conversion is:

```text
force_N = 3 * voltage_V
```

Therefore the force amplitudes are:

```text
[1.8, 3.6, 5.4, 7.2, 9.0] N
```

Each case is a pure sinusoidal shaker force:

```text
d(t) = A_N * sin(2*pi*f*t)
```

where `A_N = 3 * A_V`. The full default sweep contains:

```text
25 disturbance cases = 5 frequencies * 5 amplitudes
625 hardware rollouts = 25 cases * 5 controllers * 5 repeats
```

Each generated disturbance is saved under:

```text
data/disturbances/{case_id}.csv
```

Example case identifiers:

```text
f01_v0d6 = 1 Hz, 0.6 V, 1.8 N
f05_v3d0 = 5 Hz, 3.0 V, 9.0 N
```

The complete sweep plan is written to:

```text
data/sweep_plan.csv
```

## Timing

Each rollout uses:

```text
duration = 9.0 s
dt = 0.03 s
samples = 301
control interval = 30 ms
```

The saved samples include both `t = 0.00 s` and `t = 9.00 s`.

## Controller Implementations

`shaker_disturbance_sweep/main.py` imports the controller support code from:

```text
../shaker_disturbance/main.py
```

This keeps the sweep controllers matched to the fixed-trajectory hardware experiment.

All controllers use the current filtered state and current shaker force. MPC and DeePC do not receive future disturbance preview; the current shaker force is held constant across their prediction horizons.

Controller sources:

| Controller | Source |
|---|---|
| Passive | zero force |
| ADRC | tuned parameters from `../tune_parameter/data/best_by_method.csv` |
| MPC | tuned no-preview MPC parameters from `../tune_parameter/data/best_by_method.csv` |
| DeePC | tuned DeePC parameters plus Hankel dictionary from `../hardware_stratified_protocol/data/trajectory_*.csv` |
| AR-PPO | `../control_learning_compare/models/adaptive_residual_seed_666.pt`, with tuned ADRC as base controller |

The voice-coil command is clipped to:

```text
u in [-2, 2] N
```

The shaker command is clipped to:

```text
shaker voltage in [-3, 3] V
shaker force in [-9, 9] N
```

## Measurement Filtering

The measurement pipeline matches the final hardware protocol:

1. Before each rollout, the system settles for `2 s`, and the mean raw sensor reading is used as the displacement zero offset.
2. Raw displacements `x1_raw` and `x2_raw` are offset by this settling mean.
3. Displacements are filtered online by a causal first-order low-pass filter with `8 Hz` cutoff.
4. Velocities are reconstructed by timestamped finite differences of the filtered displacement and then filtered by the same `8 Hz` filter.
5. Accelerations are reconstructed from filtered velocity and filtered again by the same `8 Hz` filter.

Rollout CSV files store:

```text
x1, v1, a1, x2, v2, a2
x1_filtered, v1_filtered, a1_filtered, x2_filtered, v2_filtered, a2_filtered
x1_raw, x2_raw
```

Metrics and figures use the filtered channels.

## Safety And Resume Logic

The script applies the same restart protection as the fixed-trajectory hardware experiment:

```text
sudden raw displacement jump threshold = 0.008 m
absolute raw displacement threshold = 0.04 m
settle time before rollout = 2.0 s
post-rollout wait = 2.0 s
max retries per rollout = 10
```

Outputs are stopped and buffers are cleared on both success and failure paths.

Every successful rollout is saved immediately:

```text
data/rollouts/{case_id}/{controller}_repeat_{repeat}.csv
```

Re-running `main.py` skips valid existing rollout files unless `--overwrite` is passed.

## Metrics

`data/repeat_metrics.csv` stores one row per rollout:

```text
case_id
controller
repeat
frequency_hz
voltage_amplitude_v
force_amplitude_n
x2_rms, x2_mean_abs, x2_peak_abs
v2_rms, v2_mean_abs, v2_peak_abs
a2_rms, a2_mean_abs, a2_peak_abs
compute_time_mean_ms, compute_time_std_ms, compute_time_max_ms
saturation_count, fallback_count
```

`data/aggregate_statistics.csv` stores one row per `(case_id, controller)` pair. Each metric is reported as the mean and standard deviation across the five repeats for that case.

The plotting script then averages the 25 case-level means to build the sweep-level bar figure.

## Figures

`plot.py` is standalone and reads only local CSV files in this experiment directory.

It writes:

```text
figures/frequency_test_all_plot_a2_v2_contours_all.pdf
figures/frequency_test_all_plot_rms_all_bar.pdf
```

The contour figure shows the frequency-amplitude response surfaces for:

```text
x2_rms
v2_rms
a2_rms
```

The bar figure reports the mean and standard deviation of those RMS values across all 25 disturbance cases.

## Commands

Collect or resume the full sweep:

```bash
python main.py
```

Redo existing rollouts:

```bash
python main.py --overwrite
```

Run a short hardware subset:

```bash
python main.py --frequencies 1 --voltage-amplitudes 0.6 --repeats 1
```

Plot after data collection:

```bash
python plot.py
```
