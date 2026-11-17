# Shaker Disturbance Hardware Experiment

## Purpose

This experiment evaluates the final hardware response of five controllers under the same fixed shaker-side disturbance:

```text
Passive
ADRC
MPC
DeePC
AR-PPO
```

Each controller is tested for `5` independent hardware repeats. The purpose is to compare closed-loop vibration suppression under a repeatable external road-like excitation. Unlike the tuning experiment, this experiment runs on the physical seat hardware.

The main questions are:

1. How much does each active controller reduce payload displacement, velocity, and acceleration relative to Passive?
2. Does the learned AR-PPO controller improve comfort-relevant velocity and acceleration metrics on hardware?
3. Are the controllers fast enough and free from saturation/fallback events during the fixed hardware test?

## Fixed Shaker Trajectory

The shaker input is loaded from:

```text
../tune_parameter/data/fixed_shaker_disturbance.csv
```

and copied to:

```text
data/fixed_test_disturbance.csv
```

If the tuning copy is missing, `main.py` regenerates the same signal with `utils.fixed_disturbance.fixed_shaker_disturbance()`.

Default trajectory parameters:

```text
seed = 20260516
duration = 9.0 s
dt = 0.03 s
samples = 301
force range in current CSV = -7.663 N to 8.280 N
shaker hardware limit = +/- 9 N
voltage mapping = force_N / 3
```

The actual trajectory is a deterministic mixed-frequency force signal:

```text
d_raw(t) = A_mod(t) * (
    0.58 * multisine(t)
  + 0.28 * up_chirp(t)
  + 0.22 * down_chirp(t)
)
```

The multisine component contains six fixed tones:

| Frequency [Hz] | Weight | Phase [rad] |
|---:|---:|---:|
| 0.75 | 0.35 | 1.5591 |
| 1.25 | 0.50 | 1.9840 |
| 1.85 | 0.65 | 2.1194 |
| 2.65 | 0.60 | 6.0519 |
| 3.55 | 0.45 | 1.0753 |
| 4.65 | 0.30 | 2.7374 |

The chirp terms are:

```text
up_chirp:   0.70 Hz -> 5.00 Hz, phase = 2.9409 rad
down_chirp: 5.00 Hz -> 0.70 Hz, phase = 5.1347 rad
```

The slow amplitude modulation is:

```text
A_mod(t) = 1 + 0.18*sin(2*pi*0.35*t + 4.7738)
```

After summation, the signal is smoothed by a `0.6 s` start/stop envelope, uniformly scaled to the target peak `0.92 * 9 N = 8.28 N`, clipped to `+/- 9 N`, and forced to exactly `0 N` at the first and last samples.

## Hardware Signals

Sampling and duration:

```text
dt = 0.03 s
duration = 9.0 s
samples per repeat = 301
control interval = 30 ms
```

Shaker side:

```text
force range = +/- 9 N
analog voltage range = +/- 3 V
voltage command = shaker_force_n / 3
```

Voice-coil side:

```text
force range = +/- 2 N
```

All controller outputs are clipped to `+/- 2 N` before being written to hardware.

## Controllers

All controllers use the current filtered state and current shaker-force sample. They do not receive future disturbance preview.

### Passive

Passive applies:

```text
u = 0
```

### ADRC

ADRC uses the selected tuning result from:

```text
../tune_parameter/data/best_by_method.csv
```

Current selected parameters:

```text
omega_o = 16.0
kp = 30.0
kd = 6.0
b0_scale = 1.0
b0 = 1 / m2
```

### MPC

MPC also loads its selected parameters from `best_by_method.csv`. It uses the nominal discrete two-mass model and solves a box-constrained action sequence by projected gradient descent.

Current selected parameters:

```text
variant = mpc_rand_08
horizon = 3
q_x2 = 469485.442
q_v2 = 6601.093
r_u = 0.008674
r_du = 0.018208
ki_x2 = 0.0
```

The disturbance forecast across the MPC horizon is the current shaker force held constant.

### DeePC

DeePC loads its selected parameters from `best_by_method.csv` and rebuilds the Hankel dictionary from:

```text
../hardware_stratified_protocol/data/trajectory_*.csv
```

The dictionary channels are:

```text
u = voice_force_n
d = shaker_force_n
y = [x1_filtered, v1_filtered, x2_filtered, v2_filtered]
```

Current selected parameters:

```text
variant = deepc_rand_09
t_ini = 8
n_f = 3
max_columns = 80
action_alpha = 0.7241
dictionary_source = measured_csv
```

If DeePC initialization or solving fails, the script records `fallback_flag = 1` and applies the bounded stabilizing fallback force. In the current result, no selected DeePC repeat uses fallback.

### AR-PPO

AR-PPO loads:

```text
../control_learning_compare/models/adaptive_residual_seed_666.pt
```

It uses ADRC as the base controller and applies a learned residual:

```text
u = clip(u_ADRC + u_residual, -2, 2)
```

The policy observation is:

```text
[x1, v1, x2, v2, u_prev, disturbance]
```

The observation scale is stored in the checkpoint and defaults to:

```text
[0.02, 0.6, 0.02, 0.6, 2.0, 9.0]
```

## Measurement Filtering

Before every repeat, the hardware is allowed to settle and the mean sensor reading is used as the zero offset. During the rollout, the reported state is computed online:

1. Raw displacements `x1_raw` and `x2_raw` are offset by the settling mean.
2. Displacements are filtered by a causal first-order low-pass filter with `8 Hz` cutoff.
3. Velocities are reconstructed by timestamped finite differencing and then filtered by the same `8 Hz` filter.
4. Accelerations are reconstructed from filtered velocity and filtered again by the same `8 Hz` filter.

The rollout CSV stores both filtered metric signals and raw displacement signals:

```text
x1, v1, a1, x2, v2, a2
x1_filtered, v1_filtered, a1_filtered, x2_filtered, v2_filtered, a2_filtered
x1_raw, x2_raw
```

Metrics and plots use the filtered signals.

## Safety And Resume Logic

Before each repeat, the script stops the shaker and voice coil, clears serial buffers, waits for settling, and recomputes the displacement offset.

During each repeat:

```text
sudden jump abort threshold = 0.008 m
absolute displacement abort threshold = 0.04 m
max retries = 10
post-repeat wait = 2.0 s
```

Every successful repeat is saved immediately as:

```text
data/rollouts/{controller}_repeat_{repeat}.csv
```

Existing valid rollout files are skipped by default, so interrupted hardware collection can resume. Passing `--overwrite` repeats all requested trials.

Collection order is interleaved by repeat:

```text
repeat 1: Passive, ADRC, MPC, DeePC, AR-PPO
repeat 2: Passive, ADRC, MPC, DeePC, AR-PPO
...
```

This reduces the chance that slow hardware drift is confounded with controller identity.

## Metrics

For each repeat, `repeat_metrics.csv` reports:

```text
x2_rms, x2_mean_abs, x2_peak_abs
v2_rms, v2_mean_abs, v2_peak_abs
a2_rms, a2_mean_abs, a2_peak_abs
compute_time_mean_ms, compute_time_std_ms, compute_time_max_ms
saturation_count, fallback_count
```

For each controller, `aggregate_statistics.csv` reports the mean and standard deviation of those repeat-level metrics across the five repeats.

## Outputs

The experiment writes:

```text
data/fixed_test_disturbance.csv
data/rollouts/*.csv
data/repeat_metrics.csv
data/aggregate_statistics.csv
figures/fixed_trajectory_comparison.pdf
```

The plotting script reads only the local CSV files in this experiment directory.

## Commands

Collect or resume hardware data:

```bash
python main.py
```

Redo all requested repeats:

```bash
python main.py --overwrite
```

Run only a subset of controllers:

```bash
python main.py --controllers passive adrc ar_ppo
```

Plot the default first `2 s` window:

```bash
python plot.py
```

Plot the full `9 s` trajectory:

```bash
python plot.py --time-window-s 0
```
