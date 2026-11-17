# Shaker Disturbance Hardware Results

## Data Analyzed

The current result files are:

```text
data/fixed_test_disturbance.csv
data/rollouts/*.csv
data/repeat_metrics.csv
data/aggregate_statistics.csv
figures/fixed_trajectory_comparison.pdf
```

The hardware comparison contains:

```text
controllers = Passive, ADRC, MPC, DeePC, AR-PPO
repeats per controller = 5
total successful rollouts = 25
samples per rollout = 301
duration = 9.0 s
dt = 0.03 s
```

All metrics are computed from the online filtered kinematic signals. Displacement is filtered by the causal `8 Hz` first-order filter; velocity and acceleration are reconstructed by timestamped finite differencing and filtered by the same `8 Hz` causal filter.

## Fixed Disturbance

All controllers are tested under the same deterministic shaker-force trajectory. The trajectory spans:

```text
minimum force = -7.663 N
maximum force = 8.280 N
frequency band = 0.70-5.00 Hz
```

It combines six multisine tones, an up-chirp, a down-chirp, and a slow amplitude modulation:

```text
d_raw(t) = A_mod(t) * (
    0.58 * multisine(t)
  + 0.28 * up_chirp(t)
  + 0.22 * down_chirp(t)
)
```

The signal uses a `0.6 s` smooth start/stop envelope and starts and ends at exactly `0 N`. Because the same fixed trajectory is used for all methods and repeats, controller differences are not caused by different disturbance samples.

## Main Finding

All active controllers substantially reduce hardware payload motion relative to Passive. The best method depends on the metric:

```text
lowest x2 RMS: MPC
lowest v2 RMS: AR-PPO
lowest a2 RMS: AR-PPO
lowest x2 peak: AR-PPO
lowest v2 peak: AR-PPO
lowest a2 peak: AR-PPO
```

MPC gives the smallest displacement RMS, reducing `x2` RMS by `67.3%` relative to Passive. AR-PPO gives the strongest velocity and acceleration suppression, reducing `v2` RMS by `77.0%` and `a2` RMS by `77.3%` relative to Passive.

This is the key hardware interpretation: MPC is strongest for displacement tracking, while AR-PPO is strongest for comfort-oriented velocity and acceleration suppression.

## RMS Metrics

Values are mean +/- standard deviation across five repeats.

| Controller | x2 RMS [m] | vs Passive | v2 RMS [m/s] | vs Passive | a2 RMS [m/s^2] | vs Passive |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 0.00141647 +/- 0.00000903 | 0.0% | 0.0149396 +/- 0.000145 | 0.0% | 0.170396 +/- 0.001667 | 0.0% |
| ADRC | 0.000618785 +/- 0.0000396 | 56.3% | 0.00566365 +/- 0.000287 | 62.1% | 0.0635997 +/- 0.002271 | 62.7% |
| MPC | 0.000463500 +/- 0.00000972 | 67.3% | 0.00687760 +/- 0.000216 | 54.0% | 0.115757 +/- 0.003965 | 32.1% |
| DeePC | 0.000570617 +/- 0.0000462 | 59.7% | 0.00569938 +/- 0.000328 | 61.9% | 0.0676210 +/- 0.003144 | 60.3% |
| AR-PPO | 0.000599504 +/- 0.0000500 | 57.7% | 0.00343057 +/- 0.000291 | 77.0% | 0.0386255 +/- 0.003173 | 77.3% |

The displacement ranking is:

```text
MPC > DeePC > AR-PPO > ADRC > Passive
```

The velocity and acceleration rankings are:

```text
AR-PPO > ADRC > DeePC > MPC > Passive
```

where larger rank means lower RMS response.

## Peak Metrics

Peak absolute responses show the same qualitative trend.

| Controller | x2 peak [m] | vs Passive | v2 peak [m/s] | vs Passive | a2 peak [m/s^2] | vs Passive |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 0.00389731 +/- 0.000200 | 0.0% | 0.0391020 +/- 0.000762 | 0.0% | 0.441134 +/- 0.011223 | 0.0% |
| ADRC | 0.00180197 +/- 0.000144 | 53.8% | 0.0176862 +/- 0.000713 | 54.8% | 0.194150 +/- 0.009446 | 56.0% |
| MPC | 0.00147811 +/- 0.0000273 | 62.1% | 0.0201081 +/- 0.001838 | 48.6% | 0.306630 +/- 0.015651 | 30.5% |
| DeePC | 0.00167672 +/- 0.000206 | 57.0% | 0.0189521 +/- 0.001431 | 51.5% | 0.212311 +/- 0.013123 | 51.9% |
| AR-PPO | 0.00145030 +/- 0.000133 | 62.8% | 0.0112821 +/- 0.001460 | 71.1% | 0.137391 +/- 0.015885 | 68.9% |

AR-PPO has the best peak response in all three reported channels, with the largest advantage in velocity and acceleration.

## Computation And Reliability

Computation time is also reported as mean +/- standard deviation across five repeats. The first timing column averages the per-repeat mean computation time. The second timing column averages the per-repeat maximum computation time.

| Controller | Mean compute [ms] | Max compute [ms] | Saturation count | Fallback count |
|---|---:|---:|---:|---:|
| Passive | 0.0024 +/- 0.0004 | 0.0064 +/- 0.0021 | 0 | 0 |
| ADRC | 0.0583 +/- 0.0102 | 0.1246 +/- 0.0335 | 0 | 0 |
| MPC | 2.0979 +/- 0.4985 | 3.8991 +/- 1.9472 | 0 | 0 |
| DeePC | 0.6183 +/- 0.0946 | 2.7677 +/- 2.3465 | 0 | 0 |
| AR-PPO | 0.4865 +/- 0.0863 | 0.8164 +/- 0.1020 | 0 | 0 |

All controllers remain well below the `30 ms` control interval. MPC is the most computationally expensive controller, but still runs within the hardware timing margin in these repeats. No controller shows actuator saturation or fallback events, so the comparison reflects normal closed-loop behavior rather than clipping or recovery logic.

## Paper-Ready Interpretation

A concise interpretation suitable for the paper is:

```text
In the fixed shaker-disturbance hardware experiment, all active controllers reduce payload motion relative to the passive system. MPC achieves the lowest displacement RMS, reducing x2 RMS by 67.3%, but it produces higher velocity and acceleration than ADRC, DeePC, and AR-PPO. The adaptive residual PPO controller provides the strongest comfort-oriented response: it reduces velocity RMS by 77.0% and acceleration RMS by 77.3% relative to Passive, and also achieves the lowest peak displacement, velocity, and acceleration among the tested controllers. Across five repeats, no controller triggers saturation or fallback, and all computation times remain below the 30 ms control interval.
```

Recommended claims:

- Strong claim: all active methods improve over Passive on the hardware fixed-disturbance test.
- Strong claim: MPC gives the best displacement RMS.
- Strong claim: AR-PPO gives the best velocity and acceleration RMS, and the best peak responses.
- Supported claim: AR-PPO is the strongest comfort-oriented controller in this hardware test.

Avoid claiming:

- that AR-PPO has the lowest displacement RMS;
- that MPC is the best overall comfort controller;
- that these five repeats alone prove robustness to actuator faults. Actuator fault robustness is evaluated separately in `actuator_disturbance_sweep`.

## Recommended Figure

Use:

```text
figures/fixed_trajectory_comparison.pdf
```

This figure shows the mean trajectory and repeat-to-repeat standard-deviation bands for `x2`, `v2`, `a2`, control action, and the common shaker disturbance. The default plotting window is the first `2 s`; the full `9 s` trajectory can be regenerated with:

```bash
python plot.py --time-window-s 0
```

## Limitations

This experiment is a hardware fixed-trajectory comparison. It is strong evidence for repeatable response under one broadband shaker excitation, but it should be framed with these limitations:

- The disturbance trajectory is fixed rather than randomly resampled.
- There are five repeats per controller.
- The experiment tests shaker-side disturbance rejection, not actuator-side additive faults.
- The comparison uses filtered displacement-derived velocity and acceleration, so the reported acceleration is a processed comfort proxy rather than a direct accelerometer channel.
- MPC and DeePC use no future disturbance preview; the current disturbance is held constant across their prediction horizons.
