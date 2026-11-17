# Shaker Disturbance Sweep Hardware Results

## Data Analyzed

The current result files are:

```text
data/sweep_plan.csv
data/disturbances/*.csv
data/rollouts/*/*.csv
data/repeat_metrics.csv
data/aggregate_statistics.csv
figures/frequency_test_all_plot_a2_v2_contours_all.pdf
figures/frequency_test_all_plot_rms_all_bar.pdf
```

The completed dataset contains:

```text
disturbance cases = 25
controllers = 5
repeats per case/controller = 5
successful rollouts = 625
samples per rollout = 301
duration per rollout = 9.0 s
```

`repeat_metrics.csv` contains one row per rollout. `aggregate_statistics.csv` contains one row per `(case_id, controller)` pair, with mean and standard deviation across the five repeats.

Unless stated otherwise, the sweep-level values below are computed from the `25` case-level means in `aggregate_statistics.csv`. The reported `+/-` value is therefore the standard deviation across the sweep grid, not the within-case repeat standard deviation.

## Main Finding

Across the full frequency-amplitude sweep, all active controllers reduce payload motion relative to Passive. AR-PPO has the best average RMS response over the sweep:

```text
lowest average x2 RMS: AR-PPO
lowest average v2 RMS: AR-PPO
lowest average a2 RMS: AR-PPO
```

The advantage is strongest for velocity and acceleration. Relative to Passive, AR-PPO reduces:

```text
x2 RMS by 55.2%
v2 RMS by 75.4%
a2 RMS by 76.4%
```

Displacement suppression is competitive among the active controllers, while velocity and acceleration suppression clearly favor AR-PPO.

## Sweep-Averaged RMS Metrics

Values are mean +/- standard deviation across the 25 disturbance cases.

| Controller | x2 RMS [m] | vs Passive | v2 RMS [m/s] | vs Passive | a2 RMS [m/s^2] | vs Passive |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 0.00111871 +/- 0.00133415 | 0.0% | 0.0123993 +/- 0.0152463 | 0.0% | 0.157503 +/- 0.175417 | 0.0% |
| ADRC | 0.000509662 +/- 0.000411636 | 54.4% | 0.00513184 +/- 0.00366098 | 58.6% | 0.0675953 +/- 0.0425871 | 57.1% |
| MPC | 0.000521340 +/- 0.000382270 | 53.4% | 0.00747894 +/- 0.00634663 | 39.7% | 0.125122 +/- 0.105045 | 20.6% |
| DeePC | 0.000521921 +/- 0.000469178 | 53.4% | 0.00556037 +/- 0.00535137 | 55.2% | 0.0762712 +/- 0.0620993 | 51.6% |
| AR-PPO | 0.000501588 +/- 0.000205481 | 55.2% | 0.00304551 +/- 0.00178502 | 75.4% | 0.0371616 +/- 0.0186824 | 76.4% |

The average `x2` result is close among the active controllers. AR-PPO is best on average, but the difference from ADRC, MPC, and DeePC is small compared with the frequency-amplitude variation. For `v2` and `a2`, AR-PPO has a much clearer margin.

## Winner Counts

The table below counts, for each metric, how many of the 25 disturbance cases are won by each controller. Lower response is better.

| Metric | Passive | ADRC | MPC | DeePC | AR-PPO |
|---|---:|---:|---:|---:|---:|
| x2 RMS | 1 | 8 | 7 | 2 | 7 |
| v2 RMS | 2 | 0 | 5 | 1 | 17 |
| a2 RMS | 2 | 0 | 2 | 3 | 18 |
| x2 peak abs | 1 | 6 | 8 | 4 | 6 |
| v2 peak abs | 3 | 0 | 4 | 2 | 16 |
| a2 peak abs | 3 | 2 | 0 | 2 | 18 |

This shows the main pattern: displacement is shared across multiple active controllers, but velocity and acceleration are dominated by AR-PPO.

The few Passive wins occur in low-excitation cases, where active control effort can slightly increase a processed RMS or peak metric. They do not change the overall trend over the sweep.

## Amplitude Trend

At the three larger force amplitudes:

```text
5.4 N
7.2 N
9.0 N
```

AR-PPO has the lowest amplitude-averaged `x2`, `v2`, and `a2` RMS among all controllers. This is important because those cases are the more demanding parts of the sweep and are closer to the shaker force limit.

At the two smaller amplitudes, displacement RMS is sometimes won by DeePC or MPC. AR-PPO still has the lowest average velocity and acceleration RMS at every force amplitude.

## Peak Metrics

Values are mean +/- standard deviation across the 25 disturbance cases.

| Controller | x2 peak [m] | vs Passive | v2 peak [m/s] | vs Passive | a2 peak [m/s^2] | vs Passive |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 0.00200246 +/- 0.00197966 | 0.0% | 0.0226697 +/- 0.0230390 | 0.0% | 0.284840 +/- 0.273038 | 0.0% |
| ADRC | 0.00111981 +/- 0.000823965 | 44.1% | 0.0104690 +/- 0.00608530 | 53.8% | 0.139419 +/- 0.0750738 | 51.1% |
| MPC | 0.00105793 +/- 0.000602921 | 47.2% | 0.0140112 +/- 0.00927144 | 38.2% | 0.236602 +/- 0.155942 | 16.9% |
| DeePC | 0.00110627 +/- 0.000776499 | 44.8% | 0.0112800 +/- 0.00803679 | 50.2% | 0.156814 +/- 0.0988060 | 45.0% |
| AR-PPO | 0.00111949 +/- 0.000504672 | 44.1% | 0.00751261 +/- 0.00304699 | 66.9% | 0.0972557 +/- 0.0330195 | 65.9% |

MPC has the lowest average displacement peak, but AR-PPO has the lowest average velocity and acceleration peaks.

## Computation And Reliability

Computation time is reported as mean +/- standard deviation across the 25 disturbance cases, using the case-level five-repeat means.

| Controller | Mean compute [ms] | Max compute [ms] | Saturation count | Fallback count |
|---|---:|---:|---:|---:|
| Passive | 0.0027 +/- 0.0003 | 0.0125 +/- 0.0049 | 0 | 0 |
| ADRC | 0.0537 +/- 0.0086 | 0.1320 +/- 0.0207 | 0 | 0 |
| MPC | 1.9204 +/- 0.3716 | 3.0079 +/- 0.7755 | 0 | 0 |
| DeePC | 0.5874 +/- 0.0753 | 1.5376 +/- 0.4962 | 0 | 0 |
| AR-PPO | 0.4596 +/- 0.0584 | 0.8211 +/- 0.1229 | 0 | 0 |

All controllers remain far below the `30 ms` control interval. No controller shows saturation or fallback events in the completed sweep.

## Paper-Ready Interpretation

A concise interpretation suitable for the paper is:

```text
The shaker-disturbance sweep evaluates hardware performance over 25 sinusoidal disturbances spanning 1-5 Hz and 1.8-9.0 N. Averaged over the sweep, all active controllers reduce payload motion relative to Passive. AR-PPO achieves the lowest average RMS response in all three reported channels, reducing x2, v2, and a2 RMS by 55.2%, 75.4%, and 76.4%, respectively. Displacement suppression is competitive across ADRC, MPC, DeePC, and AR-PPO, but AR-PPO dominates the velocity and acceleration metrics, winning 17 of 25 v2 RMS cases and 18 of 25 a2 RMS cases. No saturation or fallback events are observed, and all controllers run comfortably within the 30 ms hardware control interval.
```

Recommended claims:

- Strong claim: AR-PPO has the best sweep-averaged RMS response over the full frequency-amplitude grid.
- Strong claim: AR-PPO is especially strong for velocity and acceleration suppression.
- Supported claim: displacement RMS is close among active controllers, so the x2 advantage should be stated modestly.
- Strong claim: all controllers meet the hardware computation-time budget in this sweep.

Avoid claiming:

- that AR-PPO wins displacement in every individual case;
- that Passive is always worse in every low-amplitude metric;
- that this sweep tests actuator-side additive disturbances. That is covered separately by `actuator_disturbance_sweep`.

## Recommended Figures

Use both generated figures:

```text
figures/frequency_test_all_plot_a2_v2_contours_all.pdf
figures/frequency_test_all_plot_rms_all_bar.pdf
```

The contour figure is the most direct evidence for frequency-amplitude robustness. The bar figure provides the compact paper summary across all 25 cases.

## Limitations

The sweep is a strong hardware robustness test for shaker-side sinusoidal disturbances, but it should be framed carefully:

- The grid contains five frequencies and five amplitudes, not a continuous disturbance distribution.
- Each case has five repeats.
- The disturbance is sinusoidal and shaker-side; actuator-side faults are tested elsewhere.
- Velocity and acceleration are derived from filtered displacement signals rather than directly measured by accelerometers.
- MPC and DeePC use no future disturbance preview; the current disturbance is held constant across their prediction horizons.
