# Actuator Disturbance Sweep Hardware Results

## Data Analyzed

The completed hardware sweep contains:

```text
actuator disturbance cases = 9
controllers = 6
repeats per controller and case = 5
total hardware rollouts = 270
samples per rollout = 301
rollout duration = 9.0 s
```

The analysis uses:

```text
data/sweep_plan.csv
data/fixed_test_disturbance.csv
data/unknown_forces/*.csv
data/rollouts/{case}/{controller}_repeat_{repeat}.csv
data/repeat_metrics.csv
data/aggregate_statistics.csv
```

Event metrics use the code-defined event window:

```text
event_start = 2.0 s
event_end = 2.0 s + pulse_duration + 0.45 s
```

The `0.45 s` extension captures the immediate recovery after the actuator-side pulse.

## Main Finding

`DGS-AR-PPO` substantially improves the nominal `AR-PPO` controller under unobserved actuator-side force pulses. Averaged case-by-case over the nine force-duration cases, DGS reduces AR-PPO event-window RMS by:

```text
x2 displacement: 31.1%
v2 velocity:     43.5%
a2 acceleration: 25.2%
```

DGS improves displacement and velocity in every tested case. Acceleration improves in eight of the nine cases. The exception is `fm20_d30`, the longest and strongest pulse, where DGS has `14.0%` higher event acceleration RMS than AR-PPO while still reducing displacement by `14.1%` and velocity by `22.4%`.

The best interpretation is that DGS turns AR-PPO from a nominal learned controller into a fault-aware hybrid controller. When the DDM residual gate detects an actuator-side event, DGS temporarily borrows the robustness of MPC; after residual recovery, it returns to AR-PPO.

## Aggregate Event Performance

Values are mean +/- standard deviation across all 45 rollouts per controller. Displacement and velocity are converted to `mm` and `mm/s`.

| Controller | x2 event RMS [mm] | v2 event RMS [mm/s] | a2 event RMS [m/s^2] | x2 full RMS [mm] | v2 full RMS [mm/s] | a2 full RMS [m/s^2] |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 3.798 +/- 1.049 | 19.001 +/- 5.384 | 0.221 +/- 0.053 | 1.891 +/- 0.339 | 15.712 +/- 1.352 | 0.177 +/- 0.016 |
| ADRC | 4.896 +/- 1.434 | 23.218 +/- 6.260 | 0.201 +/- 0.051 | 1.894 +/- 0.639 | 10.242 +/- 2.408 | 0.097 +/- 0.021 |
| MPC | 3.032 +/- 0.956 | 14.554 +/- 3.446 | 0.170 +/- 0.037 | 1.164 +/- 0.361 | 8.833 +/- 0.600 | 0.137 +/- 0.007 |
| DeePC | 4.254 +/- 1.246 | 20.263 +/- 5.754 | 0.184 +/- 0.054 | 1.575 +/- 0.493 | 8.991 +/- 1.409 | 0.093 +/- 0.013 |
| AR-PPO | 4.715 +/- 1.157 | 26.826 +/- 9.495 | 0.251 +/- 0.095 | 1.760 +/- 0.446 | 10.287 +/- 2.854 | 0.098 +/- 0.030 |
| DGS-AR-PPO | 3.236 +/- 0.910 | 14.392 +/- 3.476 | 0.170 +/- 0.033 | 1.249 +/- 0.346 | 5.961 +/- 1.157 | 0.070 +/- 0.014 |

Key observations:

- `DGS-AR-PPO` has the lowest pooled event velocity RMS, `14.392 mm/s`, slightly lower than MPC, `14.554 mm/s`.
- `DGS-AR-PPO` and MPC have nearly identical pooled event acceleration RMS, both about `0.170 m/s^2`.
- MPC has the lowest event displacement RMS, `3.032 mm`; DGS is close at `3.236 mm`.
- On full-trajectory velocity and acceleration RMS, DGS is the best controller in this dataset.

## Case-Mean Reductions

The table reports the mean percentage reduction of `DGS-AR-PPO` relative to each baseline. Reductions are computed from case means in `aggregate_statistics.csv` and then averaged over the nine cases. The range in parentheses is the minimum to maximum reduction over the cases.

Positive values mean DGS is better. Negative values mean DGS is worse.

| Baseline | x2 event RMS reduction | v2 event RMS reduction | a2 event RMS reduction |
|---|---:|---:|---:|
| Passive | +14.8% (+12.1 to +18.4) | +22.0% (-1.4 to +37.5) | +20.0% (-8.9 to +38.7) |
| ADRC | +33.6% (+29.5 to +37.0) | +37.1% (+25.7 to +45.0) | +13.4% (-6.3 to +30.2) |
| MPC | -8.1% (-19.0 to -3.2) | +1.1% (-4.2 to +8.4) | -0.8% (-21.3 to +10.7) |
| DeePC | +23.6% (+20.4 to +28.4) | +27.8% (+13.8 to +34.7) | +3.6% (-33.4 to +25.0) |
| AR-PPO | +31.1% (+14.1 to +43.7) | +43.5% (+22.4 to +59.2) | +25.2% (-14.0 to +55.7) |

This supports three claims:

- DGS consistently dominates its own base policy, AR-PPO, on displacement and velocity robustness.
- DGS is competitive with full-time MPC: slightly worse in displacement, essentially tied in acceleration, and slightly better in velocity on average.
- DGS improves over DeePC, ADRC, and Passive on the main displacement and velocity event metrics, while acceleration comparisons are more case-dependent.

## DGS Versus AR-PPO By Case

| Case | x2 event RMS reduction | v2 event RMS reduction | a2 event RMS reduction |
|---|---:|---:|---:|
| fm10_d10 | +41.5% | +49.7% | +37.7% |
| fm10_d20 | +28.6% | +38.6% | +7.9% |
| fm10_d30 | +23.2% | +34.7% | +4.2% |
| fm15_d10 | +43.7% | +50.9% | +42.6% |
| fm15_d20 | +35.0% | +53.3% | +41.1% |
| fm15_d30 | +18.4% | +32.3% | +6.3% |
| fm20_d10 | +42.3% | +50.0% | +45.6% |
| fm20_d20 | +33.5% | +59.2% | +55.7% |
| fm20_d30 | +14.1% | +22.4% | -14.0% |

The strongest DGS gains appear in short and medium pulses. Gains are smaller for the longest pulses, especially `fm20_d30`, because the system spends more time in the fallback and recovery regime and acceleration becomes more sensitive to the policy transition.

## Best Controller Counts

The best controller in each case, using five-repeat case means, is:

```text
x2 event RMS: MPC wins 9 / 9 cases
v2 event RMS: DGS wins 4 / 9, MPC wins 4 / 9, Passive wins 1 / 9
a2 event RMS: DGS wins 3 / 9, MPC wins 3 / 9, DeePC wins 2 / 9, AR-PPO wins 1 / 9
```

DGS should therefore not be described as uniformly better than MPC. The stronger and more accurate claim is:

```text
DGS recovers most of the robustness of MPC during actuator-side faults while retaining the AR-PPO nominal policy structure and using only a residual gate to decide when fallback is needed.
```

## DGS Switching Diagnostics

The DGS detector triggered once in every DGS rollout:

```text
45 / 45 DGS rollouts triggered fallback
mean event_trigger_count = 1.0
mean switch-to-fallback count = 1.0
mean switch-back-to-AR-PPO count = 1.0
```

Across all DGS rollouts:

```text
mean trigger delay from requested event start = 51.8 ms
trigger delay range = 40.2 ms to 78.0 ms
mean fallback-active duration = 26.8 samples = 0.804 s
fallback-active range = 15 samples to 58 samples
mean return time = 2.859 s
```

Per-case switching statistics:

| Case | Trigger delay [ms] | Fallback active [s] | Detection samples | Return time [s] |
|---|---:|---:|---:|---:|
| fm10_d10 | 60.2 +/- 15.6 | 0.468 +/- 0.027 | 8.8 +/- 1.3 | 2.533 +/- 0.038 |
| fm10_d20 | 56.2 +/- 17.0 | 0.720 +/- 0.030 | 17.2 +/- 2.0 | 2.778 +/- 0.038 |
| fm10_d30 | 49.5 +/- 4.8 | 1.086 +/- 0.146 | 27.4 +/- 1.5 | 3.135 +/- 0.143 |
| fm15_d10 | 51.6 +/- 11.3 | 0.468 +/- 0.027 | 10.0 +/- 0.7 | 2.523 +/- 0.024 |
| fm15_d20 | 44.4 +/- 4.4 | 0.792 +/- 0.113 | 20.4 +/- 0.5 | 2.840 +/- 0.114 |
| fm15_d30 | 47.0 +/- 13.0 | 1.248 +/- 0.295 | 30.0 +/- 0.7 | 3.301 +/- 0.289 |
| fm20_d10 | 50.5 +/- 3.8 | 0.462 +/- 0.027 | 10.8 +/- 0.4 | 2.512 +/- 0.024 |
| fm20_d20 | 56.0 +/- 14.1 | 0.792 +/- 0.132 | 21.2 +/- 0.8 | 2.853 +/- 0.125 |
| fm20_d30 | 50.6 +/- 11.8 | 1.200 +/- 0.139 | 30.6 +/- 1.1 | 3.251 +/- 0.140 |

The switching behavior matches the intended state machine:

- Short pulses return earlier, with fallback active for about `0.46-0.47 s`.
- Medium pulses keep fallback active for about `0.72-0.79 s`.
- Long pulses keep fallback active for about `1.09-1.25 s`.
- Every rollout switches back to AR-PPO after residual recovery, so DGS does not become full-time MPC.

## Computation And Reliability

Computation time is reported as mean +/- standard deviation across the 45 repeat-level metrics for each controller. The raw-sample `p95` and maximum are computed from all saved per-sample timings.

| Controller | Mean compute [ms] | Max compute [ms] | Raw p95 [ms] | Raw max [ms] | Saturation count | Fallback count |
|---|---:|---:|---:|---:|---:|---:|
| Passive | 0.0028 +/- 0.0003 | 0.0096 +/- 0.0088 | 0.0042 | 0.0622 | 0 | 0 |
| ADRC | 0.0568 +/- 0.0119 | 0.1457 +/- 0.0380 | 0.0965 | 0.2294 | 0 | 0 |
| MPC | 2.1467 +/- 0.6232 | 3.4059 +/- 1.1598 | 3.7488 | 6.2512 | 0 | 0 |
| DeePC | 0.6603 +/- 0.0910 | 2.1962 +/- 1.7021 | 0.9798 | 10.1004 | 0 | 0 |
| AR-PPO | 0.4456 +/- 0.0553 | 0.7835 +/- 0.1337 | 0.6298 | 1.2080 | 36 | 0 |
| DGS-AR-PPO | 0.9280 +/- 0.1566 | 4.5890 +/- 8.4660 | 2.1164 | 59.4295 | 0 | 0 |

All controllers satisfy the `30 ms` control interval in mean and p95 timing. There is one isolated DGS logged raw-sample outlier above `30 ms` (`59.43 ms`), so hard real-time claims should be stated conservatively. DGS still has lower mean compute time than full-time MPC because it runs AR-PPO outside fallback intervals and only adds residual checking plus temporary MPC activation.

The nonzero AR-PPO saturation count indicates that standalone AR-PPO sometimes requests force beyond the `+/- 2 N` command limit under actuator faults. DGS avoids saturation in this sweep by switching to MPC during detected events.

## Recommended Figures

Use:

```text
figures/actuator_disturbance_pulse_sweep_rms_all_bar.pdf
figures/actuator_disturbance_pulse_sweep_a2_v2_contours_all.pdf
```

Recommended time-response examples:

```text
figures/time_response_cases/actuator_disturbance_pulse_sweep_fm20_d20_time_response_comparison.pdf
figures/time_response_cases/actuator_disturbance_pulse_sweep_fm10_d10_time_response_comparison.pdf
```

`fm20_d20` is severe enough to show the AR-PPO transient clearly without being as edge-case-heavy as `fm20_d30`. `fm10_d10` is useful as a mild-disturbance example showing that the residual gate still detects the smallest tested actuator pulse.

## Paper-Ready Interpretation

A concise interpretation suitable for the paper is:

```text
Under a 3-by-3 hardware sweep of unobserved negative actuator-side force pulses, the proposed DGS-AR-PPO supervisor detected every injected event and switched exactly once to the MPC fallback before returning to AR-PPO. Relative to the nominal AR-PPO policy, DGS reduced event-window payload displacement RMS by 31.1% and payload velocity RMS by 43.5% on average across the nine force-duration cases. Its event velocity and acceleration performance was comparable to full-time MPC, while its displacement response remained slightly worse than MPC. These results indicate that residual-gated fallback can recover most of the robustness of a model-based controller without permanently replacing the learned policy.
```

Recommended claims:

- Strong claim: DGS substantially improves AR-PPO under unobserved actuator-side input disturbances.
- Strong claim: DGS detects all tested actuator-side pulses using a force-agnostic DDM residual threshold.
- Supported but nuanced claim: DGS is competitive with MPC in event velocity and acceleration, while MPC remains best for event displacement.
- Supported claim: DGS reduces AR-PPO command saturation under these faults.

Avoid claiming:

- DGS is uniformly best across all metrics and cases.
- DGS estimates the actuator disturbance magnitude. The default DGS controller detects residual events and switches policy; it does not use `F_ud_est` as compensation.
- The result covers all actuator faults. This sweep tests negative additive force pulses only.

## Limitations

The sweep is strong but bounded:

- Only negative actuator-side pulses were tested.
- The pulse start time was fixed at `2.0 s`.
- The shaker disturbance was fixed rather than randomized across rollouts.
- Each case has five repeats.
- The detector threshold was calibrated from nominal DDM residuals, but this sweep does not include a no-actuator-fault false-positive case.
- The actuator-side disturbance is injected as an independent additive input force after command saturation. Gain loss, delay, dead-zone, and saturation shrinkage are not covered here.

These limitations do not weaken the main conclusion of this sweep: DGS gives the nominal AR-PPO controller a reliable residual-gated fallback mechanism under structured actuator-side pulse disturbances.
