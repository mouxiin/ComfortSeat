# Tune Parameter Results

## Data Analyzed

The current result files are:

```text
data/fixed_shaker_disturbance.csv
data/tuning_summary.csv
data/best_by_method.csv
data/comparison_rollouts.csv
figures/tune_parameter_comparison.pdf
```

The default run uses:

```text
seed = 20260516
duration = 9.0 s
dt = 0.03 s
samples = 301
total candidates = 759
```

Candidate counts:

| Method | Candidates |
|---|---:|
| Passive | 1 |
| ADRC | 631 |
| MPC | 72 |
| DeePC | 55 |

The fixed shaker disturbance spans `-7.663 N` to `8.280 N`.

## Fixed Disturbance Trajectory

The comparison is driven by one deterministic shaker-force trajectory, not by a new random signal for each controller. It is a `9.0 s` mixed-frequency input generated with seed `20260516`.

The raw waveform combines:

- six multisine tones at `0.75, 1.25, 1.85, 2.65, 3.55, 4.65 Hz`;
- an up-chirp from `0.70 Hz` to `5.00 Hz`;
- a down-chirp from `5.00 Hz` to `0.70 Hz`;
- a slow `0.35 Hz` amplitude modulation.

In compact form:

```text
d_raw(t) = A_mod(t) * (
    0.58 * multisine(t)
  + 0.28 * up_chirp(t)
  + 0.22 * down_chirp(t)
)
```

The signal is then given a `0.6 s` smooth start/stop envelope and uniformly scaled to a target peak of `8.28 N`, which is `92%` of the `+/- 9 N` shaker force limit. The first and last samples are exactly `0 N`.

This trajectory is useful for tuning because it excites the nominal seat model over the main low-frequency band while remaining within the hardware shaker range. Since all controllers see the exact same trajectory, differences in the result table come from controller behavior rather than disturbance sampling variation.

## Main Finding

All active controllers substantially reduce the comfort-oriented objective relative to Passive. The best nominal-plant controller is MPC:

```text
Passive objective = 0.151590
ADRC objective    = 0.011333
MPC objective     = 0.002132
DeePC objective   = 0.009125
```

Objective reduction relative to Passive:

| Method | Reduction |
|---|---:|
| ADRC | 92.5% |
| MPC | 98.6% |
| DeePC | 94.0% |

MPC also reduces the objective by `81.2%` relative to the best ADRC candidate. This is expected on the ideal nominal plant because MPC uses the same model structure for prediction. The result should be interpreted as a strong nominal-model benchmark rather than as a direct hardware superiority claim.

## Best Candidates

| Method | Best variant | Objective | x2 RMS [mm] | v2 RMS [mm/s] | a2 RMS [m/s^2] | u RMS [N] | Mean compute [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| Passive | `passive` | 0.151590 | 1.5329 | 18.530 | 0.27305 | 0.000 | 0.000 |
| ADRC | `adrc_o16_kp30_kd6_b1` | 0.011333 | 0.3733 | 4.169 | 0.06854 | 0.162 | 0.000 |
| MPC | `mpc_rand_08` | 0.002132 | 0.0154 | 0.064 | 0.00115 | 0.182 | 0.801 |
| DeePC | `deepc_rand_09` | 0.009125 | 0.2671 | 3.385 | 0.05989 | 0.177 | 2.872 |

No best candidate has fallback or saturation steps:

```text
fallback_count = 0
saturation_count = 0
```

The selected ADRC parameters are:

```text
omega_o = 16.0
kp = 30.0
kd = 6.0
b0_scale = 1.0
```

These are the parameters loaded downstream when ADRC is used as a baseline or as the residual-learning base controller.

## MPC Result

The best MPC candidate is:

```text
variant = mpc_rand_08
horizon = 3
q_x2 = 469485.442
q_v2 = 6601.093
r_u = 0.008674
r_du = 0.018208
ki_x2 = 0.0
```

It gives the lowest objective and the smallest payload displacement, velocity, and reconstructed acceleration RMS in this experiment. Its mean solve time is `0.801 ms` and maximum solve time is `5.821 ms`, both below the `30 ms` control interval in this nominal simulation.

The short optimal horizon suggests that, for this fixed disturbance and nominal model, strong near-term suppression with smooth action is enough to dominate the cost.

## DeePC Result

The best DeePC candidate is:

```text
variant = deepc_rand_09
dictionary_source = measured_csv
t_ini = 8
n_f = 3
max_columns = 80
action_alpha = 0.7241
```

DeePC improves over ADRC by `19.5%` in objective and uses comparable control effort. It remains worse than MPC on the ideal nominal plant. This gap is reasonable because DeePC relies on a finite measured trajectory dictionary and regularized least-squares matching, while MPC directly uses the exact nominal model used by the simulator.

The best DeePC mean compute time is `2.872 ms`, with maximum `14.874 ms`, also below the control interval in this offline simulation.

## Paper-Ready Interpretation

A concise interpretation suitable for the paper is:

```text
The parameter tuning experiment establishes Passive, ADRC, MPC, and DeePC baselines under an identical fixed shaker disturbance on the nominal two-mass plant. All active controllers substantially reduce the weighted comfort objective relative to Passive. The selected ADRC controller reduces the objective by 92.5% and provides the active baseline used in subsequent learning experiments. MPC achieves the best nominal result, with a 98.6% objective reduction relative to Passive, while DeePC achieves a 94.0% reduction using a measured-trajectory dictionary. Because this experiment is conducted on the ideal plant, the MPC result is best interpreted as a nominal-model benchmark, whereas downstream DDM and disturbance-sweep experiments evaluate learned-model behavior and robustness.
```

Recommended claims:

- Strong claim: tuned ADRC is a much stronger baseline than Passive under the fixed shaker disturbance.
- Strong claim: MPC is the best controller in this nominal-plant tuning experiment.
- Supported claim: measured-dictionary DeePC is competitive with ADRC and improves the objective by `19.5%` relative to ADRC.
- Nuanced claim: the nominal MPC result should not be used alone as evidence of hardware robustness.

Avoid claiming:

- that MPC is the best hardware controller;
- that the tuning result proves robustness to actuator-side disturbances;
- that the random MPC or DeePC search is globally optimal.

## Recommended Figure

Use:

```text
figures/tune_parameter_comparison.pdf
```

This figure is the most direct visual summary. It shows payload displacement, velocity, acceleration, action, the common shaker disturbance, and the final objective comparison for the selected candidate from each method.

## Limitations

The result should be framed as a baseline tuning study:

- The evaluation uses the ideal nominal two-mass plant.
- The tuning uses one fixed shaker disturbance and one random seed.
- MPC uses the same model structure as the simulator, so it has an advantage in this setting.
- DeePC uses measured hardware trajectories for its dictionary but is still evaluated on the nominal plant.
- No actuator-side disturbance, sensor noise, model mismatch, or hardware delay is injected here.

These limitations are acceptable because this experiment is intended to select and document baseline parameters. Robustness and fault-response claims should come from the downstream actuator-disturbance and DDM-based experiments.
