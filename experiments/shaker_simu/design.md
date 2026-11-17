# Shaker Simu

## Goal

This experiment reruns the `shaker_disturbance` fixed-shaker benchmark inside the selected data-driven model. It is meant to answer whether the DDM surrogate reproduces the real hardware statistics saved in:

```text
shaker_disturbance/data/aggregate_statistics.csv
```

The experiment uses the same fixed shaker disturbance, the same tuned controller parameters, and the same AR-PPO checkpoint as the hardware `shaker_disturbance` experiment.

## Controllers

The simulated controllers are:

- `Passive`
- `ADRC`
- `MPC`
- `DeePC`
- `AR-PPO`

All controllers are created through `shaker_disturbance/main.py` via the existing controller factory. Therefore:

- ADRC, MPC, and DeePC parameters come from `tune_parameter/data/best_by_method.csv`.
- AR-PPO comes from `control_learning_compare/models/adaptive_residual_seed_666.pt`.
- DeePC uses the same measured CSV dictionary configuration as the hardware comparison.

## Surrogate Dynamics

The plant is the selected DDM checkpoint:

```text
ddm_ablation/models/selected_model.pt
```

The DDM state is `[x1, v1, x2, v2]`. Acceleration `a2` is reconstructed from finite differencing the simulated filtered `v2` and applying the same causal 8 Hz first-order low-pass filter used elsewhere in the project.

## Outputs

- `data/fixed_test_disturbance.csv`: fixed shaker force used in the run.
- `data/rollouts/*.csv`: one simulated rollout per controller.
- `data/simulated_metrics.csv`: simulated RMS, mean absolute, and peak metrics.
- `data/hardware_aggregate_statistics.csv`: copied hardware aggregate statistics.
- `data/simulation_vs_hardware.csv`: side-by-side hardware mean/std and DDM simulation values.
- `figures/shaker_simu_vs_hardware_rms.pdf`: RMS comparison against hardware mean and std.
- `figures/shaker_simu_time_response.pdf`: simulated time responses for all controllers.

## Usage

```bash
python main.py
python plot.py
```

Use `--duration-s 6.0` to truncate the fixed disturbance to a 6-second comparison.
