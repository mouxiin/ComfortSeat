# Shaker Simu Result

## Summary

This experiment reran the fixed `shaker_disturbance` benchmark in the selected DDM surrogate and compared the simulated RMS values with the real hardware aggregate statistics in `shaker_disturbance/data/aggregate_statistics.csv`.

The main observation is that the DDM surrogate matches the passive response reasonably well, but does not reproduce all closed-loop controller behavior equally:

- Passive is close to hardware: DDM is about `1.03x`, `1.06x`, and `1.07x` hardware for `x2`, `v2`, and `a2` RMS.
- ADRC and DeePC are systematically larger in simulation, especially for velocity and acceleration.
- MPC is much worse in simulation than hardware, about `3.6x` to `3.8x` hardware RMS. This suggests the DDM rollout plus MPC interaction is not representing the real closed-loop behavior reliably.
- AR-PPO is much better in simulation than hardware, about `0.43x`, `0.54x`, and `0.61x` hardware RMS. This suggests the DDM surrogate is optimistic for this learned controller.

## RMS Comparison

| Controller | DDM x2 RMS (m) | Hardware x2 RMS mean ± std (m) | DDM v2 RMS (m/s) | Hardware v2 RMS mean ± std (m/s) | DDM a2 RMS (m/s^2) | Hardware a2 RMS mean ± std (m/s^2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Passive | 0.001458 | 0.001416 ± 0.000009 | 0.015762 | 0.014940 ± 0.000145 | 0.182474 | 0.170396 ± 0.001667 |
| ADRC | 0.000754 | 0.000619 ± 0.000040 | 0.007971 | 0.005664 ± 0.000287 | 0.096410 | 0.063600 ± 0.002271 |
| MPC | 0.001657 | 0.000463 ± 0.000010 | 0.026403 | 0.006878 ± 0.000216 | 0.429009 | 0.115757 ± 0.003965 |
| DeePC | 0.000647 | 0.000571 ± 0.000046 | 0.007351 | 0.005699 ± 0.000328 | 0.088525 | 0.067621 ± 0.003144 |
| AR-PPO | 0.000255 | 0.000600 ± 0.000050 | 0.001851 | 0.003431 ± 0.000291 | 0.023470 | 0.038625 ± 0.003173 |

## Interpretation

The passive match means the fixed disturbance amplitude and basic DDM scale are probably not globally wrong. The larger mismatch appears mainly after controller feedback is introduced, so the likely issue is closed-loop distribution shift: the controller drives the DDM into action/state/history combinations that are not represented accurately enough by the training data.

For future controller design, the DDM simulation is useful for relative debugging and fast stress tests, but the current result suggests hardware validation is still necessary before trusting controller rankings. In particular, AR-PPO looks too good in DDM simulation, while MPC looks too bad.

## Files

- `data/simulation_vs_hardware.csv`: full numeric comparison.
- `figures/shaker_simu_vs_hardware_rms.pdf`: RMS bar comparison.
- `figures/shaker_simu_time_response.pdf`: simulated time response comparison.
