# Control Learning Compare Results

## Data Analyzed

The current result files are:

```text
data/final_summary.csv
data/average_return_comparison.csv
data/training_returns_non_residual.csv
data/training_returns_fixed_residual.csv
data/training_returns_adaptive_residual.csv
data/fixed_eval_disturbances.csv
```

The comparison uses:

```text
seed = 666
training episodes = 300
evaluation disturbances = 10
control steps per episode = 200
DDM checkpoint = ../ddm_ablation/models/selected_model.pt
```

The selected DDM checkpoint is the `M4`, `N_h = 7` model from `ddm_ablation`.

## Main Finding

All PPO controllers substantially outperform Passive and tuned ADRC on the DDM surrogate. The best final controller is `adaptive_residual` (`AR-PPO`):

```text
AR-PPO final average return = -0.10860
NR-PPO final average return = -0.11499
FR-PPO final average return = -0.11682
ADRC final average return   = -0.46958
Passive final average return = -1.44325
```

Because return is negative cost, values closer to zero are better. Relative to ADRC, final AR-PPO reduces the return magnitude by:

```text
76.9%
```

Relative to Passive, final AR-PPO reduces the return magnitude by:

```text
92.5%
```

## Final Summary

| Method | Label | Best episode | Best average return | Final average return | Final return std |
|---|---|---:|---:|---:|---:|
| passive | Passive | 0 | -1.44325 | -1.44325 | 1.07076 |
| adrc | ADRC | 0 | -0.46958 | -0.46958 | 0.19117 |
| non_residual | NR-PPO | 280 | -0.10898 | -0.11499 | 0.04793 |
| fixed_residual | FR-PPO | 235 | -0.11614 | -0.11682 | 0.05243 |
| adaptive_residual | AR-PPO | 295 | -0.10783 | -0.10860 | 0.04984 |

The ranking by final return is:

```text
AR-PPO > NR-PPO > FR-PPO > ADRC > Passive
```

The three PPO methods are close by the end of training, but AR-PPO has both the best final return and the best observed return.

## Improvement Over Baselines

Final average return magnitude reduction:

| Method | vs Passive | vs ADRC |
|---|---:|---:|
| NR-PPO | 92.0% | 75.5% |
| FR-PPO | 91.9% | 75.1% |
| AR-PPO | 92.5% | 76.9% |

Interpretation: the learned policies are not merely small refinements of ADRC on this DDM surrogate. They discover substantially lower-cost actions under the fixed disturbance suite.

## Learning Behavior

The initial episode-0 returns show the policy initialization:

- NR-PPO starts like Passive because the zero-initialized actor outputs approximately zero force.
- FR-PPO and AR-PPO start like ADRC because residual policies are initialized around the ADRC base controller.

Representative evaluation returns:

| Episode | NR-PPO | FR-PPO | AR-PPO |
|---:|---:|---:|---:|
| 0 | -1.44325 | -0.46958 | -0.46958 |
| 5 | -0.14922 | -0.80969 | -0.37608 |
| 50 | -0.17148 | -0.42915 | -0.25447 |
| 100 | -0.31574 | -0.21027 | -0.15562 |
| 150 | -0.17634 | -0.19884 | -0.12275 |
| 200 | -0.12773 | -0.13442 | -0.11580 |
| 250 | -0.11297 | -0.12724 | -0.11105 |
| 300 | -0.11499 | -0.11682 | -0.10860 |

NR-PPO learns quickly, but its curve has large fluctuations. FR-PPO initially performs poorly after early updates because it has immediate access to the full residual range around ADRC. AR-PPO is more conservative early on, then improves steadily as its residual range expands.

## Adaptive Residual Range

AR-PPO starts with:

```text
residual_u_max = 0.8 N
```

During training it expands through:

```text
0.8 -> 1.08 -> 1.458 -> 1.9683 -> 2.0 N
```

By episode `75`, AR-PPO has reached the full `2.0 N` residual range. Its final actor standard deviation is lower than the other policies:

```text
NR-PPO final actor std = 0.1723
FR-PPO final actor std = 0.1808
AR-PPO final actor std = 0.0753
```

This supports the intended behavior of AR-PPO: start with a restricted residual range to avoid destabilizing the ADRC base controller, then expand once the evaluated policy stabilizes.

## Paper-Ready Interpretation

A concise interpretation suitable for the paper is:

```text
On the selected DDM surrogate, PPO control learning substantially improves the comfort-oriented return relative to both Passive and tuned ADRC baselines. The adaptive residual policy achieved the best final average return, -0.1086, corresponding to a 76.9% reduction in return magnitude relative to ADRC. The non-residual policy learned quickly but showed larger performance fluctuations, while the fixed residual policy initially degraded the ADRC baseline before recovering. The adaptive residual range therefore provides a useful curriculum: it preserves the stabilizing ADRC prior early in training and gradually exposes the policy to the full actuator authority.
```

Recommended claims:

- Strong claim: all PPO variants outperform Passive and ADRC on the learned DDM surrogate.
- Strong claim: AR-PPO gives the best final and best observed average return in this run.
- Supported claim: adaptive residual scaling improves training behavior compared with fixed full-range residual learning.
- Nuanced claim: NR-PPO is competitive in final return, so the benefit of residual learning is mainly stability and curriculum structure rather than a huge final-return gap.

Avoid claiming:

- that this result alone proves hardware superiority. It is a DDM-surrogate training comparison.
- that AR-PPO is statistically superior across random seeds. The current generated result is one seed, `666`.
- that the learned policy is robust to actuator-side faults. That question is tested separately in `actuator_disturbance_sweep`.

## Recommended Figure

Use:

```text
figures/average_return_comparison.pdf
```

This figure is the most direct visual evidence:

- horizontal lines show Passive and ADRC baselines;
- training curves show NR-PPO, FR-PPO, and AR-PPO;
- higher return means lower predicted control cost.

For the paper, pair this figure with the DDM ablation result to make clear that the learning comparison is performed on the selected hardware-trained DDM surrogate.

## Limitations

The result should be framed carefully:

- It is based on a learned DDM surrogate, not direct hardware training.
- It uses one random seed for PPO training.
- The fixed evaluation suite has `10` disturbance cases.
- The reward is a model-based comfort proxy, so hardware validation is still needed.
- The disturbance suite contains smooth shaker excitations, not actuator-side fault pulses.

These limitations are fine as long as this experiment is presented as the controller-learning stage. The hardware robustness stage is covered by the later actuator disturbance experiments.
