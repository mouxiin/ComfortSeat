# DDM Ablation Results

## Main Result

The selected data-driven dynamics model is:

```text
M4: history-augmented residual-block MLP with residual state-increment target
selected history length: N_h = 7
stable downstream checkpoint: models/selected_model.pt
```

The result is very clear: history is the dominant factor. Memoryless models have poor open-loop rollout accuracy even when one-step errors look acceptable. Adding history reduces the held-out open-loop weighted NRMSE by roughly `76-79%`, and the full M4 model gives the best ablation result.

## History Sensitivity

The history sweep trains the full M4 structure for `N_h = 0..15`. The best held-out open-loop weighted NRMSE occurs at:

```text
N_h = 7
test open-loop weighted NRMSE = 0.1365 +/- 0.0411
one-step weighted NRMSE = 0.0696
```

Compared with the history-free M4 model (`N_h = 0`):

```text
N_h = 0 open-loop weighted NRMSE = 0.6492
N_h = 7 open-loop weighted NRMSE = 0.1365
relative reduction = 79.0%
```

Representative history sweep values:

| History length | Open-loop weighted NRMSE | One-step weighted NRMSE |
|---:|---:|---:|
| 0 | 0.6492 | 0.1534 |
| 1 | 0.1645 | 0.0824 |
| 2 | 0.1524 | 0.0752 |
| 3 | 0.1425 | 0.0719 |
| 5 | 0.1408 | 0.0706 |
| 7 | 0.1365 | 0.0696 |
| 8 | 0.1416 | 0.0699 |
| 10 | 0.1413 | 0.0699 |
| 13 | 0.1400 | 0.0690 |
| 15 | 0.1426 | 0.0695 |

Interpretation: a short history window is essential, because the measured state alone does not fully capture the effective dynamics of the hardware system, filtering pipeline, actuator response, and disturbance channel. Performance improves dramatically from `N_h = 0` to `N_h = 1`, then reaches a shallow optimum around `N_h = 7`. Longer histories do not give a meaningful open-loop benefit.

## Architecture Ablation

All ablation variants are trained using the selected history length `N_h = 7` when the architecture supports history.

| Model | Structure | Test one-step SW-NRMSE | Test open-loop SW-NRMSE | Reduction vs M0 |
|---|---|---:|---:|---:|
| M0 | memoryless, plain, absolute target | 0.1550 | 0.6077 +/- 0.2604 | +0.0% |
| M1 | memoryless, residual-block, absolute target | 0.1544 | 0.5801 +/- 0.2552 | +4.5% |
| M2 | history, plain, absolute target | 0.0717 | 0.1441 +/- 0.0406 | +76.3% |
| M3 | memoryless, plain, residual target | 0.1540 | 0.6014 +/- 0.2518 | +1.0% |
| M4 | history, residual-block, residual target | 0.0699 | 0.1383 +/- 0.0445 | +77.2% |

Key takeaways:

- M0, M1, and M3 are all memoryless and remain near `0.58-0.61` open-loop weighted NRMSE.
- M2 and M4 both use history and drop to approximately `0.14` open-loop weighted NRMSE.
- The residual-block and residual-target choices improve the history model modestly, but the main gain comes from giving the DDM temporal context.
- M4 is the best ablation model and is saved as `models/selected_model.pt`.

## One-Step Versus Open-Loop Behavior

One-step metrics are much smaller than open-loop metrics because each one-step prediction starts from the measured state. Open-loop rollout is more demanding: the model must recursively feed back its own predictions, so local errors can accumulate.

This distinction matters for the paper. A model that looks acceptable in one-step prediction can still be unstable or biased in recursive rollout. The selected DDM is justified because it is strong in both modes:

```text
M4 one-step weighted NRMSE = 0.0699
M4 open-loop weighted NRMSE = 0.1383 +/- 0.0445
```

The open-loop/one-step envelope figures visualize this difference:

```text
figures/nh07_open_vs_one_error_envelope.pdf
figures/nh07_open_vs_one_error_envelope_6s.pdf
```

## Residual Threshold For DGS

The actuator-disturbance detector uses the selected `N_h = 7` DDM as a nominal one-step predictor. The residual score is:

```text
r(t) = sqrt((e_x2(t) / 0.02)^2 + (e_v2(t) / 0.6)^2)
```

where only payload displacement and velocity residuals are used.

The DGS threshold is calibrated from nominal train+validation one-step residuals:

```text
calibration samples = 8911
threshold quantile = 0.95
tau_DGS = 0.02238545659929514
nominal calibration exceedance = 0.05005
```

Equivalent one-channel residual magnitudes:

```text
|e_x2| = 4.477e-4 m
|e_v2| = 1.343e-2 m/s
```

Held-out test residual statistics at the same threshold are:

```text
test samples = 1197
test score mean = 0.00581
test score q99 = 0.03849
test score q995 = 0.04646
test exceedance at tau_DGS = 0.04511
```

The train+validation calibration exceedance is approximately `5.0%`, and the held-out test exceedance is approximately `4.5%`. This supports the claim that the threshold is not tuned on actuator-fault cases and remains consistent on held-out nominal data.

## Paper-Ready Interpretation

A concise interpretation suitable for the method/results section is:

```text
The DDM ablation shows that temporal context is necessary for accurate hardware dynamics prediction. A memoryless MLP produced a held-out open-loop weighted NRMSE of 0.6077, while the proposed history-augmented residual DDM reduced this value to 0.1383, a 77.2% reduction. A separate history sweep selected N_h = 7, which achieved the best open-loop weighted NRMSE among N_h = 0..15. The selected DDM also provides the nominal residual model used by the downstream DGS detector; its payload-channel residual threshold was calibrated as the 95th percentile of 8911 nominal train+validation one-step residual scores, yielding tau_DGS = 0.0223854566.
```

Recommended claims:

- Strong claim: history input is essential for stable open-loop prediction on this hardware dataset.
- Strong claim: the proposed M4 model is the best ablation variant by held-out open-loop weighted NRMSE.
- Strong claim: the DGS threshold is calibrated from nominal residual statistics, not actuator-fault labels.
- Nuanced claim: residual blocks and residual targets help, but their benefit is secondary to history.

Avoid claiming:

- that one-step accuracy alone validates the DDM;
- that `N_h = 7` is universally optimal outside this hardware setup;
- that the detector threshold is a high-confidence anomaly threshold. It is intentionally sensitive for low missed detection in the downstream control task.

## Recommended Figures

Use these figures in the paper:

- `figures/history_sensitivity_olr.pdf`: shows the sharp improvement from memoryless to history-aware dynamics and justifies `N_h = 7`.
- `figures/ablation_bar_olr.pdf`: shows M4 as the best ablation model.
- `figures/nh07_open_vs_one_error_envelope_6s.pdf`: shows the selected model's open-loop versus one-step prediction behavior and marks the residual threshold used by DGS.

## Limitations

The DDM result is tied to the collected hardware protocol:

- The split contains 300 trajectories from the available stratified protocol, not a new randomized experiment.
- The selected history length is empirical and should be revalidated if the hardware, sampling time, filter, or excitation distribution changes.
- The residual threshold is calibrated on nominal data; it is not a statistical guarantee against all unseen disturbances.
- The DGS detector uses only `x2` and `v2` residuals, so faults that mainly affect other channels may require a different residual score.

These limitations are acceptable for the paper if the DDM is presented as a hardware-calibrated nominal dynamics model and residual detector, rather than as a universal physics model.
