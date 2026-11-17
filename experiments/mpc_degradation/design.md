# MPC Degradation Experiment

## Purpose

This experiment reorganizes the old `comfort_seat_4/system_contribution` study as `mpc_degradation`.
It compares the nominal MPC response under four degradation sources:

- Variant I: nonlinear springs.
- Variant II: one-sample command delay.
- Variant III: first-order voice-coil motor dynamics.
- Variant IV: nonlinear springs, command delay, and motor dynamics together.

The ideal zero-response baseline is included for reference.

## Filtering

The plotted `x2`, `v2`, and `a2` channels use the same online measurement-processing rule as the hardware experiments:

1. Measured displacement is processed by a causal first-order low-pass filter with an 8 Hz cutoff.
2. Velocity is obtained by finite differencing the filtered displacement with the timestamped sample interval, then filtering by the same causal 8 Hz first-order low-pass filter.
3. Acceleration is reconstructed by differentiating the filtered velocity estimate and filtering the result with the same causal 8 Hz first-order low-pass filter.

## Files

- `main.py`: runs the simulation and saves `data/mpc_degradation.csv`.
- `plot.py`: standalone plotting script that writes `figures/mpc_degradation_x2_v2_a2.pdf`.
- `data/`: simulation data archive.
- `figures/`: generated PDF figure.
