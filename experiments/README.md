# comfort_seat_final

Clean experiment workspace for the comfort-seat hardware and simulation studies.

## Structure

- `utils/`: shared runtime code.
  - `Hardware.py`: copied from `comfort_seat_4/Hardware.py` with the original hardware interface preserved.
  - `filters.py`: causal 8 Hz first-order low-pass filtering and online kinematics reconstruction.
  - `signals.py`: deterministic stratified excitation-plan and signal generation utilities.
  - `hardware_io.py`: quiet hardware wrappers, settling checks, and force-to-voltage conversions.
- `mpc_degradation/`: simulation experiment corresponding to the old `system_contribution`.
- `hardware_stratified_protocol/`: hardware data collection using the B1/B2/B3 stratified protocol.
- `ddm_ablation/`: data-driven dynamics model ablation and history-window sensitivity using collected CSV trajectories.
- `tune_parameter/`: ideal-system parameter tuning for baseline controllers under a fixed reusable shaker disturbance.

Each experiment directory contains:

- `design.md`: experimental design notes.
- `main.py`: data generation or collection.
- `plot.py`: standalone PDF plotting script.
- `data/`: generated CSV data files.
- `figures/`: generated PDF figures.

All figures are written as PDF files.
