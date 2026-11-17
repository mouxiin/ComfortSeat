# Active vibration control experiments

Experimental code, recorded data and trained models for learning aggregate
system dynamics and supervisory vibration control.

## Experiments

All experiments are under `experiments/`, with their original relative paths
preserved. Each module includes its code, saved `data/`, and experimental notes.
Trained checkpoints are retained in `models/` where required.

| Module | Purpose |
|---|---|
| `hardware_stratified_protocol` | Identification data collection |
| `ddm_ablation` | History-length study and dynamics-model ablation |
| `tune_parameter` | Baseline controller tuning |
| `control_learning_compare` | Residual policy training and comparison |
| `shaker_disturbance` | Fixed-excitation hardware tests |
| `shaker_disturbance_sweep` | Frequency/amplitude sweep tests |
| `actuator_disturbance_smoke_test` | Preliminary switching-controller comparison |
| `actuator_disturbance_sweep` | Repeated pulse-disturbance tests |
| `mpc_degradation`, `shaker_simu` | Supporting simulations |
| `utils` | Shared signal processing, hardware interfaces and firmware |

## Setup

Use Python 3.10+ in a virtual environment. Install computational dependencies:

```bash
python -m pip install -r requirements.txt
```

For hardware experiments, install `requirements-hardware.txt` and the NI-DAQmx
system drivers. Configure channels and serial ports in
`experiments/utils/Hardware.py` for the laboratory rig.

Run each experiment's `main.py` for training, simulation or hardware collection;
consult its `design.md` and command-line options first. Experiment-specific
`plot.py` scripts process the saved data. Runs write to their module directories;
use a separate working copy to preserve the supplied records.

The archive contains 300 identification trajectories, 625 nominal sweep runs,
270 pulse-test runs, 72 preliminary runs and 26 model checkpoints.
Historical absolute paths in CSV metadata are repository-relative; numerical
measurements are unchanged. Legacy DDM columns named `test` denote the manuscript's
model-selection subset. In supervisor logs, residuals and switching flags are
backfilled into the preceding row; their observation time is the following row.
# ComfortSeat
# ComfortSeat
# ComfortSeat
# ComfortSeat
