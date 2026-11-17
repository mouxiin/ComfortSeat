# Actuator Disturbance Smoke Test

This directory is a small hardware test harness for the actuator-side input disturbance problem. By default it now runs the same force-duration grid as `actuator_disturbance_sweep`:

```text
F_ud = {-1 N, -1.5 N, -2 N}
start time = 2.0 s
duration = {10, 20, 30} control periods
```

The goal is to smoke-test the finalized DGS-AR-PPO method under the force-duration grid before running the heavier sweep. It reuses the core hardware rollout implementation from:

```text
../actuator_disturbance_sweep/support.py
```

## Compared Controllers

The default controller set is:

- Passive
- ADRC
- MPC
- DeePC
- AR-PPO
- DGS-AR-PPO

DGS-AR-PPO uses the same DDM residual detector and the same `95%` one-step residual threshold as the sweep. Its fallback policy is MPC.

The current optimized DGS switching parameters are:

```text
detection threshold = 0.0223854566
recovery threshold = 0.5 * detection threshold = 0.0111927283
minimum fallback dwell = 12 samples
recovery consecutive samples = 4
post-return refractory = 8 samples
```

This deliberately separates detection from recovery. The gate remains sensitive to the start of an actuator-side event, but it avoids immediately switching back to AR-PPO when the residual briefly drops during the transient. This is especially important for the `-2 N`, 10-period pulse, where one or two unnecessary switches occupy a large fraction of the disturbance response.

## Commands

Collect one smoke pass over all 9 cases:

```bash
python main.py
```

Collect three repeats over all 9 cases:

```bash
python main.py --repeats 3
```

Run only the original single smoke case:

```bash
python main.py --forces -2 --durations 10
```

Run only the finalized DGS comparison:

```bash
python main.py --controllers ar_ppo mpc dgs_ar_ppo
```

Redo saved data:

```bash
python main.py --overwrite
```

Plot the collected data:

```bash
python plot.py
```

## Outputs

Data files:

- `data/test_plan.csv`
- `data/unknown_forces/{case}.csv`
- `data/rollouts/{case}/{controller}_repeat_{repeat}.csv`
- `data/repeat_metrics.csv`
- `data/aggregate_statistics.csv`

Figures:

- `figures/{case}_event_metric_comparison.pdf`
- `figures/{case}_time_response_repeat_01.pdf`
