# SmartScaling Restaurant RL

Production-style simulation of an autoscaling backend where a **tabular Q-learning agent** learns when to scale server capacity up/down based on traffic, CPU pressure, queue buildup, and infrastructure cost.

This project is a **simulation for ML systems research/demo** (not a real cloud deployment).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How the RL System Works](#how-the-rl-system-works)
- [Training Pipeline](#training-pipeline)
- [Dashboard Features](#dashboard-features)
- [Quick Start](#quick-start)
- [Artifacts and Outputs](#artifacts-and-outputs)
- [Deployment Notes](#deployment-notes)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Project Overview

The system simulates a time-varying service workload (morning/lunch/dinner/night, weekend boost, random spikes).  
At each step, an RL policy decides one of three actions:

- `0` -> Scale Down
- `1` -> Stay
- `2` -> Scale Up

The objective is to balance:

- Service reliability (SLA)
- Queue/latency control
- Resource efficiency and cost

A **rule-based baseline** is also evaluated for comparison.

---

## Tech Stack

- **Language**: Python 3.10+
- **Data/Math**: NumPy, Pandas
- **RL**: Tabular Q-learning (no deep learning)
- **Visualization**: Streamlit + Plotly
- **Storage**: CSV + NPZ

---

## Architecture

High-level flow:

1. `RestaurantEnv` simulates workload and system behavior.
2. `QLearningAgent` selects autoscaling actions and updates Q-values.
3. `MetricsLogger` stores per-step telemetry.
4. Training exports:
   - RL metrics
   - Baseline metrics
   - Q-table checkpoint
5. Streamlit dashboard loads artifacts and renders:
   - KPIs
   - Time-series telemetry
   - Q-table heatmap
   - RL vs baseline comparison

---

## Project Structure

```text
.
├─ smartscaling_restaurant_rl/
│  ├─ env/
│  │  └─ restaurant_env.py
│  ├─ agents/
│  │  └─ q_learning_agent.py
│  ├─ training/
│  │  └─ train_rl.py
│  ├─ baseline/
│  │  └─ rule_based_scaler.py
│  ├─ utils/
│  │  ├─ metrics_logger.py
│  │  └─ decision_explainer.py
│  ├─ dashboard/
│  │  └─ streamlit_app.py
│  ├─ config.py
│  ├─ requirements.txt
│  └─ README.md
├─ training/
│  └─ train_rl.py              # wrapper entry point
├─ dashboard/
│  └─ streamlit_app.py         # wrapper entry point
├─ requirements.txt
└─ README.md
```

> Wrapper scripts at workspace root allow simple run commands from repo root.

---

## How the RL System Works

### 1) State Representation (discrete)

Continuous system signals are bucketized into:

- Traffic bucket
- CPU bucket
- Queue bucket
- Server-count bucket

State tuple:

`(traffic_bucket, cpu_bucket, queue_bucket, server_index)`

### 2) Action Space

- `0`: scale down one server (bounded by min)
- `1`: keep server count unchanged
- `2`: scale up one server (bounded by max)

### 3) Reward Design

Reward combines:

- Positive reward for healthy CPU utilization band (around 40–70%)
- Positive reward for low queue
- Penalty for infra cost
- Heavy penalty for SLA violations / queue overflow

### 4) Q-learning Update

\[
Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a')\right]
\]

### 5) Exploration

- Epsilon-greedy policy
- Epsilon decays over episodes
- Minimum epsilon floor retained

---

## Training Pipeline

Training script:

- Runs at least 500 episodes (configured in `TrainingConfig`)
- Logs per-step metrics:
  - traffic
  - CPU
  - queue
  - servers
  - reward
  - action
  - booking success rate
  - infra cost
  - SLA flags
- Saves:
  - RL metrics (`metrics.csv`, `metrics.npz`)
  - Baseline metrics (`metrics_baseline.csv`, `metrics_baseline.npz`)
  - Q-table model (`models/q_table.npz`)

Baseline policy:

- `CPU > 80` -> scale up
- `CPU < 40` -> scale down
- otherwise stay

---

## Dashboard Features

Main dashboard includes:

- Top KPI cards (live/system snapshots)
- Telemetry charts:
  - traffic over time
  - CPU over time (safe band highlighted)
  - servers over time
  - reward over time
- Action distribution chart
- Q-table heatmap slice
- Decision panel with human-readable explanation
- RL vs baseline policy comparison
- Numeric interpretation block (cost and SLA comparison)

---

## Quick Start

From repository root:

```bash
pip install -r requirements.txt
python training/train_rl.py
streamlit run dashboard/streamlit_app.py
```

---

## Artifacts and Outputs

Generated under `smartscaling_restaurant_rl/`:

- `metrics.csv`
- `metrics.npz`
- `metrics_baseline.csv`
- `metrics_baseline.npz`
- `models/q_table.npz`

---

## Deployment Notes

- The root `dashboard/streamlit_app.py` is a wrapper that forwards to the actual dashboard module.
- It supports both folder names:
  - `smartscaling_restaurant_rl`
  - `smartscaling_rla`
- This avoids import-path issues across local vs hosted environments.

---

## Troubleshooting

### Import error in deploy (`from utils...`)

Use the root wrapper entrypoint:

```bash
streamlit run dashboard/streamlit_app.py
```

and ensure repository contains one of:

- `smartscaling_restaurant_rl/dashboard/streamlit_app.py`
- `smartscaling_rla/dashboard/streamlit_app.py`

### Dashboard says metrics not found

Run training first:

```bash
python training/train_rl.py
```

### Push rejected (`non-fast-forward`)

```bash
git pull --rebase origin main
git push
```

---

## Future Improvements

- Better reward calibration and policy evaluation metrics
- Multi-objective tuning (cost vs SLA frontiers)
- More realistic scaling delay/cooldown behavior
- Experiment tracking and hyperparameter sweeps
- Optional API streaming backend for true live telemetry

