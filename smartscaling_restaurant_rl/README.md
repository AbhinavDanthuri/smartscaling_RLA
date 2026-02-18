# SmartScaling Restaurant RL

Production-style simulation of a restaurant booking backend where a **tabular Q-learning agent** autoscaling policy learns to scale servers based on traffic, CPU pressure, queue buildup, and cost.

This is **not** a real cloud deployment. It’s a business + ML **simulation** designed for research demos and visualization.

## Architecture

Project structure:

```
smartscaling_restaurant_rl/
  env/restaurant_env.py            # traffic + system dynamics + reward + discrete state
  agents/q_learning_agent.py       # tabular Q-learning implementation
  training/train_rl.py             # training + logging + baseline comparison
  baseline/rule_based_scaler.py    # CPU-threshold autoscaler baseline
  utils/metrics_logger.py          # centralized logger -> CSV + NPZ
  utils/decision_explainer.py      # state/action -> human explanation
  dashboard/streamlit_app.py       # dark SaaS analytics dashboard (Plotly + Streamlit)
  config.py
  requirements.txt
```

Data artifacts (generated after training):
- `metrics.csv` / `metrics.npz` (RL run)
- `metrics_baseline.csv` / `metrics_baseline.npz` (rule-based baseline)
- `models/q_table.npz` (saved Q-table)

## How RL Works (Tabular Q-learning)

We discretize the continuous system signals into buckets:
- traffic bucket
- CPU bucket
- queue bucket
- server count bucket

Actions:
- 0: scale down
- 1: stay
- 2: scale up

Q-learning update:

\[
Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a')\right]
\]

Exploration uses epsilon-greedy with decay over episodes.

## How to Run

### Install dependencies

From your workspace root (this folder):

```bash
pip install -r requirements.txt
```

### Train the RL agent (and baseline)

```bash
python training/train_rl.py
```

Outputs:
- `metrics.csv`, `metrics.npz`
- `metrics_baseline.csv`, `metrics_baseline.npz`
- `models/q_table.npz`

### Run the dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Future Improvements

- Add explicit “scale action cooldown” / scaling delay
- Add separate read/write DB load, cache hit-rate, and multi-tier cost model
- Add evaluation-only rollouts (greedy) separate from training exploration metrics
- Add automated hyperparameter sweeps and experiment tracking

