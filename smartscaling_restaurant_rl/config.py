"""
Global configuration for the SmartScaling Restaurant RL simulation.

The goal of this file is to centralize "knobs" so:
- the environment is easy to tune
- training and the dashboard stay consistent

Everything is intentionally simple (NumPy/Pandas only) to keep the project
beginner-friendly while still feeling production-like.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """
    Restaurant autoscaling simulation configuration.

    Time:
    - We simulate one "day" per episode. By default that is 240 steps.
      That means each step represents 1440/240 = 6 minutes.
    """

    # --- Episode timing ---
    episode_length_steps: int = 240
    day_minutes: int = 24 * 60

    # --- Servers ---
    min_servers: int = 1
    max_servers: int = 12
    start_servers: int = 3
    requests_capacity_per_server_per_min: float = 22.0  # per real minute equivalent

    # --- Queue and latency ---
    max_queue: int = 500
    queue_sla_threshold: int = 180  # "pressure" threshold
    base_latency_ms: float = 120.0
    latency_ms_per_queued_per_server: float = 2.4
    latency_ms_per_cpu_over_70: float = 8.0
    latency_sla_ms: float = 700.0

    # --- Cost model ---
    infra_cost_per_server_per_min: float = 0.08  # abstract dollars/min

    # --- Traffic generator (requests per minute) ---
    weekday_multiplier: float = 1.00
    weekend_multiplier: float = 1.20
    festival_spike_probability_per_episode: float = 0.28
    festival_multiplier_min: float = 1.4
    festival_multiplier_max: float = 2.2
    festival_duration_min_steps: int = 12
    festival_duration_max_steps: int = 40

    # Baseline traffic levels; spikes are added on top in the generator
    base_low_rpm: float = 18.0
    base_morning_rpm: float = 28.0
    base_afternoon_rpm: float = 40.0
    base_night_rpm: float = 14.0
    lunch_peak_rpm: float = 95.0
    dinner_peak_rpm: float = 150.0

    # --- Noise / realism ---
    cpu_noise_std: float = 3.0

    # --- Discretization buckets (inclusive lower bounds; last bucket is ">= last") ---
    traffic_bucket_edges: tuple[int, ...] = (0, 30, 70, 120, 180)
    cpu_bucket_edges: tuple[int, ...] = (0, 40, 60, 75, 90, 101)  # 101 so 100 falls inside
    queue_bucket_edges: tuple[int, ...] = (0, 15, 60, 140, 240, 501)  # 501 so max_queue falls inside

    # --- Reward shaping weights ---
    reward_cpu_in_band: float = 1.2
    reward_low_queue: float = 0.8
    penalty_cost_weight: float = 1.0
    penalty_queue_weight: float = 1.2
    penalty_sla_violation: float = 10.0
    penalty_queue_overflow: float = 18.0


@dataclass(frozen=True)
class AgentConfig:
    # Q-learning hyperparameters
    alpha: float = 0.18  # learning rate
    gamma: float = 0.95  # discount factor

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.992  # multiply each episode

    # Randomness
    seed: int = 7


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 500
    debug_print_every_steps: int = 1200  # across the whole run

    # Where to save outputs (relative paths from project root)
    model_path: str = "models/q_table.npz"
    metrics_path_csv: str = "metrics.csv"
    metrics_path_npz: str = "metrics.npz"
    baseline_metrics_path_csv: str = "metrics_baseline.csv"
    baseline_metrics_path_npz: str = "metrics_baseline.npz"
