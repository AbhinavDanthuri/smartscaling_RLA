"""
Training pipeline for SmartScaling Restaurant RL.

Run (from workspace root):
    python training/train_rl.py

What it does:
- trains a tabular Q-learning agent for >= 500 episodes
- logs per-step metrics to CSV + NPZ
- runs the rule-based baseline on the same traffic seeds for comparison
- saves the learned Q-table to models/q_table.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

# Make project root importable when running as: python training/train_rl.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.q_learning_agent import QLearningAgent  # noqa: E402
from baseline.rule_based_scaler import RuleBasedScaler  # noqa: E402
from config import AgentConfig, EnvConfig, TrainingConfig  # noqa: E402
from env.restaurant_env import RestaurantEnv  # noqa: E402
from utils.metrics_logger import MetricsLogger  # noqa: E402


def ensure_dirs() -> None:
    (ROOT / "models").mkdir(parents=True, exist_ok=True)


def run_episode(
    env: RestaurantEnv,
    policy_fn: Callable[[Tuple[int, int, int, int], Dict], int],
    logger: MetricsLogger,
    episode_idx: int,
    seed: int,
    training: bool,
    agent: QLearningAgent | None = None,
    debug_every_global_steps: int = 0,
    global_step_start: int = 0,
) -> int:
    """
    Run one full episode and log every step.
    Returns the number of steps executed (episode length).
    """
    state = env.reset(seed=seed)
    total_steps = 0

    # We log the initial "pre-step" observation as a row with action=None? For simplicity,
    # we log only after each action (more standard in RL logs).
    done = False
    global_step = global_step_start

    while not done:
        action = int(policy_fn(state, {"episode": episode_idx}))
        next_state, reward, done, info = env.step(action)

        if training and agent is not None:
            agent.update(state, action, reward, next_state, done)

        logger.log(
            run=logger.run_name,
            episode=int(episode_idx),
            step_in_episode=int(total_steps),
            global_step=int(global_step),
            epsilon=float(agent.epsilon) if agent is not None else None,
            state_traffic_b=int(state[0]),
            state_cpu_b=int(state[1]),
            state_queue_b=int(state[2]),
            state_server_idx=int(state[3]),
            action=int(action),
            reward=float(reward),
            traffic_rpm=float(info["traffic_rpm"]),
            cpu_utilization=float(info["cpu_utilization"]),
            queue_size=int(info["queue_size"]),
            servers=int(info["servers"]),
            latency_ms=float(info["latency_ms"]),
            booking_success_rate=float(info["booking_success_rate"]),
            infra_cost=float(info["infra_cost"]),
            sla_violation=bool(info["sla_violation"]),
            queue_overflow=bool(info["queue_overflow"]),
            day_of_week=int(info["day_of_week"]),
            is_weekend=bool(info["is_weekend"]),
            festival_active=bool(info["festival_active"]),
        )

        if debug_every_global_steps and (global_step % debug_every_global_steps == 0):
            print(
                f"[debug] ep={episode_idx:03d} step={total_steps:03d} "
                f"traffic={info['traffic_rpm']:.0f}/min cpu={info['cpu_utilization']:.1f}% "
                f"q={info['queue_size']} servers={info['servers']} "
                f"r={reward:+.3f} a={action} cost={info['infra_cost']:.3f} "
                f"sla={int(info['sla_violation'])} eps={(agent.epsilon if agent else 0):.3f}"
            )

        state = next_state
        total_steps += 1
        global_step += 1

    return total_steps


def main() -> None:
    ensure_dirs()

    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()
    train_cfg = TrainingConfig()

    env = RestaurantEnv(config=env_cfg, seed=agent_cfg.seed)

    agent = QLearningAgent(
        traffic_buckets=env.num_traffic_buckets,
        cpu_buckets=env.num_cpu_buckets,
        queue_buckets=env.num_queue_buckets,
        server_buckets=env.num_server_buckets,
        num_actions=env.num_actions,
        cfg=agent_cfg,
    )

    rl_logger = MetricsLogger(run_name="rl")
    baseline_logger = MetricsLogger(run_name="baseline")

    # -----------------------
    # Training (RL)
    # -----------------------
    print(f"Training Q-learning agent for {train_cfg.episodes} episodes...")

    global_step = 0
    for ep in range(train_cfg.episodes):
        # Make training reproducible across runs, and comparable to baseline:
        # each episode uses a deterministic seed offset.
        seed = int(agent_cfg.seed + 10_000 + ep)

        def rl_policy(s, _ctx):
            return agent.choose_action(s, greedy=False)

        steps = run_episode(
            env=env,
            policy_fn=rl_policy,
            logger=rl_logger,
            episode_idx=ep,
            seed=seed,
            training=True,
            agent=agent,
            debug_every_global_steps=train_cfg.debug_print_every_steps,
            global_step_start=global_step,
        )
        global_step += steps
        agent.decay_epsilon()

    # Save artifacts
    model_path = str(ROOT / train_cfg.model_path)
    agent.save_model(model_path)

    metrics_csv = str(ROOT / train_cfg.metrics_path_csv)
    metrics_npz = str(ROOT / train_cfg.metrics_path_npz)
    rl_logger.export(csv_path=metrics_csv, npz_path=metrics_npz)

    print(f"Saved RL metrics to: {metrics_csv}")
    print(f"Saved RL metrics to: {metrics_npz}")
    print(f"Saved Q-table model to: {model_path}")

    # -----------------------
    # Baseline evaluation on SAME traffic seeds
    # -----------------------
    print("Running rule-based baseline (same traffic seeds for fair comparison)...")
    scaler = RuleBasedScaler()

    global_step = 0
    for ep in range(train_cfg.episodes):
        seed = int(agent_cfg.seed + 10_000 + ep)

        def baseline_policy(_s, _ctx):
            # Baseline uses only CPU (as required).
            return scaler.decide(env.last_cpu)

        steps = run_episode(
            env=env,
            policy_fn=baseline_policy,
            logger=baseline_logger,
            episode_idx=ep,
            seed=seed,
            training=False,
            agent=None,
            debug_every_global_steps=train_cfg.debug_print_every_steps,
            global_step_start=global_step,
        )
        global_step += steps

    baseline_csv = str(ROOT / train_cfg.baseline_metrics_path_csv)
    baseline_npz = str(ROOT / train_cfg.baseline_metrics_path_npz)
    baseline_logger.export(csv_path=baseline_csv, npz_path=baseline_npz)

    print(f"Saved baseline metrics to: {baseline_csv}")
    print(f"Saved baseline metrics to: {baseline_npz}")
    print("Done.")


if __name__ == "__main__":
    main()

