"""
Tabular Q-learning agent for autoscaling.

We store a Q-table:
    Q[s, a] ≈ expected discounted return starting from state s and taking action a.

Update rule (classic Q-learning):

    Q(s, a) ← (1 - α) Q(s, a) + α [ r + γ max_a' Q(s', a') ]

Where:
- α (alpha) is the learning rate (how fast we overwrite old estimates)
- γ (gamma) is the discount factor (how much we value the future)
- r is the immediate reward
- s' is the next state after action a

Action selection uses epsilon-greedy exploration:
- with probability ε: pick a random action (explore)
- otherwise: pick argmax_a Q(s, a) (exploit)

This project intentionally avoids deep learning (DQN, etc.) to keep everything
fully interpretable and lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from config import AgentConfig


State = Tuple[int, int, int, int]


@dataclass
class QLearningAgent:
    traffic_buckets: int
    cpu_buckets: int
    queue_buckets: int
    server_buckets: int
    num_actions: int = 3
    cfg: AgentConfig = AgentConfig()

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)

        self.epsilon: float = float(self.cfg.epsilon_start)
        self.q_table = np.zeros(
            (self.traffic_buckets, self.cpu_buckets, self.queue_buckets, self.server_buckets, self.num_actions),
            dtype=np.float32,
        )

    # -----------------------
    # Policy
    # -----------------------
    def choose_action(self, state: State, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        greedy=True forces exploitation (useful for evaluation/dashboard decisions).
        """
        if (not greedy) and (float(self.rng.random()) < self.epsilon):
            return int(self.rng.integers(0, self.num_actions))

        q = self.q_table[state]
        # Random tie-break among max actions for stability (avoids bias).
        max_q = float(np.max(q))
        best = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best))

    # -----------------------
    # Learning
    # -----------------------
    def update(self, state: State, action: int, reward: float, next_state: State, done: bool) -> float:
        """
        Apply the Q-learning update and return the TD error (debug signal).
        """
        current = float(self.q_table[state + (action,)])
        next_max = 0.0 if done else float(np.max(self.q_table[next_state]))

        target = float(reward + self.cfg.gamma * next_max)
        td_error = target - current

        new_value = (1.0 - self.cfg.alpha) * current + self.cfg.alpha * target
        self.q_table[state + (action,)] = np.float32(new_value)
        return float(td_error)

    def decay_epsilon(self) -> float:
        self.epsilon = float(max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay))
        return self.epsilon

    # -----------------------
    # Persistence
    # -----------------------
    def save_model(self, path: str) -> None:
        """
        Save Q-table + epsilon to a compressed NPZ file.
        """
        np.savez_compressed(
            path,
            q_table=self.q_table,
            epsilon=np.asarray(self.epsilon, dtype=np.float32),
            meta=np.asarray(
                [
                    self.traffic_buckets,
                    self.cpu_buckets,
                    self.queue_buckets,
                    self.server_buckets,
                    self.num_actions,
                ],
                dtype=np.int32,
            ),
        )

    def load_model(self, path: str, strict_shape: bool = True) -> None:
        data = np.load(path, allow_pickle=False)
        q = data["q_table"]
        eps = float(data["epsilon"])

        if strict_shape and tuple(q.shape) != tuple(self.q_table.shape):
            raise ValueError(f"Q-table shape mismatch: file={q.shape} vs agent={self.q_table.shape}")

        self.q_table = q.astype(np.float32, copy=False)
        self.epsilon = float(eps)

    # -----------------------
    # Introspection helpers
    # -----------------------
    def q_values(self, state: State) -> np.ndarray:
        return self.q_table[state].copy()

    def best_action(self, state: State) -> int:
        return int(np.argmax(self.q_table[state]))

