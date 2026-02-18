"""
Restaurant autoscaling environment (Gym-style without any gym dependency).

We simulate a booking backend that must autoscale server capacity to handle
time-varying traffic while controlling:
- CPU utilization
- queue pressure (backlog)
- latency (proxy for user experience)
- infra cost

The RL agent observes a DISCRETE state (bucketized) and chooses an action:
0 = scale down, 1 = stay, 2 = scale up

This file is intentionally self-contained and beginner-friendly, but built
to be modular enough for research demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import EnvConfig


Action = int
State = Tuple[int, int, int, int]  # (traffic_bucket, cpu_bucket, queue_bucket, server_index)


def _bucketize(value: float, edges: Tuple[int, ...]) -> int:
    """
    Convert a scalar into a discrete bucket index based on monotonic edges.

    We treat edges as inclusive lower bounds:
    - bucket 0 covers [edges[0], edges[1])
    - bucket 1 covers [edges[1], edges[2])
    - ...
    - last bucket covers [edges[-1], +inf) (but we often add a sentinel edge)
    """
    # np.searchsorted returns the insertion index to keep edges sorted.
    # With side="right": values equal to an edge go to the bucket AFTER it.
    # We want inclusive lower bounds, so we use "right" and subtract 1.
    idx = int(np.searchsorted(np.asarray(edges), value, side="right") - 1)
    return max(0, idx)


@dataclass
class FestivalSpike:
    start_step: int
    end_step: int
    multiplier: float

    def active(self, t: int) -> bool:
        return self.start_step <= t < self.end_step


class RestaurantEnv:
    """
    A minimal step-based simulation.

    reset() -> initial discrete state
    step(action) -> next_state, reward, done, info

    "info" includes the raw continuous signals needed for logging and dashboards.
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        # Derived: minutes per step. We simulate 1 day per episode.
        self.minutes_per_step = self.cfg.day_minutes / float(self.cfg.episode_length_steps)

        # Simulation state variables (mutable per episode)
        self.t: int = 0
        self.day_of_week: int = 0  # 0=Mon ... 6=Sun
        self.festival: Optional[FestivalSpike] = None

        self.servers: int = self.cfg.start_servers
        self.queue: int = 0
        self.last_cpu: float = 0.0
        self.last_latency_ms: float = self.cfg.base_latency_ms

        self._last_traffic_rpm: float = 0.0

    # -----------------------
    # Public API
    # -----------------------
    def reset(self, seed: Optional[int] = None) -> State:
        """
        Reset the environment to the start of a new simulated day (episode).

        If `seed` is provided, the environment RNG is re-seeded. This is useful
        for fair RL-vs-baseline comparisons on identical traffic realizations.
        """
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = 0
        self.day_of_week = int(self.rng.integers(0, 7))
        self.festival = self._maybe_create_festival_spike()

        self.servers = int(np.clip(self.cfg.start_servers, self.cfg.min_servers, self.cfg.max_servers))
        self.queue = 0

        # Initialize with a first observation so logs can start at t=0.
        traffic = self._sample_traffic_rpm(self.t)
        cpu, latency_ms, *_ = self._simulate_system(traffic_rpm=traffic, servers=self.servers, queue=self.queue)

        self._last_traffic_rpm = traffic
        self.last_cpu = cpu
        self.last_latency_ms = latency_ms

        return self._get_discrete_state(traffic_rpm=traffic, cpu=cpu, queue=self.queue, servers=self.servers)

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0 (down), 1 (stay), or 2 (up).")

        # Apply scaling action.
        if action == 0:
            self.servers = max(self.cfg.min_servers, self.servers - 1)
        elif action == 2:
            self.servers = min(self.cfg.max_servers, self.servers + 1)

        # Sample traffic for this step and advance system dynamics by one step.
        traffic = self._sample_traffic_rpm(self.t)
        cpu, latency_ms, processed, successes, cost, sla_violation, queue_overflow, next_queue = self._simulate_system(
            traffic_rpm=traffic,
            servers=self.servers,
            queue=self.queue,
        )

        booking_success_rate = 1.0 if processed <= 0 else float(successes / processed)

        reward = self._compute_reward(
            cpu=cpu,
            queue=next_queue,
            cost=cost,
            sla_violation=sla_violation,
            queue_overflow=queue_overflow,
        )

        # Update environment state.
        self.queue = next_queue
        self.last_cpu = cpu
        self.last_latency_ms = latency_ms
        self._last_traffic_rpm = traffic

        # Advance time.
        self.t += 1
        done = self.t >= self.cfg.episode_length_steps

        next_state = self._get_discrete_state(traffic_rpm=traffic, cpu=cpu, queue=self.queue, servers=self.servers)

        info = {
            "t": self.t,
            "day_of_week": self.day_of_week,
            "is_weekend": self.day_of_week in (5, 6),
            "festival_active": bool(self.festival and self.festival.active(self.t - 1)),
            "traffic_rpm": float(traffic),
            "cpu_utilization": float(cpu),
            "queue_size": int(self.queue),
            "latency_ms": float(latency_ms),
            "servers": int(self.servers),
            "processed_requests": int(processed),
            "successful_bookings": int(successes),
            "booking_success_rate": float(booking_success_rate),
            "infra_cost": float(cost),
            "sla_violation": bool(sla_violation),
            "queue_overflow": bool(queue_overflow),
        }
        return next_state, float(reward), bool(done), info

    # -----------------------
    # Core simulation pieces
    # -----------------------
    def _maybe_create_festival_spike(self) -> Optional[FestivalSpike]:
        if float(self.rng.random()) >= self.cfg.festival_spike_probability_per_episode:
            return None

        start = int(self.rng.integers(0, max(1, self.cfg.episode_length_steps - 1)))
        remaining = int(max(0, self.cfg.episode_length_steps - start))
        # If we're too close to the end of the episode, either clamp duration
        # or skip the festival spike entirely. This avoids low>=high errors.
        if remaining <= 1:
            return None

        dur_low = int(min(self.cfg.festival_duration_min_steps, remaining))
        dur_high = int(min(self.cfg.festival_duration_max_steps, remaining))
        if dur_low > dur_high:
            return None
        duration = int(self.rng.integers(dur_low, dur_high + 1))
        mult = float(self.rng.uniform(self.cfg.festival_multiplier_min, self.cfg.festival_multiplier_max))
        return FestivalSpike(start_step=start, end_step=min(self.cfg.episode_length_steps, start + duration), multiplier=mult)

    def _time_of_day_hour(self, step: int) -> float:
        minute_of_day = step * self.minutes_per_step
        return float(minute_of_day / 60.0)

    def _base_traffic_rate(self, hour: float) -> float:
        """
        Coarse daily pattern:
        - morning: low -> moderate
        - lunch: spike
        - afternoon: moderate
        - dinner: strong spike
        - night: low
        """
        # Piecewise baseline (requests per minute).
        if 0 <= hour < 6:
            base = self.cfg.base_night_rpm
        elif 6 <= hour < 11:
            # ramp up through morning
            frac = (hour - 6) / 5.0
            base = self.cfg.base_low_rpm + frac * (self.cfg.base_morning_rpm - self.cfg.base_low_rpm)
        elif 11 <= hour < 15:
            base = self.cfg.base_afternoon_rpm
        elif 15 <= hour < 18:
            base = self.cfg.base_afternoon_rpm * 0.9
        elif 18 <= hour < 22:
            base = self.cfg.base_afternoon_rpm
        else:
            base = self.cfg.base_night_rpm

        # Add smooth lunch/dinner spikes (Gaussian bumps).
        lunch = self.cfg.lunch_peak_rpm * np.exp(-0.5 * ((hour - 12.6) / 1.0) ** 2)
        dinner = self.cfg.dinner_peak_rpm * np.exp(-0.5 * ((hour - 19.4) / 1.3) ** 2)
        return float(base + lunch + dinner)

    def _sample_traffic_rpm(self, step: int) -> float:
        """
        Sample booking request arrivals for this step, as requests-per-minute.

        We then treat each step as a chunk of time (minutes_per_step). To keep the
        interpretation consistent, we convert "rpm" into "requests this step" by
        scaling inside `_simulate_system`.
        """
        hour = self._time_of_day_hour(step)
        rate = self._base_traffic_rate(hour)

        # Weekend boost
        weekend_mult = self.cfg.weekend_multiplier if self.day_of_week in (5, 6) else self.cfg.weekday_multiplier
        rate *= weekend_mult

        # Festival spike (temporary multiplier)
        if self.festival and self.festival.active(step):
            rate *= self.festival.multiplier

        # Random day-to-day variation
        rate *= float(self.rng.normal(loc=1.0, scale=0.06))
        rate = max(0.0, rate)

        # Poisson arrivals per minute; we keep as rpm-like scalar.
        # For very high rates, Poisson is still fine for a demo.
        sampled = float(self.rng.poisson(lam=rate))
        return sampled

    def _simulate_system(
        self,
        traffic_rpm: float,
        servers: int,
        queue: int,
    ) -> Tuple[float, float, int, int, float, bool, bool, int]:
        """
        Given traffic and current servers/queue, simulate one step of:
        - processing capacity
        - CPU utilization (proxy)
        - latency (proxy)
        - booking success rate (via binomial successes)
        - cost and SLA violations
        """
        # Convert arrivals from per-minute to "requests in this step".
        arrivals = int(round(float(traffic_rpm) * self.minutes_per_step))

        capacity_per_step = float(self.cfg.requests_capacity_per_server_per_min) * self.minutes_per_step
        total_capacity = max(1.0, capacity_per_step * float(servers))

        demand = float(queue + arrivals)
        processed = int(min(demand, total_capacity))

        remaining = int(demand - processed)
        next_queue = int(min(self.cfg.max_queue, max(0, remaining)))
        queue_overflow = remaining > self.cfg.max_queue

        # CPU utilization is driven by "load / capacity", plus a bit of noise.
        raw_cpu = 100.0 * (processed / total_capacity)
        cpu = float(np.clip(raw_cpu + self.rng.normal(0.0, self.cfg.cpu_noise_std), 0.0, 100.0))

        # Latency increases with queue pressure per server and with very high CPU.
        latency_ms = float(
            self.cfg.base_latency_ms
            + self.cfg.latency_ms_per_queued_per_server * (next_queue / max(1, servers))
            + self.cfg.latency_ms_per_cpu_over_70 * max(0.0, cpu - 70.0)
        )

        # Booking success probability: a simple, interpretable model.
        # We assume:
        # - high latency harms conversions
        # - very high CPU/queue increases failures/timeouts
        # Kept intentionally bounded and stable.
        p = 0.985
        p -= 0.0007 * max(0.0, latency_ms - 250.0)
        p -= 0.0025 * max(0.0, cpu - 80.0)
        p -= 0.0009 * max(0.0, next_queue - 120.0)
        success_prob = float(np.clip(p, 0.05, 0.99))

        successes = int(self.rng.binomial(n=max(0, processed), p=success_prob))

        cost = float(self.cfg.infra_cost_per_server_per_min * servers * self.minutes_per_step)

        sla_violation = bool(
            (cpu > 90.0)
            or (next_queue > self.cfg.queue_sla_threshold)
            or (latency_ms > self.cfg.latency_sla_ms)
        )

        return cpu, latency_ms, processed, successes, cost, sla_violation, bool(queue_overflow), next_queue

    def _compute_reward(self, cpu: float, queue: int, cost: float, sla_violation: bool, queue_overflow: bool) -> float:
        """
        Reward shaping (business + system objective):
        - Prefer CPU between 40% and 70% (efficient but not overloaded)
        - Prefer low queue (fast user experience)
        - Penalize cost (too many servers)
        - Heavy penalties for SLA violations / overflow
        """
        reward = 0.0

        if 40.0 <= cpu <= 70.0:
            reward += self.cfg.reward_cpu_in_band
        else:
            # Smooth penalty as you move away from the target band.
            # This avoids extremely spiky rewards that make tabular learning unstable.
            dist = min(abs(cpu - 55.0), 55.0) / 55.0  # 0..1
            reward -= 0.6 * dist

        # Queue shaping: reward low queue, penalize large queues.
        if queue <= 20:
            reward += self.cfg.reward_low_queue

        reward -= self.cfg.penalty_queue_weight * (queue / float(self.cfg.max_queue))
        reward -= self.cfg.penalty_cost_weight * cost

        if sla_violation:
            reward -= self.cfg.penalty_sla_violation
        if queue_overflow:
            reward -= self.cfg.penalty_queue_overflow

        return float(reward)

    # -----------------------
    # State representation
    # -----------------------
    def _get_discrete_state(self, traffic_rpm: float, cpu: float, queue: int, servers: int) -> State:
        traffic_b = _bucketize(traffic_rpm, self.cfg.traffic_bucket_edges)
        cpu_b = _bucketize(cpu, self.cfg.cpu_bucket_edges)
        queue_b = _bucketize(queue, self.cfg.queue_bucket_edges)

        # Server index is 0-based so it fits naturally into a Q-table tensor.
        server_idx = int(np.clip(servers, self.cfg.min_servers, self.cfg.max_servers) - self.cfg.min_servers)
        return int(traffic_b), int(cpu_b), int(queue_b), int(server_idx)

    # Convenience for dashboards/baselines.
    @property
    def num_traffic_buckets(self) -> int:
        return max(1, len(self.cfg.traffic_bucket_edges))

    @property
    def num_cpu_buckets(self) -> int:
        return max(1, len(self.cfg.cpu_bucket_edges))

    @property
    def num_queue_buckets(self) -> int:
        return max(1, len(self.cfg.queue_bucket_edges))

    @property
    def num_server_buckets(self) -> int:
        return int(self.cfg.max_servers - self.cfg.min_servers + 1)

    @property
    def num_actions(self) -> int:
        return 3

