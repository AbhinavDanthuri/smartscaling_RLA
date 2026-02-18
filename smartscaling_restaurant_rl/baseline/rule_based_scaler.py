"""
Rule-based autoscaling baseline.

Classic threshold policy:
- If CPU > 80% -> scale up
- If CPU < 40% -> scale down
- Else -> stay

This is intentionally simple and acts as a strong baseline for comparison charts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleBasedScaler:
    up_threshold: float = 80.0
    down_threshold: float = 40.0

    def decide(self, cpu_utilization: float) -> int:
        """
        Return action: 0=down, 1=stay, 2=up
        """
        if cpu_utilization > self.up_threshold:
            return 2
        if cpu_utilization < self.down_threshold:
            return 0
        return 1

