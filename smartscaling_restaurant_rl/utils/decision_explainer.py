"""
Human-readable explanations for autoscaling decisions.

The dashboard uses this to show "why" the agent chose an action.
We intentionally keep it simple and interpretable, based on discrete buckets.
"""

from __future__ import annotations

from typing import Tuple


State = Tuple[int, int, int, int]  # (traffic_bucket, cpu_bucket, queue_bucket, server_index)


ACTION_LABELS = {
    0: "Scale Down",
    1: "Stay",
    2: "Scale Up",
}


def explain_decision(state: State, action: int) -> str:
    traffic_b, cpu_b, queue_b, server_idx = state

    traffic_desc = _describe_traffic(traffic_b)
    cpu_desc = _describe_cpu(cpu_b)
    queue_desc = _describe_queue(queue_b)
    action_desc = ACTION_LABELS.get(action, f"Unknown({action})")

    # Explanation templates
    if action == 2:
        reasons = []
        if traffic_b >= 3:
            reasons.append("incoming traffic is high")
        if cpu_b >= 4:
            reasons.append("CPU pressure is approaching overload")
        if queue_b >= 3:
            reasons.append("the booking queue is building up")
        if not reasons:
            reasons.append("demand signals suggest scaling headroom is needed")
        why = " and ".join(reasons)
        return (
            f"{traffic_desc}, {cpu_desc}, {queue_desc}. "
            f"{why.capitalize()}. {action_desc} to reduce latency and prevent booking failures."
        )

    if action == 0:
        reasons = []
        if traffic_b <= 1:
            reasons.append("traffic is low")
        if cpu_b <= 1:
            reasons.append("CPU utilization is underused")
        if queue_b <= 1:
            reasons.append("queue pressure is minimal")
        if not reasons:
            reasons.append("the system appears over-provisioned")
        why = " and ".join(reasons)
        return (
            f"{traffic_desc}, {cpu_desc}, {queue_desc}. "
            f"{why.capitalize()}. {action_desc} to lower infra cost while maintaining SLA."
        )

    # Stay
    return (
        f"{traffic_desc}, {cpu_desc}, {queue_desc}. "
        "Signals look stable and within the target efficiency band. "
        f"{action_desc} to avoid unnecessary scaling churn."
    )


def _describe_traffic(bucket: int) -> str:
    if bucket <= 1:
        return "Low traffic"
    if bucket == 2:
        return "Moderate traffic"
    if bucket == 3:
        return "High traffic"
    return "Very high traffic"


def _describe_cpu(bucket: int) -> str:
    if bucket <= 1:
        return "CPU is low"
    if bucket == 2:
        return "CPU is in a healthy range"
    if bucket == 3:
        return "CPU is elevated"
    return "CPU is very high"


def _describe_queue(bucket: int) -> str:
    if bucket <= 1:
        return "Queue is small"
    if bucket == 2:
        return "Queue is growing"
    if bucket == 3:
        return "Queue is large"
    return "Queue is critical"

