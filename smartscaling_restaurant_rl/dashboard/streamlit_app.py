"""
Streamlit SaaS-style dashboard for the SmartScaling Restaurant RL project.

Run (from workspace root):
    streamlit run dashboard/streamlit_app.py

The dashboard reads:
- metrics.csv (RL run)
- metrics_baseline.csv (baseline run)
- models/q_table.npz (Q-table for heatmaps)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make project root importable when running as: streamlit run dashboard/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.decision_explainer import ACTION_LABELS, explain_decision  # noqa: E402


def set_dark_saas_style() -> None:
    st.set_page_config(
        page_title="SmartScaling Restaurant RL",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    css = """
    <style>
      :root {
        --bg: #0b1220;
        --panel: #101a2d;
        --panel2: #0f1a33;
        --text: #e7eefc;
        --muted: #96a4c6;
        --border: rgba(255,255,255,0.08);
        --accent: #7c5cff;
        --good: #32d583;
        --warn: #fdb022;
        --bad: #f97066;
      }

      html, body, [class*="css"]  {
        background: var(--bg);
        color: var(--text);
      }

      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
      }

      .panel {
        background: linear-gradient(180deg, rgba(16,26,45,0.96) 0%, rgba(16,26,45,0.75) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px 16px;
      }

      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 12px;
      }
      .kpi-card {
        background: linear-gradient(180deg, rgba(16,26,45,1) 0%, rgba(16,26,45,0.7) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px;
      }
      .kpi-title { font-size: 0.80rem; color: var(--muted); margin: 0; }
      .kpi-value { font-size: 1.6rem; font-weight: 700; margin: 4px 0 0 0; }
      .kpi-sub { font-size: 0.78rem; color: var(--muted); margin: 6px 0 0 0; }

      .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(124,92,255,0.12);
        border: 1px solid rgba(124,92,255,0.35);
        color: var(--text);
        font-size: 0.78rem;
      }

      .muted { color: var(--muted); }
      .divider { height: 1px; background: var(--border); margin: 12px 0; }

      /* Sidebar tweaks */
      section[data-testid="stSidebar"] {
        background: rgba(16,26,45,0.85);
        border-right: 1px solid var(--border);
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_q_table(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["q_table"].astype(np.float32, copy=False)


def kpi_cards(latest: pd.Series) -> None:
    # "Live users" is not directly simulated; we provide a reasonable proxy.
    traffic_rpm = float(latest.get("traffic_rpm", 0.0))
    live_users = int(round(traffic_rpm * 3.2))

    bpm = float(latest.get("traffic_rpm", 0.0))
    servers = int(latest.get("servers", 0))
    queue = int(latest.get("queue_size", 0))

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <p class="kpi-title">Live Users (est.)</p>
            <p class="kpi-value">{live_users:,}</p>
            <p class="kpi-sub">Proxy from traffic</p>
          </div>
          <div class="kpi-card">
            <p class="kpi-title">Bookings / Minute</p>
            <p class="kpi-value">{bpm:,.0f}</p>
            <p class="kpi-sub">Requests per minute</p>
          </div>
          <div class="kpi-card">
            <p class="kpi-title">Active Servers</p>
            <p class="kpi-value">{servers}</p>
            <p class="kpi-sub">Autoscaled capacity</p>
          </div>
          <div class="kpi-card">
            <p class="kpi-title">Queue Size</p>
            <p class="kpi-value">{queue:,}</p>
            <p class="kpi-sub">Backlog pressure</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_traces(line=dict(color=color, width=2.6))
    fig.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=8, r=8, t=40, b=10),
        title_font=dict(size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def action_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["action"].value_counts().sort_index()
    labels = [ACTION_LABELS.get(int(a), str(a)) for a in counts.index]
    fig = px.pie(values=counts.values, names=labels, title="Action Distribution")
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=8, r=8, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14),
        showlegend=True,
    )
    return fig


def q_table_heatmap(q_table: np.ndarray, server_idx: int, action_idx: int) -> go.Figure:
    """
    Show a 2D slice of the Q-table:
    - X axis: traffic bucket
    - Y axis: queue bucket
    - value: mean Q across CPU buckets at fixed server bucket and fixed action
    """
    # q: [traffic, cpu, queue, server, action]
    slice_ = q_table[:, :, :, server_idx, action_idx].mean(axis=1)  # -> [traffic, queue]
    z = slice_.T  # queue x traffic for nicer orientation

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="Q"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=8, r=8, t=40, b=10),
        title=f"Q-table Heatmap (server_bucket={server_idx}, action={ACTION_LABELS.get(action_idx)})",
        title_font=dict(size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title="Traffic bucket", showgrid=False)
    fig.update_yaxes(title="Queue bucket", gridcolor="rgba(255,255,255,0.08)")
    return fig


def comparison_bars(rl: pd.DataFrame, base: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    rl_cost = float(rl["infra_cost"].sum()) if not rl.empty else 0.0
    base_cost = float(base["infra_cost"].sum()) if not base.empty else 0.0

    rl_sla = float(rl["sla_violation"].mean()) if ("sla_violation" in rl.columns and not rl.empty) else 0.0
    base_sla = float(base["sla_violation"].mean()) if ("sla_violation" in base.columns and not base.empty) else 0.0

    fig_cost = go.Figure(
        data=[
            go.Bar(name="RL", x=["Total Cost"], y=[rl_cost], marker_color="#7c5cff"),
            go.Bar(name="Baseline", x=["Total Cost"], y=[base_cost], marker_color="#8aa1ff"),
        ]
    )
    fig_cost.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=8, r=8, t=40, b=10),
        title="RL vs Rule-Based — Infra Cost",
        title_font=dict(size=14),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_cost.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    fig_sla = go.Figure(
        data=[
            go.Bar(name="RL", x=["SLA Violation Rate"], y=[rl_sla], marker_color="#32d583"),
            go.Bar(name="Baseline", x=["SLA Violation Rate"], y=[base_sla], marker_color="#fdb022"),
        ]
    )
    fig_sla.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=8, r=8, t=40, b=10),
        title="RL vs Rule-Based — SLA Violations",
        title_font=dict(size=14),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_sla.update_yaxes(tickformat=".0%", gridcolor="rgba(255,255,255,0.08)")

    return fig_cost, fig_sla


def main() -> None:
    set_dark_saas_style()

    # Sidebar
    st.sidebar.markdown("## SmartScaling Restaurant RL")
    st.sidebar.markdown('<span class="tag">Production-style simulation</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    rl_csv = ROOT / "metrics.csv"
    base_csv = ROOT / "metrics_baseline.csv"
    q_path = ROOT / "models" / "q_table.npz"

    st.sidebar.markdown("### Data Sources")
    st.sidebar.caption(f"RL metrics: `{rl_csv}`")
    st.sidebar.caption(f"Baseline metrics: `{base_csv}`")
    st.sidebar.caption(f"Q-table: `{q_path}`")

    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    downsample = st.sidebar.slider("Chart Downsample (every N rows)", min_value=1, max_value=25, value=3)

    # Load
    rl = load_csv(rl_csv)
    base = load_csv(base_csv)
    q_table = load_q_table(q_path)

    if rl.empty:
        st.error("No `metrics.csv` found yet. Run training first: `python training/train_rl.py`")
        st.stop()

    # Downsample for smoother dashboards on large logs.
    if downsample > 1:
        rl_view = rl.iloc[::downsample].copy()
    else:
        rl_view = rl

    latest = rl.iloc[-1]
    kpi_cards(latest)

    # Main layout
    left, center, right = st.columns([0.26, 0.48, 0.26], gap="large")

    with center:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Core Telemetry")
        x = "global_step"

        st.plotly_chart(line_chart(rl_view, x, "traffic_rpm", "Traffic Over Time (requests/min)", "#8aa1ff"), use_container_width=True)
        st.plotly_chart(line_chart(rl_view, x, "cpu_utilization", "CPU Utilization Over Time (%)", "#fdb022"), use_container_width=True)
        st.plotly_chart(line_chart(rl_view, x, "servers", "Server Count Over Time", "#7c5cff"), use_container_width=True)
        st.plotly_chart(line_chart(rl_view, x, "booking_success_rate", "Booking Success Rate Over Time", "#32d583"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Agent Analytics")
        st.plotly_chart(action_distribution(rl), use_container_width=True)

        st.markdown("#### Q-table Heatmap")
        if q_table is None:
            st.warning("No Q-table model found yet. Train first to enable heatmaps.")
        else:
            server_idx = st.slider("Server bucket", min_value=0, max_value=q_table.shape[3] - 1, value=min(2, q_table.shape[3] - 1))
            action_idx = st.selectbox("Action", options=[0, 1, 2], format_func=lambda a: ACTION_LABELS.get(int(a), str(a)))
            st.plotly_chart(q_table_heatmap(q_table, server_idx=server_idx, action_idx=int(action_idx)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Agent Decision Panel")

        # Prefer true discrete state from training logs (best for explainability).
        if {"state_traffic_b", "state_cpu_b", "state_queue_b", "state_server_idx"}.issubset(set(rl.columns)):
            state = (
                int(latest.get("state_traffic_b", 0)),
                int(latest.get("state_cpu_b", 0)),
                int(latest.get("state_queue_b", 0)),
                int(latest.get("state_server_idx", 0)),
            )
        else:
            # Fallback for older logs: reconstruct coarse bucket-like categories.
            traffic_b = int(pd.qcut(rl["traffic_rpm"], 5, labels=False, duplicates="drop").iloc[-1]) if rl["traffic_rpm"].nunique() > 1 else 0
            cpu_b = int(pd.qcut(rl["cpu_utilization"], 6, labels=False, duplicates="drop").iloc[-1]) if rl["cpu_utilization"].nunique() > 1 else 0
            queue_b = int(pd.qcut(rl["queue_size"], 6, labels=False, duplicates="drop").iloc[-1]) if rl["queue_size"].nunique() > 1 else 0
            server_idx = int(max(0, int(latest.get("servers", 1)) - 1))
            state = (traffic_b, cpu_b, queue_b, server_idx)

        action = int(latest.get("action", 1))
        explanation = explain_decision(state, action)

        st.markdown("**Current State (bucketed)**")
        st.code(f"traffic_bucket={state[0]}, cpu_bucket={state[1]}, queue_bucket={state[2]}, server_index={state[3]}")

        st.markdown("**Recommended Action (last action taken)**")
        st.markdown(f'<span class="tag">{ACTION_LABELS.get(action)}</span>', unsafe_allow_html=True)

        st.markdown("**Explanation**")
        st.write(explanation)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Latest raw signals**")
        st.write(
            {
                "traffic_rpm": float(latest.get("traffic_rpm", 0.0)),
                "cpu_utilization": float(latest.get("cpu_utilization", 0.0)),
                "queue_size": int(latest.get("queue_size", 0)),
                "latency_ms": float(latest.get("latency_ms", 0.0)),
                "booking_success_rate": float(latest.get("booking_success_rate", 0.0)),
                "infra_cost": float(latest.get("infra_cost", 0.0)),
                "sla_violation": bool(latest.get("sla_violation", False)),
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom comparison
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Policy Comparison (RL vs Rule-based)")
    c1, c2 = st.columns(2, gap="large")
    fig_cost, fig_sla = comparison_bars(rl, base if not base.empty else pd.DataFrame(columns=rl.columns))
    with c1:
        st.plotly_chart(fig_cost, use_container_width=True)
    with c2:
        st.plotly_chart(fig_sla, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

