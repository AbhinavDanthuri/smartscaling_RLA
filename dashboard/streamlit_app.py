"""
Workspace-root convenience wrapper.

Run:
    streamlit run dashboard/streamlit_app.py

This wrapper forwards execution to the actual dashboard module under:
    smartscaling_restaurant_rl/dashboard/streamlit_app.py
or (deployment rename fallback):
    smartscaling_rla/dashboard/streamlit_app.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "smartscaling_restaurant_rl" / "dashboard" / "streamlit_app.py",
        root / "smartscaling_rla" / "dashboard" / "streamlit_app.py",
    ]

    target = None
    for candidate in candidates:
        if candidate.exists():
            target = candidate
            break

    if target is None:
        raise FileNotFoundError(
            "Could not find dashboard target. Expected one of: "
            "smartscaling_restaurant_rl/dashboard/streamlit_app.py or "
            "smartscaling_rla/dashboard/streamlit_app.py"
        )

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()



