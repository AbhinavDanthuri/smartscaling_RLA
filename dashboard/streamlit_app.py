"""
Workspace-root convenience wrapper.

This file exists so you can run exactly:
    streamlit run dashboard/streamlit_app.py

It delegates to:
    smartscaling_restaurant_rl/dashboard/streamlit_app.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "smartscaling_restaurant_rl" / "dashboard" / "streamlit_app.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

