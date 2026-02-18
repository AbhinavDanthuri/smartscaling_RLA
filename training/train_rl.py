"""
Workspace-root convenience wrapper.

This file exists so you can run exactly:
    python training/train_rl.py

It delegates to:
    smartscaling_restaurant_rl/training/train_rl.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "smartscaling_restaurant_rl" / "training" / "train_rl.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

