"""
Centralized metrics logger.

Why a logger class?
- Training loops get noisy fast if you manually append dozens of lists.
- Streamlit dashboards like "tidy" tabular data (Pandas DataFrame).
- NPZ gives you compact arrays; CSV is convenient for inspection and Plotly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class MetricsLogger:
    run_name: str = "rl"
    _data: Dict[str, List[Any]] = field(default_factory=dict)

    def log(self, **metrics: Any) -> None:
        """
        Log a single row of metrics.

        This method is flexible: you can add new keys at any time and they will
        be automatically tracked.
        """
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(v)

        # Ensure all keys have same length (fill missing keys with None).
        target_len = max(len(v) for v in self._data.values())
        for k, lst in self._data.items():
            if len(lst) < target_len:
                lst.extend([None] * (target_len - len(lst)))

    def dataframe(self) -> pd.DataFrame:
        """
        Return a Streamlit-friendly DataFrame view.
        """
        if not self._data:
            return pd.DataFrame()
        return pd.DataFrame(self._data)

    def to_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.dataframe()
        df.to_csv(path, index=False)

    def to_npz(self, path: str) -> None:
        """
        Save metrics to NPZ.

        NPZ stores each column as an array. Non-numeric columns are stored as
        object arrays, which is still fine for research demos.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.dataframe()
        arrays: Dict[str, np.ndarray] = {}
        for col in df.columns:
            series = df[col]
            # Preserve strings/bools without forcing float conversion.
            if pd.api.types.is_numeric_dtype(series):
                arrays[col] = series.to_numpy()
            else:
                arrays[col] = series.astype("object").to_numpy()
        arrays["run_name"] = np.asarray(self.run_name, dtype="object")
        np.savez_compressed(path, **arrays)

    def export(self, csv_path: str, npz_path: str) -> None:
        self.to_csv(csv_path)
        self.to_npz(npz_path)

