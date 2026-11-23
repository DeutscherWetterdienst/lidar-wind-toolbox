"""Time-grid model for binned retrievals."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    edges: np.ndarray
    timestamps: np.ndarray
    bounds: np.ndarray
    sample_bin: np.ndarray
