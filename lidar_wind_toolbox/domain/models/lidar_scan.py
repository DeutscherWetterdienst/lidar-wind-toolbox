"""Normalized NumPy lidar scan model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarScan:
    """Normalized level-1 observations represented exclusively by ndarrays."""

    time: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    radial_velocity: np.ndarray
    snr: np.ndarray
    beta: np.ndarray
    range: np.ndarray
    range_bounds: np.ndarray
    spectral_width: np.ndarray | None = None
    prf: float | None = None
    samples_per_gate: float | None = None
    bandwidth: float | None = None
    gate_length: float | None = None
    height: np.ndarray | None = None

    def __post_init__(self) -> None:
        ray_count, gate_count = self.radial_velocity.shape
        for name in ("time", "azimuth", "elevation"):
            value = np.asarray(getattr(self, name))
            if value.shape != (ray_count,):
                raise ValueError(f"{name} must have shape ({ray_count},), got {value.shape}")
        for name in ("snr", "beta"):
            value = np.asarray(getattr(self, name))
            if value.shape != (ray_count, gate_count):
                raise ValueError(f"{name} must match radial_velocity shape")
        if self.range.shape != (gate_count,):
            raise ValueError("range must have one value per gate")
        if self.range_bounds.shape != (gate_count, 2):
            raise ValueError("range_bounds must have shape (gate, 2)")
        if self.spectral_width is not None and self.spectral_width.shape != (ray_count, gate_count):
            raise ValueError("spectral_width must match radial_velocity shape")
