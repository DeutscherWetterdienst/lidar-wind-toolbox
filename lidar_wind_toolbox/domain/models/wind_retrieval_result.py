"""Mutable NumPy-backed wind retrieval result."""

from __future__ import annotations

import numpy as np

from lidar_wind_toolbox.domain.models.retrieval_parameters import RetrievalParameters


class WindRetrievalResult:
    """Mutable array-backed result populated during one retrieval."""

    def __init__(
        self,
        *,
        time: np.ndarray,
        time_bounds: np.ndarray,
        height: np.ndarray,
        height_bounds: np.ndarray,
        horizontal_width: np.ndarray,
    ) -> None:
        self.time = np.asarray(time)
        self.time_bounds = np.asarray(time_bounds)
        self.height = np.asarray(height)
        self.height_bounds = np.asarray(height_bounds)
        self.horizontal_width = np.asarray(horizontal_width)

        bins = self.time.size
        gates = self.height.size
        self.wind = np.full((bins, gates, 3), np.nan)
        self.wind_uncertainty = np.full_like(self.wind, np.nan)
        self.speed = np.full((bins, gates), np.nan)
        self.speed_uncertainty = np.full_like(self.speed, np.nan)
        self.direction = np.full_like(self.speed, np.nan)
        self.direction_uncertainty = np.full_like(self.speed, np.nan)
        self.r2 = np.full_like(self.speed, np.nan)
        self.condition_number = np.full_like(self.speed, np.nan)
        self.radial_velocity_count = np.zeros_like(self.speed)
        self.quality = np.zeros((bins, gates), dtype=bool)
        self.component_quality = np.zeros((bins, gates, 3), dtype=bool)
        self.binary_direction = np.full((bins, gates), -999, dtype=np.int64)
        self.snr = np.full_like(self.speed, np.nan)
        self.snr_min = np.full_like(self.speed, np.nan)
        self.snr_max = np.full_like(self.speed, np.nan)
        self.beta = np.full_like(self.speed, np.nan)
        self.spectral_width = np.full_like(self.speed, np.nan)

    def record_consensus(
        self,
        time_index: int,
        beams: np.ndarray,
        beam_snr: np.ndarray,
        beam_beta: np.ndarray,
        beam_spectral_width: np.ndarray,
    ) -> np.ndarray:
        """Reduce and store consensus diagnostics for one time bin."""

        arrays = (beams, beam_snr, beam_beta, beam_spectral_width)
        if any(array.shape != beams.shape for array in arrays[1:]):
            raise ValueError("consensus arrays must all have shape (beam, gate)")
        if beams.ndim != 2:
            raise ValueError("beams must have shape (beam, gate)")

        valid = np.isfinite(beams)
        self.radial_velocity_count[time_index] = valid.sum(axis=0)
        self.binary_direction[time_index] = valid.T.dot(1 << np.arange(beams.shape[0] - 1, -1, -1))
        self.snr[time_index] = np.nanmedian(beam_snr, axis=0)
        self.snr_min[time_index] = np.nanmin(beam_snr, axis=0)
        self.snr_max[time_index] = np.nanmax(beam_snr, axis=0)
        self.beta[time_index] = np.nanmedian(beam_beta, axis=0)
        self.spectral_width[time_index] = np.nanmedian(beam_spectral_width, axis=0)
        return valid

    def record_gate_solution(
        self,
        time_index: int,
        gate_index: int,
        *,
        wind: np.ndarray,
        wind_uncertainty: np.ndarray,
        speed: float,
        speed_uncertainty: float,
        direction: float,
        direction_uncertainty: float,
        r2: float,
        condition_number: float,
    ) -> None:
        """Store the solved wind products for one time/gate cell."""

        self.wind[time_index, gate_index] = wind
        self.wind_uncertainty[time_index, gate_index] = wind_uncertainty
        self.speed[time_index, gate_index] = speed
        self.speed_uncertainty[time_index, gate_index] = speed_uncertainty
        self.direction[time_index, gate_index] = direction
        self.direction_uncertainty[time_index, gate_index] = direction_uncertainty
        self.r2[time_index, gate_index] = r2
        self.condition_number[time_index, gate_index] = condition_number

    def finalize_quality(self, parameters: RetrievalParameters) -> None:
        """Compute aggregate and component quality masks in place."""

        self.quality[...] = (
            np.isfinite(self.speed)
            & (self.radial_velocity_count >= parameters.minimum_radial_velocities)
            & (self.condition_number <= parameters.condition_number_threshold)
            & (self.r2 >= parameters.r2_threshold)
        )
        self.component_quality[...] = self.quality[..., None]
