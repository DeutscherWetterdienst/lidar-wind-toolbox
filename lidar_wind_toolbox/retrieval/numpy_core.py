"""Array-only wind retrieval domain model and numerical pipeline.

This module deliberately has no dependency on xarray.  xarray is used by the
small adapters in :mod:`standard` and :mod:`windcube` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

import numpy as np

from lidar_wind_toolbox.signal_calc import CN_est
from lidar_wind_toolbox.wind_calc import find_num_dir
from lidar_wind_toolbox.retrieval.consensus import consensus
from lidar_wind_toolbox.retrieval.geometry import build_Amatrix, uvw_2_dir, uvw_2_spd


class RetrievalMethod(StrEnum):
    STANDARD_VAD = "standard_vad"
    WINDCUBE_DBS = "windcube_dbs"


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


@dataclass(frozen=True)
class RetrievalParameters:
    number_of_directions: int
    averaging_minutes: int
    consensus_range_mps: float
    consensus_percentage: float
    snr_threshold_db: float
    minimum_radial_velocities: int
    condition_number_threshold: float
    r2_threshold: float
    blindzone_gates: int = 0
    azimuth_offset_deg: float = 0.0
    excluded_azimuths_deg: tuple[float, ...] = ()
    horizontal_directions: int | None = None
    vertical_consensus_range_mps: float = 2.0
    vertical_consensus_percentage: float = 30.0
    vertical_snr_threshold_db: float = 0.0
    utc_offset_hours: int = 0

    @classmethod
    def from_legacy_config(cls, config: dict[str, str], gates: int) -> "RetrievalParameters":
        def f(name: str, default: float | None = None) -> float:
            value = config.get(name)
            return float(default if value is None else value)

        def i(name: str, default: int | None = None) -> int:
            value = config.get(name)
            return int(default if value is None else value)

        excluded = tuple(float(v) for v in config.get("EX_AZI", "").split(",") if v.strip())
        return cls(
            number_of_directions=i("NUMBER_OF_DIRECTIONS", 4),
            averaging_minutes=i("AVG_MIN", 30),
            consensus_range_mps=f("CNS_RANGE", 3),
            consensus_percentage=f("CNS_PERCENTAGE", 60),
            snr_threshold_db=f("SNR_THRESHOLD", 0),
            minimum_radial_velocities=i("N_VRAD_THRESHOLD", 4),
            condition_number_threshold=f("CN_THRESHOLD", 100),
            r2_threshold=f("R2_THRESHOLD", 0),
            blindzone_gates=i("BLINDZONE_GATES", 0),
            azimuth_offset_deg=f("AZIMUTH_OFFSET", f("BEARING", 0)),
            excluded_azimuths_deg=excluded,
            horizontal_directions=(
                i("N_HORI_DIRECTIONS") if "N_HORI_DIRECTIONS" in config else None
            ),
            vertical_consensus_range_mps=f("CNS_RANGE_VERTICAL", 2),
            vertical_consensus_percentage=f("CNS_PERCENTAGE_VERTICAL", 30),
            vertical_snr_threshold_db=f("SNR_THRESHOLD_VERTICAL", 0),
            utc_offset_hours=i("UTC_OFFSET", 0),
        )


@dataclass(frozen=True)
class TimeGrid:
    edges: np.ndarray
    timestamps: np.ndarray
    bounds: np.ndarray
    sample_bin: np.ndarray


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
        *,
        radial_velocity_count: np.ndarray,
        binary_direction: np.ndarray,
        snr: np.ndarray,
        snr_min: np.ndarray,
        snr_max: np.ndarray,
        beta: np.ndarray,
        spectral_width: np.ndarray,
    ) -> None:
        """Store diagnostics produced for one time bin."""

        self.radial_velocity_count[time_index] = radial_velocity_count
        self.binary_direction[time_index] = binary_direction
        self.snr[time_index] = snr
        self.snr_min[time_index] = snr_min
        self.snr_max[time_index] = snr_max
        self.beta[time_index] = beta
        self.spectral_width[time_index] = spectral_width

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


def build_time_grid(
    time_values: np.ndarray, date: datetime, minutes: int, offset: int = 0
) -> TimeGrid:
    start = date.replace(tzinfo=timezone.utc) - timedelta(hours=offset)
    step = np.timedelta64(minutes, "m")
    edges = np.arange(
        np.datetime64(start),
        np.datetime64(start + timedelta(days=1) + timedelta(minutes=minutes)),
        step,
    )
    values = np.asarray(time_values).astype("datetime64[ns]")
    sample_bin = np.searchsorted(edges, values, side="right") - 1
    sample_bin[(sample_bin < 0) | (sample_bin >= len(edges) - 1)] = -1
    timestamps = edges[1:].astype("datetime64[s]").astype(np.int64)
    bounds = np.column_stack((edges[:-1], edges[1:])).astype("datetime64[s]").astype(np.int64)
    return TimeGrid(edges, timestamps, bounds, sample_bin)


def _consensus_beams(
    scan: LidarScan,
    indices: np.ndarray,
    parameters: RetrievalParameters,
    method: RetrievalMethod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elevation = scan.elevation[indices]
    horizontal = elevation < 89
    vertical = elevation > 89
    azimuth = (scan.azimuth[indices] + parameters.azimuth_offset_deg) % 360
    n_horizontal = parameters.horizontal_directions or (
        parameters.number_of_directions - int(np.any(vertical))
    )
    if n_horizontal < 3 or not np.any(horizontal):
        return tuple(np.empty((0, scan.radial_velocity.shape[1])) for _ in range(3)) + (
            np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        )  # type: ignore[return-value]

    horizontal_indices = np.flatnonzero(horizontal)
    local_groups = [np.flatnonzero(np.ones(horizontal_indices.size, dtype=bool))]
    ok, n_horizontal, _, _ = find_num_dir(n_horizontal, local_groups, azimuth[horizontal], 0)
    if not ok:
        return tuple(np.empty((0, scan.radial_velocity.shape[1])) for _ in range(3)) + (
            np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        )  # type: ignore[return-value]
    # ``find_num_dir`` validates the number of populated sectors but returns
    # sectors anchored at north.  The physical scan can have an arbitrary
    # phase, so recover that phase from the observed azimuths before grouping.
    sector_width = 360 / n_horizontal
    phase = float(np.nanmedian(np.mod(azimuth[horizontal], sector_width)))
    centers = (phase + np.arange(n_horizontal) * sector_width) % 360
    excluded = np.asarray(parameters.excluded_azimuths_deg)
    if excluded.size:
        distance = np.abs(((centers[:, None] - excluded[None, :] + 180) % 360) - 180)
        centers = centers[np.all(distance > 1e-9, axis=1)]
    radius = 180 / n_horizontal
    velocity = scan.radial_velocity[indices]
    snr = scan.snr[indices]
    beta = scan.beta[indices]
    beams: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    snr_out: list[np.ndarray] = []
    beta_out: list[np.ndarray] = []
    width_out: list[np.ndarray] = []
    beam_azimuth: list[float] = []
    beam_elevation: list[float] = []
    for center in centers:
        mask = horizontal & (np.abs(((azimuth - center + 180) % 360) - 180) <= radius)
        if not np.any(mask):
            continue
        values, selected, uncertainty = consensus(
            velocity[mask], snr[mask], beta[mask], parameters.consensus_range_mps,
            parameters.consensus_percentage, parameters.snr_threshold_db,
            _velocity_half_range(scan, method),
        )
        beams.append(values)
        uncertainties.append(uncertainty)
        selected_values = np.where(selected, snr[mask], np.nan)
        snr_out.append(np.nanmean(selected_values, axis=0))
        beta_out.append(np.nanmean(np.where(selected, beta[mask], np.nan), axis=0))
        width_out.append(np.nanmean(np.where(selected, _width(scan, mask), np.nan), axis=0))
        beam_azimuth.append(float(center))
        beam_elevation.append(float(np.median(elevation[mask])))
    if np.any(vertical):
        mask = vertical
        values, selected, uncertainty = consensus(
            velocity[mask], snr[mask], beta[mask], parameters.vertical_consensus_range_mps,
            parameters.vertical_consensus_percentage, parameters.vertical_snr_threshold_db,
            _velocity_half_range(scan, method),
        )
        beams.append(values)
        uncertainties.append(uncertainty)
        snr_out.append(np.nanmean(np.where(selected, snr[mask], np.nan), axis=0))
        beta_out.append(np.nanmean(np.where(selected, beta[mask], np.nan), axis=0))
        width_out.append(np.nanmean(np.where(selected, _width(scan, mask), np.nan), axis=0))
        beam_azimuth.append(0.0)
        beam_elevation.append(90.0)
    return (
        np.asarray(beams), np.asarray(uncertainties), np.asarray(snr_out), np.asarray(beta_out),
        np.asarray(width_out), np.asarray(beam_azimuth), np.asarray(beam_elevation),
    )


def _velocity_half_range(scan: LidarScan, method: RetrievalMethod) -> float:
    if method is RetrievalMethod.WINDCUBE_DBS:
        return float((np.nanmax(scan.radial_velocity) - np.nanmin(scan.radial_velocity)) / 2)
    return float(scan.bandwidth or np.nanmax(np.abs(scan.radial_velocity)))


def _width(scan: LidarScan, mask: np.ndarray) -> np.ndarray:
    if scan.spectral_width is None:
        return np.full((int(np.count_nonzero(mask)), scan.radial_velocity.shape[1]), np.nan)
    return scan.spectral_width[mask]


def retrieve_array(
    scan: LidarScan,
    parameters: RetrievalParameters,
    date: datetime,
    method: RetrievalMethod,
) -> WindRetrievalResult:
    gates = scan.radial_velocity.shape[1]
    grid = build_time_grid(
        scan.time, date, parameters.averaging_minutes, parameters.utc_offset_hours
    )
    elev = np.nanmedian(scan.elevation[scan.elevation < 89])
    heights = scan.height if scan.height is not None else scan.range
    if scan.height is None:
        heights = scan.range * np.sin(np.deg2rad(elev))
    height_bounds = scan.range_bounds.copy()
    height_bounds *= np.sin(np.deg2rad(elev))
    width = 2 * scan.range * np.cos(np.deg2rad(elev))
    result = WindRetrievalResult(
        time=grid.timestamps,
        time_bounds=grid.bounds,
        height=heights,
        height_bounds=height_bounds,
        horizontal_width=width,
    )
    for index in range(result.time.size):
        rays = np.flatnonzero(grid.sample_bin == index)
        if not rays.size:
            continue
        (
            beams,
            beam_unc,
            beam_snr,
            beam_beta,
            beam_spec,
            azimuth,
            elevation,
        ) = _consensus_beams(scan, rays, parameters, method)
        if not len(beams):
            continue
        valid = np.isfinite(beams)
        result.record_consensus(
            index,
            radial_velocity_count=valid.sum(axis=0),
            binary_direction=valid.T.dot(1 << np.arange(len(beams) - 1, -1, -1)),
            snr=np.nanmedian(beam_snr, axis=0),
            snr_min=np.nanmin(beam_snr, axis=0),
            snr_max=np.nanmax(beam_snr, axis=0),
            beta=np.nanmedian(beam_beta, axis=0),
            spectral_width=np.nanmedian(beam_spec, axis=0),
        )
        geometry = np.round(build_Amatrix(azimuth, elevation), 6)
        for gate in range(gates):
            good = valid[:, gate]
            if good.sum() < max(3, parameters.minimum_radial_velocities):
                continue
            matrix = geometry[good]
            values = beams[good, gate]
            estimate, *_ = np.linalg.lstsq(matrix, values, rcond=None)
            pseudoinverse = np.linalg.pinv(matrix)
            covariance = pseudoinverse @ np.diag(np.square(beam_unc[good, gate])) @ pseudoinverse.T
            unc = np.sqrt(np.clip(np.diag(covariance), 0, None))
            predicted = matrix @ estimate
            residual = np.sum((values - predicted) ** 2)
            total = np.sum((values - np.mean(values)) ** 2)
            current_r2 = np.nan if total == 0 else 1 - residual / total
            current_cn = CN_est(np.ma.masked_array(matrix, mask=False))
            current_speed, current_speed_uncertainty = tuple(
                uvw_2_spd(estimate, unc).values()
            )
            current_direction, current_direction_uncertainty = tuple(
                uvw_2_dir(estimate, unc).values()
            )
            result.record_gate_solution(
                index,
                gate,
                wind=estimate,
                wind_uncertainty=unc,
                speed=current_speed,
                speed_uncertainty=current_speed_uncertainty,
                direction=current_direction,
                direction_uncertainty=current_direction_uncertainty,
                r2=current_r2,
                condition_number=current_cn,
            )
    result.finalize_quality(parameters)
    return result

