"""NumPy retrieval calculations operating on domain models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from lidar_wind_toolbox.domain.models import (
    LidarScan,
    RetrievalMethod,
    RetrievalParameters,
    TimeGrid,
    WindRetrievalResult,
)
from lidar_wind_toolbox.retrieval.consensus import consensus
from lidar_wind_toolbox.retrieval.geometry import build_Amatrix, uvw_2_dir, uvw_2_spd
from lidar_wind_toolbox.signal_calc import CN_est
from lidar_wind_toolbox.wind_calc import find_num_dir


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
            velocity[mask],
            snr[mask],
            beta[mask],
            parameters.consensus_range_mps,
            parameters.consensus_percentage,
            parameters.snr_threshold_db,
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
            velocity[mask],
            snr[mask],
            beta[mask],
            parameters.vertical_consensus_range_mps,
            parameters.vertical_consensus_percentage,
            parameters.vertical_snr_threshold_db,
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
        np.asarray(beams),
        np.asarray(uncertainties),
        np.asarray(snr_out),
        np.asarray(beta_out),
        np.asarray(width_out),
        np.asarray(beam_azimuth),
        np.asarray(beam_elevation),
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
        valid = result.record_consensus(index, beams, beam_snr, beam_beta, beam_spec)
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
