from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import diagsvd

from lidar_wind_toolbox.retrieval.consensus import consensus
from lidar_wind_toolbox.retrieval.geometry import build_Amatrix, uvw_2_dir, uvw_2_spd
from lidar_wind_toolbox.signal_calc import CN_est


@dataclass(frozen=True)
class WindcubeTimestepResult:
    """Retrieval before quality control; vectors are (gates, 3), others (gates,)."""

    uvw: NDArray[np.float64]
    uvw_uncertainty: NDArray[np.float64]
    speed: NDArray[np.float64]
    speed_uncertainty: NDArray[np.float64]
    direction: NDArray[np.float64]
    direction_uncertainty: NDArray[np.float64]
    r2: NDArray[np.float64]
    condition_number: NDArray[np.float64]
    n_good: NDArray[np.float64]


def calculate_windcube_timestep(
    radial_velocity: NDArray[np.float64],
    cnr: NDArray[np.float64],
    backscatter: NDArray[np.float64],
    azimuth: NDArray[np.float64],
    elevation: NDArray[np.float64],
    direction_centres: NDArray[np.float64],
    angular_half_width: float,
    *,
    consensus_range: int,
    consensus_percentage: int,
    snr_threshold: int,
    velocity_bandwidth: float,
) -> WindcubeTimestepResult:
    """Reconstruct one nonempty WindCube timestep, without modifying inputs.

    Velocity (m/s), linear CNR and backscatter have shape (rays, gates).
    Azimuth/elevation have shape (rays,), centres (directions,); angles are
    degrees. Consensus range and velocity bandwidth are in m/s; the latter
    is the half-width used by consensus. The CNR threshold is in dB.
    Rays below 89 degrees form directional consensus; rays above 89 degrees
    form one additional vertical consensus. Exactly 89 degrees is excluded.
    Uncertainty uses consensus spread. Gates with fewer than four valid
    consensus directions return NaN retrievals and retain their valid count.
    """
    averaged = _calculate_direction_consensus(
        radial_velocity,
        cnr,
        backscatter,
        azimuth,
        elevation,
        direction_centres,
        angular_half_width,
        consensus_range,
        consensus_percentage,
        snr_threshold,
        velocity_bandwidth,
    )
    if np.any(elevation > 89):
        vertical_velocity, vertical_uncertainty = _calculate_vertical_consensus(
            radial_velocity[elevation > 89], velocity_bandwidth
        )
        averaged = _DirectionConsensus(
            np.vstack([averaged.radial_velocity, vertical_velocity]),
            np.vstack([averaged.uncertainty, vertical_uncertainty]),
            np.hstack([averaged.azimuth, 0]),
            np.hstack([averaged.elevation, 90]),
        )
    # Inputs are already Doppler-derived velocities. Geometry projects the 3D wind
    # onto each viewing direction; reconstruction inverts those projections.
    n_good = (~np.isnan(averaged.radial_velocity)).sum(axis=0).astype(float)
    insufficient_directions = n_good < 4
    excluded = np.isnan(averaged.radial_velocity) | np.tile(
        insufficient_directions, (len(averaged.azimuth), 1)
    )
    geometry = build_Amatrix(averaged.azimuth, averaged.elevation)
    gate_geometry = np.tile(geometry, (radial_velocity.shape[1], 1, 1))
    reconstructed = _solve_wind_components(averaged.radial_velocity, geometry, excluded)
    uvw = np.full((radial_velocity.shape[1], 3), np.nan)
    uvw[...] = np.squeeze(reconstructed)
    uvw_uncertainty = _propagate_wind_uncertainty(
        averaged.radial_velocity, averaged.uncertainty, geometry
    )
    r2 = _calculate_reconstruction_r_squared(averaged.radial_velocity, gate_geometry, reconstructed)
    condition_number = _calculate_geometry_condition_number(gate_geometry, excluded)
    # Apply the existing four-direction requirement uniformly, retaining valid counts.
    for values in (uvw, uvw_uncertainty, r2, condition_number):
        values[insufficient_directions] = np.nan
    speed, speed_uncertainty, direction, direction_uncertainty = _calculate_speed_and_direction(
        uvw, uvw_uncertainty
    )
    return WindcubeTimestepResult(
        uvw=uvw,
        uvw_uncertainty=uvw_uncertainty,
        speed=speed,
        speed_uncertainty=speed_uncertainty,
        direction=direction,
        direction_uncertainty=direction_uncertainty,
        r2=r2,
        condition_number=condition_number,
        n_good=n_good,
    )


@dataclass(frozen=True)
class _DirectionConsensus:
    """Velocities/uncertainties (directions, gates), angles (directions,)."""

    radial_velocity: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    azimuth: NDArray[np.float64]
    elevation: NDArray[np.float64]


def _select_rays_in_azimuth_sector(
    azimuth: NDArray[np.float64],
    centre: float,
    half_width: float,
) -> NDArray[np.bool_]:
    """Select rays within an inclusive circular sector; all angles are degrees."""
    # Nested modulo preserves selection across the 0/360-degree boundary.
    return (
        np.mod(360 - np.mod(np.mod(azimuth - centre, 360) - half_width, 360), 360) <= 2 * half_width
    )


def _solve_wind_components(
    radial_velocity: NDArray[np.float64],
    geometry: NDArray[np.float64],
    excluded: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Solve radial_velocity = geometry @ wind; return (gates, 3, 1) in m/s.

    Velocity and exclusion masks are (directions, gates), geometry (directions, 3).
    Missing observations zero their geometry rows; excluded velocities are masked.
    """
    masked_velocity = np.ma.masked_where(excluded, radial_velocity)
    # Retain historical six-decimal rounding only for the reconstruction solve.
    rounded_geometry = np.round(geometry, 6)
    left_vectors, inverse_singular_values, right_vectors_transposed = [], [], []
    for missing in np.isnan(radial_velocity).T:
        gate_geometry = np.ma.masked_where(np.tile(missing, (3, 1)).T, rounded_geometry).filled(0)
        u, singular_values, vh = np.linalg.svd(gate_geometry, full_matrices=True)
        left_vectors.append(u)
        inverse_singular_values.append(
            np.linalg.pinv(diagsvd(singular_values, u.shape[0], vh.shape[0]))
        )
        right_vectors_transposed.append(vh)
    left_transposed = np.einsum("...ij->...ji", np.array(left_vectors))
    right_vectors = np.einsum("...ij->...ji", np.array(right_vectors_transposed))
    # SVD pseudoinverse: project onto U, scale by inverse singular values, then apply V.
    projected_velocity = np.nansum(left_transposed * masked_velocity.T[:, None, :], axis=2)[
        ..., None
    ]
    scaled_velocity = np.einsum(
        "...ik,...kj->...ij", np.array(inverse_singular_values), projected_velocity
    )
    return np.einsum("...ik,...kj->...ij", right_vectors, scaled_velocity)


def _propagate_wind_uncertainty(
    radial_velocity: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    geometry: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Propagate directional errors (m/s) to component errors (gates, 3).

    Inputs are (directions, gates); geometry is (directions, 3), unrounded.
    Missing radial velocities mask the corresponding uncertainty entries.
    """
    inverse_geometry = np.tile(np.linalg.pinv(geometry), (radial_velocity.shape[1], 1, 1))
    inverse_transposed = np.einsum("...ij->...ji", inverse_geometry)
    masked_uncertainty = np.ma.masked_where(np.isnan(radial_velocity), uncertainty).T
    # Covariance = A+ diag(sigma²) (A+)ᵀ. Preserve the original complex square root conversion.
    covariance = np.einsum(
        "nik,nk,nkl->nil", inverse_geometry, masked_uncertainty**2, inverse_transposed
    )
    return abs(np.einsum("...ii->...i", np.sqrt(covariance.astype(complex)).real))


def _calculate_reconstruction_r_squared(
    radial_velocity: NDArray[np.float64],
    gate_geometry: NDArray[np.float64],
    wind: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return R² per gate, masking NaN observations (directions, gates).

    Geometry is (gates, directions, 3); wind is (gates, 3, 1), in m/s.
    """
    observed = np.ma.masked_where(np.isnan(radial_velocity), radial_velocity).T[..., None]
    predicted = gate_geometry @ wind
    residual_sum_squares = ((observed - predicted) ** 2).sum(axis=1)
    total_sum_squares = ((observed - observed.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)
    result = np.full(radial_velocity.shape[1], np.nan)
    # Assign into an ndarray to retain historical masked-array conversion behavior.
    result[...] = np.squeeze(1 - residual_sum_squares / total_sum_squares)
    return result


def _calculate_geometry_condition_number(
    gate_geometry: NDArray[np.float64],
    excluded: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return dimensionless condition numbers for normalized viewing geometry.

    Geometry is (gates, directions, 3); exclusions are (directions, gates).
    """
    geometry_mask = np.tile(excluded.T[..., None], (1, 1, 3))
    transposed = np.einsum("...ij->...ji", gate_geometry)
    column_scaling = np.apply_along_axis(
        np.diag, 1, 1 / np.sqrt(np.einsum("...ii->...i", transposed @ gate_geometry))
    )
    normalized = np.ma.masked_where(geometry_mask, gate_geometry @ column_scaling)
    result = np.full(gate_geometry.shape[0], np.nan)
    result[...] = np.squeeze(np.array([CN_est(matrix) for matrix in normalized]))
    return result


def _calculate_speed_and_direction(
    wind: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert (gates, 3) components to speed/error (m/s), direction/error (degrees).

    Direction follows the meteorological wind-from convention; speed is horizontal.
    """
    speed, speed_error = np.vstack(
        [
            np.fromiter(uvw_2_spd(value, error).values(), dtype=float)
            for value, error in zip(wind, uncertainty)
        ]
    ).T
    direction, direction_error = np.vstack(
        [
            np.fromiter(uvw_2_dir(value, error).values(), dtype=float)
            for value, error in zip(wind, uncertainty)
        ]
    ).T
    return speed, speed_error, direction, direction_error


def _calculate_direction_consensus(
    radial_velocity: NDArray[np.float64],
    signal_ratio: NDArray[np.float64],
    backscatter: NDArray[np.float64],
    azimuth: NDArray[np.float64],
    elevation: NDArray[np.float64],
    direction_centres: NDArray[np.float64],
    angular_half_width: float,
    consensus_range: int,
    consensus_percentage: int,
    snr_threshold: int,
    velocity_bandwidth: float,
) -> _DirectionConsensus:
    """Average (rays, gates) velocities by direction; angles in degrees, signal linear."""
    oblique = elevation < 89
    radial_velocity, signal_ratio, backscatter = (
        radial_velocity[oblique],
        signal_ratio[oblique],
        backscatter[oblique],
    )
    azimuth, elevation = azimuth[oblique], elevation[oblique]
    shape = (len(direction_centres), radial_velocity.shape[1])
    velocities = np.full(shape, np.nan)
    uncertainties = np.full(shape, np.nan)
    representative_elevation = np.full(len(direction_centres), np.nan)
    for index, centre in enumerate(direction_centres):
        selected = _select_rays_in_azimuth_sector(azimuth, centre, angular_half_width)
        representative_elevation[index] = np.median(elevation[selected])
        velocities[index], accepted, spread = consensus(
            radial_velocity[selected],
            signal_ratio[selected],
            backscatter[selected],
            consensus_range,
            consensus_percentage,
            snr_threshold,
            velocity_bandwidth,
        )
        # WindCube uses the spread of accepted radial velocities directly.
        uncertainties[index] = spread
    return _DirectionConsensus(
        velocities, uncertainties, direction_centres, representative_elevation
    )


def _calculate_vertical_consensus(
    radial_velocity: NDArray[np.float64],
    velocity_bandwidth: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return vertical mean/spread (gates,) in m/s from (rays, gates) velocities."""
    # Preserve the vertical-specific range, percentage and disabled signal threshold.
    velocity, _, spread = consensus(
        radial_velocity,
        np.ones(radial_velocity.shape),
        np.ones(radial_velocity.shape),
        2,
        30,
        0,
        velocity_bandwidth,
    )
    return velocity, spread
