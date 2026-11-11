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
    azi_mean = direction_centres
    r_phi = angular_half_width
    B = velocity_bandwidth
    n_gates = radial_velocity.shape[1]
    uvw = np.full((n_gates, 3), np.nan)
    uvw_uncertainty = np.full((n_gates, 3), np.nan)
    speed = np.full((n_gates,), np.nan)
    speed_uncertainty = np.full((n_gates,), np.nan)
    direction = np.full((n_gates,), np.nan)
    direction_uncertainty = np.full((n_gates,), np.nan)
    r2 = np.full((n_gates,), np.nan)
    condition_number = np.full((n_gates,), np.nan)
    n_good = np.full((n_gates,), np.nan)

    VR = radial_velocity[elevation < 89]
    # only if vertical values exist
    if np.any(elevation > 89):
        WR = radial_velocity[elevation > 89]
        # estimate consensus of vertical velocity data
        w_cns, idx_tmp, tmp_tmp = consensus(WR, np.ones(WR.shape), np.ones(WR.shape), 2, 30, 0, B)
        WR_SPEC = tmp_tmp
    CNR = cnr[elevation < 89]
    BETA = backscatter[elevation < 89]
    azi = azimuth[elevation < 89]
    ele = elevation[elevation < 89]

    VR_CNSmax = np.full((len(azi_mean), n_gates), np.nan)
    VR_CNSunc = np.full((len(azi_mean), n_gates), np.nan)
    CNR_CNS = np.full((len(azi_mean), n_gates), np.nan)
    SIGMA_CNS = np.full((len(azi_mean), n_gates), np.nan)
    ele_cns = np.full((len(azi_mean),), np.nan)

    for ii, azi_i in enumerate(azi_mean):
        azi_idx = np.mod(360 - np.mod(np.mod(azi - azi_i, 360) - r_phi, 360), 360) <= 2 * r_phi
        ele_cns[ii] = np.median(ele[azi_idx])
        ## calculate consensus average
        VR_CNSmax[ii, :], idx_tmp, VR_CNSunc[ii, :] = consensus(
            VR[azi_idx],
            CNR[azi_idx],
            BETA[azi_idx],
            consensus_range,
            consensus_percentage,
            snr_threshold,
            B,
        )
        CNR_CNS[ii, :] = np.ma.masked_where(~idx_tmp, CNR[azi_idx]).mean(axis=0).filled(np.nan)

        SIGMA_CNS[ii, :] = VR_CNSunc[ii, :]
    if np.any(elevation > 89):
        #         Add vertical Stares to azimuth consensus
        VR_CNSmax = np.vstack([VR_CNSmax, w_cns])
        SIGMA_CNS = np.vstack([SIGMA_CNS, WR_SPEC])
        azi_mean = np.hstack([azi_mean, 0])
        ele_cns = np.hstack([ele_cns, 90])
    n_good_kk = (~np.isnan(VR_CNSmax)).sum(axis=0)

    n_good[...] = n_good_kk
    V_r = np.ma.masked_where(
        (np.isnan(VR_CNSmax)),
        VR_CNSmax,
    ).T[..., None]
    mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
    V_in = np.ma.masked_where(mask_V_in, VR_CNSmax)
    A = build_Amatrix(azi_mean, ele_cns)
    A_r = np.tile(A, (VR_CNSmax.shape[1], 1, 1))
    A_r_MP = np.tile(np.linalg.pinv(A), (VR_CNSmax.shape[1], 1, 1))
    A_r_MP_T = np.einsum("...ij->...ji", A_r_MP)
    SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T

    condi = np.isnan(VR_CNSmax)
    A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
    U, S, Vh = [], [], []
    for c_nn in condi.T:
        u, s, vh = np.linalg.svd(
            np.ma.masked_where(np.tile(c_nn, (3, 1)).T, A).filled(0), full_matrices=True
        )

        U.append(u)
        S.append(np.linalg.pinv(diagsvd(s, u.shape[0], vh.shape[0])))
        Vh.append(vh)
    U, S, Vh = np.array(U), np.array(S), np.array(Vh)

    U_T = np.einsum("...ij->...ji", U)
    Vh_T = np.einsum("...ij->...ji", Vh)
    K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
    K2 = np.einsum("...ik,...kj->...ij", S, K1)

    V_k = np.einsum("...ik,...kj->...ij", Vh_T, K2)
    uvw[...] = np.squeeze(V_k)
    # Require at least four valid consensus directions for wind retrieval.
    uvw[np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

    uvw_uncertainty[...] = abs(
        np.einsum(
            "...ii->...i",
            np.sqrt(
                np.einsum("nik,nk,nkl->nil", A_r_MP, SIGMA_r**2, A_r_MP_T).astype(complex)
            ).real,
        )
    )
    # Require at least four valid consensus directions for wind retrieval.
    uvw_uncertainty[np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(
        np.full((3, 1), np.nan)
    )

    V_r_est = A_r @ V_k
    ss_e = ((V_r - V_r_est) ** 2).sum(axis=1)
    ss_t = ((V_r - V_r.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)
    r2[...] = np.squeeze(1 - ss_e / ss_t)
    r2[np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

    mask_A = np.tile(mask_V_in.T[..., None], (1, 1, 3))
    A_r_T = np.einsum("...ij->...ji", A_r)
    Spp = np.apply_along_axis(np.diag, 1, 1 / np.sqrt(np.einsum("...ii->...i", A_r_T @ A_r)))
    Z = np.ma.masked_where(mask_A, A_r @ Spp)
    condition_number[...] = np.squeeze(np.array([CN_est(X) for X in Z]))
    condition_number[np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

    speed[...], speed_uncertainty[...] = np.vstack(
        [
            np.fromiter(uvw_2_spd(val, unc).values(), dtype=float)
            for val, unc in zip(uvw[...], uvw_uncertainty[...])
        ]
    ).T
    direction[...], direction_uncertainty[...] = np.vstack(
        [
            np.fromiter(uvw_2_dir(val, unc).values(), dtype=float)
            for val, unc in zip(uvw[...], uvw_uncertainty[...])
        ]
    ).T

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
