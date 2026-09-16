import numpy as np


def build_Amatrix(azimuth_vec, elevation_vec):
    return np.einsum(
        "ij -> ji",
        np.vstack(
            [
                np.sin((np.pi / 180) * azimuth_vec) * np.sin((np.pi / 180) * (90 - elevation_vec)),
                np.cos((np.pi / 180) * azimuth_vec) * np.sin((np.pi / 180) * (90 - elevation_vec)),
                np.cos((np.pi / 180) * (90 - elevation_vec)),
            ]
        ),
    )


def uvw_2_spd(uvw, uvw_unc):
    if (np.isfinite(uvw[0]) * np.isfinite(uvw[1])) & (~np.isnan(uvw[0]) * ~np.isnan(uvw[1])):
        speed = np.sqrt((uvw[0]) ** 2.0 + (uvw[1]) ** 2.0)
    else:
        speed = np.nan
    if speed > 0:
        df_du = uvw[0] * 1 / speed
        df_dv = uvw[1] * 1 / speed
        error = np.sqrt((df_du * uvw_unc[0]) ** 2 + (df_dv * uvw_unc[1]) ** 2)
    else:
        error = np.nan
    return {"speed": speed, "error": error}


def uvw_2_dir(uvw, uvw_unc):
    if (np.isfinite(uvw[0]) * np.isfinite(uvw[1])) & (~np.isnan(uvw[0]) * ~np.isnan(uvw[1])):
        wdir = np.arctan2(uvw[0], uvw[1]) * 180 / np.pi + 180
    else:
        wdir = np.nan
    if np.isfinite(wdir):
        error = (
            (180 / np.pi)
            * np.sqrt((uvw[0] * uvw_unc[0]) ** 2 + (uvw[1] * uvw_unc[1]) ** 2)
            / (uvw[0] ** 2 + uvw[1] ** 2)
        )
    else:
        error = np.nan
    return {"wdir": wdir, "error": error}
