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
