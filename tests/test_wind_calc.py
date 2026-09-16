import numpy as np

from lidar_wind_toolbox.retrieval.geometry import build_Amatrix
from lidar_wind_toolbox.wind_calc import uvw_2_dir, uvw_2_spd


def test_build_amatrix_for_horizontal_eastward_ray() -> None:
    matrix = build_Amatrix(
        azimuth_vec=np.array([90.0]),
        elevation_vec=np.array([0.0]),
    )

    np.testing.assert_allclose(
        matrix,
        np.array([[1.0, 0.0, 0.0]]),
        atol=1e-12,
    )


def test_wind_speed_for_3_4_5_triangle() -> None:
    result = uvw_2_spd(
        uvw=np.array([3.0, 4.0, 0.0]),
        uvw_unc=np.array([0.1, 0.1, 0.1]),
    )

    assert result["speed"] == 5.0
    assert np.isfinite(result["error"])


def test_wind_direction_for_northward_wind() -> None:
    result = uvw_2_dir(
        uvw=np.array([0.0, 10.0, 0.0]),
        uvw_unc=np.array([0.1, 0.1, 0.1]),
    )

    assert result["wdir"] == 180.0
