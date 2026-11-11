"""Focused checks for the mathematical stages used by both retrieval cores."""

import numpy as np
import pytest

from lidar_wind_toolbox.retrieval import vad, windcube_core
from lidar_wind_toolbox.retrieval.geometry import build_Amatrix


@pytest.mark.parametrize("core", [vad, windcube_core])
def test_sector_wraparound_and_inclusive_boundaries(core):
    selected = core._select_rays_in_azimuth_sector(
        np.array([349.0, 350.0, 359.0, 0.0, 10.0, 11.0, 360.0]), 0.0, 10.0
    )
    np.testing.assert_array_equal(selected, [False, True, True, True, True, False, True])


@pytest.mark.parametrize("core", [vad, windcube_core])
def test_reconstruction_with_missing_direction_and_diagnostics(core):
    geometry = build_Amatrix(np.arange(0.0, 360.0, 60.0), np.full(6, 60.0))
    expected = np.array([[3.0, 4.0, 0.5], [-2.0, 5.0, 1.0]])
    radial = geometry @ expected.T
    radial[0, 1] = np.nan
    excluded = np.isnan(radial)
    wind = core._solve_wind_components(radial, geometry, excluded)
    np.testing.assert_allclose(wind[..., 0], expected, atol=1e-5)
    gates = np.tile(geometry, (2, 1, 1))
    np.testing.assert_allclose(
        core._calculate_reconstruction_r_squared(radial, gates, wind), 1.0, atol=1e-10
    )
    condition = core._calculate_geometry_condition_number(gates, excluded)
    assert condition.shape == (2,)
    assert np.all(np.isfinite(condition))
    assert np.all(condition >= 1.0)
    perturbed = wind.copy()
    perturbed[:, 0, 0] += 1.0
    assert np.all(core._calculate_reconstruction_r_squared(radial, gates, perturbed) < 0.99)


@pytest.mark.parametrize("core", [vad, windcube_core])
def test_uncertainty_propagates_radial_variance(core):
    geometry = build_Amatrix(np.arange(0.0, 360.0, 90.0), np.full(4, 60.0))
    radial = geometry @ np.array([[3.0], [4.0], [0.5]])
    uncertainty = np.full_like(radial, 0.2)
    inverse = np.linalg.pinv(geometry)
    expected = np.sqrt(np.diag(inverse @ np.diag(uncertainty[:, 0] ** 2) @ inverse.T))
    actual = core._propagate_wind_uncertainty(radial, uncertainty, geometry)
    assert actual.shape == (1, 3)
    np.testing.assert_allclose(actual[0], expected)
