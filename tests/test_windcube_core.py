"""WindCube consensus, vertical rays and timestep collection."""

from dataclasses import fields

import numpy as np
import pytest

from lidar_wind_toolbox.retrieval.geometry import build_Amatrix
from lidar_wind_toolbox.retrieval.windcube_core import calculate_windcube_timestep
from tests.test_vad import dataset_scan, scan


def windcube_scan(gates=1, vertical=False):
    inputs, winds = scan(gates)
    inputs["cnr"] = inputs.pop("snr")
    inputs["velocity_bandwidth"] = inputs.pop("nyquist_velocity")
    inputs.pop("samples_per_gate")
    inputs.pop("pulses")
    if vertical:
        for key in ["radial_velocity", "cnr", "backscatter"]:
            extra = (
                np.tile(winds[:, 2], (3, 1)) if key == "radial_velocity" else np.ones((3, gates))
            )
            inputs[key] = np.vstack([inputs[key], extra])
        inputs["azimuth"] = np.r_[inputs["azimuth"], [0.0, 0.0, 0.0]]
        inputs["elevation"] = np.r_[inputs["elevation"], [90.0, 90.0, 90.0]]
    # Symmetric repeated-ray spread preserves the mean and gives known uncertainty.
    inputs["radial_velocity"] += np.tile([-0.1, 0.0, 0.1], len(inputs["azimuth"]) // 3)[:, None]
    return inputs, winds


@pytest.mark.parametrize("gates", [1, 3])
@pytest.mark.parametrize("vertical", [False, True])
def test_known_wind_spread_and_unchanged_inputs(gates, vertical):
    inputs, winds = windcube_scan(gates, vertical)
    before = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
    result = calculate_windcube_timestep(**inputs)
    np.testing.assert_allclose(result.uvw, winds, atol=1e-4)
    np.testing.assert_allclose(result.speed, np.hypot(winds[:, 0], winds[:, 1]), atol=1e-4)
    np.testing.assert_allclose(
        result.direction, 180 + np.degrees(np.arctan2(winds[:, 0], winds[:, 1])), atol=1e-3
    )
    np.testing.assert_allclose(result.r2, 1, atol=1e-8)
    np.testing.assert_array_equal(result.n_good, 4 + vertical)
    centres = np.r_[inputs["direction_centres"], 0] if vertical else inputs["direction_centres"]
    elevations = np.r_[np.full(4, 60.0), 90.0] if vertical else np.full(4, 60.0)
    inverse = np.linalg.pinv(build_Amatrix(centres, elevations))
    expected_unc = np.sqrt(np.diag(inverse @ inverse.T) * np.var([-0.1, 0, 0.1]))
    np.testing.assert_allclose(
        result.uvw_uncertainty, np.tile(expected_unc, (gates, 1)), atol=1e-12
    )
    for field in fields(result):
        value = getattr(result, field.name)
        assert value.shape == ((gates, 3) if field.name.startswith("uvw") else (gates,))
        assert np.all(np.isfinite(value))
    for key, value in before.items():
        np.testing.assert_array_equal(inputs[key], value)


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize(
    "key,invalid", [("cnr", 1e-5), ("radial_velocity", 30.0), ("backscatter", -1.0)]
)
def test_rejected_directions(key, invalid, vertical):
    inputs, _ = windcube_scan(3, vertical)
    inputs[key][:3, 1] = invalid
    inputs[key][:12, 2] = invalid
    result = calculate_windcube_timestep(**inputs)
    np.testing.assert_array_equal(result.n_good, [4 + vertical, 3 + vertical, int(vertical)])
    for field in fields(result):
        if field.name != "n_good":
            assert np.all(np.isnan(getattr(result, field.name)[2]))
            assert np.all(np.isfinite(getattr(result, field.name)[1])) == vertical


def test_rejected_vertical_consensus_and_89_degree_exclusion():
    inputs, _ = windcube_scan(2, True)
    inputs["radial_velocity"][-3:, 0] = 30.0
    inputs["elevation"][-3:] = 89.0
    result = calculate_windcube_timestep(**inputs)
    np.testing.assert_array_equal(result.n_good, [4, 4])
    inputs["elevation"][-3:] = 90.0
    result = calculate_windcube_timestep(**inputs)
    np.testing.assert_array_equal(result.n_good, [4, 5])


def windcube_dataset(vertical=False):
    import xarray as xr

    _, date, config, _ = dataset_scan()
    inputs, _ = windcube_scan(2, vertical)
    count = len(inputs["azimuth"])
    velocity = np.vstack([inputs["radial_velocity"], inputs["radial_velocity"] * 2])
    ds = xr.Dataset(
        {
            "radial_wind_speed": (["time", "gate"], velocity),
            "cnr": (["time", "gate"], np.tile(10 * np.log10(inputs["cnr"]), (2, 1))),
            "relative_beta": (["time", "gate"], np.tile(inputs["backscatter"], (2, 1))),
            "azimuth": ("time", np.tile(inputs["azimuth"], 2)),
            "elevation": ("time", np.tile(inputs["elevation"], 2)),
            "range": (["time", "gate"], np.tile([100.0, 200.0], (count * 2, 1))),
            "range_gate_length": 100.0,
        },
        coords={"time": np.r_[1767225600 + np.arange(count), 1767232800 + np.arange(count)]},
    )
    return ds, date, config, inputs


@pytest.mark.parametrize("vertical", [False, True])
def test_dataset_collection_and_empty_windows(vertical):
    from lidar_wind_toolbox.retrieval.windcube import lvl2wcdbs

    ds, date, config, inputs = windcube_dataset(vertical)
    inputs["velocity_bandwidth"] = float(
        (ds.radial_wind_speed.max() - ds.radial_wind_speed.min()) / 2
    )
    result = lvl2wcdbs(ds, date, config)
    mapping = dict(
        speed="wspeed",
        speed_uncertainty="errwspeed",
        direction="wdir",
        direction_uncertainty="errwdir",
        r2="r2",
        condition_number="cn",
        n_good="nvrad",
    )
    for index, factor in [(0, 1), (2, 2)]:
        core = calculate_windcube_timestep(
            **{**inputs, "radial_velocity": inputs["radial_velocity"] * factor}
        )
        good = (
            np.isfinite(core.speed)
            & (core.n_good >= 4)
            & (core.r2 >= 0.9)
            & (core.condition_number <= 10)
        )
        np.testing.assert_array_equal(result.qwind[index], good)
        for field, variable in mapping.items():
            np.testing.assert_allclose(
                result[variable][index], np.where(good, getattr(core, field), -999), rtol=1e-6
            )
        for component, name in enumerate(["u", "v", "w"]):
            np.testing.assert_allclose(
                result[name][index], np.where(good, core.uvw[:, component], -999), rtol=1e-6
            )
            np.testing.assert_allclose(
                result["err" + name][index],
                np.where(good, core.uvw_uncertainty[:, component], -999),
                rtol=1e-6,
            )
    np.testing.assert_array_equal(result.qwind[1], 0)
    np.testing.assert_array_equal(result.wspeed[1], -999.0)
