"""Synthetic scans exercise consensus and reconstruction without an xarray dataset."""

from dataclasses import fields

import numpy as np
import pytest

from lidar_wind_toolbox.retrieval.geometry import build_Amatrix
from lidar_wind_toolbox.retrieval.vad import calculate_vad_timestep


def scan(gates=1, directions=4):
    centres = np.arange(directions) * (360.0 / directions)
    azimuth = np.repeat(centres, 3)
    elevation = np.full(azimuth.shape, 60.0)
    winds = np.array([[3.0 + gate, 4.0, 0.5] for gate in range(gates)])
    velocity = build_Amatrix(azimuth, elevation) @ winds.T
    return dict(
        radial_velocity=velocity,
        snr=np.full_like(velocity, 0.1),
        backscatter=np.ones_like(velocity),
        azimuth=azimuth,
        elevation=elevation,
        direction_centres=centres,
        angular_half_width=180 / directions,
        consensus_range=3,
        consensus_percentage=60,
        snr_threshold=-20,
        samples_per_gate=64,
        pulses=10000,
        nyquist_velocity=20.0,
    ), winds


@pytest.mark.parametrize("gates,directions", [(1, 4), (3, 8)])
def test_known_wind_and_input_preservation(gates, directions):
    inputs, winds = scan(gates, directions)
    before = {key: value.copy() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    result = calculate_vad_timestep(**inputs)
    np.testing.assert_allclose(result.uvw, winds, atol=1e-4)
    np.testing.assert_allclose(result.speed, np.hypot(winds[:, 0], winds[:, 1]), atol=1e-4)
    np.testing.assert_allclose(
        result.direction, np.degrees(np.arctan2(winds[:, 0], winds[:, 1])) + 180, atol=1e-3
    )
    np.testing.assert_allclose(result.r2, 1, atol=1e-8)
    np.testing.assert_array_equal(result.n_good, directions)
    assert np.all(result.condition_number >= 1)
    for field in fields(result):
        values = getattr(result, field.name)
        assert values.shape == ((gates, 3) if field.name.startswith("uvw") else (gates,))
        assert np.all(np.isfinite(values))
        if "uncertainty" in field.name:
            assert np.all(values > 0)
    for key, value in before.items():
        np.testing.assert_array_equal(inputs[key], value)


@pytest.mark.parametrize("rejection", ["snr", "velocity", "backscatter"])
def test_four_three_and_zero_valid_directions(rejection):
    inputs, winds = scan(3)
    key, invalid = {
        "snr": ("snr", 1e-5),
        "velocity": ("radial_velocity", 30),
        "backscatter": ("backscatter", -1),
    }[rejection]
    inputs[key][:3, 1] = invalid
    inputs[key][:, 2] = invalid
    result = calculate_vad_timestep(**inputs)
    np.testing.assert_array_equal(result.n_good, [4, 3, 0])
    np.testing.assert_allclose(result.uvw[0], winds[0], atol=1e-4)
    for field in fields(result):
        if field.name != "n_good":
            assert np.all(np.isnan(getattr(result, field.name)[1:]))


def dataset_scan():
    import datetime

    import xarray as xr

    inputs, _ = scan(2)
    count = len(inputs["azimuth"])
    # Two populated windows separated by an empty window.
    times = np.concatenate(
        [
            np.datetime64("2026-01-01") + np.arange(count).astype("timedelta64[s]"),
            np.datetime64("2026-01-01T02:00") + np.arange(count).astype("timedelta64[s]"),
        ]
    )
    velocity = np.concatenate([inputs["radial_velocity"], inputs["radial_velocity"] * 2])
    dataset = xr.Dataset(
        {
            "dv": (["time", "range"], velocity),
            "intensity": (["time", "range"], np.full_like(velocity, 1.1)),
            "beta": (["time", "range"], np.ones_like(velocity)),
            "azi": ("time", np.tile(inputs["azimuth"], 2)),
            "zenith": ("time", np.tile(90 - inputs["elevation"], 2)),
            "range_bnds": (["range", "nv"], [[50.0, 150.0], [150.0, 250.0]]),
            "prf": 10000,
            "nsmpl": 64,
            "nqv": 20.0,
        },
        coords={"time": times, "range": [100.0, 200.0]},
    )
    config = dict(
        NUMBER_OF_GATES="2",
        NUMBER_OF_DIRECTIONS="4",
        AVG_MIN="60",
        CNS_RANGE="3",
        CNS_PERCENTAGE="60",
        SNR_THRESHOLD="-20",
        R2_THRESHOLD="0.9",
        CN_THRESHOLD="10",
        N_VRAD_THRESHOLD="4",
        SYSTEM_LATITUDE="50",
        SYSTEM_LONGITUDE="10",
        SYSTEM_ALTITUDE="100",
        SYSTEM_WAVELENGTH="1.5e-6",
    )
    return dataset, datetime.datetime(2026, 1, 1), config, inputs


def test_dataset_collects_results_and_preserves_empty_windows():
    from lidar_wind_toolbox.retrieval.standard import lvl2vad_standard

    dataset, date, config, inputs = dataset_scan()
    result = lvl2vad_standard(dataset, date, config)
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
        timestep = calculate_vad_timestep(
            **{**inputs, "radial_velocity": inputs["radial_velocity"] * factor}
        )
        for field, variable in mapping.items():
            np.testing.assert_allclose(result[variable][index], getattr(timestep, field), rtol=1e-6)
        for component, variable in enumerate(["u", "v", "w"]):
            np.testing.assert_allclose(
                result[variable][index], timestep.uvw[:, component], rtol=1e-6
            )
            np.testing.assert_allclose(
                result["err" + variable][index], timestep.uvw_uncertainty[:, component], rtol=1e-6
            )
        np.testing.assert_array_equal(result.qwind[index], 1)
    np.testing.assert_array_equal(result.qwind[1], 0)
    np.testing.assert_array_equal(result.wspeed[1], -999.0)
    assert result.sizes == {"time": 24, "height": 2, "nv": 2}
