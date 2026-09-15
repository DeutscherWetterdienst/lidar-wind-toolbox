import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.exceptions import InputDatasetError
from lidar_wind_toolbox.validation import validate_normalized_windcube_level1


def _legacy_normalized_dataset() -> xr.Dataset:
    """Dataset with legacy normalized variables (dv, azi, zenith, intensity)."""
    return xr.Dataset(
        data_vars={
            "dv": (("time", "range"), np.ones((2, 2))),
            "intensity": (("time", "range"), np.ones((2, 2))),
            "beta": (("time", "range"), np.ones((2, 2))),
            "delv": (("time", "range"), np.ones((2, 2))),
            "azi": (("time",), np.array([0.0, 90.0])),
            "zenith": (("time",), np.array([15.0, 15.0])),
            "nsmpl": ((), 10.0),
            "prf": ((), 10000.0),
            "nqv": ((), 19.0),
            "range_bnds": (
                ("range", "nv"),
                np.array([[25.0, 75.0], [75.0, 125.0]]),
            ),
        },
        coords={
            "time": np.array([1_700_000_000.0, 1_700_000_010.0]),
            "range": np.array([50.0, 100.0]),
            "nv": np.array([0, 1]),
        },
    )


def _native_windcube_vad_dataset() -> xr.Dataset:
    """Dataset with native WindCube VAD variables produced by read_wc_type."""
    return xr.Dataset(
        data_vars={
            "radial_wind_speed": (("time", "gate_index"), np.ones((2, 2))),
            "azimuth": (("time",), np.array([0.0, 90.0])),
            "elevation": (("time",), np.array([75.0, 75.0])),
            "cnr": (("time", "gate_index"), np.full((2, 2), -5.0)),
            "doppler_spectrum_width": (("time", "gate_index"), np.ones((2, 2))),
        },
        coords={
            "time": np.array([1_700_000_000.0, 1_700_000_010.0]),
            "gate_index": np.array([0, 1]),
        },
    )


def test_validate_accepts_legacy_normalized_dataset() -> None:
    validate_normalized_windcube_level1(_legacy_normalized_dataset())


def test_validate_accepts_native_windcube_vad_dataset() -> None:
    validate_normalized_windcube_level1(_native_windcube_vad_dataset())


def test_validate_rejects_missing_variable_in_legacy_dataset() -> None:
    dataset = _legacy_normalized_dataset().drop_vars("intensity")

    with pytest.raises(InputDatasetError, match="intensity"):
        validate_normalized_windcube_level1(dataset)


def test_validate_rejects_missing_variable_in_native_dataset() -> None:
    dataset = _native_windcube_vad_dataset().drop_vars("cnr")

    with pytest.raises(InputDatasetError, match="cnr"):
        validate_normalized_windcube_level1(dataset)


def test_validate_rejects_dataset_without_velocity_variable() -> None:
    dataset = xr.Dataset(
        data_vars={"dummy": (("time",), [1.0])},
        coords={"time": [1_700_000_000.0]},
    )

    with pytest.raises(InputDatasetError, match="radial_wind_speed.*dv"):
        validate_normalized_windcube_level1(dataset)
