import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.exceptions import InputDatasetError
from lidar_wind_toolbox.validation import validate_normalized_windcube_level1


def _valid_windcube_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "radial_wind_speed": (("time", "gate_index"), np.ones((2, 2))),
            "azimuth": (("time",), np.array([0.0, 90.0])),
            "elevation": (("time",), np.array([75.0, 75.0])),
            "cnr": (("time", "gate_index"), np.full((2, 2), -5.0)),
        },
        coords={
            "time": np.array([1_700_000_000.0, 1_700_000_010.0]),
            "gate_index": np.array([0, 1]),
        },
    )


def test_validate_normalized_windcube_level1_accepts_expected_dataset() -> None:
    validate_normalized_windcube_level1(_valid_windcube_dataset())


def test_validate_normalized_windcube_level1_rejects_missing_variable() -> None:
    dataset = _valid_windcube_dataset().drop_vars("cnr")

    with pytest.raises(InputDatasetError, match="cnr"):
        validate_normalized_windcube_level1(dataset)


def test_validate_normalized_windcube_level1_rejects_invalid_dimensions() -> None:
    dataset = _valid_windcube_dataset()
    dataset["radial_wind_speed"] = (("time",), np.array([1.0, 2.0]))

    with pytest.raises(InputDatasetError, match="radial_wind_speed must be a 2D array"):
        validate_normalized_windcube_level1(dataset)
