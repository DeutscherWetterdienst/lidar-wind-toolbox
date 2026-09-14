import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.exceptions import InputDatasetError
from lidar_wind_toolbox.validation import validate_normalized_windcube_level1


def _dataset() -> xr.Dataset:
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


def test_validate_normalized_windcube_level1_accepts_expected_dataset() -> None:
    validate_normalized_windcube_level1(_dataset())


def test_validate_normalized_windcube_level1_rejects_missing_variable() -> None:
    dataset = _dataset().drop_vars("beta")

    with pytest.raises(InputDatasetError, match="beta"):
        validate_normalized_windcube_level1(dataset)
