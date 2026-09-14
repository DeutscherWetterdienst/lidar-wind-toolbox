import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.exceptions import InputDatasetError
from lidar_wind_toolbox.validation import validate_windcube_vad_input


def _dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "radial_wind_speed": (("time", "range"), np.ones((2, 2))),
            "cnr": (("time", "range"), np.ones((2, 2))),
            "doppler_spectrum_width": (("time", "range"), np.ones((2, 2))),
            "azimuth": (("time",), np.array([0.0, 90.0])),
            "elevation": (("time",), np.array([75.0, 75.0])),
            "range_gate_length": ((), 50.0),
        },
        coords={
            "time": np.array([1_700_000_000.0, 1_700_000_010.0]),
            "range": np.array([50.0, 100.0]),
        },
    )


def test_validate_windcube_vad_input_accepts_expected_dataset() -> None:
    validate_windcube_vad_input(_dataset())


def test_validate_windcube_vad_input_rejects_missing_variable() -> None:
    dataset = _dataset().drop_vars("cnr")

    with pytest.raises(InputDatasetError, match="cnr"):
        validate_windcube_vad_input(dataset)
