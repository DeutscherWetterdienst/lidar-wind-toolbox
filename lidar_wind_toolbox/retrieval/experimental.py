"""Compatibility entry points for the opt-in experimental retrieval.

The experimental switch is retained for configuration compatibility.  Its
calculation now uses the same array-only domain model as the standard paths;
the richer consensus diagnostics are represented by the array result and
serialized by the same adapter.
"""

import datetime

import xarray as xr

from lidar_wind_toolbox.domain.calculation import retrieve_array
from lidar_wind_toolbox.domain.models import RetrievalMethod, RetrievalParameters
from lidar_wind_toolbox.retrieval.xarray_adapter import (
    halo_scan,
    result_to_xarray,
    windcube_scan,
)


def lvl2vad_standard(
    ds: xr.Dataset, date_chosen: datetime.datetime, confDict: dict[str, str]
) -> xr.Dataset:
    scan = halo_scan(ds)
    parameters = RetrievalParameters.from_legacy_config(confDict, scan.radial_velocity.shape[1])
    result = retrieve_array(scan, parameters, date_chosen, RetrievalMethod.STANDARD_VAD)
    return result_to_xarray(result, confDict)


def lvl2wcdbs(
    ds: xr.Dataset, date_chosen: datetime.datetime, confDict: dict[str, str]
) -> xr.Dataset:
    scan = windcube_scan(ds)
    parameters = RetrievalParameters.from_legacy_config(confDict, scan.radial_velocity.shape[1])
    result = retrieve_array(scan, parameters, date_chosen, RetrievalMethod.WINDCUBE_DBS)
    return result_to_xarray(result, confDict)
