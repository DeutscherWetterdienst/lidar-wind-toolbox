"""Compatibility adapter for the WindCube DBS/VAD retrieval."""

import datetime

import xarray as xr

from .numpy_core import RetrievalMethod, RetrievalParameters, retrieve_array
from .xarray_adapter import result_to_xarray, windcube_scan


def lvl2wcdbs(
    ds_comb: xr.Dataset, date_chosen: datetime.datetime, confDict: dict[str, str]
) -> xr.Dataset:
    """Run WindCube retrieval using the NumPy retrieval core."""

    scan = windcube_scan(ds_comb)
    parameters = RetrievalParameters.from_legacy_config(confDict, scan.radial_velocity.shape[1])
    result = retrieve_array(scan, parameters, date_chosen, RetrievalMethod.WINDCUBE_DBS)
    return result_to_xarray(result, confDict)
