"""Compatibility adapter for the standard VAD retrieval."""

import datetime

import xarray as xr

from .numpy_core import RetrievalMethod, RetrievalParameters, retrieve_array
from .xarray_adapter import halo_scan, result_to_xarray


def lvl2vad_standard(
    ds_tmp: xr.Dataset, date_chosen: datetime.datetime, confDict: dict[str, str]
) -> xr.Dataset:
    """Run standard VAD using the NumPy retrieval core."""

    scan = halo_scan(ds_tmp)
    parameters = RetrievalParameters.from_legacy_config(confDict, scan.radial_velocity.shape[1])
    result = retrieve_array(scan, parameters, date_chosen, RetrievalMethod.STANDARD_VAD)
    return result_to_xarray(result, confDict)
