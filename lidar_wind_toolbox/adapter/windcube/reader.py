from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.adapter.common import (
    NormalizedRadialData,
    build_normalized_radial_dataset,
)
from lidar_wind_toolbox.domain.constants import SPEED_OF_LIGHT
from lidar_wind_toolbox.hpl_files import hpl_files
from lidar_wind_toolbox.retrieval.uncertainty import calc_sigma_single


def read_wc_type(file_content: bytes) -> xr.Dataset | Literal[False]:
    try:
        ds_root = xr.open_dataset(file_content)
    except OSError:
        print("corrupted netCDF: '{}'")
        return False
    sweep_list = list(ds_root.sweep_group_name.data)
    ds_ind = xr.concat(
        (
            xr.open_dataset(file_content, group=sweep_ii, decode_times=False)
            for sweep_ii in sweep_list
        ),
        dim="time",
        data_vars="minimal",
        compat="override",
        coords="minimal",
    )
    return ds_ind


def read_wcsradial(
    file_content: bytes, confDict: dict[str, str]
) -> xr.Dataset | Literal[False]:
    ## the windcube netCDF l1-files are in cf-radial format
    # this requires a workaround to open to access the radial data,
    # when relying on xarray package alone

    # read root attributes
    try:
        ds_root = xr.open_dataset(file_content)
    except OSError:
        print("corrupted netCDF: '{}'")
        return False

    if "time_reference" in list(ds_root.keys()):
        time_reference = ds_root.time_reference.data
    else:
        time_reference = None
    sweep_list = list(ds_root.sweep_group_name.data)
    # read radial data in sweep group
    ds_tmp = xr.concat(
        (
            xr.open_dataset(file_content, group=sweep_ii, decode_times=False)
            for sweep_ii in sweep_list
        ),
        dim="time",
        data_vars="minimal",
        compat="override",
        coords="minimal",
    )
    if time_reference is None:
        time_reference = ds_tmp.time_reference.data

    range_mid = hpl_files.range_calc(ds_tmp.gate_index.data.astype(int), confDict)
    dr = (np.float32(confDict["PULS_DURATION"]) * SPEED_OF_LIGHT / 4).astype("f4")
    range_bnds = np.array([range_mid - dr, range_mid + dr]).T
    tgint = 2 * (range_bnds[0, 1] - range_bnds[0, 0]) / SPEED_OF_LIGHT

    zenith = np.array([90 - ds_root.sweep_fixed_angle.data[0]] * ds_tmp.time.size)

    ## calculate measurement uncertainty
    sigma_tmp = calc_sigma_single(
        ds_tmp.cnr.data,
        int(confDict["NUMBER_OF_GATE_POINTS"]),
        int(confDict["PULSES_PER_DIRECTION"]),
        ((ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data / 2).astype("f4"),
        1.316,
    )
    return build_normalized_radial_dataset(
        NormalizedRadialData(
            dv=ds_tmp.radial_wind_speed.data,
            errdv=sigma_tmp.astype("f4"),
            intensity=(10 ** (ds_tmp.cnr.data / 10) + 1).astype("f4"),
            beta=ds_tmp.relative_beta.data.astype("f4"),
            delv=ds_tmp.doppler_spectrum_width.data.astype("f4"),
            azi=ds_tmp.azimuth.data.astype("f4"),
            zenith=zenith.astype("f4"),
            time=ds_tmp.time.data,
            time_units="seconds since {}".format(
                pd.to_datetime(time_reference).strftime("%Y-%m-%d %H:%M:%S")
            ),
            range=ds_tmp.range.data.astype("f4"),
            range_bnds=range_bnds.astype("f4"),
            lat=np.float32(confDict["SYSTEM_LATITUDE"]),
            lon=np.float32(confDict["SYSTEM_LONGITUDE"]),
            zsl=np.float32(confDict["SYSTEM_ALTITUDE"]),
            wl=np.float32(confDict["SYSTEM_WAVELENGTH"]),
            pd=np.float32(confDict["PULS_DURATION"]),
            nfft=np.float32(confDict["FFT_POINTS"]),
            nrg=np.float32(ds_tmp.dims["range"]),
            lrg=np.float32(ds_tmp.range_gate_length.data),
            nsmpl=np.float32(confDict["NUMBER_OF_GATE_POINTS"]),
            prf=np.float32(confDict["PULS_REPETITION_FREQ"]),
            npls=np.float32(confDict["PULSES_PER_DIRECTION"]),
            focus=np.float32(confDict["FOCUS"]),
            resv=(
                (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data
                / float(confDict["FFT_POINTS"])
            ).astype("f4"),
            nqf=(
                (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data
                / float(confDict["SYSTEM_WAVELENGTH"])
            ).astype("f4"),
            nqv=((ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data / 2).astype(
                "f4"
            ),
            smplf=np.float32(confDict["NUMBER_OF_GATE_POINTS"]) / tgint,
            resf=(
                np.float32(confDict["NUMBER_OF_GATE_POINTS"])
                / tgint
                / float(confDict["FFT_POINTS"])
            ).astype("f4"),
            tgint=tgint,
        )
    )
