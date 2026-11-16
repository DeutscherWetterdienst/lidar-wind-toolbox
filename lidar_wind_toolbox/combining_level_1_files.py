#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.readers.read_byte_content import read_byte_content
from lidar_wind_toolbox.readers.windcube_bytes import read_wcsradial_bytes, read_wc_type_bytes


def combine_lvl1_to_ds(
    file_contents: list[bytes],
    confDict: dict[str, str],
    date_chosen: datetime.datetime,
    time_chosen: datetime.datetime | None = None,
) -> xr.Dataset:
    from .readers.windcube import read_wc_type

    if confDict["SYSTEM"] == "halo":
        ds = xr.concat(
            (read_byte_content(iit, confDict) for iit in file_contents),
            dim="time",
            data_vars="minimal",
            coords="minimal",
        )
        ds["nqv"].values = ((ds.dv.max() - ds.dv.min()).data / 2).astype("f4")
        ds["nqf"].values = (2 * ds.nqv.data / float(confDict["SYSTEM_WAVELENGTH"])).astype("f4")
        ds["resv"].values = (2 * ds.nqv.data / float(confDict["FFT_POINTS"])).astype("f4")
        ## delete 'delv' variable, if all entries are NaN.
        if (ds.delv == -999.0).all():
            ds = ds.drop_vars(["delv"])

    elif confDict["SYSTEM"] == "windcube":
        if ("fixed".lower() in confDict["SCAN_TYPE"].lower()) or ("stare".lower()) in confDict[
            "SCAN_TYPE"
        ].lower():
            print("processing 'Windcube-fixed/-stare' setting!")
            ds = xr.concat(
                (
                    read_wcsradial_bytes(iit, confDict)
                    for iit in file_contents
                    if read_wcsradial_bytes(iit, confDict) is not False
                ),
                dim="time",
                data_vars="minimal",
                compat="override",
                coords="minimal",
            )
            ds["nqv"].values = ((ds.dv.max() - ds.dv.min()).data / 2).astype("f4")
            ds["nqf"].values = (2 * ds.nqv.data / float(confDict["SYSTEM_WAVELENGTH"])).astype("f4")
            ds["resv"].values = (2 * ds.nqv.data / float(confDict["FFT_POINTS"])).astype("f4")
            ## delete 'delv' variable, if all entries are NaN.
            if (ds.delv == -999.0).all():
                ds = ds.drop_vars(["delv"])
        else:
            print("processing 'WindCube-dbs/-vad/-pp' setting!")
            ds = xr.concat(
                (read_wc_type_bytes(iit) for iit in file_contents if read_wc_type_bytes(iit) is not False),
                dim="time",
                data_vars="minimal",
                compat="override",
                coords="minimal",
            )
            ds["nqv"] = ((ds.radial_wind_speed.max() - ds.radial_wind_speed.min()).data / 2).astype(
                "f4"
            )
            ds["nqf"] = (2 * ds.nqv.data / float(confDict["SYSTEM_WAVELENGTH"])).astype("f4")
            ds["resv"] = (2 * ds.nqv.data / float(confDict["FFT_POINTS"])).astype("f4")
            # print('dropping "delv" / "spectral width", because all are NaN!')
    # if os.name == 'nt':
    #  ds = ds._drop_vars(['delv'])
    # else:
    #  ds = ds.drop_vars(['delv'])
    ##!!!NOTE!!!##
    # There was an issue under windows, possible due to a version problem,
    # so in case an Attribute error occurs change line 126 to following
    # ds = ds._drop_vars(['delv'])
    ## choose only timestamp within range...
    if time_chosen is not None:
        # ...a time window of range AVG_MIN
        start_dt = (
            pd.to_datetime(time_chosen - datetime.timedelta(minutes=int(confDict["AVG_MIN"])))
            - pd.Timestamp("1970-01-01")
        ) / pd.Timedelta("1s")
        end_dt = (pd.to_datetime(time_chosen) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")
    else:
        # ...a daily range
        start_dt = (pd.to_datetime(date_chosen.date()) - pd.Timestamp("1970-01-01")) / pd.Timedelta(
            "1s"
        )
        end_dt = (
            pd.to_datetime(date_chosen + datetime.timedelta(days=+1)) - pd.Timestamp("1970-01-01")
        ) / pd.Timedelta("1s")
    print(start_dt, ds.time[0].data)
    print(end_dt, ds.time[-1].data)
    ds = ds.isel(time=np.where((ds.time >= start_dt) & (ds.time <= end_dt))[0])
    ds.attrs["title"] = confDict["NC_TITLE"]
    ds.attrs["institution"] = confDict["NC_INSTITUTION"]
    ds.attrs["site_location"] = confDict["NC_SITE_LOCATION"]
    ds.attrs["source"] = confDict["NC_SOURCE"]
    ds.attrs["instrument_type"] = confDict["NC_INSTRUMENT_TYPE"]
    ds.attrs["instrument_mode"] = confDict["NC_INSTRUMENT_MODE"]
    if "NC_INSTRUMENT_FIRMWARE_VERSION" in confDict:
        ds.attrs["instrument_firmware_version"] = confDict["NC_INSTRUMENT_FIRMWARE_VERSION"]
    else:
        ds.attrs["instrument_firmware_version"] = "N/A"
    ds.attrs["instrument_contact"] = confDict["NC_INSTRUMENT_CONTACT"]
    if "NC_INSTRUMENT_ID" in confDict:
        ds.attrs["instrument_id"] = confDict["NC_INSTRUMENT_ID"]
    else:
        ds.attrs["instrument_id"] = "N/A"
    ds.attrs["conventions"] = confDict["NC_CONVENTIONS"]
    ds.attrs["processing_date"] = str(pd.to_datetime(datetime.datetime.now())) + " UTC"
    ds.attrs["instrument_contact"] = confDict["NC_INSTRUMENT_CONTACT"]
    ds.attrs["data_policy"] = confDict["NC_DATA_POLICY"]
    # attributes for operational use of netCDFs, see E-Profile wind profiler netCDF version 1.7
    if "NC_WIGOS_STATION_ID" in confDict:
        ds.attrs["wigos_station_id"] = confDict["NC_WIGOS_STATION_ID"]
    else:
        ds.attrs["wigos_station_id"] = "N/A"
    if "NC_WMO_ID" in confDict:
        ds.attrs["wmo_id"] = confDict["NC_WMO_ID"]
    else:
        ds.attrs["wmo_id"] = "N/A"
    if "NC_PI_ID" in confDict:
        ds.attrs["principal_investigator"] = confDict["NC_PI_ID"]
    else:
        ds.attrs["principal_investigator"] = "N/A"
    if "NC_INSTRUMENT_SERIAL_NUMBER" in confDict:
        ds.attrs["instrument_serial_number"] = confDict["NC_INSTRUMENT_SERIAL_NUMBER"]
    else:
        ds.attrs["instrument_serial_number"] = " "
    ds.attrs["history"] = (
        confDict["NC_HISTORY"]
        + " version "
        + confDict["VERSION"]
        + " on "
        + str(pd.to_datetime(datetime.datetime.now()))
        + " UTC"
    )
    ds.attrs["comments"] = confDict["NC_COMMENTS"]
    ## add configuration as attribute used to create the file
    configuration = """"""
    for dd in confDict:
        configuration += dd + "=" + confDict[dd] + "\n"
    ds.attrs["File_Configuration"] = configuration

    # adjust time variable to double (aka float64)
    ds.time.data.astype(np.float64)

    if "UTC_OFFSET" in confDict:
        time_delta = int(confDict["UTC_OFFSET"])
    else:
        time_delta = 0

    ds.time.attrs["units"] = (
        "seconds since 1970-01-01 00:00:00",
        "seconds since 1970-01-01 00:00:00 {:+03d}".format(time_delta),
    )[abs(np.sign(time_delta))]
    ds.time.encoding["units"] = (
        "seconds since 1970-01-01 00:00:00",
        "seconds since 1970-01-01 00:00:00 {:+03d}".format(time_delta),
    )[abs(np.sign(time_delta))]

    return ds
