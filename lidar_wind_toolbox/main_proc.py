import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.retrieval import experimental
from lidar_wind_toolbox.retrieval.standard import lvl2vad_standard
from lidar_wind_toolbox.retrieval.windcube import lvl2wcdbs
from lidar_wind_toolbox.stare_proc import build_wind_l2_filename, process_stare_l2_from_ds


def process_dataset(
    ds_tmp: xr.Dataset, date_chosen: datetime.datetime, confDict: dict[str, str]
) -> xr.Dataset:
    ds_lvl2 = calculate_wind(confDict, date_chosen, ds_tmp)
    add_attributes_from_confict_dict(confDict, ds_lvl2)
    return ds_lvl2


def calculate_wind(confDict, date_chosen, ds_tmp):
    system = confDict["SYSTEM"].lower()

    if system == "windcube":
        return process_windcube(confDict, date_chosen, ds_tmp)
    if system == "halo":
        return process_halo(confDict, date_chosen, ds_tmp)


def process_halo(confDict, date_chosen, ds_tmp) -> xr.Dataset:
    experimental_wind = confDict.get("EXPERIMENTAL_WIND_RETRIEVAL", "0") == "1"
    retrieve_standard = experimental.lvl2vad_standard if experimental_wind else lvl2vad_standard
    scan_type = confDict["SCAN_TYPE"].lower()

    if (len(ds_tmp.range.dims) > 1) & (scan_type == "dbs"):
        print("processing 'Streamline-dbs' setting!")
        ds_lvl2 = retrieve_standard(ds_tmp, date_chosen, confDict)
    if ("vad" in scan_type) | ("user" in scan_type):
        print("processing 'Streamline-VAD' setting!")
        ds_lvl2 = retrieve_standard(ds_tmp, date_chosen, confDict)
    if "stare" in scan_type:
        print("processing 'Streamline-Stare' setting!")
        with xr.open_dataset(build_wind_l2_filename(confDict, date_chosen)) as ds_wind:
            ds_lvl2 = process_stare_l2_from_ds(ds_tmp, ds_wind, date_chosen, confDict)
        return ds_lvl2
    raise ValueError(f"Unsupported scan_type {scan_type}")


def process_windcube(confDict, date_chosen, ds_tmp) -> xr.Dataset:
    experimental_wind = confDict.get("EXPERIMENTAL_WIND_RETRIEVAL", "0") == "1"
    retrieve_standard = experimental.lvl2vad_standard if experimental_wind else lvl2vad_standard
    retrieve_windcube = experimental.lvl2wcdbs if experimental_wind else lvl2wcdbs
    scan_type = confDict["SCAN_TYPE"].lower()
    if "fixed" not in scan_type:
        if ("dbs" in scan_type) or ("vad" in scan_type) or ("ppi" in scan_type):
            print("processing 'Windcube-dbs/-vad' setting!")
            ds_lvl2 = retrieve_windcube(ds_tmp, date_chosen, confDict)
        if "rhi" in scan_type:
            print("settings for RHI not yet implemented!")
            # create place holder dataset
            ds_lvl2 = xr.Dataset()
    if "fixed" in scan_type:
        if "vad" in scan_type:
            print("processing 'Windcube-vad-fixed' setting!...for old system version!!")
            ds_lvl2 = retrieve_standard(ds_tmp, date_chosen, confDict)
        if "stare" in scan_type:
            print("processing 'WindCube-stare' setting!")
            print("coming soon!")
            # create place holder dataset
            ds_lvl2 = xr.Dataset()
    return ds_lvl2


def add_attributes_from_confict_dict(confDict, ds_lvl2) -> None:
    ds_lvl2.attrs["title"] = confDict["NC_TITLE"]
    ds_lvl2.attrs["institution"] = confDict["NC_INSTITUTION"]
    ds_lvl2.attrs["site_location"] = confDict["NC_SITE_LOCATION"]
    ds_lvl2.attrs["source"] = confDict["NC_SOURCE"]
    ds_lvl2.attrs["instrument_type"] = confDict["NC_INSTRUMENT_TYPE"]
    ds_lvl2.attrs["instrument_mode"] = confDict["NC_INSTRUMENT_MODE"]
    if "NC_INSTRUMENT_FIRMWARE_VERSION" in confDict:
        ds_lvl2.attrs["instrument_firmware_version"] = confDict["NC_INSTRUMENT_FIRMWARE_VERSION"]
    else:
        ds_lvl2.attrs["instrument_firmware_version"] = "N/A"
    if "NC_INSTRUMENT_ID" in confDict:
        ds_lvl2.attrs["instrument_id"] = confDict["NC_INSTRUMENT_ID"]
    else:
        ds_lvl2.attrs["instrument_id"] = "N/A"
    ds_lvl2.attrs["instrument_contact"] = confDict["NC_INSTRUMENT_CONTACT"]

    ds_lvl2.attrs["conventions"] = confDict["NC_CONVENTIONS"]
    ds_lvl2.attrs["processing_date"] = str(pd.to_datetime(datetime.datetime.now())) + " UTC"

    ds_lvl2.attrs["data_policy"] = confDict["NC_DATA_POLICY"]

    if "NC_WIGOS_STATION_ID" in confDict:
        ds_lvl2.attrs["wigos_station_id"] = confDict["NC_WIGOS_STATION_ID"]
    else:
        ds_lvl2.attrs["wigos_station_id"] = "N/A"
    if "NC_WMO_ID" in confDict:
        ds_lvl2.attrs["wmo_id"] = confDict["NC_WMO_ID"]
    else:
        ds_lvl2.attrs["wmo_id"] = "N/A"
    if "NC_PI_ID" in confDict:
        ds_lvl2.attrs["principal_investigator"] = confDict["NC_PI_ID"]
    else:
        ds_lvl2.attrs["principal_investigator"] = "N/A"
    if "NC_INSTRUMENT_SERIAL_NUMBER" in confDict:
        ds_lvl2.attrs["instrument_serial_number"] = confDict["NC_INSTRUMENT_SERIAL_NUMBER"]
    else:
        ds_lvl2.attrs["instrument_serial_number"] = " "
    ds_lvl2.attrs["history"] = (
        confDict["NC_HISTORY"]
        + " version "
        + confDict["VERSION"]
        + " on "
        + str(pd.to_datetime(datetime.datetime.now()))
        + " UTC"
    )
    if "OPERATIONAL" in confDict:
        if bool(int(confDict["OPERATIONAL"])):
            # Blocking statur, only important for operational use
            ds_lvl2.attrs["references"] = confDict[
                "NC_REFERENCES"
            ]  # 'Doppler lidar PPI-based retrieval, see VAD'
            ds_lvl2.attrs["data_blocking_status"] = confDict["NC_DATA_BLOCKING_STATUS"]
            # set all uncertainties to NaN-Value
            for item in ["erru", "errv", "errw", "errwspeed", "errwdir"]:
                ds_lvl2[item] = ds_lvl2[item].where(ds_lvl2[item] == -999.0, other=-999.0)

    ds_lvl2.attrs["comments"] = confDict["NC_COMMENTS"]


def write_netcdf(ds: xr.Dataset, filename: str | Path, confDict: dict[str, str]) -> None:
    # compress variables
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in np.hstack([ds.data_vars, ds.coords])}

    # set UTC offset
    if "UTC_OFFSET" in confDict:
        time_delta = int(confDict["UTC_OFFSET"])
    else:
        time_delta = 0

    ds.time.encoding["units"] = (
        "seconds since 1970-01-01 00:00:00",
        "seconds since 1970-01-01 00:00:00 {:+03d}".format(time_delta),
    )[abs(np.sign(time_delta))]

    ds.to_netcdf(filename, unlimited_dims={"time": True}, encoding=encoding)
