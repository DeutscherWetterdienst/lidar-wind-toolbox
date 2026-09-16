#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .readers.file_discovery import try_date
from .readers.halo import read_hpl
from .readers.windcube import read_wc_type, read_wcsradial


class hpl_files(object):
    name = []
    time = []

    def __init__(self, name, time):
        self.name = name
        self.time = time

    @staticmethod
    def make_file_list(
        date_chosen: datetime.datetime, confDict: dict[str, str], url: str | Path
    ) -> "hpl_files":
        path = (
            Path(url)
            / date_chosen.strftime("%Y")
            / date_chosen.strftime("%Y%m")
            / date_chosen.strftime("%Y%m%d")
        )

        if confDict["SYSTEM"] == "halo":
            scan_type = confDict["SCAN_TYPE"]
            mylist = list(path.glob("**/" + scan_type + "*.hpl"))

        elif confDict["SYSTEM"] == "windcube":
            scan_type = confDict["SCAN_TYPE"].lower().split("_")[0]
            l_rg = "*" + confDict["RANGE_GATE_LENGTH"] + "m"
            if (
                abs(
                    (
                        date_chosen
                        - datetime.datetime(date_chosen.year, date_chosen.month, date_chosen.day)
                    ).total_seconds()
                )
                > 0
            ):
                mylist = list(
                    path.glob(
                        "**/*" + date_chosen.strftime("%Y-%m-%d_%H*") + scan_type + l_rg + "*.nc"
                    )
                )
            else:
                mylist = list(path.glob("**/*" + scan_type + l_rg + "*.nc"))

            if "TP".lower() in confDict["SCAN_TYPE"].lower():
                mylist = list(filter(lambda k: "TP" in k.stem, mylist))
            else:
                mylist = list(filter(lambda k: "TP" not in k.stem, mylist))

        return hpl_files.filelist_to_hpl_files(mylist, confDict["SYSTEM"])

    @staticmethod
    def filelist_to_hpl_files(
        files: list[Path] | list[Path | str], inst_type: str, base_filename: str | None = None
    ) -> "hpl_files":
        fileparts_separator = (
            "_"  # separator between parts of filename, e.g. date and instrument id
        )

        # this routine can be entry point to the hpl_files class, therefore:
        #   - ensure files are Path objects and not strings
        #   - ensure files in list are unique to avoid segmentation faults

        files = list(set(Path(file) for file in files))

        if base_filename is not None:
            # input checking
            if not isinstance(base_filename, str):
                raise ValueError("type of input argument 'base_filename' must be string or None")
            if files[0].name[len(base_filename)] == fileparts_separator:
                raise ValueError(
                    "input argument 'base_filename' must also contain the tailing fileparts separator "
                    + fileparts_separator
                )
            # get number of fileparts in base_filename (as fileparts_separator is last digit, empty part will be split)
            len_basename_parts = (
                len(base_filename.split(fileparts_separator)) - 1
            )  # correct for final empty part
        else:
            len_basename_parts = None

        if inst_type.lower() == "halo":
            if base_filename is None:
                len_basename_parts = 2
            ind_date = slice(len_basename_parts, None)
        elif inst_type.lower() == "windcube":
            if base_filename is None:
                len_basename_parts = 1
            ind_date = slice(len_basename_parts, len_basename_parts + 2)
        else:
            raise ValueError(
                "allowed values for inst_type are 'halo' and 'windcube' but found this: "
                + inst_type
            )

        file_time = [
            try_date("T".join(re.sub("-", "", x.stem).split(fileparts_separator)[ind_date]))
            for x in files
        ]
        files_sorted = [files[idx] for idx in np.argsort(file_time).astype(int)]

        return hpl_files(files_sorted, np.sort(file_time))

    @staticmethod
    def range_calc(rg_vec, confDict: dict[str, str]):
        """Calculate range bounds, also accounting for overlapping gates. If your hpl-files contain overlapping gates please add the "OVERLAPPING_GATES" argument to the configuration file."""
        if "OVERLAPPING_GATES" in confDict:
            r = lambda x, idx: (
                (
                    x
                    / (1, float(confDict["NUMBER_OF_GATE_POINTS"]))[
                        int(confDict["OVERLAPPING_GATES"])
                    ]
                    + idx
                )
                * float(confDict["RANGE_GATE_LENGTH"])
            )
        else:
            r = lambda x, idx: (x + idx) * float(confDict["RANGE_GATE_LENGTH"])

        return r(rg_vec, 0.5).astype("f4")

    @staticmethod
    def split_header(string):
        return [x.strip() for x in re.split("[:\=\-]", re.sub("[\n\t]", "", string), 1)]

    @staticmethod
    def split_data(string):
        return re.split("\s+", re.sub("\n", "", string).strip())

    @staticmethod
    def split_default(string):
        return string

    @staticmethod
    def switch(case: bool, string: str) -> list[str] | str:
        return {True: hpl_files.split_header(string), False: hpl_files.split_data(string)}.get(
            case, hpl_files.split_default
        )

    @staticmethod
    def combine_lvl1(
        hpl_list: "hpl_files",
        confDict: dict[str, str],
        date_chosen: datetime.datetime,
        time_chosen: datetime.datetime | None = None,
    ) -> Path:
        print(hpl_list.time)
        print(time_chosen)
        ds = hpl_files.combine_lvl1_to_ds(hpl_list, confDict, date_chosen, time_chosen)

        if time_chosen is not None:
            path = Path(confDict["NC_L1_PATH"])
            path.mkdir(parents=True, exist_ok=True)
            # use time of start of processing
            path = path / Path(
                confDict["NC_L1_BASENAME"]
                + "v"
                + confDict["VERSION"]
                + "_"
                + time_chosen.strftime("%Y%m%d%H%M")
                + ".nc"
            )
        else:
            path = Path(
                confDict["NC_L1_PATH"]
                + "/"
                + date_chosen.strftime("%Y")
                + "/"
                + date_chosen.strftime("%Y%m")
            )
            path.mkdir(parents=True, exist_ok=True)
            # use daily processing
            path = path / Path(
                confDict["NC_L1_BASENAME"]
                + "v"
                + confDict["VERSION"]
                + "_"
                + date_chosen.strftime("%Y%m%d")
                + ".nc"
            )

        # compress variables
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in np.hstack([ds.data_vars, ds.coords])}

        try:
            ds.to_netcdf(path, encoding=encoding)
        except RuntimeError:
            print("CRITICAL - writing NetCDF file failed, re-trying without timestamps")
            for ts in ["timestamp", "timestamp_local"]:
                ds = ds.drop_vars(ts)
                encoding.pop(ts)
            ds.to_netcdf(path, encoding=encoding)
        ds.close()
        return path

    @staticmethod
    def combine_lvl1_to_ds(
        hpl_list: "hpl_files",
        confDict: dict[str, str],
        date_chosen: datetime.datetime,
        time_chosen: datetime.datetime | None = None,
    ) -> xr.Dataset:
        if confDict["SYSTEM"] == "halo":
            ds = xr.concat(
                (read_hpl(iit, confDict) for iit in hpl_list.name),
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
                        read_wcsradial(iit, confDict)
                        for iit in hpl_list.name
                        if read_wcsradial(iit, confDict) is not False
                    ),
                    dim="time",
                    data_vars="minimal",
                    compat="override",
                    coords="minimal",
                )
                ds["nqv"].values = ((ds.dv.max() - ds.dv.min()).data / 2).astype("f4")
                ds["nqf"].values = (2 * ds.nqv.data / float(confDict["SYSTEM_WAVELENGTH"])).astype(
                    "f4"
                )
                ds["resv"].values = (2 * ds.nqv.data / float(confDict["FFT_POINTS"])).astype("f4")
                ## delete 'delv' variable, if all entries are NaN.
                if (ds.delv == -999.0).all():
                    ds = ds.drop_vars(["delv"])
            else:
                print("processing 'WindCube-dbs/-vad/-pp' setting!")
                ds = xr.concat(
                    (read_wc_type(iit) for iit in hpl_list.name if read_wc_type(iit) is not False),
                    dim="time",
                    data_vars="minimal",
                    compat="override",
                    coords="minimal",
                )
                ds["nqv"] = (
                    (ds.radial_wind_speed.max() - ds.radial_wind_speed.min()).data / 2
                ).astype("f4")
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
            start_dt = (
                pd.to_datetime(date_chosen.date()) - pd.Timestamp("1970-01-01")
            ) / pd.Timedelta("1s")
            end_dt = (
                pd.to_datetime(date_chosen + datetime.timedelta(days=+1))
                - pd.Timestamp("1970-01-01")
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
