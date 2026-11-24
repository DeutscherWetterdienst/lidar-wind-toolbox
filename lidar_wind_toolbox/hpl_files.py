#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.readers.file_discovery import try_date


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
        return [x.strip() for x in re.split(r"[:=\-]", re.sub("[\n\t]", "", string), 1)]

    @staticmethod
    def split_data(string):
        return re.split(r"\s+", re.sub("\n", "", string).strip())

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
        """Combine discovered files through the byte-content/domain boundary."""
        from lidar_wind_toolbox.domain.merging.level_1 import combine_lvl1_to_ds

        return combine_lvl1_to_ds(
            [path.read_bytes() for path in hpl_list.name],
            confDict,
            None,
            None,
            date_chosen,
            time_chosen,
        )
