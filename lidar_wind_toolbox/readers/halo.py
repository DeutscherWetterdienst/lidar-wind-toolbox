import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.hpl_files import hpl_files
from lidar_wind_toolbox.retrieval.uncertainty import calc_sigma_single


def read_hpl(filename: Path, confDict: dict[str, str]) -> xr.Dataset:
    if not filename.exists():
        print("Oops, file doesn't exist!")
    else:
        print("reading file: " + filename.name)
    with filename.open() as infile:
        header_info = True
        mheader = {}
        for line in infile:
            if line.startswith("****"):
                header_info = False
                ## Adjust header in order to extract data formats more easily
                ## 1st for 'Data line 1' , i.e. time of beam etc.
                tmp = [x.split() for x in mheader["Data line 1"].split("  ")]
                if len(tmp) > 3:
                    tmp.append(" ".join([tmp[2][2], tmp[2][3]]))
                    tmp.append(" ".join([tmp[2][4], tmp[2][5]]))
                tmp[0] = " ".join(tmp[0])
                tmp[1] = " ".join(tmp[1])
                tmp[2] = " ".join([tmp[2][0], tmp[2][1]])
                mheader["Data line 1"] = tmp
                tmp = mheader["Data line 1 (format)"].split(",1x,")
                tmp.append(tmp[-1])
                tmp.append(tmp[-1])
                mheader["Data line 1 (format)"] = tmp
                ## Adjust header in order to extract data formats more easily
                ## 2st for 'Data line 2' , i.e. actual data
                tmp = [x.split() for x in mheader["Data line 2"].split("  ")]
                tmp[0] = " ".join(tmp[0])
                tmp[1] = " ".join(tmp[1])
                tmp[2] = " ".join(tmp[2])
                tmp[3] = " ".join(tmp[3])
                mheader["Data line 2"] = tmp
                tmp = mheader["Data line 2 (format)"].split(",1x,")
                mheader["Data line 2 (format)"] = tmp
                ## start counter for time and range gates
                counter_jj = 0
                continue  # stop the loop and continue with the next line

            tmp = hpl_files.switch(header_info, line)
            ## this temporary variable indicates whether the a given data line includes
            # the spectral width or not, so 2d information can be distinguished from
            # 1d information.
            indicator = len(line[:10].split())

            if header_info:
                try:
                    if tmp[0][0:1] == "i":
                        tmp_tmp = {"Data line 2 (format)": tmp[0]}
                    else:
                        tmp_tmp = {tmp[0]: tmp[1]}
                except:
                    if tmp[0][0] == "f":
                        tmp_tmp = {"Data line 1 (format)": tmp[0]}
                    else:
                        tmp_tmp = {"blank": "nothing"}
                mheader.update(tmp_tmp)
            elif not header_info:
                if counter_jj == 0:
                    n_o_rays = (len(filename.open().read().splitlines()) - 17) // (
                        int(mheader["Number of gates"]) + 1
                    )
                    mbeam = np.recarray(
                        (n_o_rays,),
                        dtype=np.dtype(
                            [
                                ("time", "f8"),
                                ("azimuth", "f4"),
                                ("elevation", "f4"),
                                ("pitch", "f4"),
                                ("roll", "f4"),
                            ]
                        ),
                    )
                    mdata = np.recarray(
                        (n_o_rays, int(mheader["Number of gates"])),
                        dtype=np.dtype(
                            [
                                ("range gate", "i2"),
                                ("velocity", "f4"),
                                ("snrp1", "f4"),
                                ("beta", "f4"),
                                ("dels", "f4"),
                            ]
                        ),
                    )
                    mdata[:, :] = np.full(mdata.shape, -999.0)

                # store tmp in time array
                if indicator == 1:
                    dt = np.dtype(
                        [
                            ("time", "f8"),
                            ("azimuth", "f4"),
                            ("elevation", "f4"),
                            ("pitch", "f4"),
                            ("roll", "f4"),
                        ]
                    )
                    if len(tmp) < 4:
                        tmp.extend(["-999"] * 2)
                    if counter_jj < n_o_rays:
                        mbeam[counter_jj] = np.array(tuple(tmp), dtype=dt)
                        counter_jj = counter_jj + 1
                # store tmp in range gate array
                elif indicator == 2:
                    dt = np.dtype(
                        [
                            ("range gate", "i2"),
                            ("velocity", "f4"),
                            ("snrp1", "f4"),
                            ("beta", "f4"),
                            ("dels", "f4"),
                        ]
                    )
                    ii_index = np.array(tmp[0], dtype=dt[0])

                    if len(tmp) == 4:
                        tmp.append("-999")
                        mdata[counter_jj - 1, ii_index] = np.array(tuple(tmp), dtype=dt)
                    elif len(tmp) == 5:
                        mdata[counter_jj - 1, ii_index] = np.array(tuple(tmp), dtype=dt)

    # set time information
    time_tmp = (
        pd.to_numeric(
            pd.to_timedelta(pd.DataFrame(mbeam)["time"], unit="h")
            + pd.to_datetime(
                datetime.datetime.strptime(mheader["Start time"], "%Y%m%d %H:%M:%S.%f").date()
            )
        ).values
        / 10**9
    )
    time_ds = [
        x + (datetime.timedelta(days=1)).total_seconds() if time_tmp[0] - x > 0 else x
        for x in time_tmp
    ]

    # calculate range in meters from range gate number, gate length
    range_mid = hpl_files.range_calc(mdata["range gate"][0, :], confDict)
    dr = (np.float32(confDict["PULS_DURATION"]) * 299792458 / 4).astype("f4")
    range_bnds = np.array([range_mid - dr, range_mid + dr]).T
    tgint = (2 * np.array(confDict["RANGE_GATE_LENGTH"], dtype="f4") / 299792458).astype("f4")

    SNR_tmp = np.copy(mdata["snrp1"]) - 1
    SNR_tmp[abs(SNR_tmp) <= np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    ## calculate SNR in dB
    SNR_dB = 10 * np.log10(SNR_tmp.astype(complex)).real
    ## calculate measurement uncertainty, with consensus indices
    sigma_tmp = calc_sigma_single(
        SNR_dB,
        int(mheader["Gate length (pts)"]),
        int(confDict["PULSES_PER_DIRECTION"]),
        float(mheader["Gate length (pts)"]) / tgint / 2 * float(confDict["SYSTEM_WAVELENGTH"]),
        1.316,
    )

    return xr.Dataset(
        {
            "dv": (
                ["time", "range"],
                mdata["velocity"],
                {
                    "units": "m s-1",
                    "long_name": "radial velocity of scatterers away from instrument",
                    "standard_name": "doppler_velocity",
                    "comments": "A velocity is a vector quantity; the component of the velocity of the scatterers along the line of sight of the instrument where positive implies movement away from the instrument",
                    "_FillValue": -999.0,
                    "_CoordinateAxes": "time range",
                },
            ),
            "errdv": (
                ["time", "range"],
                sigma_tmp.astype("f4"),
                {
                    "units": "m s-1",
                    "long_name": "error of Doppler velocity",
                    "standard_name": "doppler_velocity_error",
                    "comments": "error of radial velocity calculated from Cramer-Rao lower bound (CRLB)",
                    "_FillValue": -999.0,
                    "_CoordinateAxes": "time range",
                },
            ),
            "intensity": (
                ["time", "range"],
                mdata["snrp1"],
                {
                    "units": "1",
                    "long_name": "backscatter intensity: b_int = snr+1, where snr denotes the signal-to-noise-ratio",
                    "standard_name": "backscatter_intensity",
                    "comments": "backscatter intensity: b_int = snr+1",
                    "_FillValue": -999.0,
                    "_CoordinateAxes": "time range",
                },
            ),
            "beta": (
                ["time", "range"],
                mdata["beta"],
                {
                    "units": "m-1 sr-1",
                    "long_name": "attenuated backscatter coefficient",
                    "standard_name": "volume_attenuated_backwards_scattering_function_in_air",
                    "comments": "determined from SNR by means of lidar equation; uncalibrated and uncorrected",
                    "_FillValue": -999.0,
                    "_CoordinateAxes": "time range",
                },
            ),
            "delv": (
                ["time", "range"],
                mdata["dels"],
                {
                    "units": "m s-1",
                    "long_name": "spectral width of detected signal",
                    "standard_name": "spectral_width",
                    "comments": "currently not part of the standard data product",
                    "_FillValue": -999.0,
                    "_CoordinateAxes": "time range",
                },
            ),
            "azi": (
                "time",
                mbeam["azimuth"],
                {
                    "units": "degree",
                    "long_name": "sensor azimuth due reference point",
                    "standard_name": "sensor_azimuth_angle",
                    "_CoordinateAxes": "time",
                    "comments": 'sensor_azimuth_angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction. A comment attribute should be added to a data variable with this standard name to specify the reference direction. A standard name also exists for platform_azimuth_angle, where "platform" refers to the vehicle from which observations are made e.g. aeroplane, ship, or satellite. For some viewing geometries the sensor and the platform cannot be assumed to be close enough to neglect the difference in calculated azimuth angle.',
                },
            ),
            #                         , 'ele': ('time'
            #                                  , mbeam['elevation']
            #                                  , {'units' : 'degree'
            #                                    ,'long_name' : 'beam direction due elevation'
            #                                    ,'standard_name' : 'elevation_angle'
            #                                    ,'comments' : 'elevation angle of the beam to local horizone; a value of 90 is directly overhead'
            #                                    }
            #                                   )
            "zenith": (
                "time",
                90 - mbeam["elevation"],
                {
                    "units": "degree",
                    "long_name": "beam direction due zenith",
                    "standard_name": "zenith_angle",
                    "_CoordinateAxes": "time",
                    "comments": "zenith angle of the beam to the local vertical; a value of zero is directly overhead",
                },
            ),
            "lat": (
                [],
                np.float32(confDict["SYSTEM_LATITUDE"]),
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "comments": "latitude of sensor",
                    "_FillValue": -999.0,
                },
            ),
            "lon": (
                [],
                np.float32(confDict["SYSTEM_LONGITUDE"]),
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "comments": "longitude of sensor",
                    "_FillValue": -999.0,
                },
            ),
            "zsl": (
                [],
                np.float32(confDict["SYSTEM_ALTITUDE"]),
                {
                    "units": "m",
                    "comments": "system altitude above mean sea level",
                    "standard_name": "altitude",
                    "_FillValue": -999.0,
                },
            ),
            "wl": (
                [],
                np.float32(confDict["SYSTEM_WAVELENGTH"]),
                {
                    "units": "m",
                    "long_name": "laser center wavelength",
                    "standard_name": "radiation_wavelength",
                    "_FillValue": -999.0,
                },
            ),
            "pd": (
                [],
                np.float32(confDict["PULS_DURATION"]),
                {
                    "units": "seconds",
                    "long_name": "laser duration",
                    "comments": "duration of the transmitted pulse pd = 2 dr / c",
                    "_FillValue": -999.0,
                },
            ),
            "nfft": (
                [],
                np.float32(confDict["FFT_POINTS"]),
                {
                    "units": "1",
                    "long_name": "number of fft points",
                    "comments": "according to the manufacturer",
                    "_FillValue": -999.0,
                },
            ),
            #                         , 'id': ([]
            #                                 , confDict['SYSTEM_ID']
            #                                 , {'long_name': 'system identification number'}
            #                                 )
            "nrg": (
                [],
                np.float32(mheader["Number of gates"]),
                {
                    "long_name": "total number of range gates per ray",
                    "units": "1",
                    "_FillValue": -999.0,
                },
            ),
            "lrg": (
                [],
                np.float32(mheader["Range gate length (m)"]),
                {"units": "m", "long_name": "range gate length", "_FillValue": -999.0},
            ),
            "nsmpl": (
                [],
                np.float32(mheader["Gate length (pts)"]),
                {"long_name": "points per range gate", "units": "1"},
            ),
            "prf": (
                [],
                np.float32(confDict["PULS_REPETITION_FREQ"]),
                {
                    "units": "s-1",
                    "long_name": "pulse repetition frequency",
                    "_FillValue": -999.0,
                },
            ),
            "npls": (
                [],
                np.float32(confDict["PULSES_PER_DIRECTION"]),
                {"long_name": "number of pulses per ray", "units": "1", "_FillValue": -999.0},
            ),
            "focus": (
                [],
                np.float32(mheader["Focus range"]),
                {"units": "m", "long_name": "telescope focus length", "_FillValue": -999.0},
            ),
            "resv": (
                [],
                np.float32(mheader["Resolution (m/s)"]),
                {
                    "units": "m s-1",
                    "long_name": "resolution of Doppler velocity",
                    "_FillValue": -999.0,
                },
            ),
            "nqf": (
                [],
                (np.float32(mheader["Gate length (pts)"]) / tgint / 2).astype("f4"),
                {
                    "long_name": "nyquist frequency",
                    "comments": "half of the detector sampling frequency; detector bandwidth",
                },
            ),
            "nqv": (
                [],
                (
                    np.float32(mheader["Gate length (pts)"])
                    / tgint
                    / 2
                    * np.float32(confDict["SYSTEM_WAVELENGTH"])
                    / 2
                ).astype("f4"),
                {
                    "long_name": "nyquist velocity",
                    "comments": "nq_freq*lambda/2; signal bandwidth",
                },
            ),
            "smplf": (
                [],
                np.float32(mheader["Gate length (pts)"]) / tgint,
                {
                    "long_name": "sampling frequency",
                    "units": "s-1",
                    "comments": "nsmpl / tgint",
                },
            ),
            "resf": (
                [],
                (
                    np.float32(mheader["Gate length (pts)"]) / tgint / float(confDict["FFT_POINTS"])
                ).astype("f4"),
                {
                    "long_name": "frequency resolution",
                    "units": "s-1",
                    "comments": "smplf / nfft",
                },
            ),
            "tgint": (
                [],
                tgint,
                {
                    "long_name": "total observation time per range gate",
                    "units": "s",
                    "comments": "time window used for time gating the time series of the signal received on the detector: tgint = (2 X) / c, with X = range_bnds[range,1] - range_bnds[range,0]",
                },
            ),
            "range_bnds": (
                ["range", "nv"],
                range_bnds.astype("f4"),
                {"units": "m", "_FillValue": -999.0},
            ),
        },
        coords={
            "time": (
                ["time"],
                time_ds,
                {
                    "units": "seconds since 1970-01-01 00:00:00",
                    "standard_name": "time",
                    "long_name": "Time",
                    "calendar": "gregorian",
                    "_CoordinateAxisType": "time",
                },
            ),
            "range": (
                ["range"],
                range_mid.astype("f4"),
                {
                    "units": "m",
                    "long_name": "line of sight distance towards the center of each range gate",
                    "_FillValue": -999.0,
                    "_CoordinateAxisType": "range",
                },
            ),
            "nv": (["nv"], np.arange(0, 2).astype(np.int8)),
        },
    )
