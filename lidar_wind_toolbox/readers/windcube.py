import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.hpl_files import hpl_files
from lidar_wind_toolbox.retrieval.uncertainty import calc_sigma_single


def read_wc_type(filename):
    while True:
        if not filename.exists():
            print("Oops, no such file or directory '{}'".format(filename))
            break
        else:
            print("reading file '{}'".format(filename))
            try:
                ds_root = xr.open_dataset(filename)
            except OSError:
                print("corrupted netCDF: '{}'".format(filename))
                return False
            sweep_list = list(ds_root.sweep_group_name.data)
            ds_ind = xr.concat(
                (
                    xr.open_dataset(filename, group=sweep_ii, decode_times=False)
                    for sweep_ii in sweep_list
                ),
                dim="time",
                data_vars="minimal",
                compat="override",
                coords="minimal",
            )
        return ds_ind


def read_wcsradial(filename, confDict: dict[str, str]):
    while True:
        if not filename.exists():
            print("Oops, no such file or directory '{}'".format(filename))
            break
        else:
            print("reading file '{}'".format(filename))

        ## the windcube netCDF l1-files are in cf-radial format
        # this requires a workaround to open to access the radial data,
        # when relying on xarray package alone

        # read root attributes
        try:
            ds_root = xr.open_dataset(filename)
        except OSError:
            print("corrupted netCDF: '{}'".format(filename))
            return False

        if "time_reference" in list(ds_root.keys()):
            time_reference = ds_root.time_reference.data
        else:
            time_reference = None
        sweep_list = list(ds_root.sweep_group_name.data)
        # read radial data in sweep group
        ds_tmp = xr.concat(
            (
                xr.open_dataset(filename, group=sweep_ii, decode_times=False)
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
        dr = (np.float32(confDict["PULS_DURATION"]) * 299792458 / 4).astype("f4")
        range_bnds = np.array([range_mid - dr, range_mid + dr]).T
        tgint = 2 * (range_bnds[0, 1] - range_bnds[0, 0]) / 299792458

        zenith = np.array([90 - ds_root.sweep_fixed_angle.data[0]] * ds_tmp.time.size)

        ## calculate measurement uncertainty
        sigma_tmp = calc_sigma_single(
            ds_tmp.cnr.data,
            int(confDict["NUMBER_OF_GATE_POINTS"]),
            int(confDict["PULSES_PER_DIRECTION"]),
            ((ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data / 2).astype(
                "f4"
            ),
            1.316,
        )
        return xr.Dataset(
            {
                "dv": (
                    ["time", "range"],
                    ds_tmp.radial_wind_speed.data,
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
                    (10 ** (ds_tmp.cnr.data / 10) + 1).astype("f4"),
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
                    ds_tmp.relative_beta.data.astype("f4"),
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
                    ds_tmp.doppler_spectrum_width.data.astype("f4"),
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
                    ds_tmp.azimuth.data.astype("f4"),
                    {
                        "units": "degree",
                        "long_name": "sensor azimuth due reference point",
                        "standard_name": "sensor_azimuth_angle",
                        "_CoordinateAxes": "time",
                        "comments": 'sensor_azimuth_angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction. A comment attribute should be added to a data variable with this standard name to specify the reference direction. A standard name also exists for platform_azimuth_angle, where "platform" refers to the vehicle from which observations are made e.g. aeroplane, ship, or satellite. For some viewing geometries the sensor and the platform cannot be assumed to be close enough to neglect the difference in calculated azimuth angle.',
                    },
                ),
                "zenith": (
                    "time",
                    zenith.astype("f4"),
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
                "nrg": (
                    [],
                    np.float32(ds_tmp.dims["range"]),
                    {
                        "long_name": "total number of range gates per ray",
                        "units": "1",
                        "_FillValue": -999.0,
                    },
                ),
                "lrg": (
                    [],
                    np.float32(ds_tmp.range_gate_length.data),
                    {"units": "m", "long_name": "range gate length", "_FillValue": -999.0},
                ),
                "nsmpl": (
                    [],
                    np.float32(confDict["NUMBER_OF_GATE_POINTS"]),
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
                    {
                        "long_name": "number of pulses per ray",
                        "units": "1",
                        "_FillValue": -999.0,
                    },
                ),
                "focus": (
                    [],
                    np.float32(confDict["FOCUS"]),
                    {"units": "m", "long_name": "telescope focus length", "_FillValue": -999.0},
                ),
                "resv": (
                    [],
                    (
                        (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data
                        / float(confDict["FFT_POINTS"])
                    ).astype("f4"),
                    {
                        "units": "m s-1",
                        "long_name": "resolution of Doppler velocity",
                        "_FillValue": -999.0,
                    },
                ),
                "nqf": (
                    [],
                    (
                        (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data
                        / float(confDict["SYSTEM_WAVELENGTH"])
                    ).astype("f4"),
                    {
                        "long_name": "nyquist frequency",
                        "comments": "half of the detector sampling frequency; detector bandwidth",
                    },
                ),
                "nqv": (
                    [],
                    (
                        (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data / 2
                    ).astype("f4"),
                    {
                        "long_name": "nyquist velocity",
                        "comments": "nq_freq*lambda/2; signal bandwidth",
                    },
                ),
                "smplf": (
                    [],
                    np.float32(confDict["NUMBER_OF_GATE_POINTS"]) / tgint,
                    {
                        "long_name": "sampling frequency",
                        "units": "s-1",
                        "comments": "nsmpl / tgint",
                    },
                ),
                "resf": (
                    [],
                    (
                        np.float32(confDict["NUMBER_OF_GATE_POINTS"])
                        / tgint
                        / float(confDict["FFT_POINTS"])
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
                    ds_tmp.time.data,
                    {
                        "units": "seconds since {}".format(
                            pd.to_datetime(time_reference).strftime("%Y-%m-%d %H:%M:%S")
                        ),
                        "standard_name": "time",
                        "long_name": "Time",
                        "calendar": "gregorian",
                        "_CoordinateAxisType": "time",
                    },
                ),
                "range": (
                    ["range"],
                    ds_tmp.range.data.astype("f4"),
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
