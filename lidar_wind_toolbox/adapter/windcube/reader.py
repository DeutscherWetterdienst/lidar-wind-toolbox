from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.adapter.common import (
    NormalizedRadialData,
    build_normalized_radial_dataset,
)
from lidar_wind_toolbox.adapter.range_calc import range_calc
from lidar_wind_toolbox.domain.constants import SPEED_OF_LIGHT
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


def read_wcsradial(file_content: bytes, confDict: dict[str, str]) -> xr.Dataset | Literal[False]:
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

    range_values = range_calc(ds_tmp.gate_index.data.astype(int), confDict)
    range_gate_half_width = (np.float32(confDict["PULS_DURATION"]) * SPEED_OF_LIGHT / 4).astype(
        "f4"
    )
    range_bounds = np.array(
        [range_values - range_gate_half_width, range_values + range_gate_half_width]
    ).T
    gate_time_interval = 2 * (range_bounds[0, 1] - range_bounds[0, 0]) / SPEED_OF_LIGHT
    range_coordinate_values = ds_tmp.range.data.astype("f4")

    zenith_values = np.array([90 - ds_root.sweep_fixed_angle.data[0]] * ds_tmp.time.size)
    radial_velocity_span = (ds_tmp.radial_wind_speed.max() - ds_tmp.radial_wind_speed.min()).data

    ## calculate measurement uncertainty
    sigma_tmp = calc_sigma_single(
        ds_tmp.cnr.data,
        int(confDict["NUMBER_OF_GATE_POINTS"]),
        int(confDict["PULSES_PER_DIRECTION"]),
        (radial_velocity_span / 2).astype("f4"),
        1.316,
    )
    intensity_values = (10 ** (ds_tmp.cnr.data / 10) + 1).astype("f4")
    beta_values = ds_tmp.relative_beta.data.astype("f4")
    spectral_width_values = ds_tmp.doppler_spectrum_width.data.astype("f4")
    azimuth_values = ds_tmp.azimuth.data.astype("f4")
    time_units = "seconds since {}".format(
        pd.to_datetime(time_reference).strftime("%Y-%m-%d %H:%M:%S")
    )
    velocity_resolution = (radial_velocity_span / float(confDict["FFT_POINTS"])).astype("f4")
    nyquist_frequency = (radial_velocity_span / float(confDict["SYSTEM_WAVELENGTH"])).astype("f4")
    nyquist_velocity = (radial_velocity_span / 2).astype("f4")
    sampling_frequency = np.float32(confDict["NUMBER_OF_GATE_POINTS"]) / gate_time_interval
    frequency_resolution = (sampling_frequency / float(confDict["FFT_POINTS"])).astype("f4")
    return build_normalized_radial_dataset(
        NormalizedRadialData(
            radial_velocity=ds_tmp.radial_wind_speed.data,
            radial_velocity_error=sigma_tmp.astype("f4"),
            intensity=intensity_values,
            beta=beta_values,
            spectral_width=spectral_width_values,
            azimuth=azimuth_values,
            zenith=zenith_values.astype("f4"),
            time=ds_tmp.time.data,
            time_units=time_units,
            range=range_coordinate_values,
            range_bounds=range_bounds.astype("f4"),
            lat=np.float32(confDict["SYSTEM_LATITUDE"]),
            lon=np.float32(confDict["SYSTEM_LONGITUDE"]),
            altitude=np.float32(confDict["SYSTEM_ALTITUDE"]),
            wavelength=np.float32(confDict["SYSTEM_WAVELENGTH"]),
            pulse_duration=np.float32(confDict["PULS_DURATION"]),
            number_of_fft_points=np.float32(confDict["FFT_POINTS"]),
            number_of_range_gates=np.float32(ds_tmp.dims["range"]),
            range_gate_length=np.float32(ds_tmp.range_gate_length.data),
            points_per_range_gate=np.float32(confDict["NUMBER_OF_GATE_POINTS"]),
            pulse_repetition_frequency=np.float32(confDict["PULS_REPETITION_FREQ"]),
            pulses_per_direction=np.float32(confDict["PULSES_PER_DIRECTION"]),
            focus=np.float32(confDict["FOCUS"]),
            velocity_resolution=velocity_resolution,
            nyquist_frequency=nyquist_frequency,
            nyquist_velocity=nyquist_velocity,
            sampling_frequency=sampling_frequency,
            frequency_resolution=frequency_resolution,
            gate_time_interval=gate_time_interval,
        )
    )
