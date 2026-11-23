"""Thin xarray adapters for the NumPy retrieval domain model."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from lidar_wind_toolbox.domain.models import (
    LidarScan,
    RetrievalParameters,
    WindRetrievalResult,
)


def halo_scan(dataset: xr.Dataset) -> LidarScan:
    elevation = 90 - np.asarray(dataset["zenith"].data)
    radial = np.asarray(dataset["dv"].data)
    return LidarScan(
        time=np.asarray(dataset["time"].data),
        azimuth=np.asarray(dataset["azi"].data),
        elevation=elevation,
        radial_velocity=radial,
        snr=np.asarray(dataset["intensity"].data) - 1,
        beta=np.asarray(dataset["beta"].data),
        range=np.asarray(dataset["range"].data),
        range_bounds=np.asarray(dataset["range_bnds"].data).copy(),
        spectral_width=(np.asarray(dataset["delv"].data) if "delv" in dataset else None),
        prf=float(np.asarray(dataset["prf"].data)),
        samples_per_gate=float(np.asarray(dataset["nsmpl"].data)),
        bandwidth=float(np.asarray(dataset["nqv"].data)),
    )


def windcube_scan(dataset: xr.Dataset) -> LidarScan:
    radial = np.asarray(dataset["radial_wind_speed"].data)
    range_values = np.asarray(dataset["range"].data)
    gate_length = float(np.asarray(dataset["range_gate_length"].data))
    if "measurement_height" in dataset:
        height = np.asarray(dataset["measurement_height"].data).reshape(-1)
        height = np.broadcast_to(height[0], range_values.shape).copy()
    else:
        height = np.sin(np.deg2rad(np.asarray(dataset["elevation"].data)))
        height = np.nanmedian(height) * range_values
    return LidarScan(
        time=np.asarray(dataset["time"].data).astype("datetime64[s]"),
        azimuth=np.asarray(dataset["azimuth"].data),
        elevation=np.asarray(dataset["elevation"].data),
        radial_velocity=radial,
        snr=10 ** (np.asarray(dataset["cnr"].data) / 10),
        beta=(
            np.asarray(dataset["relative_beta"].data)
            if "relative_beta" in dataset
            else np.ones_like(radial)
        ),
        range=range_values,
        range_bounds=np.column_stack((height - gate_length / 2, height + gate_length / 2)),
        spectral_width=(
            np.asarray(dataset["doppler_spectrum_width"].data)
            if "doppler_spectrum_width" in dataset
            else None
        ),
        gate_length=gate_length,
        height=height,
    )


def result_to_xarray(
    result: WindRetrievalResult,
    config: dict[str, str],
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    altitude: float | None = None,
    wavelength: float | None = None,
) -> xr.Dataset:
    if latitude is None and "SYSTEM_LATITUDE" in config:
        latitude = float(config["SYSTEM_LATITUDE"])
    if longitude is None and "SYSTEM_LONGITUDE" in config:
        longitude = float(config["SYSTEM_LONGITUDE"])
    if altitude is None and "SYSTEM_ALTITUDE" in config:
        altitude = float(config["SYSTEM_ALTITUDE"])
    if wavelength is None and "SYSTEM_WAVELENGTH" in config:
        wavelength = float(config["SYSTEM_WAVELENGTH"])
    n = int(config.get("BLINDZONE_GATES", 0))
    sl = slice(n, None)
    quality = result.quality[:, sl]
    component_quality = result.component_quality[:, sl]
    values = result.wind[:, sl]
    uncertainties = result.wind_uncertainty[:, sl]

    def numeric(value: np.ndarray, fill: float = -999.0) -> np.ndarray:
        out = np.asarray(value, dtype=np.float32).copy()
        out[~np.isfinite(out)] = fill
        return out

    def flag(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.int8)

    data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "wspeed": (("time", "height"), numeric(result.speed[:, sl])),
        "errwspeed": (("time", "height"), numeric(result.speed_uncertainty[:, sl])),
        "wdir": (("time", "height"), numeric(result.direction[:, sl])),
        "errwdir": (("time", "height"), numeric(result.direction_uncertainty[:, sl])),
        "u": (("time", "height"), numeric(values[..., 0])),
        "erru": (("time", "height"), numeric(uncertainties[..., 0])),
        "v": (("time", "height"), numeric(values[..., 1])),
        "errv": (("time", "height"), numeric(uncertainties[..., 1])),
        "w": (("time", "height"), numeric(values[..., 2])),
        "errw": (("time", "height"), numeric(uncertainties[..., 2])),
        "r2": (("time", "height"), numeric(result.r2[:, sl])),
        "cn": (("time", "height"), numeric(result.condition_number[:, sl])),
        "nvrad": (("time", "height"), numeric(result.radial_velocity_count[:, sl])),
        "qwind": (("time", "height"), flag(quality)),
        "qu": (("time", "height"), flag(component_quality[..., 0])),
        "qv": (("time", "height"), flag(component_quality[..., 1])),
        "qw": (("time", "height"), flag(component_quality[..., 2])),
        "binary_dir": (("time", "height"), numeric(result.binary_direction[:, sl])),
        "snr": (("time", "height"), numeric(result.snr[:, sl])),
        "snr_min": (("time", "height"), numeric(result.snr_min[:, sl])),
        "snr_max": (("time", "height"), numeric(result.snr_max[:, sl])),
        "beta": (("time", "height"), numeric(result.beta[:, sl])),
        "spec_w": (("time", "height"), numeric(result.spectral_width[:, sl])),
        "time_bnds": (("time", "nv"), result.time_bounds.astype(np.float64)),
        "height_bnds": (("height", "nv"), result.height_bounds[n:].astype(np.float32)),
    }
    coords = {
        "time": ("time", result.time.astype(np.float64)),
        "height": ("height", result.height[sl].astype(np.float32)),
        "nv": ("nv", np.arange(2, dtype=np.int8)),
    }
    dataset = xr.Dataset(data, coords=coords)
    dataset["config"] = xr.DataArray(
        "".join(f"{key}={value}\n" for key, value in config.items()),
        attrs={"standard_name": "configuration_file"},
    )
    dataset["time"].attrs.update(
        {"units": "seconds since 1970-01-01 00:00:00", "bounds": "time_bnds"}
    )
    dataset["height"].attrs.update(
        {"units": "m", "bounds": "height_bnds", "standard_name": "height"}
    )
    for name in ("wspeed", "u", "v", "w"):
        dataset[name].attrs["_FillValue"] = -999.0
    if latitude is not None:
        dataset["lat"] = xr.DataArray(np.float32(latitude))
    if longitude is not None:
        dataset["lon"] = xr.DataArray(np.float32(longitude))
    if altitude is not None:
        dataset["zsl"] = xr.DataArray(np.float32(altitude))
    if wavelength is not None:
        dataset["frequency"] = xr.DataArray(np.float32(299792458 / wavelength))
    dataset["hor_width"] = xr.DataArray(
        result.horizontal_width[n:].astype(np.float32), dims=("height",)
    )
    dataset["vert_res"] = xr.DataArray(np.float32(np.nanmean(np.diff(result.height))))
    dataset.attrs["File_Configuration"] = dataset["config"].item()
    return dataset

