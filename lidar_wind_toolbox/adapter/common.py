"""Shared construction of normalized radial lidar datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class NormalizedRadialData:
    """Instrument-independent values required by the normalized Level-1 schema."""

    dv: np.ndarray
    errdv: np.ndarray
    intensity: np.ndarray
    beta: np.ndarray
    delv: np.ndarray
    azi: np.ndarray
    zenith: np.ndarray
    time: np.ndarray
    time_units: str
    range: np.ndarray
    range_bnds: np.ndarray
    lat: np.float32
    lon: np.float32
    zsl: np.float32
    wl: np.float32
    pd: np.float32
    nfft: np.float32
    nrg: np.float32
    lrg: np.float32
    nsmpl: np.float32
    prf: np.float32
    npls: np.float32
    focus: np.float32
    resv: np.float32
    nqf: np.float32
    nqv: np.float32
    smplf: np.float32
    resf: np.float32
    tgint: np.float32


def build_normalized_radial_dataset(data: NormalizedRadialData) -> xr.Dataset:
    """Build the common normalized dataset from instrument-specific values."""

    radial_attrs = {
        "units": "m s-1",
        "long_name": "radial velocity of scatterers away from instrument",
        "standard_name": "doppler_velocity",
        "comments": "A velocity is a vector quantity; the component of the velocity of the scatterers along the line of sight of the instrument where positive implies movement away from the instrument",
        "_FillValue": -999.0,
        "_CoordinateAxes": "time range",
    }
    error_attrs = {
        "units": "m s-1",
        "long_name": "error of Doppler velocity",
        "standard_name": "doppler_velocity_error",
        "comments": "error of radial velocity calculated from Cramer-Rao lower bound (CRLB)",
        "_FillValue": -999.0,
        "_CoordinateAxes": "time range",
    }
    intensity_attrs = {
        "units": "1",
        "long_name": "backscatter intensity: b_int = snr+1, where snr denotes the signal-to-noise-ratio",
        "standard_name": "backscatter_intensity",
        "comments": "backscatter intensity: b_int = snr+1",
        "_FillValue": -999.0,
        "_CoordinateAxes": "time range",
    }
    beta_attrs = {
        "units": "m-1 sr-1",
        "long_name": "attenuated backscatter coefficient",
        "standard_name": "volume_attenuated_backwards_scattering_function_in_air",
        "comments": "determined from SNR by means of lidar equation; uncalibrated and uncorrected",
        "_FillValue": -999.0,
        "_CoordinateAxes": "time range",
    }
    spectral_attrs = {
        "units": "m s-1",
        "long_name": "spectral width of detected signal",
        "standard_name": "spectral_width",
        "comments": "currently not part of the standard data product",
        "_FillValue": -999.0,
        "_CoordinateAxes": "time range",
    }
    azi_attrs = {
        "units": "degree",
        "long_name": "sensor azimuth due reference point",
        "standard_name": "sensor_azimuth_angle",
        "_CoordinateAxes": "time",
    }
    zenith_attrs = {
        "units": "degree",
        "long_name": "beam direction due zenith",
        "standard_name": "zenith_angle",
        "_CoordinateAxes": "time",
        "comments": "zenith angle of the beam to the local vertical; a value of zero is directly overhead",
    }
    scalar_attrs = {
        "lat": {"units": "degrees_north", "long_name": "latitude", "standard_name": "latitude", "comments": "latitude of sensor", "_FillValue": -999.0},
        "lon": {"units": "degrees_east", "long_name": "longitude", "standard_name": "longitude", "comments": "longitude of sensor", "_FillValue": -999.0},
        "zsl": {"units": "m", "comments": "system altitude above mean sea level", "standard_name": "altitude", "_FillValue": -999.0},
        "wl": {"units": "m", "long_name": "laser center wavelength", "standard_name": "radiation_wavelength", "_FillValue": -999.0},
        "pd": {"units": "seconds", "long_name": "laser duration", "comments": "duration of the transmitted pulse pd = 2 dr / c", "_FillValue": -999.0},
        "nfft": {"units": "1", "long_name": "number of fft points", "comments": "according to the manufacturer", "_FillValue": -999.0},
        "nrg": {"units": "1", "long_name": "total number of range gates per ray", "_FillValue": -999.0},
        "lrg": {"units": "m", "long_name": "range gate length", "_FillValue": -999.0},
        "nsmpl": {"units": "1", "long_name": "points per range gate"},
        "prf": {"units": "s-1", "long_name": "pulse repetition frequency", "_FillValue": -999.0},
        "npls": {"units": "1", "long_name": "number of pulses per ray", "_FillValue": -999.0},
        "focus": {"units": "m", "long_name": "telescope focus length", "_FillValue": -999.0},
        "resv": {"units": "m s-1", "long_name": "resolution of Doppler velocity", "_FillValue": -999.0},
        "nqf": {"long_name": "nyquist frequency", "comments": "half of the detector sampling frequency; detector bandwidth"},
        "nqv": {"long_name": "nyquist velocity", "comments": "nq_freq*lambda/2; signal bandwidth"},
        "smplf": {"units": "s-1", "long_name": "sampling frequency", "comments": "nsmpl / tgint"},
        "resf": {"units": "s-1", "long_name": "frequency resolution", "comments": "smplf / nfft"},
        "tgint": {"units": "s", "long_name": "total observation time per range gate", "comments": "time window used for time gating the time series of the signal received on the detector: tgint = (2 X) / c, with X = range_bnds[range,1] - range_bnds[range,0]"},
    }

    dataset = xr.Dataset(
        {
            "dv": (["time", "range"], data.dv, radial_attrs),
            "errdv": (["time", "range"], data.errdv, error_attrs),
            "intensity": (["time", "range"], data.intensity, intensity_attrs),
            "beta": (["time", "range"], data.beta, beta_attrs),
            "delv": (["time", "range"], data.delv, spectral_attrs),
            "azi": (["time"], data.azi, azi_attrs),
            "zenith": (["time"], data.zenith, zenith_attrs),
            "lat": ([], data.lat, scalar_attrs["lat"]),
            "lon": ([], data.lon, scalar_attrs["lon"]),
            "zsl": ([], data.zsl, scalar_attrs["zsl"]),
            "wl": ([], data.wl, scalar_attrs["wl"]),
            "pd": ([], data.pd, scalar_attrs["pd"]),
            "nfft": ([], data.nfft, scalar_attrs["nfft"]),
            "nrg": ([], data.nrg, scalar_attrs["nrg"]),
            "lrg": ([], data.lrg, scalar_attrs["lrg"]),
            "nsmpl": ([], data.nsmpl, scalar_attrs["nsmpl"]),
            "prf": ([], data.prf, scalar_attrs["prf"]),
            "npls": ([], data.npls, scalar_attrs["npls"]),
            "focus": ([], data.focus, scalar_attrs["focus"]),
            "resv": ([], data.resv, scalar_attrs["resv"]),
            "nqf": ([], data.nqf, scalar_attrs["nqf"]),
            "nqv": ([], data.nqv, scalar_attrs["nqv"]),
            "smplf": ([], data.smplf, scalar_attrs["smplf"]),
            "resf": ([], data.resf, scalar_attrs["resf"]),
            "tgint": ([], data.tgint, scalar_attrs["tgint"]),
            "range_bnds": (["range", "nv"], data.range_bnds, {"units": "m", "_FillValue": -999.0}),
        },
        coords={
            "time": (["time"], data.time, {"units": data.time_units, "standard_name": "time", "long_name": "Time", "calendar": "gregorian", "_CoordinateAxisType": "time"}),
            "range": (["range"], data.range, {"units": "m", "long_name": "line of sight distance towards the center of each range gate", "_FillValue": -999.0, "_CoordinateAxisType": "range"}),
            "nv": (["nv"], np.arange(0, 2).astype(np.int8)),
        },
    )
    return dataset
