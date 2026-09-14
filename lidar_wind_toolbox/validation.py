import xarray as xr

from lidar_wind_toolbox.exceptions import InputDatasetError


def require_variables(dataset: xr.Dataset, names: set[str]) -> None:
    """Raise InputDatasetError when required variables are absent."""

    missing = names - set(dataset.variables)
    if missing:
        raise InputDatasetError(f"Dataset is missing required variables: {sorted(missing)!r}")


def validate_windcube_vad_input(dataset: xr.Dataset) -> None:
    """Validate the minimum internal contract for WindCube VAD retrieval."""

    require_variables(
        dataset,
        {
            "time",
            "range",
            "azimuth",
            "elevation",
            "cnr",
            "radial_wind_speed",
            "doppler_spectrum_width",
            "range_gate_length",
        },
    )

    for name in ("cnr", "radial_wind_speed", "doppler_spectrum_width"):
        if dataset[name].dims != ("time", "range"):
            raise InputDatasetError(f"{name} must have dimensions ('time', 'range')")

    for name in ("azimuth", "elevation"):
        if dataset[name].dims != ("time",):
            raise InputDatasetError(f"{name} must have dimensions ('time',)")

    if dataset["range"].dims != ("range",):
        raise InputDatasetError("range must have dimensions ('range',)")
