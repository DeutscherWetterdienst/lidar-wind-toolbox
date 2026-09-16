import xarray as xr

from .exceptions import InputDatasetError


def require_variables(dataset: xr.Dataset, names: set[str]) -> None:
    """Raise InputDatasetError when required variables are absent."""

    missing = names - set(dataset.variables)
    if missing:
        raise InputDatasetError(f"Dataset is missing required variables: {sorted(missing)!r}")


def validate_normalized_windcube_level1(dataset: xr.Dataset) -> None:
    """Validate the WindCube Level-1 dataset contract for VAD retrieval."""

    require_variables(
        dataset,
        {
            "time",
            "radial_wind_speed",
            "azimuth",
            "elevation",
            "cnr",
        },
    )

    if dataset["radial_wind_speed"].ndim != 2:
        raise InputDatasetError("radial_wind_speed must be a 2D array (time, gates)")

    for name in ("azimuth", "elevation"):
        if dataset[name].ndim != 1:
            raise InputDatasetError(f"{name} must be a 1D array along time")
