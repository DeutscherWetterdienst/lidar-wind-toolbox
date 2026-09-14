import xarray as xr

from .exceptions import InputDatasetError


def require_variables(dataset: xr.Dataset, names: set[str]) -> None:
    """Raise InputDatasetError when required variables are absent."""

    missing = names - set(dataset.variables)
    if missing:
        raise InputDatasetError(f"Dataset is missing required variables: {sorted(missing)!r}")


def validate_normalized_windcube_level1(dataset: xr.Dataset) -> None:
    """Validate the normalized internal Level-1 contract for WindCube retrieval."""

    require_variables(
        dataset,
        {
            "time",
            "range",
            "dv",
            "intensity",
            "beta",
            "azi",
            "zenith",
            "nsmpl",
            "prf",
            "nqv",
            "range_bnds",
        },
    )

    for name in ("dv", "intensity", "beta"):
        if dataset[name].dims != ("time", "range"):
            raise InputDatasetError(f"{name} must have dimensions ('time', 'range')")

    if "delv" in dataset.variables and dataset["delv"].dims != ("time", "range"):
        raise InputDatasetError("delv must have dimensions ('time', 'range')")

    for name in ("azi", "zenith"):
        if dataset[name].dims != ("time",):
            raise InputDatasetError(f"{name} must have dimensions ('time',)")

    if dataset["range"].dims != ("range",):
        raise InputDatasetError("range must have dimensions ('range',)")

    if dataset["range_bnds"].dims != ("range", "nv"):
        raise InputDatasetError("range_bnds must have dimensions ('range', 'nv')")
