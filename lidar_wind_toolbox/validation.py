import xarray as xr

from .exceptions import InputDatasetError


def require_variables(dataset: xr.Dataset, names: set[str]) -> None:
    """Raise InputDatasetError when required variables are absent."""

    missing = names - set(dataset.variables)
    if missing:
        raise InputDatasetError(f"Dataset is missing required variables: {sorted(missing)!r}")


def validate_normalized_windcube_level1(dataset: xr.Dataset) -> None:
    """Validate the normalized internal Level-1 contract for WindCube retrieval."""

    # Native WindCube VAD files use the original WindCube variable names.
    if "radial_wind_speed" in dataset.variables:
        require_variables(
            dataset,
            {
                "time",
                "radial_wind_speed",
                "azimuth",
                "elevation",
                "cnr",
                "doppler_spectrum_width",
            },
        )
        return

    # Some legacy readers produce the normalized internal variable names.
    if "dv" in dataset.variables:
        require_variables(
            dataset,
            {
                "time",
                "dv",
                "azi",
                "zenith",
                "intensity",
            },
        )
        return

    raise InputDatasetError(
        "Dataset must contain either 'radial_wind_speed' or 'dv' as the radial velocity variable."
    )
