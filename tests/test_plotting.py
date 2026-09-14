from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lidar_wind_toolbox.plotting import (
    plot_level1_backscatter_quicklook,
    plot_level2_wind_quicklook,
)


def make_level2_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "u": (("time", "height"), np.array([[3.0, 4.0]], dtype=np.float32)),
            "v": (("time", "height"), np.array([[4.0, 3.0]], dtype=np.float32)),
            "wspeed": (("time", "height"), np.array([[5.0, 5.0]], dtype=np.float32)),
            "qwind": (("time", "height"), np.array([[1, 1]], dtype=np.int8)),
        },
        coords={
            "time": np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]"),
            "height": np.array([100.0, 200.0], dtype=np.float32),
        },
    )


def make_level1_dataset() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "beta": (("time", "range"), np.array([[1e-7, 2e-7]], dtype=np.float32)),
            "azi": (("time",), np.array([0.0], dtype=np.float32)),
            "elevation": (("time",), np.array([75.0], dtype=np.float32)),
        },
        coords={
            "time": np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]"),
            "range": np.array([50.0, 100.0], dtype=np.float32),
        },
    )


def test_plot_level2_wind_quicklook_returns_figure() -> None:
    fig = plot_level2_wind_quicklook(
        make_level2_dataset(),
        day=date(2026, 1, 1),
    )

    assert fig.axes
    plt.close(fig)


def test_plot_level1_backscatter_quicklook_returns_figure() -> None:
    fig = plot_level1_backscatter_quicklook(
        make_level1_dataset(),
        day=date(2026, 1, 1),
        system="windcube",
    )

    assert fig.axes
    plt.close(fig)
