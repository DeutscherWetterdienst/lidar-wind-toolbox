from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lidar_wind_toolbox.io import save_figure, write_dataset


def test_write_dataset_writes_netcdf(tmp_path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "x": (("time",), np.array([1.0], dtype=np.float32)),
        },
        coords={
            "time": np.array([1.0], dtype=np.float64),
        },
    )

    path = tmp_path / "result.nc"
    write_dataset(dataset, path)

    assert path.exists()
    assert not list(tmp_path.glob("*.part"))

    written = xr.open_dataset(path)
    try:
        assert "x" in written.data_vars
    finally:
        written.close()


def test_save_figure_writes_file(tmp_path: Path) -> None:
    fig = plt.figure()
    path = tmp_path / "plot.png"

    try:
        save_figure(fig, path)
        assert path.exists()
    finally:
        plt.close(fig)
