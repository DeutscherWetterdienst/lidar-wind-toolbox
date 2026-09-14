import os
import tempfile
from pathlib import Path

import xarray as xr
from matplotlib.figure import Figure


def write_dataset(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".part",
        dir=path.parent,
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        comp = {"zlib": True, "complevel": 9}
        encoding = {name: comp for name in [*dataset.data_vars, *dataset.coords]}

        dataset.to_netcdf(
            tmp_path,
            unlimited_dims={"time": True},
            encoding=encoding,
        )
        os.replace(tmp_path, path)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, transparent=False, bbox_inches="tight")
