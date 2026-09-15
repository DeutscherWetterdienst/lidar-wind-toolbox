from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .plot_helpers import ql_helper

ColorKey = Literal["red", "green", "blue", "alpha"]


def _cmap_discretize(cmap: str | mcolors.Colormap, n: int) -> mcolors.Colormap:
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors_i = np.concatenate((np.linspace(0, 1.0, n), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, n + 1)

    cdict: dict[ColorKey, Sequence[tuple[float, ...]]] = {}
    keys: tuple[ColorKey, ...] = ("red", "green", "blue")

    for ki, key in enumerate(keys):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(n + 1)
        ]

    return mcolors.LinearSegmentedColormap(cmap.name + f"_{n}", cdict, 1024)


def plot_level2_wind_quicklook(
    dataset: xr.Dataset,
    *,
    day: date,
    utc_offset_hours: int = 0,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    fig.set_facecolor("w")
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    x_mesh, y_mesh = np.meshgrid(dataset.time.data, dataset.height.data)
    u = np.copy(dataset.u.data)

    if np.all(np.isnan(u)):
        return fig

    v = np.copy(dataset.v.data)
    wind_speed = np.copy(dataset.wspeed.data)
    qwind = np.copy(dataset.qwind.data)

    qwind = qwind * (
        np.sqrt(u**2 + v**2, out=np.zeros_like(u), where=~np.isnan(u**2 + v**2)) >= 2.5
    )
    mask = qwind < 1

    masked_u = np.ma.masked_where(mask, u)
    masked_v = np.ma.masked_where(mask, v)
    masked_ws = np.ma.masked_where(mask, wind_speed)

    flattened_ws = masked_ws.filled(np.nan).flatten()
    finite_ws = flattened_ws[np.isfinite(flattened_ws)]

    if finite_ws.size == 0:
        return fig

    wsmax = max(np.round(np.nanpercentile(finite_ws, 95), -1) + 10, 10)
    palette = plt.get_cmap(_cmap_discretize(cm.jet, int(wsmax))).with_extremes(under="white")

    day_start = datetime(day.year, day.month, day.day) - timedelta(hours=utc_offset_hours)
    day_end = day_start + timedelta(days=1)

    barbs = ax.barbs(
        x_mesh.T,
        y_mesh.T,
        masked_u,
        masked_v,
        masked_ws,
        clim=[0, wsmax],
        pivot="middle",
        barb_increments=dict(half=2.5, full=5, flag=25),
        sizes=dict(emptybarb=0.25, spacing=0.1, height=0.5, width=0.3),
        cmap=palette,
    )

    ax.set_xlim(mdates.date2num(day_start), mdates.date2num(day_end))
    ax.set_aspect("auto")

    cticks = np.linspace(0, wsmax, int(wsmax / 5 + 1))
    cbar = fig.colorbar(barbs, ax=ax, extend="both", pad=0.02, ticks=cticks)
    cbar.set_label(
        r"$\rm{wind\;speed}\;/\;\rm{m}\,\rm{s}^{-1}$",
        rotation=270,
        fontsize=22,
        labelpad=30,
    )
    cbar.ax.tick_params(labelsize=18, length=0, width=2, direction="in")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - utc_offset_hours, 24 - utc_offset_hours, 6), 24))
    )
    ax.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - utc_offset_hours, 24 - utc_offset_hours, 1), 24))
    )
    ax.set_xlabel(day.strftime("%Y-%m-%d") + "\n" + "time (UTC)", fontsize=22)
    ax.set_ylabel(r"$\rm{height}\;/\;\rm{m}$", fontsize=22)

    valid_height_indices = np.unravel_index(qwind == 1, wind_speed.shape)
    if np.any(np.sum(valid_height_indices[1], axis=0).astype(bool)):
        hmax = dataset.height.data[np.sum(valid_height_indices[1], axis=0).astype(bool)][-1]
    else:
        hmax = float(dataset.height.data[-1])

    if hmax >= 6000:
        delmajor = 2000
        delnom = 4
    elif (hmax > 2000) * (hmax < 6000):
        delmajor = 1000
        delnom = 4
    else:
        delmajor = 500
        delnom = 5

    ax.yaxis.set_major_locator(MultipleLocator(delmajor))
    ax.yaxis.set_minor_locator(MultipleLocator(delmajor // delnom))
    ax.set_ylim((0.0, float(np.round(hmax, -2) + 100)))

    qwind_low = np.copy(dataset.qwind.data)
    qwind_low = qwind_low * (
        np.sqrt(
            u**2 + v**2,
            out=999.0 * np.ones_like(u),
            where=~np.isnan(u**2 + v**2),
        )
        < 2.5
    )
    low_mask = qwind_low < 1

    masked_u_low = np.ma.masked_where(low_mask, np.copy(dataset.u.data))
    masked_v_low = np.ma.masked_where(low_mask, np.copy(dataset.v.data))
    masked_ws_low = np.ma.masked_where(low_mask, np.copy(dataset.wspeed.data))

    ax.barbs(
        x_mesh.T,
        y_mesh.T,
        masked_u_low,
        masked_v_low,
        masked_ws_low,
        clim=[0, wsmax],
        rounding=False,
        pivot="middle",
        barb_increments=dict(half=0.25, full=5, flag=25),
        sizes=dict(emptybarb=0.25, spacing=0.1, height=0.0, width=0.0),
        cmap=palette,
    )

    ax.tick_params(
        axis="both",
        labelsize=18,
        length=34,
        width=2.0,
        pad=7.78,
        which="major",
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        labelsize=18,
        length=23,
        width=1.0,
        which="minor",
        direction="in",
        top=True,
        right=True,
    )

    return fig


def plot_level1_backscatter_quicklook(
    dataset: xr.Dataset,
    *,
    day: date,
    system: str,
    utc_offset_hours: int = 0,
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    fig.set_facecolor("w")
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plot_dataset = dataset.copy()
    if "zenith" not in plot_dataset.variables and "elevation" in plot_dataset.variables:
        plot_dataset["zenith"] = 90 - plot_dataset["elevation"]

    config_dict = {"SYSTEM": system}
    beta_max, time_mean, range_vec, elevation, vmin, vmax = ql_helper(plot_dataset, config_dict)

    x_mesh, y_mesh = np.meshgrid(
        mdates.date2num(pd.to_datetime(time_mean)),
        range_vec * np.sin(np.pi / 180 * elevation.mean()),
    )
    z_values = np.copy(beta_max)

    if np.all(np.isnan(z_values)):
        return fig

    mask = np.isnan(z_values)
    masked_z = np.ma.masked_where(mask, z_values)

    # Determine whether true backscatter data exists BEFORE ql_helper mutates the dataset
    has_beta = ("relative_beta" in plot_dataset.variables) or ("beta" in plot_dataset.variables)

    config_dict = {"SYSTEM": system}
    beta_max, time_mean, range_vec, elevation, vmin, vmax = ql_helper(plot_dataset, config_dict)

    x_mesh, y_mesh = np.meshgrid(
        mdates.date2num(pd.to_datetime(time_mean)),
        range_vec * np.sin(np.pi / 180 * elevation.mean()),
    )
    z_values = np.copy(beta_max)

    if np.all(np.isnan(z_values)):
        return fig

    mask = np.isnan(z_values)
    masked_z = np.ma.masked_where(mask, z_values)

    # Use LogNorm only for positive physical backscatter; linear norm for CNR in dB (vmin <= 0)
    if has_beta and vmin > 0:
        color_mesh = ax.pcolormesh(
            x_mesh.T,
            y_mesh.T,
            masked_z,
            cmap=cm.gnuplot2,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        )
        cbar_label = r"$\rm{attenuated\;backscatter}\;/\;\rm{m}^{-1}\,\rm{sr}^{-1}$"
    else:
        color_mesh = ax.pcolormesh(
            x_mesh.T,
            y_mesh.T,
            masked_z,
            cmap=cm.gnuplot2,
            vmin=vmin,
            vmax=vmax,
        )
        cbar_label = r"$\rm{CNR}\;/\;\rm{dB}$"

    cbar = fig.colorbar(color_mesh, ax=ax, extend="both", pad=0.01)
    cbar.set_label(cbar_label, rotation=270, fontsize=22, labelpad=37)
    cbar.ax.tick_params(which="major", direction="out", length=14, width=2, labelsize=22)
    cbar.ax.tick_params(which="minor", direction="out", length=8, width=2, labelsize=22)

    day_start = datetime(day.year, day.month, day.day) - timedelta(hours=utc_offset_hours)
    day_end = day_start + timedelta(days=1)

    ax.set_xlabel(day.strftime("%Y-%m-%d") + "\n" + "time (UTC)", fontsize=22)
    ax.set_ylabel(r"$\rm{height}\;/\;\rm{m}$", fontsize=22)
    ax.set_xlim(mdates.date2num(day_start), mdates.date2num(day_end))
    ax.set_aspect("auto")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - utc_offset_hours, 24 - utc_offset_hours, 6), 24))
    )
    ax.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - utc_offset_hours, 24 - utc_offset_hours, 1), 24))
    )

    hmax = range_vec[-1] * np.sin(np.pi / 180 * elevation.mean())
    if hmax >= 6000:
        delmajor = 2000
        delnom = 4
    elif (hmax > 2000) * (hmax < 6000):
        delmajor = 1000
        delnom = 4
    else:
        delmajor = 500
        delnom = 5

    ax.yaxis.set_major_locator(MultipleLocator(delmajor))
    ax.yaxis.set_minor_locator(MultipleLocator(delmajor // delnom))
    ax.set_ylim((0.0, float(np.round(hmax, -2) + 100)))
    ax.tick_params(
        axis="both",
        labelsize=18,
        length=34,
        width=2.0,
        pad=7.78,
        which="major",
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        labelsize=18,
        length=23,
        width=1.0,
        which="minor",
        direction="in",
        top=True,
        right=True,
    )

    return fig
