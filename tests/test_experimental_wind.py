"""Numerical coverage for the wind capabilities imported from experimental."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.main_proc import process_dataset
from lidar_wind_toolbox.retrieval.experimental import lvl2vad_standard, lvl2wcdbs
from lidar_wind_toolbox.retrieval.geometry import build_Amatrix


@pytest.mark.parametrize("system", ["halo", "windcube"])
@pytest.mark.parametrize("exclude", [False, True])
def test_wind_retrieval_with_interleaved_vertical_beams(system, exclude):
    # Four slanted directions plus a vertical stare in each scan cycle.
    azimuth = np.tile([0.0, 90.0, 180.0, 270.0, 0.0], 8)
    elevation = np.tile([60.0, 60.0, 60.0, 60.0, 90.0], 8)
    velocity = build_Amatrix(azimuth + 20, elevation) @ np.array([3.0, 2.0, 0.0])
    # The second gate expands the observed velocity span used for WindCube's B.
    radial = velocity[:, None] * np.array([1.0, 2.0])
    times = np.datetime64("2024-07-20") + np.arange(len(azimuth)) * np.timedelta64(5, "s")
    ds = xr.Dataset(
        {
            "dv": (("time", "range"), radial),
            "intensity": (("time", "range"), np.full(radial.shape, 2.0)),
            "beta": (("time", "range"), np.full(radial.shape, 1e-6)),
            "azi": ("time", azimuth),
            "zenith": ("time", 90 - elevation),
            "prf": ((), 10000.0),
            "nsmpl": ((), 10.0),
            "nqv": ((), 20.0),
            "range_bnds": (("range", "nv"), [[75.0, 125.0], [175.0, 225.0]]),
        },
        coords={"time": times, "range": [100.0, 200.0]},
    )
    cfg = {
        "NUMBER_OF_DIRECTIONS": "5",
        "NUMBER_OF_GATES": "2",
        "SCAN_TYPE": "DBS",
        "AVG_MIN": "30",
        "AZIMUTH_OFFSET": "20",
        "N_HORI_DIRECTIONS": "4",
        "CNS_RANGE": "3",
        "CNS_PERCENTAGE": "60",
        "SNR_THRESHOLD": "0",
        "R2_THRESHOLD": "0.95",
        "CN_THRESHOLD": "100",
        "N_VRAD_THRESHOLD": "4",
        "SYSTEM_LATITUDE": "52",
        "SYSTEM_LONGITUDE": "14",
        "SYSTEM_ALTITUDE": "100",
        "SYSTEM_WAVELENGTH": "1.5e-6",
    }
    if exclude:
        cfg["EX_AZI"] = "110"  # Exclusions apply after the bearing correction.
    if system == "windcube":
        ds = ds.rename({"dv": "radial_wind_speed", "azi": "azimuth", "beta": "relative_beta"})
        ds["elevation"] = ("time", elevation)
        ds["cnr"] = (("time", "range"), np.zeros(radial.shape))
        ds["doppler_spectrum_width"] = (("time", "range"), np.full(radial.shape, 0.5))
        ds["range_gate_length"] = 50.0
        ds = ds.assign_coords(time=times.astype("datetime64[s]").astype(float))
        retrieve = lvl2wcdbs
    else:
        retrieve = lvl2vad_standard
    result = retrieve(ds, datetime(2024, 7, 20), cfg)
    first = result.isel(time=0, height=0)
    np.testing.assert_allclose([first.u, first.v, first.w], [3, 2, 0], atol=1e-4)
    assert first.qwind == 1
    assert first.nvrad == (4 if exclude else 5)
    assert first.binary_dir == (15 if exclude else 31)
    np.testing.assert_allclose([first.snr, first.snr_min, first.snr_max], 1.0)
    assert np.isfinite(first.erru)
    assert (result.qwind.isel(time=slice(1, None)) == 0).all()
    assert (result.binary_dir.isel(time=slice(1, None)) == -999).all()
    assert (result.height_bnds[:, 1] > result.height_bnds[:, 0]).all()
    cfg["BLINDZONE_GATES"] = "1"
    trimmed = retrieve(ds, datetime(2024, 7, 20), cfg)
    np.testing.assert_equal(trimmed.height, result.height[1:])


def test_general_halo_stare_processing_loads_wind_support(tmp_path, monkeypatch):
    wind_path = tmp_path / "2024" / "202407" / "wind_20240720.nc"
    wind_path.parent.mkdir(parents=True)
    xr.Dataset({"wspeed": ("time", [4.0])}).to_netcdf(wind_path)
    stare = xr.Dataset(coords={"range": [100.0]})
    captured = []

    def retrieve(ds_stare, ds_wind, date, cfg):
        assert ds_stare is stare
        captured.append(ds_wind.wspeed.item())
        return xr.Dataset()

    monkeypatch.setattr("lidar_wind_toolbox.main_proc.process_stare_l2_from_ds", retrieve)
    cfg = {
        "SYSTEM": "halo",
        "SCAN_TYPE": "stare",
        "NC_WIND_L2_PATH": str(tmp_path),
        "NC_WIND_L2_BASENAME": "wind_",
        **{
            key: "test"
            for key in [
                "NC_TITLE",
                "NC_INSTITUTION",
                "NC_SITE_LOCATION",
                "NC_SOURCE",
                "NC_INSTRUMENT_TYPE",
                "NC_INSTRUMENT_MODE",
                "NC_INSTRUMENT_CONTACT",
                "NC_CONVENTIONS",
                "NC_DATA_POLICY",
                "NC_HISTORY",
                "VERSION",
                "NC_COMMENTS",
            ]
        },
    }
    process_dataset(stare, datetime(2024, 7, 20), cfg)
    assert captured == [4.0]


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize(
    "system,scan_type,entry_point",
    [
        ("halo", "vad", "lvl2vad_standard"),
        ("windcube", "fixed_vad", "lvl2vad_standard"),
        ("windcube", "vad", "lvl2wcdbs"),
    ],
)
def test_processing_selects_experimental_wind_only_when_enabled(
    monkeypatch, experimental, system, scan_type, entry_point
):
    def fail_with_label(label):
        def retrieve(*args):
            raise RuntimeError(label)

        return retrieve

    for name in ("lvl2vad_standard", "lvl2wcdbs"):
        monkeypatch.setattr(
            f"lidar_wind_toolbox.main_proc.{name}", fail_with_label(f"original:{name}")
        )
        monkeypatch.setattr(
            f"lidar_wind_toolbox.main_proc.experimental.{name}",
            fail_with_label(f"experimental:{name}"),
        )
    cfg = {"SYSTEM": system, "SCAN_TYPE": scan_type}
    if experimental:
        cfg["EXPERIMENTAL_WIND_RETRIEVAL"] = "1"
    expected = f"{'experimental' if experimental else 'original'}:{entry_point}"
    with pytest.raises(RuntimeError, match=expected):
        process_dataset(xr.Dataset(coords={"range": [100.0]}), datetime(2024, 7, 20), cfg)
