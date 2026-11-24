import numpy as np

from lidar_wind_toolbox.adapter.common import (
    NormalizedRadialData,
    build_normalized_radial_dataset,
)


def test_build_normalized_radial_dataset_uses_common_level1_schema() -> None:
    time = np.array([1.0, 2.0])
    radial = np.ones((2, 3), dtype=np.float32)
    data = NormalizedRadialData(
        dv=radial,
        errdv=radial,
        intensity=radial,
        beta=radial,
        delv=radial,
        azi=np.array([10.0, 20.0], dtype=np.float32),
        zenith=np.array([15.0, 25.0], dtype=np.float32),
        time=time,
        time_units="seconds since 1970-01-01 00:00:00",
        range=np.array([50.0, 100.0, 150.0], dtype=np.float32),
        range_bnds=np.array([[25.0, 75.0], [75.0, 125.0], [125.0, 175.0]], dtype=np.float32),
        lat=np.float32(52.0),
        lon=np.float32(13.0),
        zsl=np.float32(100.0),
        wl=np.float32(1.5e-6),
        pd=np.float32(4e-7),
        nfft=np.float32(1024),
        nrg=np.float32(3),
        lrg=np.float32(50.0),
        nsmpl=np.float32(10),
        prf=np.float32(10000),
        npls=np.float32(3000),
        focus=np.float32(500),
        resv=np.float32(1),
        nqf=np.float32(2),
        nqv=np.float32(3),
        smplf=np.float32(4),
        resf=np.float32(5),
        tgint=np.float32(6),
    )

    dataset = build_normalized_radial_dataset(data)

    assert dataset["dv"].dims == ("time", "range")
    assert dataset["range_bnds"].dims == ("range", "nv")
    assert dataset["azi"].dims == ("time",)
    assert dataset["time"].attrs["units"] == data.time_units
    assert set(("dv", "errdv", "intensity", "beta", "delv", "azi", "zenith")) <= set(
        dataset.variables
    )
