import numpy as np

from lidar_wind_toolbox.adapter.common import (
    NormalizedRadialData,
    build_normalized_radial_dataset,
)


def test_build_normalized_radial_dataset_uses_common_level1_schema() -> None:
    time = np.array([1.0, 2.0])
    radial = np.ones((2, 3), dtype=np.float32)
    data = NormalizedRadialData(
        radial_velocity=radial,
        radial_velocity_error=radial,
        intensity=radial,
        beta=radial,
        spectral_width=radial,
        azimuth=np.array([10.0, 20.0], dtype=np.float32),
        zenith=np.array([15.0, 25.0], dtype=np.float32),
        time=time,
        time_units="seconds since 1970-01-01 00:00:00",
        range=np.array([50.0, 100.0, 150.0], dtype=np.float32),
        range_bounds=np.array([[25.0, 75.0], [75.0, 125.0], [125.0, 175.0]], dtype=np.float32),
        lat=np.float32(52.0),
        lon=np.float32(13.0),
        altitude=np.float32(100.0),
        wavelength=np.float32(1.5e-6),
        pulse_duration=np.float32(4e-7),
        number_of_fft_points=np.float32(1024),
        number_of_range_gates=np.float32(3),
        range_gate_length=np.float32(50.0),
        points_per_range_gate=np.float32(10),
        pulse_repetition_frequency=np.float32(10000),
        pulses_per_direction=np.float32(3000),
        focus=np.float32(500),
        velocity_resolution=np.float32(1),
        nyquist_frequency=np.float32(2),
        nyquist_velocity=np.float32(3),
        sampling_frequency=np.float32(4),
        frequency_resolution=np.float32(5),
        gate_time_interval=np.float32(6),
    )

    dataset = build_normalized_radial_dataset(data)

    assert dataset["dv"].dims == ("time", "range")
    assert dataset["range_bnds"].dims == ("range", "nv")
    assert dataset["azi"].dims == ("time",)
    assert dataset["time"].attrs["units"] == data.time_units
    assert set(("dv", "errdv", "intensity", "beta", "delv", "azi", "zenith")) <= set(
        dataset.variables
    )
