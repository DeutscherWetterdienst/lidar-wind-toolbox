from datetime import datetime, timezone

import numpy as np
import pytest

from lidar_wind_toolbox.retrieval.numpy_core import (
    LidarScan,
    RetrievalParameters,
    build_time_grid,
)


def test_lidar_scan_validates_array_contract() -> None:
    scan = LidarScan(
        time=np.array(["2024-01-01T00:00:00"], dtype="datetime64[s]"),
        azimuth=np.array([0.0]),
        elevation=np.array([75.0]),
        radial_velocity=np.zeros((1, 2)),
        snr=np.ones((1, 2)),
        beta=np.ones((1, 2)),
        range=np.array([100.0, 200.0]),
        range_bounds=np.array([[75.0, 125.0], [175.0, 225.0]]),
    )
    assert scan.radial_velocity.shape == (1, 2)

    with pytest.raises(ValueError, match="snr"):
        LidarScan(
            time=scan.time,
            azimuth=scan.azimuth,
            elevation=scan.elevation,
            radial_velocity=scan.radial_velocity,
            snr=np.ones((2, 2)),
            beta=scan.beta,
            range=scan.range,
            range_bounds=scan.range_bounds,
        )


def test_time_grid_uses_array_bin_assignments() -> None:
    values = np.array(
        ["2024-01-01T00:01:00", "2024-01-01T00:31:00"], dtype="datetime64[s]"
    )
    grid = build_time_grid(values, datetime(2024, 1, 1, tzinfo=timezone.utc), 30)
    assert grid.sample_bin[:2].tolist() == [0, 1]
    assert grid.bounds.shape == (48, 2)
    assert grid.timestamps.shape == (48,)


def test_legacy_parameters_are_typed() -> None:
    parameters = RetrievalParameters.from_legacy_config(
        {
            "NUMBER_OF_DIRECTIONS": "5",
            "AVG_MIN": "10",
            "CNS_RANGE": "3",
            "CNS_PERCENTAGE": "60",
            "SNR_THRESHOLD": "0",
            "N_VRAD_THRESHOLD": "4",
            "CN_THRESHOLD": "100",
            "R2_THRESHOLD": "0.95",
            "EX_AZI": "10, 20",
        },
        gates=2,
    )
    assert parameters.number_of_directions == 5
    assert parameters.averaging_minutes == 10
    assert parameters.excluded_azimuths_deg == (10.0, 20.0)
