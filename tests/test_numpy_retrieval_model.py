from datetime import datetime, timezone

import numpy as np
import pytest

from lidar_wind_toolbox.retrieval.numpy_core import (
    LidarScan,
    RetrievalParameters,
    WindRetrievalResult,
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


def test_wind_retrieval_result_owns_and_records_output_arrays() -> None:
    result = WindRetrievalResult(
        time=np.array([1, 2]),
        time_bounds=np.array([[0, 1], [1, 2]]),
        height=np.array([100.0, 200.0]),
        height_bounds=np.array([[75.0, 125.0], [175.0, 225.0]]),
        horizontal_width=np.array([20.0, 40.0]),
    )

    assert result.wind.shape == (2, 2, 3)
    assert result.speed.shape == (2, 2)
    assert np.isnan(result.wind).all()
    assert not result.quality.any()
    assert (result.binary_direction == -999).all()

    valid = result.record_consensus(
        0,
        np.array([[1.0, 2.0], [1.1, np.nan], [0.9, 2.1], [1.0, np.nan]]),
        np.array([[1.0, 2.0], [1.0, np.nan], [1.0, 2.0], [1.0, np.nan]]),
        np.array([[1e-6, 2e-6], [1e-6, np.nan], [1e-6, 2e-6], [1e-6, np.nan]]),
        np.array([[0.4, 0.5], [0.4, np.nan], [0.4, 0.5], [0.4, np.nan]]),
    )
    np.testing.assert_array_equal(
        valid,
        [[True, True], [True, False], [True, True], [True, False]],
    )

    with pytest.raises(ValueError, match="shape"):
        result.record_consensus(
            0,
            np.ones((4, 2)),
            np.ones((4, 1)),
            np.ones((4, 2)),
            np.ones((4, 2)),
        )
    result.record_gate_solution(
        0,
        0,
        wind=np.array([3.0, 2.0, 0.0]),
        wind_uncertainty=np.array([0.1, 0.1, 0.1]),
        speed=3.6,
        speed_uncertainty=0.1,
        direction=236.3,
        direction_uncertainty=1.0,
        r2=0.99,
        condition_number=2.0,
    )

    result.finalize_quality(
        RetrievalParameters(
            number_of_directions=4,
            averaging_minutes=30,
            consensus_range_mps=3.0,
            consensus_percentage=60.0,
            snr_threshold_db=0.0,
            minimum_radial_velocities=3,
            condition_number_threshold=10.0,
            r2_threshold=0.95,
        )
    )

    np.testing.assert_allclose(result.wind[0, 0], [3.0, 2.0, 0.0])
    assert result.binary_direction[0, 0] == 15
    np.testing.assert_allclose(result.snr[0], [1.0, 2.0])
    np.testing.assert_allclose(result.snr_min[0], [1.0, 2.0])
    np.testing.assert_allclose(result.snr_max[0], [1.0, 2.0])
    assert result.quality[0, 0]
    assert not result.quality[0, 1]
