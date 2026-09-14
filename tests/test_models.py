from datetime import datetime, timezone

import pytest

from lidar_wind_toolbox.models import ProcessingWindow, WindRetrievalSettings


def test_processing_window_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ProcessingWindow(
            start=datetime(2026, 1, 1),
            end=datetime(2026, 1, 2),
        )


def test_processing_window_rejects_reversed_interval() -> None:
    with pytest.raises(ValueError, match="after start"):
        ProcessingWindow(
            start=datetime(2026, 1, 2, tzinfo=timezone.utc),
            end=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_retrieval_settings_require_three_directions() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        WindRetrievalSettings(
            number_of_directions=2,
            averaging_minutes=30,
            consensus_range_mps=3,
            consensus_percentage=60,
            snr_threshold_db=0,
            minimum_radial_velocities=3,
            condition_number_threshold=10.0,
            r2_threshold=0.95,
        )
