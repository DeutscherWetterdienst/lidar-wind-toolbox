from datetime import datetime, timezone

import xarray as xr

from lidar_wind_toolbox.metadata import add_global_metadata
from lidar_wind_toolbox.models import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindRetrievalSettings,
)


def test_add_global_metadata_uses_explicit_processing_time() -> None:
    context = ProcessingContext(
        window=ProcessingWindow(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
        retrieval=WindRetrievalSettings(
            number_of_directions=4,
            averaging_minutes=30,
            consensus_range_mps=3,
            consensus_percentage=60,
            snr_threshold_db=0,
            minimum_radial_velocities=3,
            condition_number_threshold=10.0,
            r2_threshold=0.95,
        ),
        instrument=InstrumentMetadata(
            system="windcube",
            instrument_type="WindCube",
            instrument_serial_number="123",
            latitude_deg=52.0,
            longitude_deg=14.0,
            altitude_m=100.0,
            wavelength_m=1.55e-6,
        ),
        product=ProductMetadata(
            title="Test product",
            institution="Test institution",
            site_location="Test site",
        ),
        scan_type="vad",
        processing_version="0.1.0",
        processed_at=datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc),
    )

    result = add_global_metadata(xr.Dataset(), context)

    assert result.attrs["title"] == "Test product"
    assert result.attrs["instrument_serial_number"] == "123"
    assert result.attrs["processing_date"] == "2026-01-03T12:00:00Z"
