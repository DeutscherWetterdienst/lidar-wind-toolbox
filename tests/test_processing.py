from datetime import UTC, datetime

import numpy as np
import xarray as xr

from lidar_wind_toolbox.models import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindRetrievalSettings,
)
from lidar_wind_toolbox.processing import retrieve_windcube_vad


def make_context(*, blindzone_gates: int = 0) -> ProcessingContext:
    return ProcessingContext(
        window=ProcessingWindow(
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
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
            blindzone_gates=blindzone_gates,
        ),
        instrument=InstrumentMetadata(
            system="windcube",
            instrument_type="WindCube Scan",
            instrument_serial_number="test-001",
            latitude_deg=52.0,
            longitude_deg=14.0,
            altitude_m=100.0,
            wavelength_m=1.552e-6,
        ),
        product=ProductMetadata(
            title="Synthetic test product",
            institution="Test institution",
            site_location="Test site",
        ),
        scan_type="vad",
        processing_version="0.1.0",
        processed_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )


def synthetic_dataset() -> xr.Dataset:
    azimuth = np.repeat(np.arange(0.0, 360.0, 30.0), 2)
    time = 1_767_225_600.0 + np.arange(azimuth.size) * 5.0
    ranges = np.array([50.0, 100.0], dtype=np.float32)

    return xr.Dataset(
        data_vars={
            "radial_wind_speed": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 2.0, dtype=np.float32),
            ),
            "cnr": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 5.0, dtype=np.float32),
            ),
            "doppler_spectrum_width": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 0.5, dtype=np.float32),
            ),
            "relative_beta": (
                ("time", "range"),
                np.ones((azimuth.size, ranges.size), dtype=np.float32),
            ),
            "azimuth": (("time",), azimuth.astype(np.float32)),
            "elevation": (
                ("time",),
                np.full(azimuth.size, 75.0, dtype=np.float32),
            ),
            "range_gate_length": ((), np.float32(50.0)),
        },
        coords={
            "time": time,
            "range": ranges,
        },
    )


def test_retrieve_windcube_vad_returns_level2_dataset() -> None:
    source = synthetic_dataset()
    original = source.copy(deep=True)

    result = retrieve_windcube_vad(source, make_context())

    assert set(result.data_vars) >= {
        "u",
        "v",
        "w",
        "wspeed",
        "wdir",
        "qwind",
        "cn",
        "nvrad",
        "r2",
    }

    assert result.attrs["processing_version"] == "0.1.0"
    assert result.attrs["processing_date"] == "2026-01-02T12:00:00Z"

    assert "File_Configuration" not in result.attrs
    assert "config" not in result.variables

    xr.testing.assert_identical(source, original)
