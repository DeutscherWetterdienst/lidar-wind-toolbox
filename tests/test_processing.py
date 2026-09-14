from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from lidar_wind_toolbox.models import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindCubeLevel1ReaderSettings,
    WindRetrievalSettings,
)
from lidar_wind_toolbox.processing import (
    process_windcube_vad_files,
    read_windcube_scan_files,
    retrieve_windcube_vad,
)


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


def make_reader_settings() -> WindCubeLevel1ReaderSettings:
    return WindCubeLevel1ReaderSettings(
        pulse_duration_s=4.01e-7,
        points_per_gate=10,
        pulses_per_direction=3000,
        pulse_repetition_frequency_hz=10000.0,
        fft_points=1024,
        focus_m=500.0,
    )


def normalized_level1_dataset() -> xr.Dataset:
    azimuth = np.repeat(np.arange(0.0, 360.0, 30.0), 2)
    time = 1_767_225_600.0 + np.arange(azimuth.size) * 5.0
    ranges = np.array([50.0, 100.0], dtype=np.float32)

    return xr.Dataset(
        data_vars={
            "dv": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 2.0, dtype=np.float32),
            ),
            "intensity": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 2.0, dtype=np.float32),
            ),
            "beta": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 1e-7, dtype=np.float32),
            ),
            "delv": (
                ("time", "range"),
                np.full((azimuth.size, ranges.size), 0.5, dtype=np.float32),
            ),
            "azi": (("time",), azimuth.astype(np.float32)),
            "zenith": (("time",), np.full(azimuth.size, 15.0, dtype=np.float32)),
            "nsmpl": ((), np.float32(10.0)),
            "prf": ((), np.float32(10000.0)),
            "nqv": ((), np.float32(19.0)),
            "range_bnds": (
                ("range", "nv"),
                np.array([[25.0, 75.0], [75.0, 125.0]], dtype=np.float32),
            ),
        },
        coords={
            "time": time,
            "range": ranges,
            "nv": np.array([0, 1], dtype=np.int8),
        },
    )


def test_retrieve_windcube_vad_returns_level2_dataset() -> None:
    source = normalized_level1_dataset()
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


def test_read_windcube_scan_files_uses_explicit_reader_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyFiles:
        pass

    expected_dataset = normalized_level1_dataset()

    def fake_filelist_to_hpl_files(files: list[Path], inst_type: str) -> DummyFiles:
        captured["files"] = files
        captured["inst_type"] = inst_type
        return DummyFiles()

    def fake_combine(
        files_hpl: DummyFiles,
        conf_dict: dict[str, str],
        date_chosen: datetime,
        time_chosen: datetime | None = None,
    ) -> xr.Dataset:
        captured["files_hpl"] = files_hpl
        captured["conf_dict"] = conf_dict
        captured["date_chosen"] = date_chosen
        captured["time_chosen"] = time_chosen
        return expected_dataset

    monkeypatch.setattr(
        "lidar_wind_toolbox.processing.hpl_files.filelist_to_hpl_files",
        fake_filelist_to_hpl_files,
    )
    monkeypatch.setattr(
        "lidar_wind_toolbox.processing.hpl_files.combine_lvl1_to_ds",
        fake_combine,
    )

    files = [Path("a.nc"), Path("b.nc")]
    context = make_context()
    reader_settings = make_reader_settings()

    result = read_windcube_scan_files(
        files,
        context=context,
        reader_settings=reader_settings,
    )

    assert result is expected_dataset
    assert captured["files"] == files
    assert captured["inst_type"] == "windcube"
    assert captured["files_hpl"].__class__ is DummyFiles
    assert captured["time_chosen"] is None

    conf_dict = captured["conf_dict"]
    assert isinstance(conf_dict, dict)
    assert conf_dict["PULS_DURATION"] == str(reader_settings.pulse_duration_s)
    assert conf_dict["NUMBER_OF_GATE_POINTS"] == str(reader_settings.points_per_gate)
    assert conf_dict["PULSES_PER_DIRECTION"] == str(reader_settings.pulses_per_direction)
    assert conf_dict["PULS_REPETITION_FREQ"] == str(reader_settings.pulse_repetition_frequency_hz)
    assert conf_dict["FFT_POINTS"] == str(reader_settings.fft_points)
    assert conf_dict["FOCUS"] == str(reader_settings.focus_m)


def test_process_windcube_vad_files_returns_level1_and_level2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level1 = normalized_level1_dataset()
    level2 = xr.Dataset(
        data_vars={
            "wspeed": (("time", "height"), np.array([[1.0]], dtype=np.float32)),
        },
        coords={
            "time": np.array([1.0], dtype=np.float64),
            "height": np.array([50.0], dtype=np.float32),
        },
    )

    def fake_read(
        files: list[Path],
        *,
        context: ProcessingContext,
        reader_settings: WindCubeLevel1ReaderSettings,
    ) -> xr.Dataset:
        assert files == [Path("one.nc"), Path("two.nc")]
        assert context == make_context()
        assert reader_settings == make_reader_settings()
        return level1

    def fake_retrieve(dataset: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        assert dataset is level1
        assert context == make_context()
        return level2

    monkeypatch.setattr("lidar_wind_toolbox.processing.read_windcube_scan_files", fake_read)
    monkeypatch.setattr("lidar_wind_toolbox.processing.retrieve_windcube_vad", fake_retrieve)

    result_level1, result_level2 = process_windcube_vad_files(
        [Path("one.nc"), Path("two.nc")],
        context=make_context(),
        reader_settings=make_reader_settings(),
    )

    assert result_level1 is level1
    assert result_level2 is level2
