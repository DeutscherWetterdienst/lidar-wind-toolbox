from datetime import UTC, date, datetime
from pathlib import Path

from lidar_wind_toolbox import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindCubeLevel1ReaderSettings,
    WindRetrievalSettings,
    process_windcube_vad_files,
    save_figure,
    write_dataset,
)
from lidar_wind_toolbox.plotting import (
    plot_level1_backscatter_quicklook,
    plot_level2_wind_quicklook,
)

files = [
    Path("WLS200s-197_2026-01-01_00-00-59_vad_303_50m.nc"),
    Path("WLS200s-197_2026-01-01_00-11-10_vad_303_50m.nc"),
]

context = ProcessingContext(
    window=ProcessingWindow(
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
    ),
    retrieval=WindRetrievalSettings(
        number_of_directions=12,
        averaging_minutes=10,
        consensus_range_mps=3.0,
        consensus_percentage=60.0,
        snr_threshold_db=0.0,
        minimum_radial_velocities=10,
        condition_number_threshold=10.0,
        r2_threshold=0.95,
        blindzone_gates=0,
    ),
    instrument=InstrumentMetadata(
        system="windcube",
        instrument_type="WindCube Scan",
        instrument_serial_number="197",
        latitude_deg=46.8,
        longitude_deg=6.9,
        altitude_m=490.0,
        wavelength_m=1.552e-6,
    ),
    product=ProductMetadata(
        title="Wind profile",
        institution="Example Institute",
        site_location="Example Site",
    ),
    scan_type="vad",
    processing_version="0.1.0",
    processed_at=datetime.now(UTC),
)

reader_settings = WindCubeLevel1ReaderSettings(
    pulse_duration_s=4.01e-7,
    points_per_gate=10,
    pulses_per_direction=3000,
    pulse_repetition_frequency_hz=10000.0,
    fft_points=1024,
    focus_m=500.0,
)

level1, level2 = process_windcube_vad_files(
    files=files,
    context=context,
    reader_settings=reader_settings,
)

write_dataset(level1, Path("out/level1.nc"))
write_dataset(level2, Path("out/level2.nc"))

fig1 = plot_level1_backscatter_quicklook(
    level1,
    day=date(2026, 1, 1),
    system="windcube",
)
fig2 = plot_level2_wind_quicklook(
    level2,
    day=date(2026, 1, 1),
)

save_figure(fig1, Path("out/backscatter.png"))
save_figure(fig2, Path("out/wind.png"))
