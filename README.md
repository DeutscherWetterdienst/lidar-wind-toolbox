# lidar_wind_toolbox

Processing tools for Doppler wind lidar data.

The package is currently transitioning away from the historic config-file-driven
workflow toward an explicit Python API with typed inputs and minimal side effects.

## Current status

The preferred interface is the typed Python API:

- pass explicit raw input files
- pass explicit metadata and retrieval settings as Python objects
- receive Level-1 and Level-2 `xarray.Dataset` objects
- write files only in your calling application or via the explicit helpers in `lidar_wind_toolbox.io`

The legacy config-file-based workflow is still present, but it is considered
transitional.

At the moment the public processing entry point is best understood as a
VAD wind-profile retrieval for WindCube radial data.

## Data Levels and Processing Stages

The toolbox distinguishes between three stages of data representation:

```text
Device
  │  Download
  ▼
Native Instrument Files (NetCDF)
  │  Assembly, harmonization, time-window selection
  ▼
Level-1 Dataset (xarray.Dataset)
  │  VAD inversion, consensus filtering, QC metrics
  ▼
Level-2 Product (xarray.Dataset)
```

### 1. Native Instrument Files (Raw Downloads)

The NetCDF files downloaded from the device are not uncalibrated raw detector signals (photodiode ADC samples). The instrument firmware has already computed line-of-sight Doppler spectra, peak velocities, and signal-to-noise ratios.

- **Content**: Line-of-sight observations along single laser beams.
- **Key variables**: `radial_wind_speed`, `cnr`, `doppler_spectrum_width`, `azimuth`, `elevation`, `range`.
- **Granularity**: One file typically spans a single scan sweep or an interval of 10–11 minutes.

### 2. Level 1: Harmonized Radial Observations

Level 1 combines multiple native files into a consistent, time-aligned daily observation dataset.

- **Process**: Reads sweep groups, concatenates along the time axis, filters by day bounds, and standardizes variable naming and coordinates.
- **Content**: Cleaned and continuous time series of all radial beams for the processing window.
- **Key variables**: Radial Doppler velocity, carrier-to-noise ratio (CNR), scan geometry (`azimuth`, `elevation`), gate bounds, and instrument metadata.

### 3. Level 2: Retrieved Wind Profiles

Level 2 transforms radial velocity measurements from multiple azimuth angles into a vertical profile of the three-dimensional wind vector using Velocity Azimuth Display (VAD) analysis.

- **Process**: 
  - Groups radial rays into regular temporal windows (e.g. 10 or 30 minutes).
  - Applies consensus averaging per azimuth direction to reject turbulent or noisy outliers.
  - Solves the linear geometric inversion system ($v_r = \mathbf{A} \cdot [u, v, w]^T$) via Singular Value Decomposition (SVD).
  - Evaluates geometric condition numbers ($CN$), model fit quality ($R^2$), and minimum ray counts ($N_{\text{vrad}}$).
- **Key variables**: Horizontal and vertical wind components (`u`, `v`, `w`), scalar wind speed (`wspeed`), wind direction (`wdir`), uncertainties (`erru`, `errv`, `errw`, `errwspeed`, `errwdir`), and the primary quality flag (`qwind`).

### Quality Indicators vs. Processing Level

Processing levels indicate the **degree of geophysical derivation**, not whether an individual data point is valid. Quality metrics exist at every level:

- **Level 1**: Evaluated primarily via `cnr` (Carrier-to-Noise Ratio in dB) and Doppler spectral width.
- **Level 2**: Evaluated via consensus cluster size, collinearity check ($CN \le \text{threshold}$), reconstruction fit ($R^2 \ge \text{threshold}$), and available beam count ($N_{\text{vrad}}$). Data points passing all criteria are flagged as valid (`qwind == 1`).

## Installation

```bash
uv sync
```

## Primary API

```python
from datetime import UTC, datetime
from pathlib import Path

from lidar_wind_toolbox import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindCubeLevel1ReaderSettings,
    WindRetrievalSettings,
    process_windcube_vad_files,
)

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
    files=[
        Path("WLS200s-197_2026-01-01_00-00-59_vad_303_50m.nc"),
        Path("WLS200s-197_2026-01-01_00-11-10_vad_303_50m.nc"),
    ],
    context=context,
    reader_settings=reader_settings,
)
```

The functions do not read config files and do not write output files unless you
explicitly call the I/O helpers.

## Core retrieval API

If you already have an in-memory normalized dataset, you can call the retrieval directly:

```python
from lidar_wind_toolbox import retrieve_windcube_vad

level2 = retrieve_windcube_vad(level1, context)
```

## Plotting

The package also provides dataset-based plotting helpers:

```python
from datetime import date
from pathlib import Path

from lidar_wind_toolbox.io import save_figure
from lidar_wind_toolbox.plotting import (
    plot_level1_backscatter_quicklook,
    plot_level2_wind_quicklook,
)

fig1 = plot_level1_backscatter_quicklook(
    level1,
    day=date(2026, 1, 1),
    system="windcube",
)

fig2 = plot_level2_wind_quicklook(
    level2,
    day=date(2026, 1, 1),
)

save_figure(fig1, Path("backscatter.png"))
save_figure(fig2, Path("wind.png"))
```

## Writing datasets

The package provides explicit helpers for writing netCDF datasets and figures:

```python
from pathlib import Path

from lidar_wind_toolbox.io import write_dataset

write_dataset(level1, Path("out/level1.nc"))
write_dataset(level2, Path("out/level2.nc"))
```

## Legacy-to-new mapping

The old `.conf` files mixed together:

- instrument metadata
- scientific retrieval settings
- output metadata
- file-reader settings
- file paths

The new API replaces those values with explicit Python objects.

### Typical mapping

| Legacy key              | New API field                                                |
|-------------------------|--------------------------------------------------------------|
| `SYSTEM`                | `InstrumentMetadata.system`                                  |
| `SYSTEM_LATITUDE`       | `InstrumentMetadata.latitude_deg`                            |
| `SYSTEM_LONGITUDE`      | `InstrumentMetadata.longitude_deg`                           |
| `SYSTEM_ALTITUDE`       | `InstrumentMetadata.altitude_m`                              |
| `SYSTEM_WAVELENGTH`     | `InstrumentMetadata.wavelength_m`                            |
| `NUMBER_OF_DIRECTIONS`  | `WindRetrievalSettings.number_of_directions`                 |
| `AVG_MIN`               | `WindRetrievalSettings.averaging_minutes`                    |
| `CNS_RANGE`             | `WindRetrievalSettings.consensus_range_mps`                  |
| `CNS_PERCENTAGE`        | `WindRetrievalSettings.consensus_percentage`                 |
| `SNR_THRESHOLD`         | `WindRetrievalSettings.snr_threshold_db`                     |
| `N_VRAD_THRESHOLD`      | `WindRetrievalSettings.minimum_radial_velocities`            |
| `CN_THRESHOLD`          | `WindRetrievalSettings.condition_number_threshold`           |
| `R2_THRESHOLD`          | `WindRetrievalSettings.r2_threshold`                         |
| `BLINDZONE_GATES`       | `WindRetrievalSettings.blindzone_gates`                      |
| `PULS_DURATION`         | `WindCubeLevel1ReaderSettings.pulse_duration_s`              |
| `NUMBER_OF_GATE_POINTS` | `WindCubeLevel1ReaderSettings.points_per_gate`               |
| `PULSES_PER_DIRECTION`  | `WindCubeLevel1ReaderSettings.pulses_per_direction`          |
| `PULS_REPETITION_FREQ`  | `WindCubeLevel1ReaderSettings.pulse_repetition_frequency_hz` |
| `FFT_POINTS`            | `WindCubeLevel1ReaderSettings.fft_points`                    |
| `FOCUS`                 | `WindCubeLevel1ReaderSettings.focus_m`                       |
| `NC_TITLE`              | `ProductMetadata.title`                                      |
| `NC_INSTITUTION`        | `ProductMetadata.institution`                                |
| `NC_SITE_LOCATION`      | `ProductMetadata.site_location`                              |

Filesystem paths are intentionally no longer part of the typed processing API.

## Running tests

```bash
uv run pytest
```

## Test status

Automated tests currently cover:

- typed models
- metadata handling
- validation of the minimum WindCube dataset contract
- selected wind-calculation helpers
- a synthetic conical-scan processing smoke test
- config-free file-to-Level-2 orchestration smoke tests
- explicit dataset and figure output helpers

Validation against real instrument data is still required before operational use.
