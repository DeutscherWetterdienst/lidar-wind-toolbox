# lidar_wind_toolbox

Processing tools for Doppler wind lidar data.

The package is currently transitioning away from the historic config-file-driven
workflow toward an explicit Python API with typed inputs and minimal side effects.

## Current status

The preferred interface is the typed Python API:

- pass an in-memory `xarray.Dataset`
- pass explicit metadata and retrieval settings as Python objects
- receive a processed `xarray.Dataset`
- write files only in your calling application

The legacy config-file-based workflow is still present, but it is considered
transitional.

At the moment the public processing entry point is best understood as a
VAD wind-profile retrieval for WindCube radial data.

## Installation

```bash
uv sync
```

## Primary API

```python
from datetime import UTC, datetime

from lidar_wind_toolbox import (
    InstrumentMetadata,
    ProcessingContext,
    ProcessingWindow,
    ProductMetadata,
    WindRetrievalSettings,
    retrieve_windcube_vad,
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

result = retrieve_windcube_vad(dataset, context)
```

The function does not read input files or write output files.

## Legacy-to-new mapping

The old `.conf` files mixed together:

- instrument metadata
- scientific retrieval settings
- output metadata
- file paths

The new API replaces those values with explicit Python objects.

### Typical mapping

| Legacy key             | New API field                                      |
|------------------------|----------------------------------------------------|
| `SYSTEM`               | `InstrumentMetadata.system`                        |
| `SYSTEM_LATITUDE`      | `InstrumentMetadata.latitude_deg`                  |
| `SYSTEM_LONGITUDE`     | `InstrumentMetadata.longitude_deg`                 |
| `SYSTEM_ALTITUDE`      | `InstrumentMetadata.altitude_m`                    |
| `SYSTEM_WAVELENGTH`    | `InstrumentMetadata.wavelength_m`                  |
| `NUMBER_OF_DIRECTIONS` | `WindRetrievalSettings.number_of_directions`       |
| `AVG_MIN`              | `WindRetrievalSettings.averaging_minutes`          |
| `CNS_RANGE`            | `WindRetrievalSettings.consensus_range_mps`        |
| `CNS_PERCENTAGE`       | `WindRetrievalSettings.consensus_percentage`       |
| `SNR_THRESHOLD`        | `WindRetrievalSettings.snr_threshold_db`           |
| `N_VRAD_THRESHOLD`     | `WindRetrievalSettings.minimum_radial_velocities`  |
| `CN_THRESHOLD`         | `WindRetrievalSettings.condition_number_threshold` |
| `R2_THRESHOLD`         | `WindRetrievalSettings.r2_threshold`               |
| `BLINDZONE_GATES`      | `WindRetrievalSettings.blindzone_gates`            |
| `NC_TITLE`             | `ProductMetadata.title`                            |
| `NC_INSTITUTION`       | `ProductMetadata.institution`                      |
| `NC_SITE_LOCATION`     | `ProductMetadata.site_location`                    |

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

Validation against real instrument data is still required before operational use.
