I’ll map the repository’s main data flow and inspect the reader, processing, retrieval, model, and utility modules so the responsibility boundaries are grounded in the actual code.


The repository is a Doppler lidar processing toolbox. Its central job is to turn instrument files containing line-of-sight radial velocities into quality-controlled vertical wind profiles.

The current end-to-end flow is:

```text
Instrument files
    │
    ├── Halo HPL text files
    │       └── readers/halo.py
    │
    └── WindCube NetCDF/CF-Radial files
            └── readers/windcube.py
                    │
                    ▼
        normalized Level-1 xarray.Dataset
                    │
                    ├── time-window grouping
                    ├── consensus filtering
                    ├── Doppler/VAD inversion
                    ├── uncertainty calculation
                    ├── R² and condition-number diagnostics
                    └── quality control
                    │
                    ▼
        Level-2 wind-profile xarray.Dataset
                    │
                    ├── NetCDF output
                    └── quicklook plots
```

## Repository structure

```text
lidar_wind_toolbox/
├── models.py
├── processing.py
├── metadata.py
├── validation.py
├── time_windows.py
├── io.py
├── command_line.py
├── main_proc.py
├── config.py
├── hpl_files.py
├── hpl2netCDF_client.py
│
├── readers/
│   ├── halo.py
│   ├── windcube.py
│   └── file_discovery.py
│
├── retrieval/
│   ├── consensus.py
│   ├── geometry.py
│   ├── uncertainty.py
│   ├── standard.py
│   ├── vad.py
│   ├── windcube.py
│   └── windcube_core.py
│
├── plotting.py
├── plot_helpers.py
├── signal_calc.py
├── wind_calc.py
└── exceptions.py
```

Tests are organized around models, processing, validation, plotting, time windows, numerical helpers, and the two retrieval cores.

## What tasks does the repository perform?

### 1. Configuration and command-line operation

- `config.py` reads legacy `.conf` files into string dictionaries.
- `command_line.py` parses dates and processing commands.
- `hpl2netCDF_client.py` implements the older command-based workflow.

The legacy workflow supports operations such as:

- HPL → Level 1
- Level 1 → Level 2
- HPL → Level 1 → Level 2
- near-real-time processing
- Level-1 and Level-2 quicklooks
- normalization and cleanup of generated files

### 2. File discovery and raw input handling

- `readers/file_discovery.py` extracts dates from filenames.
- `readers/halo.py` parses Halo HPL text files.
- `readers/windcube.py` opens WindCube NetCDF files and combines sweep groups.
- `hpl_files.py` contains additional file discovery, HPL grouping, range conversion, and dataset assembly logic.

The readers deal with instrument-specific representations such as:

- beam time
- azimuth and elevation
- radial velocity
- SNR/CNR
- backscatter
- spectral width
- range gates
- instrument metadata

### 3. Level-1 normalization

The normalized Level-1 representation is an `xarray.Dataset` containing radial observations aligned along time and range/gate dimensions.

The newer WindCube path in [processing.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/processing.py) validates this contract through [validation.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/validation.py).

Typical Level-1 variables include:

- `radial_wind_speed`
- `cnr`
- `azimuth`
- `elevation`
- `range`
- `range_gate_length`
- `relative_beta`
- `doppler_spectrum_width`

The older path uses aliases such as:

- `dv`
- `intensity`
- `azi`
- `zenith`
- `beta`

This naming split is one of the main architectural boundaries that should eventually be resolved.

### 4. Temporal grouping

[time_windows.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/time_windows.py) creates half-open processing intervals:

```text
[start, end)
```

The legacy retrieval functions currently calculate their own time vectors and indices internally. The newer API validates daily processing windows before calling the legacy numerical path.

### 5. Wind retrieval

The retrieval performs a VAD-style inversion:

```text
radial velocity = geometry matrix × [u, v, w]
```

The main mathematical stages are:

1. Group rays by azimuth sector.
2. Apply consensus filtering.
3. Calculate a representative radial velocity per direction and gate.
4. Build the viewing-geometry matrix.
5. Solve for `u`, `v`, and `w`.
6. Propagate radial-velocity uncertainty.
7. Calculate scalar wind speed and wind direction.
8. Calculate `R²`.
9. Calculate geometry condition number.
10. Count valid radial velocities.
11. Apply quality thresholds.

The main retrieval entry points are:

- [retrieval/vad.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/retrieval/vad.py)
- [retrieval/windcube_core.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/retrieval/windcube_core.py)
- [retrieval/standard.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/retrieval/standard.py)
- [retrieval/windcube.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/retrieval/windcube.py)

The two core modules now contain the more readable timestep-level numerical work. The two instrument-facing retrieval modules still contain dataset slicing, time iteration, quality control, and Level-2 dataset construction.

### 6. Quality control

Level-2 quality is based on:

- valid wind-speed calculation
- minimum number of radial velocities
- condition-number threshold
- `R²` threshold
- consensus validity

The primary combined quality flag is `qwind`. Component-specific flags include `qu`, `qv`, and `qw`.

### 7. Metadata and output

- `metadata.py` adds metadata from typed `ProcessingContext`.
- `io.py` writes datasets and figures.
- `main_proc.py` contains the older dataset dispatcher and NetCDF writer.
- `processing.py` is the newer typed public orchestration layer.

### 8. Visualization

[plotting.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/plotting.py) creates:

- Level-1 backscatter/CNR quicklooks
- Level-2 wind-speed and wind-vector quicklooks

`plot_helpers.py` contains legacy plotting preparation logic.

## The main responsibility problems today

### `readers/halo.py` does too much

[readers/halo.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/readers/halo.py) currently combines:

- HPL header parsing
- HPL data-line parsing
- record allocation
- date conversion
- range calculation
- SNR conversion
- measurement uncertainty calculation
- normalized xarray dataset construction

A reader should primarily decode the instrument format. Numerical uncertainty and normalized product construction are separate responsibilities.

A cleaner split would be:

```text
halo_format.py
    parse_header()
    parse_beam_record()
    parse_gate_record()

halo_reader.py
    read_hpl_file() -> RawHaloScan

halo_normalizer.py
    normalize_halo_scan() -> Level1Dataset
```

The HPL parser should return a typed intermediate representation or arrays. The normalizer should convert it into the common Level-1 schema.

### `readers/windcube.py` also mixes layers

It currently combines:

- NetCDF opening
- sweep-group discovery
- sweep concatenation
- range conversion
- vertical geometry derivation
- uncertainty calculation
- normalized dataset construction

A clearer split would be:

```text
windcube_format.py
    open_sweeps()
    concatenate_sweeps()

windcube_reader.py
    read_raw_windcube_file() -> RawWindCubeScan

windcube_normalizer.py
    normalize_windcube_scan() -> Level1Dataset
```

### `hpl_files.py` is a large legacy boundary

`hpl_files` handles:

- file naming
- date-based path creation
- file list generation
- HPL parsing helpers
- range conversion
- combining files
- xarray dataset creation

These should eventually become separate modules. The class itself is mostly a namespace for unrelated static functions rather than an object with meaningful state.

Suggested split:

```text
readers/file_discovery.py
readers/halo_format.py
readers/level1_assembly.py
readers/range_geometry.py
```

### `main_proc.py` mixes dispatch, metadata, and output

[main_proc.py](/home/sebastian/IdeaProjects/lidar-wind-toolbox/lidar_wind_toolbox/main_proc.py) currently:

- selects retrieval paths based on instrument and scan type
- calls retrieval functions
- applies legacy output metadata
- writes NetCDF
- handles operational blocking behavior

These should be separate:

```text
orchestration/dispatch.py
metadata.py
io.py
quality_policy.py
```

### Retrieval modules mix numerical and product responsibilities

The retrieval functions still contain several separate concerns:

- reading arrays from xarray
- time-window selection
- direction detection
- consensus calculation
- Doppler/VAD inversion
- quality control
- Level-2 xarray construction
- configuration serialization

The recently extracted cores improve the numerical part, but the surrounding functions could still be separated into:

```text
retrieval/
    consensus.py
    geometry.py
    uncertainty.py
    vad_core.py
    windcube_core.py
    direction_detection.py
    quality_control.py
    level2_builder.py
```

The intended data flow would be:

```text
Level-1 Dataset
    ↓
extract timestep arrays
    ↓
calculate core result
    ↓
apply quality policy
    ↓
build Level-2 dataset
```

That would make the scientific calculation independent of xarray and easier to test.

### Legacy and typed APIs coexist

There are effectively two APIs:

1. The older config-dictionary and command-line API:
   - `config.py`
   - `main_proc.py`
   - `hpl2netCDF_client.py`
   - `standard.py`
   - `windcube.py`

2. The newer typed API:
   - `models.py`
   - `processing.py`
   - `metadata.py`
   - `validation.py`
   - `time_windows.py`
   - `io.py`

The newer API still translates typed objects back into legacy dictionaries through `_legacy_config()` and `_legacy_reader_config()`.

That is a practical compatibility layer, but it means domain configuration is duplicated. The long-term direction should be:

```text
typed public API
    ↓
typed internal services
    ↓
instrument adapters
    ↓
typed retrieval settings
```

The legacy dictionary path can remain as a compatibility adapter rather than being used internally by the main implementation.

## Suggested target architecture

```text
lidar_wind_toolbox/
├── domain/
│   ├── models.py
│   ├── level1.py
│   ├── level2.py
│   └── retrieval_settings.py
│
├── readers/
│   ├── file_discovery.py
│   ├── halo_format.py
│   ├── halo_reader.py
│   ├── windcube_format.py
│   └── windcube_reader.py
│
├── normalization/
│   ├── halo.py
│   ├── windcube.py
│   └── schema.py
│
├── retrieval/
│   ├── consensus.py
│   ├── geometry.py
│   ├── uncertainty.py
│   ├── direction_detection.py
│   ├── vad_core.py
│   ├── windcube_core.py
│   ├── quality_control.py
│   └── level2_builder.py
│
├── orchestration/
│   ├── processing.py
│   ├── dispatch.py
│   └── time_windows.py
│
├── output/
│   ├── metadata.py
│   ├── dataset_writer.py
│   └── plotting.py
│
└── compatibility/
    ├── config.py
    ├── main_proc.py
    └── hpl2netCDF_client.py
```

The most valuable boundaries are:

- raw file decoding versus normalization
- normalized Level-1 data versus retrieval input extraction
- numerical retrieval versus xarray dataset handling
- scientific quality metrics versus product fill values
- orchestration versus file writing
- typed API versus legacy compatibility

## Recommended extraction order

1. Define a common Level-1 dataset contract and variable names.
2. Split Halo parsing from Halo normalization.
3. Split WindCube file opening from WindCube normalization.
4. Move time-window extraction into a shared service.
5. Keep `vad.py` and `windcube_core.py` as pure numerical modules.
6. Extract shared quality-control and Level-2 assembly code.
7. Make `processing.py` call typed internal functions directly.
8. Reduce the legacy modules to adapters.
9. Move plotting and writing behind explicit output interfaces.

The repository’s core scientific responsibility is clear: it performs Doppler lidar ingestion, radial-observation normalization, consensus-filtered VAD inversion, uncertainty estimation, quality control, and wind-profile product generation. Its main maintainability issue is that instrument decoding, scientific computation, product assembly, and legacy orchestration are currently interleaved across the same modules.
