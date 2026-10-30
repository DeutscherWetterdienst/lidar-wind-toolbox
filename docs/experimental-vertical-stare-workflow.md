# Experimental vertical-stare workflow

**Inputs:** (1) a daily **stare L1** NetCDF file containing `dv`, `intensity`, `time`, and `range`; and (2) a daily **wind L2** NetCDF file with horizontal wind speed `wspeed` and compatible time/height coverage. L1 stare observations **alone are not sufficient** for the `epsilon` retrieval.

A supplied real-world example is the AWI Halo Stream Line XR+ stare dataset for **2024-07-20** (instrument S/N 288, 8,873 profiles × 250 range gates). The repository-ready configuration is in [`examples/stare_awi_288.conf`](examples/stare_awi_288.conf), and detailed setup, input checks, outputs, and caveats are documented in **[`docs/stare-processing.md`](docs/stare-processing.md)**.

After preparing the two inputs and editing their directory roots:

```bash
wind-lidar-process -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2
```

To create the stare product **and its three quicklooks** in one step:

```bash
wind-lidar-process -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2wql
```

To generate **only quicklooks** from an already processed stare L2 file (no wind L2 input is needed for this step):

```bash
wind-lidar-process -u examples/stare_awi_288.conf -d 2024-07-20 -c starewql
```

The equivalent Python API is:

```python
from datetime import datetime
from lidar_wind_toolbox.hpl2netCDF_client import hpl2netCDFClient

client = hpl2netCDFClient(
    "examples/stare_awi_288.conf", "l1_stare_l2", datetime(2024, 7, 20)
)
l2_file, quicklook_files = client.dailystarewql()
```

## Usage

To select the new wind implementation, set `EXPERIMENTAL_WIND_RETRIEVAL = 1`
in a config file, or `use_experimental_retrieval=True` in
`WindRetrievalSettings`. Both instrument entry points can also be called directly
from `lidar_wind_toolbox.retrieval.experimental`. The settings below apply to the
experimental implementation; supplying them alone does not select it. The stare
commands remain available independently of this wind selection.

Supported wind settings include:

| Config key | `WindRetrievalSettings` field |
| --- | --- |
| `AZIMUTH_OFFSET` (fallback: `BEARING`) | `azimuth_offset_deg` |
| `EX_AZI` (comma-separated corrected azimuths) | `excluded_azimuths_deg` |
| `N_HORI_DIRECTIONS` | `horizontal_directions` |
| `CNS_RANGE_VERTICAL` | `vertical_consensus_range_mps` |
| `CNS_PERCENTAGE_VERTICAL` | `vertical_consensus_percentage` |
| `SNR_THRESHOLD_VERTICAL` | `vertical_snr_threshold_db` |
