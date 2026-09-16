import json
from dataclasses import asdict
from datetime import UTC, time, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .exceptions import UnsupportedScanTypeError
from .hpl_files import hpl_files
from .main_proc import process_dataset
from .metadata import add_global_metadata
from .models import ProcessingContext, WindCubeLevel1ReaderSettings
from .validation import validate_normalized_windcube_level1


def read_windcube_scan_files(
    files: list[Path],
    *,
    context: ProcessingContext,
    reader_settings: WindCubeLevel1ReaderSettings,
) -> xr.Dataset:
    """Read and normalize WindCube scan files into the internal retrieval dataset."""

    if context.instrument.system != "windcube":
        raise UnsupportedScanTypeError(
            f"read_windcube_scan_files requires a WindCube context, got {context.instrument.system!r}"
        )

    if context.scan_type != "vad":
        raise UnsupportedScanTypeError(
            f"read_windcube_scan_files currently requires scan_type='vad', got {context.scan_type!r}"
        )

    if not files:
        raise ValueError("files must not be empty")

    files_hpl = hpl_files.filelist_to_hpl_files(files, context.instrument.system)
    processing_day = context.window.start.astimezone(UTC).replace(tzinfo=None)

    return hpl_files.combine_lvl1_to_ds(
        files_hpl,
        _legacy_reader_config(
            context=context,
            reader_settings=reader_settings,
        ),
        processing_day,
    )


def process_windcube_vad_files(
    files: list[Path],
    *,
    context: ProcessingContext,
    reader_settings: WindCubeLevel1ReaderSettings,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Read WindCube scan files and retrieve the corresponding Level-2 VAD product."""

    level1 = read_windcube_scan_files(
        files,
        context=context,
        reader_settings=reader_settings,
    )
    level2 = retrieve_windcube_vad(level1, context)
    return level1, level2


def retrieve_windcube_vad(
    dataset: xr.Dataset,
    context: ProcessingContext,
) -> xr.Dataset:
    """Retrieve a daily Level-2 WindCube VAD product.

    This public API does not read or write files. The current implementation
    uses the legacy numerical retrieval internally and may emit diagnostic
    output to stdout.
    """

    if context.instrument.system != "windcube":
        raise UnsupportedScanTypeError(
            f"retrieve_windcube_vad requires a WindCube context, got {context.instrument.system!r}"
        )

    if context.scan_type != "vad":
        raise UnsupportedScanTypeError(
            f"retrieve_windcube_vad requires scan_type='vad', got {context.scan_type!r}"
        )

    _validate_daily_window(context)
    validate_normalized_windcube_level1(dataset)

    number_of_gates = int(dataset["radial_wind_speed"].shape[1])

    legacy_config = _legacy_config(
        context,
        number_of_gates=number_of_gates,
    )

    processing_day = context.window.start.astimezone(UTC).replace(tzinfo=None)

    # Legacy code mutates its input dataset. Preserve the caller's dataset.
    retrieval_input = dataset.copy(deep=True)

    if "range_gate_length" not in retrieval_input.variables:
        if "range_bnds" in retrieval_input.variables:
            retrieval_input["range_gate_length"] = (
                (
                    retrieval_input["range_bnds"].isel(nv=1)
                    - retrieval_input["range_bnds"].isel(nv=0)
                )
                .mean()
                .astype(np.float32)
            )
        elif "range" in retrieval_input.variables and retrieval_input["range"].size > 1:
            dim_name = retrieval_input["range"].dims[0]
            retrieval_input["range_gate_length"] = (
                retrieval_input["range"].diff(dim_name).mean().astype(np.float32)
            )

    result = process_dataset(
        retrieval_input,
        processing_day,
        legacy_config,
    )

    # Remove legacy config-file-derived provenance.
    result = result.drop_vars("config", errors="ignore")
    result.attrs.pop("File_Configuration", None)

    result = add_global_metadata(result, context)

    processed_at = context.processed_at.astimezone(UTC)
    result.attrs["history"] = (
        f"Processed by lidar_wind_toolbox {context.processing_version} "
        f"at {processed_at.isoformat().replace('+00:00', 'Z')}"
    )

    result.attrs["processing_configuration"] = json.dumps(
        asdict(context.retrieval),
        sort_keys=True,
        separators=(",", ":"),
    )

    return result


def _validate_daily_window(context: ProcessingContext) -> None:
    start = context.window.start.astimezone(UTC)
    end = context.window.end.astimezone(UTC)

    if start.time() != time.min:
        raise ValueError("WindCube VAD daily retrieval requires a window starting at 00:00 UTC")

    if end != start + timedelta(days=1):
        raise ValueError("WindCube VAD daily retrieval requires a 24-hour UTC window")


def _legacy_reader_config(
    *,
    context: ProcessingContext,
    reader_settings: WindCubeLevel1ReaderSettings,
) -> dict[str, str]:
    """Translate typed reader settings for the legacy Level-1 normalization path."""

    return {
        "SYSTEM": context.instrument.system,
        "SYSTEM_ID": context.instrument.instrument_serial_number,
        "SYSTEM_LATITUDE": str(context.instrument.latitude_deg),
        "SYSTEM_LONGITUDE": str(context.instrument.longitude_deg),
        "SYSTEM_ALTITUDE": str(context.instrument.altitude_m),
        "SYSTEM_WAVELENGTH": str(context.instrument.wavelength_m),
        "SCAN_TYPE": context.scan_type,
        "VERSION": context.processing_version,
        "PULS_DURATION": str(reader_settings.pulse_duration_s),
        "NUMBER_OF_GATE_POINTS": str(reader_settings.points_per_gate),
        "PULSES_PER_DIRECTION": str(reader_settings.pulses_per_direction),
        "PULS_REPETITION_FREQ": str(reader_settings.pulse_repetition_frequency_hz),
        "FFT_POINTS": str(reader_settings.fft_points),
        "FOCUS": str(reader_settings.focus_m) if reader_settings.focus_m is not None else "0",
        "AVG_MIN": str(context.retrieval.averaging_minutes),
        "NC_TITLE": context.product.title,
        "NC_INSTITUTION": context.product.institution,
        "NC_SITE_LOCATION": context.product.site_location,
        "NC_SOURCE": context.product.source,
        "NC_INSTRUMENT_TYPE": context.instrument.instrument_type,
        "NC_INSTRUMENT_MODE": context.scan_type,
        "NC_INSTRUMENT_CONTACT": context.instrument.contact or "N/A",
        "NC_INSTRUMENT_ID": context.instrument.instrument_id or "N/A",
        "NC_INSTRUMENT_SERIAL_NUMBER": context.instrument.instrument_serial_number,
        "NC_INSTRUMENT_FIRMWARE_VERSION": context.instrument.firmware_version or "N/A",
        "NC_CONVENTIONS": context.product.conventions,
        "NC_DATA_POLICY": context.product.data_policy,
        "NC_COMMENTS": context.product.comments,
        "NC_HISTORY": "Processed by lidar_wind_toolbox",
        "NC_WIGOS_STATION_ID": context.product.wigos_station_id or "N/A",
        "NC_WMO_ID": context.product.wmo_id or "N/A",
        "NC_PI_ID": context.product.principal_investigator or "N/A",
    }


def _legacy_config(
    context: ProcessingContext,
    *,
    number_of_gates: int,
) -> dict[str, str]:
    """Translate typed settings for the legacy retrieval implementation.

    This function is a temporary migration boundary. New code must not depend
    on its returned dictionary.
    """

    def _fmt(val: float | int) -> str:
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        return str(val)

    return {
        "SYSTEM": context.instrument.system,
        "SYSTEM_ID": context.instrument.instrument_serial_number,
        "SYSTEM_LATITUDE": str(context.instrument.latitude_deg),
        "SYSTEM_LONGITUDE": str(context.instrument.longitude_deg),
        "SYSTEM_ALTITUDE": str(context.instrument.altitude_m),
        "SYSTEM_WAVELENGTH": str(context.instrument.wavelength_m),
        "SCAN_TYPE": context.scan_type,
        "NUMBER_OF_DIRECTIONS": str(context.retrieval.number_of_directions),
        "NUMBER_OF_GATES": str(number_of_gates),
        "AVG_MIN": str(context.retrieval.averaging_minutes),
        "CN_THRESHOLD": _fmt(context.retrieval.condition_number_threshold),
        "CNS_RANGE": _fmt(context.retrieval.consensus_range_mps),
        "CNS_PERCENTAGE": _fmt(context.retrieval.consensus_percentage),
        "SNR_THRESHOLD": _fmt(context.retrieval.snr_threshold_db),
        "N_VRAD_THRESHOLD": str(context.retrieval.minimum_radial_velocities),
        "R2_THRESHOLD": str(context.retrieval.r2_threshold),
        "BLINDZONE_GATES": str(context.retrieval.blindzone_gates),
        "VERSION": context.processing_version,
        "NC_TITLE": context.product.title,
        "NC_INSTITUTION": context.product.institution,
        "NC_SITE_LOCATION": context.product.site_location,
        "NC_SOURCE": context.product.source,
        "NC_INSTRUMENT_TYPE": context.instrument.instrument_type,
        "NC_INSTRUMENT_MODE": context.scan_type,
        "NC_INSTRUMENT_CONTACT": context.instrument.contact or "N/A",
        "NC_INSTRUMENT_ID": context.instrument.instrument_id or "N/A",
        "NC_INSTRUMENT_SERIAL_NUMBER": context.instrument.instrument_serial_number,
        "NC_INSTRUMENT_FIRMWARE_VERSION": context.instrument.firmware_version or "N/A",
        "NC_CONVENTIONS": context.product.conventions,
        "NC_DATA_POLICY": context.product.data_policy,
        "NC_COMMENTS": context.product.comments,
        "NC_HISTORY": "Processed by lidar_wind_toolbox",
        "NC_WIGOS_STATION_ID": context.product.wigos_station_id or "N/A",
        "NC_WMO_ID": context.product.wmo_id or "N/A",
        "NC_PI_ID": context.product.principal_investigator or "N/A",
    }
