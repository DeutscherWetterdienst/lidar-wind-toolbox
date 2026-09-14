import json
from dataclasses import asdict
from datetime import UTC, time, timedelta

import xarray as xr

from lidar_wind_toolbox.exceptions import UnsupportedScanTypeError
from lidar_wind_toolbox.main_proc import process_dataset
from lidar_wind_toolbox.metadata import add_global_metadata
from lidar_wind_toolbox.models import ProcessingContext
from lidar_wind_toolbox.validation import validate_windcube_vad_input


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
    validate_windcube_vad_input(dataset)

    legacy_config = _legacy_config(
        context,
        number_of_gates=dataset.sizes["range"],
    )

    processing_day = context.window.start.astimezone(UTC).replace(tzinfo=None)

    # Legacy code mutates its input dataset. Preserve the caller's dataset.
    result = process_dataset(
        dataset.copy(deep=True),
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


def _legacy_config(
    context: ProcessingContext,
    *,
    number_of_gates: int,
) -> dict[str, str]:
    """Translate typed settings for the legacy retrieval implementation.

    This function is a temporary migration boundary. New code must not depend
    on its returned dictionary.
    """

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
        "CN_THRESHOLD": str(context.retrieval.condition_number_threshold),
        "CNS_RANGE": str(context.retrieval.consensus_range_mps),
        "CNS_PERCENTAGE": str(context.retrieval.consensus_percentage),
        "SNR_THRESHOLD": str(context.retrieval.snr_threshold_db),
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
