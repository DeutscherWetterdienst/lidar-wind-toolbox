from datetime import timezone

import xarray as xr

from lidar_wind_toolbox.models import ProcessingContext


def add_global_metadata(dataset: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
    """Return a dataset copy with processing metadata applied."""

    result = dataset.copy(deep=False)
    processed_at = context.processed_at.astimezone(timezone.utc)

    result.attrs.update(
        {
            "title": context.product.title,
            "institution": context.product.institution,
            "site_location": context.product.site_location,
            "source": context.product.source,
            "instrument_type": context.instrument.instrument_type,
            "instrument_mode": context.scan_type,
            "instrument_id": context.instrument.instrument_id or "N/A",
            "instrument_contact": context.instrument.contact or "N/A",
            "instrument_serial_number": context.instrument.instrument_serial_number,
            "instrument_firmware_version": context.instrument.firmware_version or "N/A",
            "Conventions": context.product.conventions,
            "data_policy": context.product.data_policy,
            "comments": context.product.comments,
            "processing_version": context.processing_version,
            "processing_date": processed_at.isoformat().replace("+00:00", "Z"),
            "wigos_station_id": context.product.wigos_station_id or "N/A",
            "wmo_id": context.product.wmo_id or "N/A",
            "principal_investigator": context.product.principal_investigator or "N/A",
        }
    )

    if context.product.references is not None:
        result.attrs["references"] = context.product.references
    if context.product.data_blocking_status is not None:
        result.attrs["data_blocking_status"] = context.product.data_blocking_status

    return result
