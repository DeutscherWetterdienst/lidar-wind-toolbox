from __future__ import annotations

from datetime import UTC, datetime, timedelta


def normalize_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")

    return value.astimezone(UTC)


def build_processing_intervals(
    start: datetime,
    end: datetime,
    averaging_minutes: int,
) -> list[tuple[datetime, datetime]]:
    """Build half-open processing intervals.

    The returned intervals use the convention:

        [start, end)

    Therefore the start timestamp is included and the end timestamp is
    excluded. This allows adjacent intervals to be processed without
    duplicate observations.

    The start must be aligned to a complete minute and to the averaging
    interval. The final interval may be shorter than the averaging interval
    if the requested window is not an exact multiple of it.
    """

    if averaging_minutes <= 0:
        raise ValueError("averaging_minutes must be positive")

    normalized_start = normalize_utc(start)
    normalized_end = normalize_utc(end)

    if normalized_end <= normalized_start:
        raise ValueError("end must be after start")

    if normalized_start.second != 0 or normalized_start.microsecond != 0:
        raise ValueError("start must be aligned to a complete minute")

    if normalized_start.minute % averaging_minutes != 0:
        raise ValueError("start must be aligned to averaging_minutes")

    interval = timedelta(minutes=averaging_minutes)
    result: list[tuple[datetime, datetime]] = []

    current = normalized_start

    while current < normalized_end:
        next_time = min(current + interval, normalized_end)
        result.append((current, next_time))
        current = next_time

    return result
