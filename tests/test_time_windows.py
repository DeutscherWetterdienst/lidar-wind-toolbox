from datetime import UTC, datetime, timedelta

import pytest

from lidar_wind_toolbox.time_windows import (
    build_processing_intervals,
    normalize_utc,
)


def test_normalize_utc_converts_timezone() -> None:
    value = datetime(
        2026,
        9,
        14,
        14,
        0,
        tzinfo=UTC,
    )

    assert normalize_utc(value) == value


def test_normalize_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_utc(datetime(2026, 9, 14, 12, 0))


def test_build_processing_intervals_for_thirty_minutes() -> None:
    start = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    end = datetime(2026, 9, 14, 12, 30, tzinfo=UTC)

    result = build_processing_intervals(
        start,
        end,
        averaging_minutes=10,
    )

    assert result == [
        (
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 10, tzinfo=UTC),
        ),
        (
            datetime(2026, 9, 14, 12, 10, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 20, tzinfo=UTC),
        ),
        (
            datetime(2026, 9, 14, 12, 20, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 30, tzinfo=UTC),
        ),
    ]


def test_intervals_are_start_inclusive_and_end_exclusive() -> None:
    start = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    end = datetime(2026, 9, 14, 12, 20, tzinfo=UTC)

    result = build_processing_intervals(
        start,
        end,
        averaging_minutes=10,
    )

    assert result[0][0] == start
    assert result[0][1] == result[1][0]
    assert result[-1][1] == end


def test_final_interval_may_be_shorter() -> None:
    start = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    end = datetime(2026, 9, 14, 12, 25, tzinfo=UTC)

    result = build_processing_intervals(
        start,
        end,
        averaging_minutes=10,
    )

    assert result[-1] == (
        datetime(2026, 9, 14, 12, 20, tzinfo=UTC),
        datetime(2026, 9, 14, 12, 25, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (
            datetime(2026, 9, 14, 12, 1, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 30, tzinfo=UTC),
            "averaging_minutes",
        ),
        (
            datetime(2026, 9, 14, 12, 0, 1, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 30, tzinfo=UTC),
            "complete minute",
        ),
        (
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            "after",
        ),
    ],
)
def test_build_processing_intervals_rejects_invalid_window(
    start: datetime,
    end: datetime,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_processing_intervals(
            start,
            end,
            averaging_minutes=10,
        )


def test_build_processing_intervals_normalizes_non_utc_timezone() -> None:
    from datetime import timezone

    start = datetime(
        2026,
        9,
        14,
        14,
        0,
        tzinfo=timezone.utc,
    )
    end = start + timedelta(minutes=20)

    result = build_processing_intervals(
        start,
        end,
        averaging_minutes=10,
    )

    assert result[0][0].tzinfo == UTC
