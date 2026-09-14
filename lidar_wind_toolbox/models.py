from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Optional

InstrumentSystem = Literal["windcube", "halo"]
ScanType = Literal["dbs", "vad", "fixed_vad", "stare", "rhi", "ppi"]


@dataclass(frozen=True)
class ProcessingWindow:
    """Timezone-aware UTC interval represented by a processed product."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("ProcessingWindow.start must be timezone-aware")
        if self.end.tzinfo is None or self.end.utcoffset() is None:
            raise ValueError("ProcessingWindow.end must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("ProcessingWindow.end must be after start")

    @classmethod
    def for_utc_day(cls, day: date) -> "ProcessingWindow":
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        return cls(start=start, end=start + timedelta(days=1))


@dataclass(frozen=True)
class WindRetrievalSettings:
    """Scientific settings for wind retrieval."""

    number_of_directions: int
    averaging_minutes: int
    consensus_range_mps: float
    consensus_percentage: float
    snr_threshold_db: float
    minimum_radial_velocities: int
    condition_number_threshold: float
    r2_threshold: float
    blindzone_gates: int = 0

    def __post_init__(self) -> None:
        if self.number_of_directions < 3:
            raise ValueError("number_of_directions must be at least 3")
        if self.averaging_minutes <= 0:
            raise ValueError("averaging_minutes must be positive")
        if not 0 <= self.consensus_percentage <= 100:
            raise ValueError("consensus_percentage must be in [0, 100]")
        if self.minimum_radial_velocities < 3:
            raise ValueError("minimum_radial_velocities must be at least 3")
        if self.condition_number_threshold <= 0:
            raise ValueError("condition_number_threshold must be positive")
        if not 0 <= self.r2_threshold <= 1:
            raise ValueError("r2_threshold must be in [0, 1]")
        if self.blindzone_gates < 0:
            raise ValueError("blindzone_gates must not be negative")


@dataclass(frozen=True)
class InstrumentMetadata:
    """Instrument and station metadata required by output products."""

    system: InstrumentSystem
    instrument_type: str
    instrument_serial_number: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    wavelength_m: float
    instrument_id: Optional[str] = None
    firmware_version: Optional[str] = None
    contact: Optional[str] = None


@dataclass(frozen=True)
class ProductMetadata:
    """Metadata describing a processed data product."""

    title: str
    institution: str
    site_location: str
    source: str = "ground based remote sensing"
    conventions: str = "CF-1.8"
    data_policy: str = "unrestricted use"
    comments: str = ""
    wigos_station_id: Optional[str] = None
    wmo_id: Optional[str] = None
    principal_investigator: Optional[str] = None
    references: Optional[str] = None
    data_blocking_status: Optional[str] = None


@dataclass(frozen=True)
class ProcessingContext:
    """All explicit non-I/O inputs required for one processing operation."""

    window: ProcessingWindow
    retrieval: WindRetrievalSettings
    instrument: InstrumentMetadata
    product: ProductMetadata
    scan_type: ScanType
    processing_version: str
    processed_at: datetime

    def __post_init__(self) -> None:
        if self.processed_at.tzinfo is None or self.processed_at.utcoffset() is None:
            raise ValueError("processed_at must be timezone-aware")


@dataclass(frozen=True)
class WindCubeLevel1ReaderSettings:
    """Explicit non-I/O settings required to normalize WindCube scan files."""

    pulse_duration_s: float
    points_per_gate: int
    pulses_per_direction: int
    pulse_repetition_frequency_hz: float
    fft_points: int
    focus_m: float | None = None

    def __post_init__(self) -> None:
        if self.pulse_duration_s <= 0:
            raise ValueError("pulse_duration_s must be positive")
        if self.points_per_gate <= 0:
            raise ValueError("points_per_gate must be positive")
        if self.pulses_per_direction <= 0:
            raise ValueError("pulses_per_direction must be positive")
        if self.pulse_repetition_frequency_hz <= 0:
            raise ValueError("pulse_repetition_frequency_hz must be positive")
        if self.fft_points <= 0:
            raise ValueError("fft_points must be positive")
