"""Typed retrieval configuration model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalParameters:
    number_of_directions: int
    averaging_minutes: int
    consensus_range_mps: float
    consensus_percentage: float
    snr_threshold_db: float
    minimum_radial_velocities: int
    condition_number_threshold: float
    r2_threshold: float
    blindzone_gates: int = 0
    azimuth_offset_deg: float = 0.0
    excluded_azimuths_deg: tuple[float, ...] = ()
    horizontal_directions: int | None = None
    vertical_consensus_range_mps: float = 2.0
    vertical_consensus_percentage: float = 30.0
    vertical_snr_threshold_db: float = 0.0
    utc_offset_hours: int = 0

    @classmethod
    def from_legacy_config(cls, config: dict[str, str], gates: int) -> "RetrievalParameters":
        def f(name: str, default: float | None = None) -> float:
            value = config.get(name)
            if value is None and default is not None:
                return default
            if value is not None:
                return float(value)
            raise ValueError(f"expected a default value or value for float ${name}")

        def i(name: str, default: int | None = None) -> int:
            value = config.get(name)
            if value is None and default is not None:
                return default
            if value is not None:
                return int(value)
            raise ValueError(f"expected a default value or value for int ${name}")


        excluded = tuple(float(v) for v in config.get("EX_AZI", "").split(",") if v.strip())
        return cls(
            number_of_directions=i("NUMBER_OF_DIRECTIONS", 4),
            averaging_minutes=i("AVG_MIN", 30),
            consensus_range_mps=f("CNS_RANGE", 3),
            consensus_percentage=f("CNS_PERCENTAGE", 60),
            snr_threshold_db=f("SNR_THRESHOLD", 0),
            minimum_radial_velocities=i("N_VRAD_THRESHOLD", 4),
            condition_number_threshold=f("CN_THRESHOLD", 100),
            r2_threshold=f("R2_THRESHOLD", 0),
            blindzone_gates=i("BLINDZONE_GATES", 0),
            azimuth_offset_deg=f("AZIMUTH_OFFSET", f("BEARING", 0)),
            excluded_azimuths_deg=excluded,
            horizontal_directions=(
                i("N_HORI_DIRECTIONS") if "N_HORI_DIRECTIONS" in config else None
            ),
            vertical_consensus_range_mps=f("CNS_RANGE_VERTICAL", 2),
            vertical_consensus_percentage=f("CNS_PERCENTAGE_VERTICAL", 30),
            vertical_snr_threshold_db=f("SNR_THRESHOLD_VERTICAL", 0),
            utc_offset_hours=i("UTC_OFFSET", 0),
        )
