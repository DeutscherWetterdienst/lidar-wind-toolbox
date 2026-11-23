"""Retrieval method identifiers."""

from enum import StrEnum


class RetrievalMethod(StrEnum):
    STANDARD_VAD = "standard_vad"
    WINDCUBE_DBS = "windcube_dbs"
