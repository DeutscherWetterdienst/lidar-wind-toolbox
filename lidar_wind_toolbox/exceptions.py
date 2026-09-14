class LidarWindToolboxError(Exception):
    """Base exception for this package."""


class InputDatasetError(LidarWindToolboxError, ValueError):
    """Raised when an input dataset does not satisfy a processing contract."""


class UnsupportedScanTypeError(LidarWindToolboxError, ValueError):
    """Raised when an instrument and scan type combination is unsupported."""
