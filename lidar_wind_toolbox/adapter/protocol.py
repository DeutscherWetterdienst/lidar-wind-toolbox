from collections.abc import Mapping
from typing import Literal, Protocol

import xarray as xr


class ByteContentReader(Protocol):
    """Read one instrument file from its in-memory byte representation."""

    def __call__(
        self, file_content: bytes, conf_dict: Mapping[str, str]
    ) -> xr.Dataset | Literal[False]: ...

