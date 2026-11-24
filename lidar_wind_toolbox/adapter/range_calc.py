from typing import Any


def range_calc(rg_vec: Any, confDict: dict[str, str]) -> Any:
    """Calculate range bounds, also accounting for overlapping gates. If your hpl-files contain overlapping gates please add the "OVERLAPPING_GATES" argument to the configuration file."""
    if "OVERLAPPING_GATES" in confDict:
        r = lambda x, idx: (
            (
                x
                / (1, float(confDict["NUMBER_OF_GATE_POINTS"]))[int(confDict["OVERLAPPING_GATES"])]
                + idx
            )
            * float(confDict["RANGE_GATE_LENGTH"])
        )
    else:
        r = lambda x, idx: (x + idx) * float(confDict["RANGE_GATE_LENGTH"])

    return r(rg_vec, 0.5).astype("f4")
