from typing import Any


def range_calc(rg_vec: Any, confDict: dict[str, str]) -> Any:
    """Calculate range bounds, also accounting for overlapping gates. If your hpl-files contain overlapping gates please add the "OVERLAPPING_GATES" argument to the configuration file."""
    range_gate = float(confDict["RANGE_GATE_LENGTH"])
    if "OVERLAPPING_GATES" in confDict:
        number_of_gates = float(confDict["NUMBER_OF_GATE_POINTS"])
        return (
            (rg_vec / (1, number_of_gates)[int(confDict["OVERLAPPING_GATES"])] + 0.5) * range_gate
        ).astype("f4")
    else:
        return ((rg_vec + 0.5) * range_gate).astype("f4")
