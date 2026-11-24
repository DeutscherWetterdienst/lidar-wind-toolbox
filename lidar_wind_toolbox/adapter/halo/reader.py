import datetime
from io import BytesIO, TextIOWrapper

import numpy as np
import pandas as pd
import xarray as xr

from lidar_wind_toolbox.adapter.common import (
    NormalizedRadialData,
    build_normalized_radial_dataset,
)
from lidar_wind_toolbox.domain.constants import SPEED_OF_LIGHT
from lidar_wind_toolbox.hpl_files import hpl_files
from lidar_wind_toolbox.retrieval.uncertainty import calc_sigma_single


def read_halo(file_content: bytes, confDict: dict[str, str]) -> xr.Dataset:
    with TextIOWrapper(BytesIO(file_content), encoding="utf-8") as infile:
        header_info = True
        mheader = {}
        for line in infile:
            if line.startswith("****"):
                header_info = False
                ## Adjust header in order to extract data formats more easily
                ## 1st for 'Data line 1' , i.e. time of beam etc.
                tmp = [x.split() for x in mheader["Data line 1"].split("  ")]
                if len(tmp) > 3:
                    tmp.append(" ".join([tmp[2][2], tmp[2][3]]))
                    tmp.append(" ".join([tmp[2][4], tmp[2][5]]))
                tmp[0] = " ".join(tmp[0])
                tmp[1] = " ".join(tmp[1])
                tmp[2] = " ".join([tmp[2][0], tmp[2][1]])
                mheader["Data line 1"] = tmp
                tmp = mheader["Data line 1 (format)"].split(",1x,")
                tmp.append(tmp[-1])
                tmp.append(tmp[-1])
                mheader["Data line 1 (format)"] = tmp
                ## Adjust header in order to extract data formats more easily
                ## 2st for 'Data line 2' , i.e. actual data
                tmp = [x.split() for x in mheader["Data line 2"].split("  ")]
                tmp[0] = " ".join(tmp[0])
                tmp[1] = " ".join(tmp[1])
                tmp[2] = " ".join(tmp[2])
                tmp[3] = " ".join(tmp[3])
                mheader["Data line 2"] = tmp
                tmp = mheader["Data line 2 (format)"].split(",1x,")
                mheader["Data line 2 (format)"] = tmp
                ## start counter for time and range gates
                counter_jj = 0
                continue  # stop the loop and continue with the next line

            tmp = hpl_files.switch(header_info, line)
            ## this temporary variable indicates whether the a given data line includes
            # the spectral width or not, so 2d information can be distinguished from
            # 1d information.
            indicator = len(line[:10].split())

            if header_info:
                try:
                    if tmp[0][0:1] == "i":
                        tmp_tmp = {"Data line 2 (format)": tmp[0]}
                    else:
                        tmp_tmp = {tmp[0]: tmp[1]}
                except Exception:
                    if tmp[0][0] == "f":
                        tmp_tmp = {"Data line 1 (format)": tmp[0]}
                    else:
                        tmp_tmp = {"blank": "nothing"}
                mheader.update(tmp_tmp)
            elif not header_info:
                if counter_jj == 0:
                    n_o_rays = (
                        len(TextIOWrapper(file_content, encoding="utf-8").read().splitlines()) - 17
                    ) // (int(mheader["Number of gates"]) + 1)
                    mbeam = np.recarray(
                        (n_o_rays,),
                        dtype=np.dtype(
                            [
                                ("time", "f8"),
                                ("azimuth", "f4"),
                                ("elevation", "f4"),
                                ("pitch", "f4"),
                                ("roll", "f4"),
                            ]
                        ),
                    )
                    mdata = np.recarray(
                        (n_o_rays, int(mheader["Number of gates"])),
                        dtype=np.dtype(
                            [
                                ("range gate", "i2"),
                                ("velocity", "f4"),
                                ("snrp1", "f4"),
                                ("beta", "f4"),
                                ("dels", "f4"),
                            ]
                        ),
                    )
                    mdata[:, :] = np.full(mdata.shape, -999.0)

                # store tmp in time array
                if indicator == 1:
                    dt = np.dtype(
                        [
                            ("time", "f8"),
                            ("azimuth", "f4"),
                            ("elevation", "f4"),
                            ("pitch", "f4"),
                            ("roll", "f4"),
                        ]
                    )
                    if len(tmp) < 4:
                        tmp.extend(["-999"] * 2)
                    if counter_jj < n_o_rays:
                        mbeam[counter_jj] = np.array(tuple(tmp), dtype=dt)
                        counter_jj = counter_jj + 1
                # store tmp in range gate array
                elif indicator == 2:
                    dt = np.dtype(
                        [
                            ("range gate", "i2"),
                            ("velocity", "f4"),
                            ("snrp1", "f4"),
                            ("beta", "f4"),
                            ("dels", "f4"),
                        ]
                    )
                    ii_index = np.array(tmp[0], dtype=dt[0])

                    if len(tmp) == 4:
                        tmp.append("-999")
                        mdata[counter_jj - 1, ii_index] = np.array(tuple(tmp), dtype=dt)
                    elif len(tmp) == 5:
                        mdata[counter_jj - 1, ii_index] = np.array(tuple(tmp), dtype=dt)

    # set time information
    time_tmp = (
        pd.to_numeric(
            pd.to_timedelta(pd.DataFrame(mbeam)["time"], unit="h")
            + pd.to_datetime(
                datetime.datetime.strptime(mheader["Start time"], "%Y%m%d %H:%M:%S.%f").date()
            )
        ).values
        / 10**9
    )
    time_ds = [
        x + (datetime.timedelta(days=1)).total_seconds() if time_tmp[0] - x > 0 else x
        for x in time_tmp
    ]

    # calculate range in meters from range gate number, gate length
    range_values = hpl_files.range_calc(mdata["range gate"][0, :], confDict)
    range_gate_half_width = (
        np.float32(confDict["PULS_DURATION"]) * SPEED_OF_LIGHT / 4
    ).astype("f4")
    range_bounds = np.array(
        [range_values - range_gate_half_width, range_values + range_gate_half_width]
    ).T
    gate_time_interval = (
        2 * np.array(confDict["RANGE_GATE_LENGTH"], dtype="f4") / SPEED_OF_LIGHT
    ).astype("f4")

    SNR_tmp = np.copy(mdata["snrp1"]) - 1
    SNR_tmp[abs(SNR_tmp) <= np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    ## calculate SNR in dB
    SNR_dB = 10 * np.log10(SNR_tmp.astype(complex)).real
    ## calculate measurement uncertainty, with consensus indices
    sigma_tmp = calc_sigma_single(
        SNR_dB,
        int(mheader["Gate length (pts)"]),
        int(confDict["PULSES_PER_DIRECTION"]),
        float(mheader["Gate length (pts)"])
        / gate_time_interval
        / 2
        * float(confDict["SYSTEM_WAVELENGTH"]),
        1.316,
    )

    gate_points = np.float32(mheader["Gate length (pts)"])
    number_of_gates = np.float32(mheader["Number of gates"])
    range_gate_length = np.float32(mheader["Range gate length (m)"])
    pulse_repetition_frequency = np.float32(confDict["PULS_REPETITION_FREQ"])
    pulses_per_direction = np.float32(confDict["PULSES_PER_DIRECTION"])
    focus_range = np.float32(mheader["Focus range"])
    velocity_resolution = np.float32(mheader["Resolution (m/s)"])
    nyquist_frequency = (gate_points / gate_time_interval / 2).astype("f4")
    nyquist_velocity = (
        gate_points
        / gate_time_interval
        / 2
        * np.float32(confDict["SYSTEM_WAVELENGTH"])
        / 2
    ).astype("f4")
    sampling_frequency = gate_points / gate_time_interval
    frequency_resolution = (
        gate_points / gate_time_interval / float(confDict["FFT_POINTS"])
    ).astype("f4")
    zenith_values = 90 - mbeam["elevation"]

    return build_normalized_radial_dataset(
        NormalizedRadialData(
            dv=mdata["velocity"],
            errdv=sigma_tmp.astype("f4"),
            intensity=mdata["snrp1"],
            beta=mdata["beta"],
            delv=mdata["dels"],
            azi=mbeam["azimuth"],
            zenith=zenith_values,
            time=np.asarray(time_ds),
            time_units="seconds since 1970-01-01 00:00:00",
            range=range_values.astype("f4"),
            range_bnds=range_bounds.astype("f4"),
            lat=np.float32(confDict["SYSTEM_LATITUDE"]),
            lon=np.float32(confDict["SYSTEM_LONGITUDE"]),
            zsl=np.float32(confDict["SYSTEM_ALTITUDE"]),
            wl=np.float32(confDict["SYSTEM_WAVELENGTH"]),
            pd=np.float32(confDict["PULS_DURATION"]),
            nfft=np.float32(confDict["FFT_POINTS"]),
            nrg=number_of_gates,
            lrg=range_gate_length,
            nsmpl=gate_points,
            prf=pulse_repetition_frequency,
            npls=pulses_per_direction,
            focus=focus_range,
            resv=velocity_resolution,
            nqf=nyquist_frequency,
            nqv=nyquist_velocity,
            smplf=sampling_frequency,
            resf=frequency_resolution,
            tgint=gate_time_interval,
        )
    )
