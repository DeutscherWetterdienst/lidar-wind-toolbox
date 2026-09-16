import xarray as xr


def read_wc_type(filename):
    while True:
        if not filename.exists():
            print("Oops, no such file or directory '{}'".format(filename))
            break
        else:
            print("reading file '{}'".format(filename))
            try:
                ds_root = xr.open_dataset(filename)
            except OSError:
                print("corrupted netCDF: '{}'".format(filename))
                return False
            sweep_list = list(ds_root.sweep_group_name.data)
            ds_ind = xr.concat(
                (
                    xr.open_dataset(filename, group=sweep_ii, decode_times=False)
                    for sweep_ii in sweep_list
                ),
                dim="time",
                data_vars="minimal",
                compat="override",
                coords="minimal",
            )
        return ds_ind
