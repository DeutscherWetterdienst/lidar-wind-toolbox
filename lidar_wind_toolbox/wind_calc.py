import itertools as it

import numpy as np


def diff_aa(x, y, c):
    """calculate aliasing independent differences"""
    return c - abs(abs(x - y) - c)


def process(lst, mon):
    # Guard clause against empty lists
    if len(lst) < 1:
        return lst

    # use an object here to work around closure limitations
    state = type("State", (object,), dict(prev=lst[0], n=0))

    def grouper_proc(x):
        if mon == 1:
            if x < state.prev:
                state.n += 1
        elif mon == -1:
            if x > state.prev:
                state.n += 1
        state.prev = x
        return state.n

    return {k: list(g) for k, g in it.groupby(lst, grouper_proc)}


def check_num_dir(n_rays, calc_idx, azimuth, idx_valid):
    h, be = np.histogram(np.mod(azimuth[calc_idx[idx_valid]], 360), bins=2 * n_rays, range=(0, 360))
    counts = np.sum(np.r_[h[-1], h[:-1]].reshape(-1, 2), axis=1)  # rotate and sum
    edges = np.r_[np.r_[be[-2], be[:-2]][::2], be[-2]]  # rotate and skip
    kk_idx = counts >= 3
    return kk_idx, np.arange(0, 360, 360 // n_rays), edges


def find_num_dir(n_rays, calc_idx, azimuth, idx_valid):
    if np.all(check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[0]):
        return (
            np.all(check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[0]),
            n_rays,
            check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[1],
            check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[2],
        )
    elif ~np.all(check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[0]):
        if n_rays > 4:
            print(
                "number of directions to high...try"
                + str(n_rays // 2)
                + "...instead of "
                + str(n_rays)
            )
            return find_num_dir(n_rays // 2, calc_idx, azimuth, idx_valid)
        elif n_rays < 4:
            print("number of directions to high...try" + str(4) + "...instead")
            return find_num_dir(4, calc_idx, azimuth, idx_valid)
        else:
            print("not enough valid directions!-->skip non-convergent time windows")
            return (
                np.all(check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[0]),
                n_rays,
                check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[1],
                check_num_dir(n_rays, calc_idx, azimuth, idx_valid)[2],
            )
