import itertools as it
import operator as op

import numpy as np


def consensus(Vr, SNR, BETA, CNS_range, CNS_percentage, SNR_threshold, B):
    """
    consensus(Vr,SNR,CNS_range,CNS_percentage,SNR_threshold)
        Calculate consensus average:
        --> row-wise, calculate the arithmetic mean using only the members of the most likely cluster
            , i.e. the cluster with the maximum number of edges, if the total number of valid clusters is greater than a specified limit.

        Parameters
        ----------
        Vr(time,height) : array_like (intended for using numpy array)
            n-dimensional array representing a signal.
        SNR(time,height) : array_like (intended for using numpy array)
            n-dimensional array of the same dimension as x representing the signal to noise ratio.
        CNS_range(scalar) : scalar value, i.e. 0-dimensional
            scalar value giving the radius for neighboring values in Vr.
        CNS_percentage(scalar) : scalar value, i.e. 0-dimensional
            scalar value stating the minimum percentage for the relative number of valid clusters compared to the totaö number of clusters.
        SNR_threshold(scalar) : scalar value, i.e. 0-dimensional
            scalar value giving the lower bounded threshold of the signal to noise threshold.

        order : {'Vr', 'SNR', 'CNS_range', 'CNS_percentage', 'SNR_threshold'}

        Returns
        -------
        (MEAN, IDX, UNC) --> [numpy array, boolean array, numpy array]
            MEAN - consensus average of array, ...
            IDX - index of values used for the consensus, ...
            UNC - standard deviation of centered around the consensus average of array, ...
            ...for each row

        Dependencies
        ------------
        functions : check_if_db(x), in_mag(snr), filter_by_snr(vr,snr,snr_threshold)

        Notes
        -----
        All array inputs must have the same dimensions, namely (time,height).
        If the input SNR is already given in dB, do NOT filter the input SNR in advance for missing values, because filtering will be done during the calculation
        Translation between dB and magnitude can be done with the functions "in_dB(x)" and "in_mag(x)".
        This function implicitely uses machine epsilon (np.float16) for the numerical value of 0, see "filter_by_snr(vr,snr,snr_threshold)".
    """
    condi_0 = SNR > 0
    if SNR_threshold == 0:
        condi_snr = condi_0
    else:
        condi_snr = (10 * np.log10(SNR.astype(complex)).real > SNR_threshold) & (BETA > 0)

    Vr_m = np.ma.masked_where(~condi_snr, Vr)
    condi_vr = abs(Vr_m.filled(-999.0)) <= B
    Vr_m = np.ma.masked_where(~condi_vr, Vr_m)
    ### calculate the number of points within the consensusrange
    ## easy-to-understand way
    # SUMlt= 1 + np.sum(
    #             np.einsum(  'ij...,ik...-> ij...'
    #                         , (abs(np.einsum('ij... -> ji...', Vr_m[None,...]) - Vr_m[None,...]) < CNS_range).filled(False).astype(int)
    #                         , np.apply_along_axis(np.diag, 0, condi_vr).astype(int))
    #             , axis=0) - (condi_vr).astype(int)
    ## performance strong way using iterators
    SUMlt = 1 + calc_node_degree(Vr_m, CNS_range, B, metric="l1norm")

    Vr_maxim = np.ma.masked_where(
        ~(
            (100 * np.max(SUMlt, axis=0) / CNS_percentage >= condi_vr.sum(axis=0))
            & (condi_vr.sum(axis=0) >= Vr.shape[0] / 100 * 60.0)
        ),
        Vr_m[-(np.argmax(np.flipud(SUMlt), axis=0) + 1), np.arange(0, SUMlt.shape[1])],
    )
    mask_m = abs(Vr_m.filled(999.0) - Vr_maxim.filled(-999.0)) < CNS_range
    Vr_m = np.ma.masked_where(~mask_m, Vr_m.filled(-999.0))
    MEAN = Vr_m.sum(axis=0).filled(np.nan) / np.max(SUMlt, axis=0)
    IDX = mask_m
    UNC = np.nanstd(Vr_m - MEAN.T, axis=0)
    UNC[np.isnan(MEAN)] = np.nan

    ### memory and time efficient option
    # SUMlt= 1 + calc_node_degree(Vr_m, CNS_range, B, metric='l1norm')

    ### this code is more efficient, but less intuitive and accounts for one-time velocity folding
    # SUMlt= 1 + calc_node_degree(Vr_m, CNS_range, B, metric='l1norm_aa')
    # Vr_maxim= np.ma.masked_where( ~((100*np.max(SUMlt, axis=0)/condi_vr.sum(axis=0) >= CNS_percentage) & (100*condi_vr.sum(axis=0)/Vr.shape[0] > 60.))
    #                                 , Vr_m[-(np.argmax(np.flipud(SUMlt),axis=0)+1), np.arange(0,SUMlt.shape[1])]
    #                                 # , Vr_m[np.argmax(SUMlt,axis=0), np.arange(0,SUMlt.shape[1])]
    #                             )
    # mask_m= diff_aa(Vr_m, V_max, B) < 3
    # Vr_m = np.ma.masked_where((mask_m), Vr).filled(Vr-np.sign(Vr-V_max)*2*B*np.heaviside(abs(Vr-V_max)-B, 1))
    # Vr_m = np.ma.masked_where(~(mask_m), Vr_m)
    # MEAN= Vr_m.mean(axis=0).filled(np.nan)
    # IDX= mask_m
    # UNC= np.nanstd(Vr_m-MEAN.T, axis=0)
    # UNC[np.isnan(MEAN)]= np.nan

    return np.round(MEAN, 4), IDX, UNC


def calc_node_degree(Vr, CNS_range, B, metric="l1norm"):
    """takes masked array as input"""
    if metric == "l1norm":

        def f_abs_pairdiff(x, y):
            return op.abs(op.sub(x, y)) < CNS_range

    elif metric == "l1norm_aa":

        def f_abs_pairdiff(x, y):
            return op.sub(B, op.abs(op.sub(op.abs(op.sub(x, y)), B))) < CNS_range

    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    with np.errstate(invalid="ignore"):
        return np.array(
            list(
                grouper(
                    it.starmap(f_abs_pairdiff, (it.permutations(Vr.filled(np.nan), 2))),
                    Vr.shape[0] - 1,
                )
            )
        ).sum(axis=1)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""

    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)
