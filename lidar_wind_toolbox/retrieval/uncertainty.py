import numpy as np


def calc_sigma_single(SNR_dB, Mpts, nsmpl, BW, delta_v):
    "calculates the instrument uncertainty: SNR in dB!"
    SNR = 10 ** (SNR_dB / 10)
    bb = np.sqrt(2.0 * np.pi) * (delta_v / BW)
    alpha = SNR / bb
    Np = Mpts * nsmpl * SNR

    a1 = 2.0 * (np.sqrt(np.ma.divide(np.sqrt(np.pi), alpha)))
    a2 = 1 + 0.16 * alpha
    a3 = np.ma.divide(delta_v, np.sqrt(Np))
    sigma = np.ma.masked_where(SNR_dB > -5, (a1 * a2 * a3).filled(np.nan)).filled(a3.filled(np.nan))

    return sigma
