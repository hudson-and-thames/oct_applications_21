"""
File that contains the Python functions written by Marcos Lopez
de Prado and Riccardo Rebonato in order to conduct their study
on Kinetic Component Analysis. This code has been modified slightly
to include variable type annotations and follow Python PEP 8 style
conventions. Function inputs/outputs have also been modified slightly in
order to more easily integrate with my trading logic elsewhere in
the codebase. The core logic of each function has remained unchanged.

The original code and accompanying paper can be
found here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2422183
"""

import enum
import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.nonparametric.smoothers_lowess as sml
import statsmodels.stats.diagnostic as sm3
from pykalman import KalmanFilter


@enum.unique
class Kinematics(str, enum.Enum):
    """
    Enumerator class that specifies the kinematic quantities
    we'll be working with
    """
    position = "position"
    velocity = "velocity"
    acceleration = "acceleration"

    @staticmethod
    def values():
        """
        Method that lists our kinematic quantities
        """
        return list(map(lambda k: k.value, Kinematics))


def fit_kca(
    signal: pd.Series, seed: float, forecast_horizon: Optional[int] = None,
) -> np.ndarray:
    """
    By MLdP on 02/22/2014 <lopezdeprado@lbl.gov>
    Kinetic Component Analysis
    """
    # 1) Set up A and Q matrices
    h = (len(signal) - 1) / len(signal)
    A = np.array([[1, h, 0.5 * h ** 2], [0, 1, h], [0, 0, 1]])
    Q = seed * np.eye(A.shape[0])

    # 2) Create Kalman filter with our parameters and apply EM
    kf = KalmanFilter(transition_matrices=A, transition_covariance=Q)
    kf = kf.em(signal)

    # 3) Generate smoothed state means and covariances
    smoothed_state_means, smoothed_state_covars = kf.smooth(signal)

    # 4) Forecast
    if forecast_horizon:
        for _ in range(forecast_horizon):
            updated_mean, updated_covar = kf.filter_update(
                filtered_state_mean=smoothed_state_means[-1],
                filtered_state_covariance=smoothed_state_covars[-1],
            )
            smoothed_state_means = np.append(
                smoothed_state_means, updated_mean.reshape(1, -1), axis=0
            )
            smoothed_state_covars = np.append(
                smoothed_state_covars, np.expand_dims(updated_covar, axis=0), axis=0
            )

    return smoothed_state_means


def select_fft(series: pd.Series(), min_alpha: Optional[float] = None,) -> np.ndarray:
    """
    By MLdP on 02/20/2014 <lopezdeprado@lbl.gov>
    FFT signal extraction with frequency selection
    """
    series_ = series
    fft_res = np.fft.fft(series_, axis=0)
    fft_res = {i: j[0] for i, j in zip(range(fft_res.shape[0]), fft_res)}
    fft_opt = np.zeros(series_.shape, dtype=complex)
    lags, crit = int(12 * (series_.shape[0] / 100.0) ** 0.25), None
    # 2) Search forward
    while True:
        key, crit_old = None, crit
        for key_ in fft_res.keys():
            fft_opt[key_, 0] = fft_res[key_]
            series__ = np.fft.ifft(fft_opt, axis=0)
            series__ = np.real(series__)
            crit_ = sm3.acorr_ljungbox(
                series_ - series__, lags=lags
            )  # test for the max # lags
            crit_ = crit_[0][-1], crit_[1][-1]
            if not crit or crit_[0] < crit[0]:
                crit, key = crit_, key_
            fft_opt[key_, 0] = 0
        if key:
            fft_opt[key, 0] = fft_res[key]
            del fft_res[key]
        else:
            break
        if min_alpha:
            if crit[1] > min_alpha:
                break
            if crit_old and crit[0] / crit_old[0] > 1 - min_alpha:
                break
    series_ = np.fft.ifft(fft_opt, axis=0)
    series_ = np.real(series_)
    return series_


def get_periodic(
    periods: int, nobs: int, scale: float, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    By MLdP on 02/20/2014 <lopezdeprado@lbl.gov>
    Kinetic Component Analysis of a periodic function
    """
    t = np.linspace(0, np.pi * periods / 2.0, nobs)
    rnd = np.random.RandomState(seed)
    signal = np.sin(t)
    z = signal + scale * rnd.randn(nobs)
    return t, signal, z


def vs_fft(main_path: Union[str, os.PathLike],) -> None:
    """
    By MLdP on 02/20/2014 <lopezdeprado@lbl.gov>
    Kinetic Component Analysis of a periodic function
    """
    # 1) Set parameters
    nobs, periods = 300, 10
    # 2) Get Periodic noisy measurements
    t, signal, z = get_periodic(periods, nobs, scale=0.5)
    # 3) Fit KCA
    x_point, x_bands = fit_kca(signal=pd.Series(z), seed=0.001)[:2]
    # 4) Plot KCA's point estimates
    color = ["b", "g", "r"]
    plt.plot(t, z, marker="x", linestyle="", label="measurements")
    plt.plot(
        t, x_point[:, 0], marker="o", linestyle="-", label="position", color=color[0]
    )
    plt.plot(
        t, x_point[:, 1], marker="o", linestyle="-", label="velocity", color=color[1]
    )
    plt.plot(
        t,
        x_point[:, 2],
        marker="o",
        linestyle="-",
        label="acceleration",
        color=color[2],
    )
    plt.legend(loc="lower left", prop={"size": 8})
    plt.savefig(main_path + "Figure1.png")
    # 5) Plot KCA's confidence intervals (2 std)
    for i in range(x_bands.shape[1]):
        plt.plot(t, x_point[:, i] - 2 * x_bands[:, i], linestyle="-", color=color[i])
        plt.plot(t, x_point[:, i] + 2 * x_bands[:, i], linestyle="-", color=color[i])
    plt.legend(loc="lower left", prop={"size": 8})
    plt.savefig(main_path + "Figure2.png")
    plt.clf()
    plt.close()  # reset pylab
    # 6) Plot comparison with FFT
    fft = select_fft(z.reshape(-1, 1), min_alpha=0.05)
    plt.plot(t, signal, marker="x", linestyle="", label="Signal")
    plt.plot(t, x_point[:, 0], marker="o", linestyle="-", label="KCA position")
    plt.plot(t, fft["series"], marker="o", linestyle="-", label="FFT position")
    plt.legend(loc="lower left", prop={"size": 8})
    plt.savefig(main_path + "Figure3.png")


def vs_lowess(main_path: Union[str, os.PathLike],) -> None:
    """
    By MLdP on 02/20/2014 <lopezdeprado@lbl.gov>
    Kinetic Component Analysis of a periodic function
    """
    # 1) Set parameters
    nobs, periods, frac = 300, 10, [0.5, 0.25, 0.1]
    # 2) Get Periodic noisy measurements
    t, signal, z = get_periodic(periods, nobs, scale=0.5)
    # 3) Fit KCA
    x_point, x_bands = fit_kca(signal=pd.Series(z), seed=0.001)[:2]
    # 4) Plot comparison with LOWESS
    plt.plot(t, z, marker="o", linestyle="", label="measurements")
    plt.plot(t, signal, marker="x", linestyle="", label="Signal")
    plt.plot(t, x_point[:, 0], marker="o", linestyle="-", label="KCA position")
    for frac_ in frac:
        lowess = sml.lowess(z.flatten(), range(z.shape[0]), frac=frac_)[:, 1].reshape(
            -1, 1
        )
        pp.plot(
            t, lowess, marker="o", linestyle="-", label="LOWESS(" + str(frac_) + ")"
        )
    plt.legend(loc="lower left", prop={"size": 8})
    plt.savefig(main_path + "Data/test/Figure4.png")
