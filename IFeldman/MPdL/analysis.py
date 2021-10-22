# by MLdP on 02/20/2014 <lopezdeprado@lbl.gov>
# Kinetic Component Analysis of a periodic function

from pathlib import Path
import numpy as np
import matplotlib.pyplot as pp
import statsmodels.nonparametric.smoothers_lowess as sml
import kca
from selectFFT import selectFFT

mainPath = Path().cwd() / "IFeldman/figures"


def getPeriodic(periods, nobs, scale, seed=0):
    t = np.linspace(0, np.pi * periods / 2, nobs)
    rnd = np.random.RandomState(seed)
    signal = np.sin(t)
    z = signal + scale * rnd.randn(nobs)
    return t, signal, z


def vsFFT():

    # 1) Set parameters
    nobs, periods = 300, 10

    # 2) Get Periodic noisy measurements
    t, signal, z = getPeriodic(periods, nobs, scale=0.5)

    # 3) Fit KCA
    x_point, x_bands = kca.fitKCA(t, z, q=0.001)[:2]

    # 4) Plot KCA'a point estimates
    color = ["b", "g", "r"]
    pp.plot(t, z, marker="x", linestyle="", label="measurements")
    pp.plot(
        t, x_point[:, 0], marker="o", linestyle="-", label="position", color=color[0]
    )
    pp.plot(
        t, x_point[:, 1], marker="o", linestyle="-", label="velocity", color=color[1]
    )
    pp.plot(
        t,
        x_point[:, 2],
        marker="o",
        linestyle="-",
        label="acceleration",
        color=color[2],
    )
    pp.legend(loc="lower left", prop={"size": 8})
    pp.savefig(mainPath / "Figure1.png")

    # 5) Plot KCA's confidence intervals (2 std)
    for i in range(x_bands.shape[1]):
        pp.plot(t, x_point[:, i] - 2 * x_bands[:, i], linestyle="-", color=color[i])
        pp.plot(t, x_point[:, i] + 2 * x_bands[:, i], linestyle="-", color=color[i])
    pp.legend(loc="lower left", prop={"size": 8})
    pp.savefig(mainPath / "Figure2.png")
    pp.clf()
    pp.close()

    # 6) Plot comparison with FFT
    fft = selectFFT(z.reshape(-1, 1), minAlpha=0.05)
    pp.plot(t, signal, marker="x", linestyle="", label="Signal")
    pp.plot(t, x_point[:, 0], marker="o", linestyle="-", label="KCA position")
    pp.plot(t, fft["series"], marker="o", linestyle="-", label="FFT position")
    pp.legend(loc="lower left", prop={"size": 8})
    pp.savefig(mainPath / "Figure3.png")

    return


def vsLOWESS():

    # 1) Set parameters
    nobs, periods, frac = 300, 10, [0.5, 0.25, 0.1]

    # 2) Get Periodic noisy measurements
    t, signal, z = getPeriodic(periods, nobs, scale=0.5)

    # 3) Fit KCA
    x_point, x_bands = kca.fitKCA(t, z, q=0.001)[:2]

    # 4) Plot comparison with LOWESS
    pp.plot(t, z, marker="o", linestyle="", label="measurements")
    pp.plot(t, signal, marker="x", linestyle="-", label="Signal")
    pp.plot(t, x_point[:, 0], marker="o", linestyle="-", label="KCA position")
    for frac_ in frac:
        lowess = sml.lowess(z.flatten(), range(z.shape[0]), frac=frac_)[:, 1].reshape(
            -1, 1
        )
        pp.plot(
            t, lowess, marker="o", linestyle="-", lable="LOWESS(" + str(frac_) + ")"
        )
    pp.legend(loc="lower left", prop={"size": 8})
    pp.savefig(mainPath / "Figure4.png")

    return


if __name__ == "__main__":
    vsFFT()
    vsLOWESS()
