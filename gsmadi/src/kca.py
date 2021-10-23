"""kca.py

Module implements Kinetic Component Analysis (KCA).

Original implementation can be found in
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2422183

Original author: MLdP on 02/22/2014 <lopezdeprado@lbl.gov>
"""

import numpy as np
from pykalman import KalmanFilter


def fitKCA(time, measurable, scalar, steps=0):
    '''Fit measurement over time using KCA
    Parameters:
        time (numpy.Array): Iterable with time indices
        measurable: Signal to perform fitting (e.g. price path)
        scalar: Scalar that multiplies the seed states covariance
        steps: number of steps to forecast (optional, default=0)
    Return:
        x[0]: smoothed state means of position velocity and acceleration
        x[1]: smoothed state covar of position velocity and acceleration
    '''
    t = time
    z = measurable
    q = scalar
    fwd = steps

    # 1) Set up matrices A,H and a seed for Q
    h = (t[-1] - t[0]) / t.shape[0]
    A = np.array([[1, h, 0.5*h**2],
                  [0, 1, h],
                  [0, 0, 1]])
    Q = q * np.eye(A.shape[0])
    # 2) Apply the filter
    kf = KalmanFilter(transition_matrices=A, transition_covariance=Q)
    # 3) EM estimates
    kf = kf.em(z)

    # 4) Smooth
    x_mean, x_covar = kf.smooth(z)
    # 5) Forecast
    for fwd_ in range(fwd):
        x_mean_, x_covar_ = kf.filter_update(filtered_state_mean=x_mean[-1],
                                             filtered_state_covariance=x_covar[-1])
        x_mean = np.append(x_mean, x_mean_.reshape(1, -1), axis=0)
        x_covar_ = np.expand_dims(x_covar_, axis=0)
        x_covar = np.append(x_covar, x_covar_, axis=0)

    # 6) Std series
    x_std = (x_covar[:, 0, 0]**.5).reshape(-1, 1)
    for i in range(1, x_covar.shape[1]):
        x_std_ = x_covar[:, i, i]**.5
        x_std = np.append(x_std, x_std_.reshape(-1, 1), axis=1)

    return x_mean, x_std, x_covar
