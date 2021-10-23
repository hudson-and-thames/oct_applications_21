"""trading.py

Module collects

"""

import random
import numpy as np
import pandas as pd

from .kca import fitKCA
from .utils import split_price_time_series


def generate_trade_signal(price_df):
    """Generate next day buy or sell signal with future price prediction
        on a set of past price observations using KCA.
    Parameters:
        price_df (pandas.DataFrame): price observetions with th
    Returns
        tuple (numpy.float64, int): contains a price prediction on the first
            slot of the tuple and a 1 or -1 on the second slot to indicate
            a buy and sell signal, respectively.
    """
    randq = random.randrange(1000, 5000)  # Generate random seed for KCA q seed

    market_price = price_df.iloc[-1][0]  # Extract last price as market price

    # Capture last four quarters during KCA fit
    last_four_quarter_signal_df = price_df.iloc[360:, :]

    price_series, time_series = split_price_time_series(
        last_four_quarter_signal_df)

    x_mu, _, _ = fitKCA(time_series, price_series, randq, steps=1)

    predicted = x_mu[-1][0]  # Extract KCA position next step prediction

    if (predicted > market_price):
        return (predicted, 1)
    else:
        return (predicted, -1)


class KCAStrategy(object):
    """KCAStrategy is a simpel trading algorithm based on KCA. It fits 360
        days worth of price obsrvations to produce a prediction signal for
        next day of trading. We use 360 days to capture at least 4 quarters
        worth of price actions.
    """
    def __init__(self, initial_observations):
        self.observations = initial_observations
        self.observations['prediction'] = np.NaN
        self.observations['decision'] = np.NaN

    def predict(self):
        """Predict next day market price and append to initial fitted observations.

        Returns:
            pandas.Series: Next day prediction row
        """
        signal = generate_trade_signal(self.observations)
        prediction = signal[0]
        decision = signal[1]

        new_index = self.observations.index.append(
                        pd.date_range(start=self.observations.index[-1],
                                      periods=2,
                                      freq='D',
                                      closed='right'))
        next_day = pd.to_datetime(new_index[-1])

        self.observations.loc[next_day] = [np.NaN, prediction, decision]

        return self.observations.iloc[-1]

    def update_latest_observation(self, new_observation):
        """Update last appended date to initial observations with actual
            observed value.
        Parameters:
            new_observation (float): Newly observed price observation to
                take into considerations next prediction fit
        Returns:
            pandas.Series: Latest date row with prediction and actual observed
                value.
        """
        self.observations.iloc[-1, 0] = new_observation

        return self.observations.iloc[-1]

    def plot():
        pass
