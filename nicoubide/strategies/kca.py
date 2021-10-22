"""
The Kinetic Component Analysis (KCA) strategy

The script houses the KCA strategy class which utilizes the
implementation of KCA put forward by Marcos López de Prado
and Riccardo Rebonato
"""

import enum
from typing import List, Optional, Union, Tuple

import datetime as dt
import pandas as pd
import numpy as np

from utils import signal_processing_methods as methods


@enum.unique
class PriceTrend(str, enum.Enum):
    """
    Enumerator class that specifies the four cases of velocity and acceleration
    pairings we'll be considering:
        1) Velocity (+), Acceleration (+): "UP"
        2) Velocity (-), Acceleration (-): "DOWN"
        3) Velocity (+), Acceleration (-): "CONVEX DOWN"
        4) Velocity (-), Acceleration (+): "CONVEX UP"
    For a trading signal to be issued, one of these cases has to occur for a
    certain number of user defined timesteps in a row
    """

    up = "UP"
    down = "DOWN"
    convex_down = "CONVEX DOWN"
    convex_up = "CONVEX UP"


@enum.unique
class Position(int, enum.Enum):
    """
    Enumerator class that specifies the two positions we can be
    in relative to a given security at any given timestep -
    "hold" means we currently have an open position and
    "out" means we have no position
    """

    hold = 1
    out = 0


@enum.unique
class Signal(int, enum.Enum):
    """
    Enumerator class that specifies the three trading signals
    we can generate at any given timestep - the "none" signal
    to more easily enforce only having 1 open position in a given
    security at a time
    """

    buy = 1
    sell = -1
    none = 0


class KCA:
    def __init__(
        self,
        universe: pd.DataFrame,
        seed: float,
        lookback_period: dt.timedelta,
        forecast_horizon: Optional[int] = None,
    ) -> None:
        """
        The KCA constructor automatically calculates the kinematic quantities
        we're interested and finds the trends in the price that our trading
        signals will be based on.
        :param universe (pd.DataFrame): Price series of the futures in our investment universe
        :param seed (float):
            The seed value that initializing the EM estimation of the states covariance for KCA.
        :param lookback_period (dt.timedelta):
            The number of time steps in a row that must occur with the same
            price trend for a trading signal to be issued
        :param forecast_horizon Optional[int]:
            Optional parameter that specifies how far into the future we want to forecast
        """

        # Historical price data for the securities in our investment universe
        self.universe = universe
        self.securities = universe.columns
        self.index = universe.index

        # Holding the lookback period
        self.lookback_period = lookback_period

        # The kinematics associated with the futures contracts in our universe
        self.kinematics = self.generate_kinematics(
            seed=seed, forecast_horizon=forecast_horizon
        )

        # Generate price trends to avoid recomputing every iteration
        self.price_trends = self.generate_price_trends()

        # Create attribute to hold signals and positions
        self.signals = None
        self.positions = None

    def generate_all_signals_and_positions(self) -> None:
        """
        Function that generates the historical signals and positions for
        each of the securities in our investment universe.
        :return None: The positions and signals generated are stored in instance attributes
        """
        # Initializes the signals and positions dataframes
        self.signals = self.initialize_signals()
        self.positions = self.initialize_positions()

        # Calculates the signals and positions for each timestep
        for timestep in self.index[self.lookback_period.days :]:
            (
                self.signals.loc[timestep],
                self.positions.loc[timestep],
            ) = self.generate_single_day_signals_and_positions(timestep)

    def generate_single_day_signals_and_positions(
        self, timestep: dt.datetime
    ) -> Tuple[List[Signal], List[Position]]:
        """
        Function generates the signals and positions for each security in our
        investment universe for one timestep
        :param timestep (dt.datetime): A timestep in the datetime index of our universe attribute
        :return Tuple[List[Signal], List[Position]]:
            Returns a tuple containing two lists - one with the signals generated this timestep
            for all our securities and the other with all the positions generated
        """

        # The window to query historical price trends for
        # This window will go back from the current timestep
        #  by the number of days in the lookback period
        window = np.argwhere(self.index < timestep).reshape(-1)[
            -self.lookback_period.days :
        ]
        signal_list = []
        positions_list = []
        # Generate signals and positions for each security
        for security in self.securities:
            historical_price_trends = self.price_trends[security][window]
            # Only issue a signal if all trends in the lookback period are the same
            # (i.e. there is only one unique entry in the series)
            if len(np.unique(historical_price_trends)) == 1:
                # Append new signal after checking against position rules
                new_signal = self.convert_price_trend_to_signal(
                    historical_price_trends[-1]
                )
                if self.get_last_issued_position(security) == Position.out:
                    # Entering new position - Signal and Position are updated
                    signal_list.append(new_signal)
                    positions_list.append(self.switch_position(security))
                else:
                    if self.get_last_issued_signal(security) == new_signal:
                        # Already in position - Signal and Position stay the same
                        signal_list.append(Signal.none)
                        positions_list.append(self.positions[security][-1])
                    else:
                        # Exiting position - Signal and Position are updated
                        signal_list.append(new_signal)
                        positions_list.append(self.switch_position(security))
            else:
                # Signal is none
                signal_list.append(Signal.none)
                # If signal is none, position is the same as yesterday
                positions_list.append(self.positions[security][-1])
        return signal_list, positions_list

    def get_last_issued_position(self, security: str) -> Position:
        """
        Function enforces our rule that we can only be in
        one position per security at a time
        """
        return self.positions[security][-1]

    def get_last_issued_signal(self, security: str) -> Signal:
        """
        Function gets the last issued signal for a given security
        """
        return self.signals[security][self.signals[security] != 0][-1]

    def initialize_positions(self):
        """
        Function Initializes all of the positions to "out"
        """
        return pd.DataFrame(
            data=Position.out,
            columns=self.securities,
            index=self.index[: self.lookback_period.days],
        )

    def initialize_signals(self):
        """
        Function initializes all of the signals to "none"
        """
        return pd.DataFrame(
            data=Signal.none,
            columns=self.securities,
            index=self.index[: self.lookback_period.days],
        )

    def switch_position(self, security: str) -> Position:
        """
        Function updates our position with respect to a given security.
        If we had an open position, we close it (and vice-versa)
        """
        # If we were holding, we're now out
        if self.positions[security][-1] == Position.hold:
            return Position.out
        # If we were out, we're now holding
        else:
            return Position.hold

    def generate_price_trends(self) -> pd.DataFrame:
        """
        Function generates the price trends at each timestep and
        stores them in an instance attribute
        :return:
        """
        pt = []
        # Iterate over all our securities
        for future in self.universe.columns:
            pt.append(
                self.kinematics[future].apply(
                    lambda x: self.determine_price_trend(x), axis=1
                )
            )
        pt_df = pd.concat(pt, axis=1)
        pt_df.columns = self.universe.columns
        pt_df.index = self.universe.index
        return pt_df

    def determine_price_trend(self, historical_kinematics: pd.Series) -> PriceTrend:
        """
        Function determines the price trend at a given timestep given the kinematic
        quantities of the security at that timestep
        """
        vel_sign = np.sign(historical_kinematics["velocity"])
        acc_sign = np.sign(historical_kinematics["acceleration"])
        if vel_sign == 1.0 and acc_sign == 1.0:
            return PriceTrend.up
        elif vel_sign == -1.0 and acc_sign == -1.0:
            return PriceTrend.down
        elif vel_sign == -1.0 and acc_sign == 1.0:
            return PriceTrend.convex_up
        elif vel_sign == 1.0 and acc_sign == -1.0:
            return PriceTrend.convex_down

    def convert_price_trend_to_signal(self, pt: PriceTrend) -> Signal:
        """
        Function converts a PriceTrend object to an Signal object
        :param pt:
        :return:
        """
        if pt in [PriceTrend.up, PriceTrend.convex_up]:
            return Signal.buy
        elif pt in [PriceTrend.down, PriceTrend.convex_down]:
            return Signal.sell
        else:
            raise ValueError("Invalid PriceTrend object")

    def generate_kinematics(
        self, seed: float, forecast_horizon: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Function generates our desired kinematic quantities using the
        implementation written by Marcos López de Prado and Riccardo Rebonato
        """
        kcas = []
        for future in self.universe.columns:
            kcas.append(
                methods.fit_kca(
                    self.universe[future], seed=seed, forecast_horizon=forecast_horizon
                )
            )
        cols = pd.MultiIndex.from_product(
            [self.universe.columns, methods.Kinematics.values()]
        )
        return pd.DataFrame(np.concatenate(kcas, axis=1), columns=cols)