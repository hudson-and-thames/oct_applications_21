from abc import ABC, abstractmethod
import functools
import numpy as np
import pandas as pd
from MPdL.kca import fitKCA


class AlphaStrategy(ABC):
    """
    Abstract implementation of alpha strategy

    :params ts: (pd.Series): Time series of asset daily closes prices.
    """

    def __init__(self, ts: pd.Series):
        self.ts = ts

    def __iter__(self):
        yield from self.signal().iteritems()

    def __repr__(self):
        return f"{self.__class__.__name__}(ts)"

    def __str__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def signal(self) -> pd.Series:
        return

    def trade(self) -> pd.Series:
        """Enter into a trade if current signal is different from previous signal."""
        return pd.Series(abs(self.signal().diff()), index=self.ts.index)


class KCAMomentum(AlphaStrategy):
    """
    Class implementation of a momentum alpha strategy using KCA algorithm
    factors of position and acceleration.

    :params ts: (pd.Series): Time series of asset daily closes prices.
    :param n: (int): Window in days for moving average acceleration
        calculation.
    :param threshold: (float): Threshold between 0 - 1 for acceleration rolling
        average test.  To include all possible signals use 0.5.
    """

    def __init__(self, ts: pd.Series, n: int = 5, threshold: float = 0.5):
        super().__init__(ts)
        self.n = n
        self.threshold = threshold

    def _bot_q(self):
        """Bottom threshold for the signal calculation."""
        return self.acc().quantile(self.threshold)

    def _top_q(self):
        """Top threshold for the signal calculation."""
        return self.acc().quantile(1 - self.threshold)

    def acc(self) -> pd.Series:
        """Returns the KCA acceleration."""
        x_points, _ = self._fitKCA()
        return pd.Series(x_points[:, 2], index=self.ts.index)

    def acc_std(self) -> pd.Series:
        """Returns the KCA position standard deviation."""
        _, x_bands = self._fitKCA()
        return pd.Series(x_bands[:, 2], index=self.ts.index)

    @functools.lru_cache()
    def _fitKCA(self):
        """Returns x_points, x_bands"""
        idx = np.arange(len(self.ts.index))
        x_points, x_bands = fitKCA(idx, self.ts.to_numpy(), q=0.1)[:2]
        return x_points, x_bands

    def macc(self) -> pd.Series:
        """Returns the moving average acceleration."""
        return self.acc().rolling(self.n).mean()

    def pos(self) -> pd.Series:
        """Returns the KCA position."""
        x_points, _ = self._fitKCA()
        return pd.Series(x_points[:, 0], index=self.ts.index)

    def pos_std(self) -> pd.Series:
        """Returns the KCA position standard deviation."""
        _, x_bands = self._fitKCA()
        return pd.Series(x_bands[:, 0], index=self.ts.index)

    def signal(self):
        """Momentum signal based on the KSA factors."""
        condlist = [
            ((self.pos() < self.ts) & (self.macc() < self._bot_q())),
            ((self.pos() > self.ts) & (self.macc() > self._top_q())),
        ]
        choicelist = [-1, 1]
        return pd.Series(
            np.select(condlist, choicelist, default=0), index=self.ts.index
        )


class KCAMeanReversion(KCAMomentum):
    """
    Class implementation of a mean revernting alpha strategy using KCA algorithm
    factors of position and acceleration.

    :params ts: (pd.Series): Time series of asset daily closes prices.
    :param n: (int): Window in days for moving average acceleration
        calculation.
    :param threshold: (float): Threshold between 0 - 1 for acceleration rolling
        average test.  To include all possible signals use 0.5.
    """

    def __init__(self, ts: pd.Series, n: int = 5, threshold: float = 0.5):
        super().__init__(ts, n, threshold)

    def signal(self):
        """Mean reverting signal based on the KSA factors."""
        condlist = [
            ((self.pos() > self.ts) & (self.macc() < self._top_q())),
            ((self.pos() < self.ts) & (self.macc() > self._bot_q())),
        ]
        choicelist = [-1, 1]
        return pd.Series(
            np.select(condlist, choicelist, default=0), index=self.ts.index
        )
