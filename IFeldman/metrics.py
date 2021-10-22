import numpy as np
import pandas as pd


class Metrics:
    """
    Metrics class implementation to produce the outputs of an alpha strategy.

    :param ts: (pd.Series): Time series of asset daily closes prices.
    :param trade: (pd.Series): Trade signal to apply to the time series.
    """

    def __init__(self, ts: pd.Series, trade: pd.Series):
        self.ts = ts
        self.trade = trade

    def pnl(self) -> pd.Series:
        """Returns the daily profit and loss."""
        pnl = self.ret() * self.trade
        return pd.Series(pnl, index=self.ts.index)

    def pnl_cum(self) -> pd.Series:
        """Returns the cummulative profit and loss."""
        pnlc = self.pnl().add(1).cumprod().sub(1)
        return pd.Series(pnlc, index=self.ts.index)

    def pnl_rolling(self, n=252) -> pd.Series:
        """Returns the rolling profit and loss.
        Default window is 252 days."""
        pnlr = self.pnl().rolling(n).mean() * n
        return pd.Series(pnlr * n, index=self.ts.index).fillna(0)

    def ret(self) -> pd.Series:
        """Returns the daily simple returns."""
        ret = np.log(self.ts.to_numpy() / self.ts.shift(1).to_numpy())
        return pd.Series(ret, index=self.ts.index).fillna(0)

    def ret_rolling(self, n=252) -> pd.Series:
        """Returns the rolling returns.
        Default window is 252 days."""
        return ((1 + self.ret()).rolling(n).apply(np.prod, raw=True) - 1).fillna(0)

    def sharpe(self, n=252) -> pd.Series:
        """Returns the sharpe ration for the strategy.
        Default window is 252 days."""
        return pd.Series(self.pnl_rolling(n).div(self.vol()), index=self.ts.index)

    def sign(self) -> pd.Series:
        """Returns the direction of the daily simple returns."""
        return pd.Series(np.sign(self.ret().to_numpy()), self.ts.index)

    def vol(self, n=252) -> pd.Series:
        """Returns the strategy volatility.
        Default window is 252 days."""
        return pd.Series((self.ret_rolling(n).std() * np.sqrt(n)), index=self.ts.index)
