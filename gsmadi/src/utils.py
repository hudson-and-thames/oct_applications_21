import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_squared_error
from math import sqrt

def split_price_time_series(df):
    """"""
    price_series = pd.Series(df.iloc[:, 0]).to_numpy(dtype=int, na_value=0)
    time_series = df.index.to_series().to_numpy(dtype=int, na_value=0)

    return price_series, time_series


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def download_futures_data():
    """Download futures data from Yahoo Finance."""
    futures_symbols = "ZN=F ES=F ZF=F ZT=F NQ=F ZB=F CL=F GC=F NG=F YM=F"
    futures = yf.download(futures_symbols,
                          start="2010-01-01",
                          end="2020-12-31")
    futures.to_csv('data/futures.csv')


def random_time_series_frame():
    """Generate random time-series dataframe for testing."""
    dates = pd.date_range(start='1/1/2018', periods=520, freq='D')
    rand_nums = np.random.randint(520, size=(520, 1))

    df = pd.DataFrame(rand_nums, index=dates, columns=['A'])

    return df

def add_kca_results(df, x_mu, x_std):
    """Add some KCA results to provided frame."""
    p, v = [], []
    p_std = []

    for i in range(len(df)):
        p.append(x_mu[i][0])
        v.append(x_mu[i][1])
        p_std.append(x_std[i][0])

    df['KCA_position'] = p
    df['KCA_velocity'] = v
    df['KCA_position_std'] = p_std

    return df

def set_outcomes_of_predictions(df):
    """Set outcome as 1 if direction of forecast was correct else set to 0."""
    condition = ((df['decision'] > 0) & (df['prediction_delta'] > 0)) | ((df['decision'] < 0) & (df['prediction_delta'] < 0))
    df['outcome'] = np.NAN
    df.loc[condition, 'outcome'] = 1
    df.loc[~condition, 'outcome'] = 0

    return df

def compute_hit_to_miss_ratio(df):
    """Compute hit-miss ratio of algorithm forecast."""
    return df['outcome'].sum() / len(df)

def execute_kca_rolling_window_fit(kca, test_sample, days=30):
    """Roll over a test set to continously predict and update actual values."""
    actual_observations = test_sample

    for day in range(days):
        prediction = kca.predict()
        actual_next_day = actual_observations.first('1D')
        actual_price = actual_next_day.iloc[0][0]
        kca.update_latest_observation(actual_price)
        actual_observations = actual_observations.iloc[1: , :]
    
    return kca