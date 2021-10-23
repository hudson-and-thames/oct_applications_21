import numpy as np

from src.trading import generate_trade_signal, KCAStrategy
from src.utils import random_time_series_frame


def test_generate_trade_signal():
    df = random_time_series_frame()

    signal = generate_trade_signal(df)
    assert type(signal) == tuple

    market_price = df.iloc[-1][0]
    predicted = signal[0]
    decision = signal[1]

    if market_price < predicted:
        assert decision == 1  # Buy decision if prediction above market price
    else:
        assert decision == -1  # Sell decision if prediction below market price


def test_kca_trading_strategy():
    df = random_time_series_frame()

    kca = KCAStrategy(df)

    print(kca.observations.tail())
    next_day_prediction = kca.predict()

    # We dont know the actual value of next day yet
    assert type(next_day_prediction[0]) != float
    print(type(next_day_prediction[0]))
    print(kca.observations.tail())
    next_day_prediction = kca.update_latest_observation(333)
    assert type(next_day_prediction[0]) == np.float64
    print(next_day_prediction)
    print(kca.observations.tail())

    # Add an additional day prediction after previous prediction
    next_day_prediction = kca.predict()
    next_day_prediction = kca.update_latest_observation(222)

    print(kca.observations.tail())
    print(type(next_day_prediction))
