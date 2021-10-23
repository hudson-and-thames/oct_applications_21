from src.kca import fitKCA
from src.utils import split_price_time_series, random_time_series_frame


def test_kca_fit():
    df = random_time_series_frame()
    price_series, time_series = split_price_time_series(df)

    x_mean, x_std, _ = fitKCA(time_series, price_series, 50)

    assert len(x_mean) == len(price_series)
    assert len(x_std) == len(price_series)

    # Make sure we return position, velocity, acceleration
    assert len(x_mean[0]) == 3


def test_kca_forward_prediction():
    df = random_time_series_frame()
    price_series, time_series = split_price_time_series(df)
    steps_forward = 20

    x_mean, x_std, _ = fitKCA(time_series, price_series, 50,
                              steps=steps_forward)

    assert len(x_mean) == (len(price_series) + steps_forward)
    assert len(x_std) == (len(price_series) + steps_forward)

    # Make sure we return position, velocity, acceleration
    assert len(x_mean[0]) == 3
