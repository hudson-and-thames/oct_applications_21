from src.utils import split_price_time_series, random_time_series_frame


def test_split_price_time_series():
    # df = pd.util.testing.makeDataFrame()
    df = random_time_series_frame()

    p, t = split_price_time_series(df)

    assert len(p) == len(df)
    assert len(t) == len(df)
