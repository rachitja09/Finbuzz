import pandas as pd
from utils.storage import save_portfolio, load_portfolio
from utils.earnings import get_vix, days_until_next_earnings


def test_storage_roundtrip(tmp_path):
    df = pd.DataFrame([{"symbol": "AAPL", "qty": 1, "avg_price": 100}])
    # save to app data path
    save_portfolio(df)
    loaded = load_portfolio()
    assert "symbol" in loaded.columns


def test_get_vix_callable():
    v = get_vix()
    # function should return None or a float
    assert v is None or isinstance(v, float)


def test_days_until_next_earnings_callable():
    d = days_until_next_earnings("AAPL")
    assert d is None or isinstance(d, int)
