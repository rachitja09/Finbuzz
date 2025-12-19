import pandas as pd
from utils.backtest import run_sma_backtest


def test_backtest_no_data():
    res = run_sma_backtest(pd.Series(dtype=float))
    assert res["final_value"] == 10000.0


def test_backtest_simple():
    prices = pd.Series([10,11,12,13,12,11,10,9,10,11,12,13,14,15,16])
    res = run_sma_backtest(prices, short=2, long=3, initial_cash=1000)
    assert "final_value" in res
    assert isinstance(res["equity_curve"], pd.Series)
