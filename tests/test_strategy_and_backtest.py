import pandas as pd
import numpy as np

from utils.strategy import compute_recommendation
from utils.backtest import run_sma_backtest


def make_mock_price_series(n=60, start_price=100.0, drift=0.001):
    # create a simple upward trending series with noise
    prices = [start_price]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + drift + (np.random.randn() * 0.002)))
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    return pd.DataFrame({'close': prices, 'ema20': pd.Series(prices).ewm(span=20).mean(), 'ema50': pd.Series(prices).ewm(span=50).mean(), 'rsi': pd.Series(prices).diff().fillna(0).rolling(14).apply(lambda x: 50)})


def test_compute_recommendation_basic():
    df = make_mock_price_series()
    res = compute_recommendation('FAKE', df, tech_w=1.0, analyst_w=0.0, consensus_w=0.0, news_w=0.0)
    assert isinstance(res, dict)
    assert 'score' in res and 'recommendation' in res and 'components' in res


def test_backtest_returns_structure():
    df = make_mock_price_series()
    price_series = pd.Series(df['close'].values, index=df.index)
    out = run_sma_backtest(price_series, short=5, long=20, initial_cash=1000.0)
    assert isinstance(out, dict)
    assert 'final_value' in out and 'returns_pct' in out and 'equity_curve' in out
    assert hasattr(out['equity_curve'], 'iloc')
