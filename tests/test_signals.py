import pandas as pd
import numpy as np
from utils import signals


def make_series(values):
    return pd.Series(values)


def test_sharpe_basic():
    s = make_series([100, 101, 102, 103, 104, 105])
    sh = signals.sharpe_ratio(s, freq=252)
    assert np.isfinite(sh)


def test_sortino_basic():
    s = make_series([100, 99, 98, 99, 100, 101])
    so = signals.sortino_ratio(s)
    assert np.isfinite(so)


def test_momentum():
    s = make_series(list(range(100, 200)))
    mom = signals.momentum(s, periods=10)
    assert np.isfinite(mom)


def test_volatility_and_rsi():
    s = make_series([100, 102, 101, 103, 104, 106, 105, 107, 106, 108, 109, 110, 111, 112, 113])
    vol = signals.volatility(s)
    rsi = signals.rsi_from_series(s, period=5)
    assert np.isfinite(vol)
    assert rsi is None or isinstance(rsi, float)


def test_atr():
    high = make_series([10, 11, 12, 13, 14, 15, 16, 17])
    low = make_series([9, 9.5, 10, 11, 12, 13, 14, 15])
    close = make_series([9.5, 10.5, 11, 12, 13, 14, 15, 16])
    a = signals.atr(high, low, close, period=3)
    assert a is None or np.isfinite(a)
