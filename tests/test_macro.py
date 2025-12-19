import pandas as pd
import numpy as np
import datetime as dt

from utils import macro


def make_shiller_like(months=240, start_year=2000):
    # create synthetic monthly series with a gentle trend
    dates = pd.date_range(start=f"{start_year}-01-01", periods=months, freq='M')
    # earnings: base + small noise
    earn = 1.0 + 0.001 * np.arange(months) + np.random.normal(0, 0.01, size=months)
    price = 15.0 * earn + 0.05 * np.arange(months)  # price roughly correlated with earn
    df = pd.DataFrame({'date': dates, 'price': price, 'earn': earn})
    return df


def test_compute_cape_basic():
    df = make_shiller_like(months=240, start_year=1990)
    cape_series = macro.compute_cape(df)
    # CAPE should be finite for months after at least some rolling period
    assert isinstance(cape_series, pd.Series)
    # Expect at least one value
    assert len(cape_series) > 0
    # Values should be positive
    assert (cape_series.dropna() > 0).all()


def test_classify_regime_rules():
    # High CAPE + inversion => Macro-Defensive
    out = macro.classify_regime(80.0, -0.5, -0.2)
    assert out['regime'] == 'Macro-Defensive'

    # High CAPE alone => Valuation-Rich
    out = macro.classify_regime(90.0, 0.5, 0.2)
    assert out['regime'] == 'Valuation-Rich'

    # inversion alone => Inversion-Warn
    out = macro.classify_regime(50.0, -0.2, -0.1)
    assert out['regime'] == 'Inversion-Warn'

    # low cape => Opportunistic
    out = macro.classify_regime(10.0, 0.1, 0.2)
    assert out['regime'] == 'Opportunistic'
