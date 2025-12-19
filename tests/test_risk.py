import pandas as pd
import numpy as np

from utils.risk import historical_var, portfolio_var


def test_historical_var_simple():
    # simple returns with clear downside
    r = pd.Series([-0.05, -0.02, 0.01, 0.02, -0.03])
    v = historical_var(r, confidence=0.8)
    assert v > 0


def test_portfolio_var_equal_weights():
    # two assets with identical returns; portfolio var should be similar to individual var / sqrt(2)
    dates = pd.date_range("2025-01-01", periods=100)
    a = pd.Series(np.random.normal(0.001, 0.02, size=100), index=dates).cumsum() + 100
    b = a * 1.0
    df = pd.DataFrame({"A": a, "B": b})
    var, cvar = portfolio_var(df, weights=None, confidence=0.95, method="historical")
    assert var >= 0
    assert cvar >= 0
