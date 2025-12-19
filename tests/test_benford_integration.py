import pandas as pd
import numpy as np
from utils.recommend import make_recommendation_enhanced


def make_uptrend_df(n=100, start_price=100.0):
    # simple synthetic uptrend with EMA20>EMA50 and oversold/overbought avoidance
    prices = np.linspace(start_price, start_price * 1.2, n)
    df = pd.DataFrame({
        "close": prices,
        "ema20": pd.Series(prices).ewm(span=20, adjust=False).mean(),
        "ema50": pd.Series(prices).ewm(span=50, adjust=False).mean(),
        "rsi": np.linspace(40, 50, n),
    })
    return df


def test_benford_penalty_applied():
    # Create an otherwise strong buy signal
    df = make_uptrend_df(120)

    # Construct a benign volume series that roughly follows Benford-like distribution
    # For simplicity, create volumes over several magnitudes using geometric progression
    volumes_good = np.logspace(3, 7, len(df)) * np.random.uniform(0.8, 1.2, len(df))
    df["volume"] = volumes_good

    verdict_good, reason_good = make_recommendation_enhanced(df, df.iloc[-1])

    # Now make an artificial bad volume series (e.g., constant repeated small values) to violate Benford
    volumes_bad = np.full(len(df), 1000.0)
    df_bad = df.copy()
    df_bad["volume"] = volumes_bad

    verdict_bad, reason_bad = make_recommendation_enhanced(df_bad, df_bad.iloc[-1])

    # The reasons for the bad case should include the Benford data-integrity mention
    assert "Data-integrity" in reason_bad or "Benford" in reason_bad

    # And the verdict for bad should be no stronger than the good verdict (reduced conviction)
    order = {"Sell": -1, "Hold": 0, "Buy": 1}
    assert order.get(verdict_bad, 0) <= order.get(verdict_good, 0)
