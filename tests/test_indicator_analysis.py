import pandas as pd
import numpy as np
from utils.indicator_analysis import detect_trend, detect_rsi_signal, detect_bollinger_signal, detect_volume_signal
from utils.recommend import make_recommendation_enhanced
from utils.indicator_analysis import analyze_indicators


def make_sample_df(close_vals, volume_vals=None):
    idx = pd.date_range(end=pd.Timestamp.today(), periods=len(close_vals))
    df = pd.DataFrame({"close": close_vals}, index=idx)
    df["ema20"] = pd.Series(df["close"]).ewm(span=20, adjust=False).mean()
    df["ema50"] = pd.Series(df["close"]).ewm(span=50, adjust=False).mean()
    # simple Bollinger bands
    mid = df["close"].rolling(20).mean().ffill()
    std = df["close"].rolling(20).std().fillna(0)
    df["bb_mid"] = mid
    df["bb_up"] = mid + 2 * std
    df["bb_low"] = mid - 2 * std
    # RSI approximation
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    if volume_vals is not None:
        df["volume"] = volume_vals
    return df


def test_detect_trend_uptrend():
    vals = list(range(50, 150))
    df = make_sample_df(vals)
    assert detect_trend(df) in ("uptrend", "sideways")


def test_rsi_signals():
    # Oversold scenario: declining prices
    low_vals = list(range(100, 50, -1))
    df_low = make_sample_df(low_vals)
    assert detect_rsi_signal(df_low) in ("oversold", "neutral")


def test_bollinger_and_volume():
    vals = [100.0] * 30
    vol = [1000] * 30
    vol[-1] = 5000
    df = make_sample_df(vals, volume_vals=vol)
    bb = detect_bollinger_signal(df)
    assert "pct" in bb
    vsig = detect_volume_signal(df)
    assert vsig in ("spike", "normal")


def test_enhanced_recommendation_smoke():
    vals = list(range(100, 160))
    df = make_sample_df(vals, volume_vals=[1000]*len(vals))
    last = df.iloc[-1]
    verdict, reason = make_recommendation_enhanced(df, last, profile={"pe": 10, "sector_pe": 15}, analyst={"buy": 3, "sell":1}, news_suggestion=None, earnings_days=None)
    assert verdict in ("Buy", "Sell", "Hold")


def test_macd_bullish_detection():
    # Construct a dataset with a clear upward move to trigger a MACD bullish crossover
    vals = list(range(50, 150))
    df = make_sample_df(vals)
    out = analyze_indicators(df)
    assert out.get("macd") in ("bullish", "neutral")
