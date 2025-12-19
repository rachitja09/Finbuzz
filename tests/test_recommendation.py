import pandas as pd
from utils.recommend import make_recommendation


def test_recommend_buy_signal():
    # create a small latest_row with oversold RSI and price > ema20 > ema50
    latest = pd.Series({
        "close": 100.0,
        "ema20": 95.0,
        "ema50": 90.0,
        "rsi": 25.0,
    })
    verdict, reason = make_recommendation(latest, profile=None, boll_pct=0.05)
    assert verdict == "Buy"
    assert "RSI oversold" in reason or "Uptrend" in reason


def test_recommend_sell_signal():
    latest = pd.Series({
        "close": 50.0,
        "ema20": 60.0,
        "ema50": 70.0,
        "rsi": 75.0,
    })
    verdict, reason = make_recommendation(latest, profile=None, boll_pct=0.95)
    assert verdict == "Sell"
    assert "Downtrend" in reason or "RSI overbought" in reason
