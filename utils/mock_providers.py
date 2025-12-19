"""Lightweight mock provider data for development.

Enable by setting environment variable DEV_MOCK=1 or when live fetchers return no data.
This module intentionally avoids secrets and provides tiny deterministic samples.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def profile(sym: str) -> dict:
    base = 150.0 + (abs(hash(sym)) % 50)
    sectors = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical"]
    sector = sectors[abs(hash(sym)) % len(sectors)]
    return {
        "symbol": sym,
        "companyName": f"{sym} Inc.",
        "sector": sector,
        "industry": "Software" if sector == "Technology" else "General",
        "price": float(base),
        "mktCap": int(50_000_000_000 + (abs(hash(sym)) % 200) * 1_000_000_00),
        "beta": float(0.8 + (abs(hash(sym)) % 5) * 0.1),
    }

def ratios_ttm(sym: str) -> dict:
    return {
        "peRatioTTM": 25.0 + (hash(sym) % 10),
        "priceToBookRatioTTM": 6.2,
        "netProfitMarginTTM": 0.21,
        "returnOnEquityTTM": 0.25,
        "returnOnAssetsTTM": 0.08,
        "currentRatioTTM": 1.6,
        "debtEquityRatioTTM": 0.4,
        "freeCashFlowPerShareTTM": 5.5,
    }

def finnhub_price_target(sym: str) -> dict:
    p = 150.0 + (hash(sym) % 50)
    return {"symbol": sym, "targetMean": p * 1.1, "targetHigh": p * 1.5, "targetLow": p * 0.8}

def finnhub_reco(sym: str) -> dict:
    return {"strongBuy": 3, "buy": 7, "hold": 5, "sell": 0, "strongSell": 0}

def ohlc(sym: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    # simple synthetic OHLC series for 90 days
    rng = pd.date_range(end=pd.Timestamp.today(), periods=90, freq='B')
    base = 100.0 + (hash(sym) % 30)
    np_random = np.random.RandomState(abs(hash(sym)) % (2**32))
    price = base + np_random.randn(len(rng)).cumsum() * 0.5
    df = pd.DataFrame({
        "Open": price + np_random.randn(len(rng)) * 0.1,
        "High": price + np.abs(np_random.randn(len(rng)) * 0.3) + 0.1,
        "Low": price - np.abs(np_random.randn(len(rng)) * 0.3) - 0.1,
        "Close": price,
        "Volume": (1000000 + (np_random.rand(len(rng)) * 100000)).astype(int),
    }, index=rng)
    return df
