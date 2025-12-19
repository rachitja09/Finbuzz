"""Lightweight indicators shim used by pages.

This module re-exports a few functions from the internal data_fetchers implementation
so pages can import `from indicators import rsi, ema` without heavy optional deps.
"""
import pandas as pd
from data_fetchers.indicators import rsi as _rsi


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    out = _rsi(series, window)
    return pd.Series(out, index=series.index, dtype=float)


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    # Simple EMA wrapper returning a pandas Series
    out = series.ewm(span=span, adjust=False).mean()
    return pd.Series(out, index=series.index, dtype=float)
