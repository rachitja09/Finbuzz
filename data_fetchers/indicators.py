# data_fetchers/indicators.py
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / (loss.replace(0, 1e-9))
    # ensure a Series return for static checkers
    out = 100 - (100 / (1 + rs))
    return pd.Series(out, index=series.index, dtype=float)


def vol_ma(vol: pd.Series, window: int = 20) -> pd.Series:
    out = vol.rolling(window=window, min_periods=1).mean()
    return pd.Series(out, index=vol.index, dtype=float)
