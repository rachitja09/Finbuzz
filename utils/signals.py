from __future__ import annotations
import math
from typing import Optional
from utils.cache import cache_get, cache_set

import numpy as np
import pandas as pd


def _annualize_factor(freq: int = 252) -> float:
    return math.sqrt(freq)


def returns_from_close(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change().dropna()


def sharpe_ratio(close: pd.Series, rf: float = 0.0, freq: int = 252, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        # cache key uses symbol and ts when provided
        if symbol:
            key = f"sharpe:{symbol}:{int(ts) if ts is not None else 'none'}:{freq}:{rf}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        r = returns_from_close(close)
        if r.empty:
            return float('nan')
        mean = r.mean() - (rf / freq)
        sd = r.std()
        if sd == 0 or not np.isfinite(sd):
            return float('nan')
        val = float(mean / sd * _annualize_factor(freq))
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return float('nan')


def sortino_ratio(close: pd.Series, rf: float = 0.0, freq: int = 252, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        if symbol:
            key = f"sortino:{symbol}:{int(ts) if ts is not None else 'none'}:{freq}:{rf}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        r = returns_from_close(close)
        if r.empty:
            return float('nan')
        downs = r[r < 0]
        if downs.empty:
            return float('nan')
        downside_std = downs.std()
        mean = r.mean() - (rf / freq)
        if downside_std == 0 or not np.isfinite(downside_std):
            return float('nan')
        val = float(mean / downside_std * _annualize_factor(freq))
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return float('nan')


def max_drawdown(close: pd.Series, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        if symbol:
            key = f"mdd:{symbol}:{int(ts) if ts is not None else 'none'}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        p = close.astype(float).dropna()
        if p.empty:
            return 0.0
        running_max = p.cummax()
        drawdown = (p - running_max) / running_max
        mdd = drawdown.min()
        val = float(abs(mdd))
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return 0.0


def calmar_ratio(close: pd.Series, freq: int = 252, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        if symbol:
            key = f"calmar:{symbol}:{int(ts) if ts is not None else 'none'}:{freq}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        r = returns_from_close(close)
        if r.empty:
            return float('nan')
        annual_return = float((1 + r.mean()) ** freq - 1)
        mdd = max_drawdown(close)
        if mdd == 0:
            return float('nan')
        val = float(annual_return / mdd)
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return float('nan')


def momentum(close: pd.Series, periods: int = 90, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        if symbol:
            key = f"mom:{symbol}:{int(ts) if ts is not None else 'none'}:{periods}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        p = close.astype(float).dropna()
        if len(p) < periods + 1:
            return float('nan')
        val = float(p.iloc[-1] / p.iloc[-periods - 1] - 1)
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return float('nan')


def volatility(close: pd.Series, freq: int = 252, symbol: str | None = None, ts: float | None = None) -> float:
    try:
        if symbol:
            key = f"vol:{symbol}:{int(ts) if ts is not None else 'none'}:{freq}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        r = returns_from_close(close)
        if r.empty:
            return float('nan')
        val = float(r.std() * _annualize_factor(freq))
        if symbol:
            cache_set(key, val)
        return val
    except Exception:
        return float('nan')


def rsi_from_series(close: pd.Series, period: int = 14, symbol: str | None = None, ts: float | None = None) -> Optional[float]:
    try:
        if symbol:
            key = f"rsi:{symbol}:{int(ts) if ts is not None else 'none'}:{period}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        p = close.astype(float).dropna()
        if len(p) < period + 1:
            return None
        delta = p.diff()
        up = delta.clip(lower=0).rolling(period).mean()
        down = -delta.clip(upper=0).rolling(period).mean()
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        out = float(val) if np.isfinite(val) else None
        if symbol and out is not None:
            cache_set(key, out)
        return out
    except Exception:
        return None


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, symbol: str | None = None, ts: float | None = None) -> Optional[float]:
    try:
        if symbol:
            key = f"atr:{symbol}:{int(ts) if ts is not None else 'none'}:{period}"
            cached = cache_get(key)
            if cached is not None:
                return float(cached)
        h = high.astype(float).dropna()
        l = low.astype(float).dropna()
        c = close.astype(float).dropna()
        if len(c) < period + 1:
            return None
        prev_close = c.shift(1)
        tr1 = h - l
        tr2 = (h - prev_close).abs()
        tr3 = (l - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atrval = tr.rolling(period).mean().iloc[-1]
        out = float(atrval) if np.isfinite(atrval) else None
        if symbol and out is not None:
            cache_set(key, out)
        return out
    except Exception:
        return None
