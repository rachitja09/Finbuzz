from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


def _safe_last(series: pd.Series) -> float:
    try:
        return float(pd.to_numeric(series.iloc[-1], errors="coerce"))
    except Exception:
        return float("nan")


def detect_trend(df: pd.DataFrame) -> str:
    """Detect a simple trend using EMA20 and EMA50.

    Returns one of: 'uptrend', 'downtrend', 'sideways', or 'unknown'.
    """
    try:
        if "ema20" in df.columns and "ema50" in df.columns:
            ema20 = pd.to_numeric(df["ema20"], errors="coerce").dropna()
            ema50 = pd.to_numeric(df["ema50"], errors="coerce").dropna()
            if len(ema20) < 3 or len(ema50) < 3:
                return "unknown"
            # Look at last values and recent slope
            last20 = ema20.iloc[-1]
            last50 = ema50.iloc[-1]
            slope20 = ema20.iloc[-1] - ema20.iloc[-3]
            slope50 = ema50.iloc[-1] - ema50.iloc[-3]
            if last20 > last50 and slope20 > 0 and slope50 >= 0:
                return "uptrend"
            if last20 < last50 and slope20 < 0 and slope50 <= 0:
                return "downtrend"
            return "sideways"
    except Exception:
        pass
    return "unknown"


def detect_rsi_signal(df: pd.DataFrame) -> str:
    """Return 'oversold', 'overbought', or 'neutral' based on RSI column."""
    try:
        if "rsi" in df.columns and len(df["rsi"]) > 0:
            val = _safe_last(df["rsi"])
            if np.isnan(val):
                return "neutral"
            if val < 30:
                return "oversold"
            if val > 70:
                return "overbought"
    except Exception:
        pass
    return "neutral"


def detect_bollinger_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Return bollinger-related signals: percent %B, bandwidth, and a label."""
    out: Dict[str, Any] = {"pct": None, "bandwidth": None, "label": "neutral"}
    try:
        if all(k in df.columns for k in ("bb_low", "bb_up", "bb_mid", "close")):
            low = _safe_last(df["bb_low"])
            up = _safe_last(df["bb_up"])
            mid = _safe_last(df["bb_mid"])
            close = _safe_last(df["close"])
            if np.isfinite(low) and np.isfinite(up) and (up - low) != 0:
                pct = (close - low) / (up - low)
                bw = (up - low) / mid if np.isfinite(mid) and mid != 0 else None
                out.update({"pct": float(pct), "bandwidth": float(bw) if bw is not None else None})
                if pct < 0.1:
                    out["label"] = "below_lower"
                elif pct > 0.9:
                    out["label"] = "above_upper"
                else:
                    out["label"] = "middle"
    except Exception:
        pass
    return out


def detect_volume_signal(df: pd.DataFrame) -> str:
    """Compare latest volume to 20-day average volume."""
    try:
        if "volume" in df.columns and len(df["volume"]) >= 1:
            vol = pd.to_numeric(df["volume"], errors="coerce").dropna()
            if len(vol) == 0:
                return "normal"
            last = vol.iloc[-1]
            v20 = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else np.nan
            if np.isfinite(v20) and v20 > 0:
                ratio = float(last / v20)
                if ratio > 2.0:
                    return "spike"
                if ratio < 0.5:
                    return "low"
                return "normal"
    except Exception:
        pass
    return "normal"


def analyze_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Run a set of quick indicator detectors and return a summary dict.

    Keys: trend, rsi, bollinger (dict), volume
    """
    out = {
        "trend": detect_trend(df),
        "rsi": detect_rsi_signal(df),
        "bollinger": detect_bollinger_signal(df),
        "volume": detect_volume_signal(df),
    }
    try:
        # add MACD crossover detection if we have enough data
        macd_sig = "neutral"
        if "close" in df.columns and len(df["close"]) >= 26:
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            if len(macd) >= 2:
                last_macd = macd.iloc[-1]
                prev_macd = macd.iloc[-2]
                last_sig = signal.iloc[-1]
                prev_sig = signal.iloc[-2]
                # bullish crossover: MACD crosses above signal
                if prev_macd <= prev_sig and last_macd > last_sig:
                    macd_sig = "bullish"
                elif prev_macd >= prev_sig and last_macd < last_sig:
                    macd_sig = "bearish"
        out["macd"] = macd_sig
    except Exception:
        out["macd"] = "neutral"
    return out
