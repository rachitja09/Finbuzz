import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from utils.helpers import fmt_number
from utils import signals


def _close_series(df: pd.DataFrame) -> pd.Series:
    try:
        colmap = {c.lower(): c for c in df.columns}
        c = colmap.get("close") or colmap.get("adjclose") or colmap.get("adj_close")
        if c is None:
            return pd.Series(dtype=float)
        return pd.Series(df[c], dtype=float).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _annualized_trend(close: pd.Series) -> Optional[float]:
    """Estimate annualized drift via linear regression on log prices; returns None if insufficient data."""
    try:
        if close is None or len(close) < 60:  # need at least ~3 months of data
            return None
        # use simple daily index as time; slope on log-price approximates drift
        y = np.log(close.to_numpy())
        x = np.arange(len(close))
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        # convert daily log-slope to annualized simple rate
        return float(np.exp(slope * 252) - 1)
    except Exception:
        return None


def compute_projection(df: pd.DataFrame, horizon_years: int = 10, symbol: str | None = None) -> dict:
    close = _close_series(df)
    if close.empty:
        return {"expected": np.nan, "growth": np.nan, "upper": np.nan, "lower": np.nan}

    last_px = float(close.iloc[-1]) if len(close) else np.nan
    drift = _annualized_trend(close)
    vol = signals.volatility(close, symbol=symbol)  # annualized

    if drift is None or not np.isfinite(drift):
        return {"expected": last_px, "growth": np.nan, "upper": np.nan, "lower": np.nan}

    expected = float(last_px * ((1 + drift) ** horizon_years)) if np.isfinite(last_px) else np.nan

    # crude fan using +/- 1 vol band on drift
    vol_band = vol if vol is not None and np.isfinite(vol) else 0.0
    upper = float(last_px * ((1 + drift + vol_band) ** horizon_years)) if np.isfinite(last_px) else np.nan
    lower = float(last_px * ((1 + max(drift - vol_band, -0.99)) ** horizon_years)) if np.isfinite(last_px) else np.nan

    return {"expected": expected, "growth": drift, "upper": upper, "lower": lower}


def render_forecast(symbol: str, df: pd.DataFrame, horizon_years: int = 10) -> None:
    """Render a lightweight, heuristic long-horizon projection banner.

    This is NOT investment advice and is intentionally conservative: it uses
    a simple log-price trend fit and volatility band to convey uncertainty.
    """
    proj = compute_projection(df, horizon_years=horizon_years, symbol=symbol)
    if proj is None:
        return

    growth = proj.get("growth")
    expected = proj.get("expected")
    upper = proj.get("upper")
    lower = proj.get("lower")

    with st.expander("Long-horizon trajectory (heuristic)", expanded=False):
        if expected is None or not np.isfinite(expected):
            st.info("Not enough history to form a trajectory.")
            return
        st.markdown("""
        These numbers are heuristic projections based on historical drift and volatility. They are **not** predictions or advice; long-horizon outcomes are highly uncertain.
        """)
        cols = st.columns(3)
        cols[0].metric(f"{horizon_years}y expected", f"${fmt_number(expected, 2)}")
        cols[1].metric("Drift (annualized)", f"{growth*100:+.2f}%" if growth is not None and np.isfinite(growth) else "—")
        if np.isfinite(upper) and np.isfinite(lower):
            cols[2].metric("Range (±vol band)", f"${fmt_number(lower,2)} – ${fmt_number(upper,2)}")
        else:
            cols[2].metric("Range (±vol band)", "—")
