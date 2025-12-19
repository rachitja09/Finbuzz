import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import fmt_number
from utils import signals


def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    try:
        colmap = {c.lower(): c for c in df.columns}
        col = colmap.get(name.lower())
        if col is None:
            return pd.Series(dtype=float)
        return pd.Series(df[col], dtype=float).dropna()
    except Exception:
        return pd.Series(dtype=float)


def compute_kpis(df: pd.DataFrame, symbol: str | None = None) -> dict:
    """Compute lightweight KPIs from price history. Safe on empty data."""
    close = _safe_series(df, "close")
    if close.empty:
        return {"sharpe": np.nan, "sortino": np.nan, "vol": np.nan, "mdd": np.nan, "mom_3m": np.nan, "mom_6m": np.nan}

    k = {"sharpe": signals.sharpe_ratio(close, symbol=symbol),
         "sortino": signals.sortino_ratio(close, symbol=symbol),
         "vol": signals.volatility(close, symbol=symbol),
         "mdd": signals.max_drawdown(close, symbol=symbol),
         "mom_3m": signals.momentum(close, periods=63, symbol=symbol),
         "mom_6m": signals.momentum(close, periods=126, symbol=symbol)}
    return k


def render_kpis(symbol: str, df: pd.DataFrame) -> None:
    """Render a compact KPI row for risk-adjusted metrics."""
    k = compute_kpis(df, symbol=symbol)
    cols = st.columns(6)
    sharpe, sortino, vol, mdd, mom3, mom6 = (k.get("sharpe"), k.get("sortino"), k.get("vol"), k.get("mdd"), k.get("mom_3m"), k.get("mom_6m"))
    cols[0].metric("Sharpe", fmt_number(sharpe, 2) if np.isfinite(sharpe) else "—")
    cols[1].metric("Sortino", fmt_number(sortino, 2) if np.isfinite(sortino) else "—")
    cols[2].metric("Vol (ann)", fmt_number(vol, 2) if np.isfinite(vol) else "—")
    cols[3].metric("Max DD", f"{mdd*100:.1f}%" if mdd is not None and np.isfinite(mdd) else "—")
    cols[4].metric("Mom 3M", f"{mom3*100:.1f}%" if mom3 is not None and np.isfinite(mom3) else "—")
    cols[5].metric("Mom 6M", f"{mom6*100:.1f}%" if mom6 is not None and np.isfinite(mom6) else "—")
