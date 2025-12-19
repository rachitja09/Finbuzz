"""Lightweight risk analytics helpers: VaR, CVaR, and factor exposures.

This module intentionally keeps dependencies minimal (numpy, pandas, scipy
when available) and is suitable for inclusion in the Streamlit app.
"""
from __future__ import annotations

from typing import Literal, Tuple
import numpy as np
import pandas as pd


def returns_from_prices(price: pd.Series) -> pd.Series:
    """Compute simple returns (pct change) from a price series.

    Returns are aligned with price.index and drop NaNs.
    """
    if price is None or price.empty:
        return pd.Series(dtype=float)
    return price.pct_change().dropna()


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (positive number as loss fraction)"""
    if returns is None or len(returns) == 0:
        return 0.0
    q = 1.0 - confidence
    val = -np.percentile(returns.dropna(), q * 100.0)
    return float(val)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR using mean and std deviation."""
    if returns is None or len(returns) == 0:
        return 0.0
    mu = float(np.nanmean(returns))
    sigma = float(np.nanstd(returns, ddof=1))
    # no-op: sqrt not required here

    try:
        # inverse CDF of normal
        import scipy.stats as ss

        z = ss.norm.ppf(1.0 - confidence)
    except Exception:
        # simple fallback: sample large normal and estimate quantile
        z = float(np.percentile(np.random.normal(size=1000000), (1.0 - confidence) * 100.0))

    var = -(mu + sigma * z)
    return float(var)


def portfolio_returns_from_prices(price_df: pd.DataFrame, weights: pd.Series | dict | None = None) -> pd.Series:
    """Compute portfolio returns from a DataFrame of prices (columns = tickers).

    weights may be a Series indexed by columns or a dict. If None, equal weights.
    """
    if price_df is None or price_df.empty:
        return pd.Series(dtype=float)
    # compute returns per asset
    ret = price_df.pct_change().dropna()
    if weights is None:
        w = pd.Series(1.0 / ret.shape[1], index=ret.columns)
    else:
        w = pd.Series(weights) if not isinstance(weights, pd.Series) else weights
        # normalize
        s = float(w.sum()) if float(w.sum()) != 0.0 else 1.0
        w = w / s
    # align
    w = w.reindex(ret.columns).fillna(0.0)
    # portfolio returns are dot product across columns
    port = ret.dot(w)
    return port


def portfolio_var(
    price_df: pd.DataFrame,
    weights: pd.Series | dict | None = None,
    confidence: float = 0.95,
    method: Literal["historical", "parametric"] = "historical",
) -> Tuple[float, float]:
    """Compute portfolio VaR and CVaR (expected shortfall) as fractions.

    Returns (VaR, CVaR). VaR is positive number representing loss fraction.
    """
    port_ret = portfolio_returns_from_prices(price_df, weights)
    if port_ret.empty:
        return 0.0, 0.0
    if method == "parametric":
        var = parametric_var(port_ret, confidence=confidence)
    else:
        var = historical_var(port_ret, confidence=confidence)

    # CVaR: average loss beyond VaR level
    losses = -port_ret[port_ret <= -var]
    if len(losses) == 0:
        cvar = var
    else:
        cvar = float(losses.mean())
    return float(var), float(cvar)


def factor_exposures(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Compute factor exposures (betas) by OLS regression of each asset in `returns` onto `factors`.

    Both inputs should be aligned on index (dates). Returns a DataFrame of betas (columns=factors).
    """
    if returns is None or factors is None or returns.empty or factors.empty:
        return pd.DataFrame()
    # align indexes
    common = returns.index.intersection(factors.index)
    if len(common) == 0:
        return pd.DataFrame()
    r = returns.loc[common]
    f = factors.loc[common]
    # add constant
    X = f.copy()
    X.insert(0, "const", 1.0)
    betas = {}
    for col in r.columns:
        y = r[col].fillna(0.0)
        try:
            # Solve OLS via numpy lstsq
            coef, *_ = np.linalg.lstsq(np.asarray(X.values), np.asarray(y.values), rcond=None)
            betas[col] = pd.Series(coef, index=X.columns)
        except Exception:
            betas[col] = pd.Series(dtype=float)
    if not betas:
        return pd.DataFrame()
    out = pd.DataFrame(betas).T
    return out
