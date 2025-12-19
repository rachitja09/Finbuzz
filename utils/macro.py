from __future__ import annotations
import math
import statistics
from typing import Dict, List, Optional, Tuple

import io
import requests
import datetime as dt
import numpy as np
import pandas as pd

from utils.cache import cache_get, cache_set
from utils import rates


SHILLER_CSV_URLS = [
    "https://www.econ.yale.edu/~shiller/data/ie_data.csv",
]


def _fetch_shiller_csv() -> Optional[str]:
    for url in SHILLER_CSV_URLS:
        try:
            try:
                from utils.retry import requests_get_with_retry
                r = requests_get_with_retry(url, timeout=10, retries=2)
            except Exception:
                r = requests.get(url, timeout=10)
            if r is not None and getattr(r, 'status_code', None) == 200 and getattr(r, 'text', None):
                return r.text
        except Exception:
            continue
    return None


def _parse_shiller_csv(txt: str) -> pd.DataFrame:
    # Attempt robust parsing: the file is CSV-like with header in first line.
    f = io.StringIO(txt)
    # Some versions have comment lines; pandas can handle via comment='#'
    df = pd.read_csv(f, comment='#')
    # Normalize columns: look for 'Date' or first column as date
    if df.shape[1] < 3:
        raise ValueError("unexpected shiller csv format")
    # heuristics: find 'P' (price) and 'E' (earnings) columns - common labels include 'P' and 'E' or 'Price'/'Earnings'
    cols = [c.strip() for c in df.columns]
    # Try a few common name variants
    def find_col(cands):
        for c in cols:
            for cand in cands:
                if cand.lower() in c.lower():
                    return c
        return None

    date_col = find_col(['Date', 'Yr', 'Year']) or df.columns[0]
    price_col = find_col(['P', 'Price', 'Real Price', 'S&P'])
    earn_col = find_col(['E', 'Earnings', 'Real Earnings'])
    cpi_col = find_col(['CPI'])

    if price_col is None or earn_col is None:
        # try fallback: some Shiller tables have Price (P) and Earnings (E) as second/third columns
        price_col = df.columns[1]
        earn_col = df.columns[2]

    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    out['price'] = pd.to_numeric(df[price_col], errors='coerce')
    out['earn'] = pd.to_numeric(df[earn_col], errors='coerce')
    if cpi_col:
        out['cpi'] = pd.to_numeric(df[cpi_col], errors='coerce')
    return out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)


def fetch_shiller_series() -> pd.DataFrame:
    """Return parsed Shiller series DataFrame with columns date, price, earn, cpi if available.

    Returns empty DataFrame on failure.
    """
    cached = cache_get('shiller:series')
    if cached is not None:
        return cached
    try:
        txt = _fetch_shiller_csv()
        if not txt:
            return pd.DataFrame()
        df = _parse_shiller_csv(txt)
        cache_set('shiller:series', df)
        return df
    except Exception:
        return pd.DataFrame()


def compute_cape(df: pd.DataFrame) -> pd.Series:
    """Compute CAPE series (price divided by 10-year average of earnings).

    Expects df with 'date' and 'earn' and 'price' as numeric columns. Returns a pd.Series indexed by date.
    """
    if df.empty or 'earn' not in df.columns or 'price' not in df.columns:
        return pd.Series(dtype=float)
    s = df.set_index('date').sort_index()
    # Use rolling window of 120 months (10 years) on earnings
    earn = s['earn'].astype(float)
    rolling_mean = earn.rolling(window=120, min_periods=60).mean()
    cape = s['price'].astype(float) / rolling_mean
    cape = cape.replace([np.inf, -np.inf], np.nan).dropna()
    return cape


def get_cape_latest() -> Tuple[Optional[float], Optional[float]]:
    """Return (cape_value, cape_percentile) or (None, None) if unavailable."""
    cached = cache_get('shiller:cape:latest')
    if cached is not None:
        return cached
    df = fetch_shiller_series()
    if df.empty:
        cache_set('shiller:cape:latest', (None, None))
        return None, None
    try:
        cape_series = compute_cape(df)
        if cape_series.empty:
            cache_set('shiller:cape:latest', (None, None))
            return None, None
        latest = float(cape_series.iloc[-1])
        # percentile over history
        pct = float((cape_series < latest).mean()) * 100.0
        cache_set('shiller:cape:latest', (latest, pct))
        return latest, pct
    except Exception:
        cache_set('shiller:cape:latest', (None, None))
        return None, None


def get_yield_spreads() -> Dict[str, Optional[float]]:
    """Return dictionary with latest yields and spreads. Uses utils.rates.fetch_rate_series when possible.

    Keys: '10y', '2y', '3m', '2y_10y', '3m_10y'
    """
    cached = cache_get('macro:yields:latest')
    if cached is not None:
        return cached
    out: Dict[str, Optional[float]] = {k: None for k in ['10y', '2y', '3m', '2y_10y', '3m_10y']}
    try:
        fred_key = None
        try:
            # use rates.fetch_rate_series which expects fred_key arg; try get runtime key via rates._get_fred_key if available
            if hasattr(rates, '_get_fred_key'):
                fred_key = rates._get_fred_key()
        except Exception:
            fred_key = None

        # Try FRED series fetch for recent points
        ten = rates.fetch_rate_series('DGS10', fred_key, points=30) or []
        two = rates.fetch_rate_series('DGS2', fred_key, points=30) or []
        three = rates.fetch_rate_series('DGS3MO', fred_key, points=30) or []
        latest10 = float(ten[-1]) if ten and not np.isnan(ten[-1]) else None
        latest2 = float(two[-1]) if two and not np.isnan(two[-1]) else None
        latest3 = float(three[-1]) if three and not np.isnan(three[-1]) else None
        out['10y'] = latest10
        out['2y'] = latest2
        out['3m'] = latest3
        if latest10 is not None and latest2 is not None:
            out['2y_10y'] = latest2 - latest10
        if latest10 is not None and latest3 is not None:
            out['3m_10y'] = latest3 - latest10
    except Exception:
        pass
    cache_set('macro:yields:latest', out)
    return out


def classify_regime(cape_pct: Optional[float], spread_3m_10: Optional[float], spread_2y_10: Optional[float]) -> Dict[str, object]:
    """Return a simple regime dict: {'regime': str, 'reason': str, 'cape_pct': float, 'spread_3m_10': float, ...}

    Regime mapping is conservative and tunable.
    """
    regime = 'Neutral'
    reasons: List[str] = []
    try:
        if cape_pct is not None:
            if cape_pct > 75:
                reasons.append('High CAPE')
            elif cape_pct < 25:
                reasons.append('Low CAPE')
            else:
                reasons.append('Normal CAPE')
        if spread_3m_10 is not None:
            if spread_3m_10 < 0:
                reasons.append('3m-10y inverted')
        if spread_2y_10 is not None:
            if spread_2y_10 < 0:
                reasons.append('2y-10y inverted')

        # Conservative rules
        if ('High CAPE' in reasons) and (('3m-10y inverted' in reasons) or ('2y-10y inverted' in reasons)):
            regime = 'Macro-Defensive'
        elif 'High CAPE' in reasons:
            regime = 'Valuation-Rich'
        elif ('3m-10y inverted' in reasons) or ('2y-10y inverted' in reasons):
            regime = 'Inversion-Warn'
        elif 'Low CAPE' in reasons:
            regime = 'Opportunistic'
        else:
            regime = 'Neutral'
    except Exception:
        regime = 'Neutral'
    return {
        'regime': regime,
        'reason': '; '.join(reasons) if reasons else 'No macro signal',
        'cape_pct': cape_pct,
        'spread_3m_10': spread_3m_10,
        'spread_2y_10': spread_2y_10,
    }


def get_macro_regime() -> Dict[str, object]:
    """Fetch inputs and return the current regime dict."""
    cached = cache_get('macro:regime')
    if cached is not None:
        return cached
    cape_val, cape_pct = get_cape_latest()
    yields = get_yield_spreads()
    out = classify_regime(cape_pct, yields.get('3m_10y'), yields.get('2y_10y'))
    out['cape'] = cape_val
    out['cape_pct'] = cape_pct
    out['yields'] = yields
    cache_set('macro:regime', out)
    return out
