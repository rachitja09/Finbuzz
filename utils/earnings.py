import streamlit as st
import requests
import datetime as dt
import pandas as pd


def _get_finnhub_key() -> str | None:
    try:
        from config import get_runtime_key

        return get_runtime_key("FINNHUB_API_KEY")
    except Exception:
        # fallback to env var
        v = __import__("os").environ.get("FINNHUB_API_KEY")
        return v if v else None


@st.cache_data(ttl=300)
def days_until_next_earnings(symbol: str) -> int | None:
    """Return days until next earnings date, or None if unknown/failed.

    Cached for 5 minutes to avoid hitting API repeatedly.
    """
    fh = _get_finnhub_key()
    if not fh:
        return None
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"symbol": symbol, "token": fh}, timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json() or {}
        # Finnhub returns 'earningsCalendar' structure; read the first upcoming
        ec = data.get("earningsCalendar") or []
        if ec:
            nxt = ec[0].get("date")
            if not nxt:
                return None
            d = dt.datetime.strptime(nxt, "%Y-%m-%d").date()
            # use timezone-aware UTC now
            now = dt.datetime.now(dt.timezone.utc).date()
            return (d - now).days
    except Exception:
        return None
    return None


@st.cache_data(ttl=300)
def get_vix() -> float | None:
    """Return a float VIX value or None. Cached to reduce calls."""
    try:
        resp = requests.get("https://query1.finance.yahoo.com/v7/finance/quote", params={"symbols": "^VIX"}, timeout=5)
        if resp.status_code != 200:
            return None
        dat = resp.json().get("quoteResponse", {}).get("result", [])
        if not dat:
            return None
        val = dat[0].get("regularMarketPrice")
        try:
            return float(val)
        except Exception:
            return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_next_earnings_info(symbol: str) -> dict:
    """Return a dict with keys: date (ISO str), source (str), estimate (float|None), raw (dict).

    Tries Finnhub first (if API key present), then falls back to yfinance calendar if available.
    Returns empty dict on failure.
    """
    from typing import Any
    out: dict[str, Any | None] = {"date": None, "source": None, "estimate": None, "raw": None}
    # Try Finnhub
    fh = _get_finnhub_key()
    if fh:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"symbol": symbol, "token": fh}, timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json() or {}
                ec = data.get("earningsCalendar") or []
                if ec:
                    first = ec[0]
                    nxt = first.get("date")
                    out["date"] = nxt
                    out["source"] = "finnhub"
                    out["estimate"] = first.get("epsEstimate") or None
                    out["raw"] = first
                    return out
        except Exception:
            pass

    # fallback to yfinance if available (no API key required but optional dep)
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        cal = getattr(t, "calendar", None) or {}
        # calendar can be a DataFrame-like dict; try to find a date-like entry
        if isinstance(cal, dict):
            # attempt common keys
            for v in cal.values():
                try:
                    # v may be a pandas Timestamp or string
                    if hasattr(v, "date"):
                        out["date"] = str(v.date())
                        out["source"] = "yfinance"
                        out["raw"] = cal
                        return out
                except Exception:
                    continue
    except Exception:
        pass

    return out


def earnings_impact_text(symbol: str, days_until: int | None) -> str:
    """Return a short human-friendly sentence about how upcoming earnings might impact the stock.

    Uses simple heuristics: nearer earnings increases volatility/risk; smaller companies or low coverage increase surprise potential.
    """
    if days_until is None:
        return "No upcoming earnings date found — impact unknown."
    if days_until < 0:
        return "Earnings have recently occurred; check the latest release and guidance for potential impact."
    if days_until == 0:
        return "Earnings today — expect elevated volatility and volume; results and guidance will likely move the stock intraday."
    if days_until <= 2:
        return "Earnings within 48 hours — many traders avoid initiating new positions due to elevated event risk; consider tightening stops."
    if days_until <= 7:
        return "Earnings this week — implied volatility often rises into earnings; options-sensitive traders should expect higher premiums."
    if days_until <= 30:
        return "Earnings within the month — news flow and analyst updates may start to appear; moderate event risk."
    return "Earnings are more than a month away — normal trading risk applies until the event window approaches."


@st.cache_data(ttl=600)
def get_last_earnings_summary(symbol: str) -> dict:
    """Return summary of the most-recent (past) earnings report.

    Returns dict with keys: date (ISO str), actual (float|None), estimate (float|None), surprise_pct (float|None), post_move_pct (float|None), source, raw
    """
    from typing import Any
    out: dict[str, Any | None] = {"date": None, "actual": None, "estimate": None, "surprise_pct": None, "post_move_pct": None, "source": None, "raw": None}
    try:
        # Try Finnhub past earnings endpoint first
        fh = _get_finnhub_key()
        if fh:
            try:
                r = requests.get("https://finnhub.io/api/v1/stock/earnings", params={"symbol": symbol, "token": fh}, timeout=10)
                if r.status_code == 200:
                    arr = r.json() or []
                    # arr is list of earnings ordered by date (may be future and past); pick most recent past date
                    import datetime as _dt
                    today = _dt.date.today()
                    past = [a for a in arr if a.get("date")]
                    cand = None
                    for a in sorted(past, key=lambda x: x.get("date"), reverse=True):
                        try:
                            d = _dt.datetime.strptime(a.get("date"), "%Y-%m-%d").date()
                            if d <= today:
                                cand = a
                                break
                        except Exception:
                            continue
                    if cand:
                        out["date"] = cand.get("date")
                        out["estimate"] = cand.get("estimate") if cand.get("estimate") is not None else None
                        out["source"] = "finnhub"
                        out["raw"] = cand
            except Exception:
                pass

        # Fallback: yfinance, use Ticker.earnings or events; then compute post-earnings move via history
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            # yfinance may provide earnings_dates or earnings_history; try several attributes
            last_date = None
            estimate = None
            # Attempt attribute 'earnings' (quarterly earnings table) or 'get_earnings_history'
            try:
                if hasattr(t, "earnings") and t.earnings is not None and not t.earnings.empty:
                    # earnings DataFrame with index as year; take last row
                    edf = t.earnings
                    # try to infer last reported year-quarter index
                    # yfinance doesn't always include date; skip detailed matching
                # try get_earnings_dates
            except Exception:
                pass
            # Try get_earnings_dates if available
            try:
                if hasattr(t, "get_earnings_dates"):
                    ed = t.get_earnings_dates(limit=8)
                    # ed may be a DataFrame-like object; ensure it's not None and not empty
                    if ed is not None and hasattr(ed, "empty") and not ed.empty:
                        edf = ed
                        # iterate rows in reverse order safely
                        try:
                            iterator = edf.iloc[::-1].iterrows()
                        except Exception:
                            try:
                                iterator = reversed(list(edf.iterrows()))
                            except Exception:
                                iterator = []
                        for _, row in iterator:
                            # row may be a pandas Series or a dict-like mapping
                            dval = None
                            try:
                                if isinstance(row, dict):
                                    dval = row.get("Earnings Date") or row.get("startdatetime") or row.get("date")
                                    estimate = row.get("EPS Estimate") or row.get("estimate") or estimate
                                else:
                                    # pandas Series: use .get or index membership
                                    if "Earnings Date" in row.index:
                                        dval = row.get("Earnings Date")
                                    elif "startdatetime" in row.index:
                                        dval = row.get("startdatetime")
                                    elif "date" in row.index:
                                        dval = row.get("date")
                                    # estimate columns may vary
                                    try:
                                        if "EPS Estimate" in row.index:
                                            estimate = row.get("EPS Estimate") or estimate
                                        elif "estimate" in row.index:
                                            estimate = row.get("estimate") or estimate
                                    except Exception:
                                        pass
                            except Exception:
                                dval = None
                            try:
                                if dval is not None:
                                    if hasattr(dval, "date"):
                                        last_date = str(dval.date())
                                    else:
                                        last_date = str(dval)
                                    break
                            except Exception:
                                continue
            except Exception:
                pass

            # If we found nothing via yfinance metadata, try earnings history via Finnhub fallback already or leave None
            # Compute post-earnings day price move if we have a date
            if out.get("date") is None and last_date:
                out["date"] = last_date
            # Use yfinance history to compute post-earnings price move if date available
            edate = out.get("date")
            if edate:
                import datetime as _dt
                try:
                    dt_obj = _dt.datetime.strptime(edate, "%Y-%m-%d").date()
                    # fetch two days of history around the date
                    hist = t.history(start=dt_obj - _dt.timedelta(days=3), end=dt_obj + _dt.timedelta(days=5), interval="1d")
                    if hist is not None and len(hist) >= 2:
                        # find the index row for the earnings date or nearest previous trading day
                        # Normalize index to dates
                        try:
                            hist_index = [pd.to_datetime(i).date() for i in hist.index]
                        except Exception:
                            hist_index = [i.date() for i in hist.index]
                        # find last trading day on or before earnings date
                        prior = None
                        post = None
                        for i, d in enumerate(hist_index):
                            if d <= dt_obj:
                                prior = i
                        # prior is index of last <= dt_obj
                        if prior is not None:
                            # try to set post = prior+1 if exists
                            if prior + 1 < len(hist_index):
                                post = prior + 1
                            # compute prices
                            try:
                                prior_close = float(hist.iloc[prior]["Close"])
                                if post is not None:
                                    post_close = float(hist.iloc[post]["Close"])
                                else:
                                    post_close = None
                                if prior_close is not None and post_close is not None:
                                    out["post_move_pct"] = (post_close - prior_close) / prior_close * 100.0
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        # Compute surprise_pct if we have actual and estimate
        try:
            a = out.get("actual")
            e = out.get("estimate")
            if a is None or e is None:
                # try to pull values from raw if Finnhub returned them as strings
                raw = out.get("raw") or {}
                if raw:
                    try:
                        if out.get("actual") is None:
                            v = raw.get("actual")
                            if v is not None:
                                try:
                                    out["actual"] = float(v)
                                except Exception:
                                    # sometimes strings like '-' or 'N/A'
                                    pass
                    except Exception:
                        pass
                    try:
                        if out.get("estimate") is None:
                            v = raw.get("estimate")
                            if v is not None:
                                try:
                                    out["estimate"] = float(v)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                a = out.get("actual")
                e = out.get("estimate")
            if a is not None and e is not None and e != 0:
                out["surprise_pct"] = (a - e) / abs(e) * 100.0
        except Exception:
            pass
    except Exception:
        pass
    return out
