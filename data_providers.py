from __future__ import annotations
import datetime as dt
from typing import Any, Dict, List, Optional
import requests
from utils.http import cached_get
import pandas as pd
# yfinance is optional for tests; guard import so modules can be imported without it.
yf: Any = None
try:
    import yfinance as _yf  # type: ignore
    yf = _yf
except Exception:  # pragma: no cover - environment dependent
    yf = None

# Curated US large-cap tickers used as a fallback when market-providers are unavailable
_MAJOR_US_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","V",
    "UNH","JNJ","PG","MA","HD","BAC","WMT","DIS","ADBE","CMCSA","NFLX",
    "INTC","ORCL","T","XOM","CVX","KO","PEP","NKE","CRM","ABNB","PYPL",
]

# Endpoints
FINNHUB = "https://finnhub.io/api/v1"     # Symbol search, quote, company news  (docs) :contentReference[oaicite:4]{index=4}
FMP = "https://financialmodelingprep.com/api"  # v3 + v4 family of endpoints (docs) :contentReference[oaicite:5]{index=5}

def _get(url: str, params: Dict[str, Any] | None = None, timeout: int = 20) -> Any:
    # Use cached_get to reduce API calls and add polite per-host throttling.
    # cached_get will attempt to parse JSON and return text otherwise.
    return cached_get(url, params=params, timeout=timeout)

# ---------------------- Symbol search / suggestions ----------------------

def suggest_symbols(query: str, token_finnhub: str, token_fmp: str = "") -> List[Dict[str, str]]:
    """
    Return list like [{'symbol': 'AAPL', 'description': 'Apple Inc'}].
    Uses Finnhub /search; falls back to FMP /v3/search when needed.  :contentReference[oaicite:6]{index=6}
    """
    out: List[Dict[str, str]] = []
    if not query or len(query.strip()) < 2:
        return out
    # Finnhub
    try:
        j = _get(f"{FINNHUB}/search", {"q": query, "token": token_finnhub})
        for it in j.get("result", [])[:20]:
            sym = it.get("symbol") or ""
            desc = it.get("description") or ""
            if sym and desc:
                out.append({"symbol": sym, "description": desc})
    except Exception:
        pass
    # FMP fallback
    if len(out) < 5 and token_fmp:
        try:
            j = _get(f"{FMP}/v3/search", {"query": query, "limit": 20, "apikey": token_fmp})
            for it in j[:20]:
                sym = it.get("symbol") or it.get("symbolInput") or ""
                name = it.get("name") or it.get("companyName") or ""
                if sym and name:
                    out.append({"symbol": sym, "description": name})
        except Exception:
            pass
    # Dedupe
    seen = set()
    dedup = []
    for it in out:
        if it["symbol"] not in seen:
            seen.add(it["symbol"])
            dedup.append(it)
    return dedup[:20]

# ---------------------- Prices / quotes ----------------------

def price_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    # Defensive: if yfinance isn't available return empty df
    ticker_cls = getattr(yf, "Ticker", None)
    if ticker_cls is None:
        return pd.DataFrame()
    try:
        df = ticker_cls(symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # normalize column names and index
        df = df.rename(columns=str.title).reset_index()
        # Normalize index column to `date`
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "date" not in df.columns and df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "date"})
        return df
    except Exception:
        return pd.DataFrame()

def quote_now(symbol: str, finnhub_token: str) -> Dict[str, Any]:
    try:
        j = _get(f"{FINNHUB}/quote", {"symbol": symbol, "token": finnhub_token})
        return {"c": j.get("c"), "pc": j.get("pc"), "d": j.get("d"), "dp": j.get("dp")}
    except Exception:
        return {}

# ---------------------- News ----------------------

def company_news(symbol: str, token_finnhub: str, days: int = 3) -> List[Dict[str, Any]]:
    """Prefer Finnhub company news (by ticker) to avoid non-finance hits.  :contentReference[oaicite:7]{index=7}"""
    try:
        to = dt.date.today()
        frm = to - dt.timedelta(days=days)
        # first attempt - correct token name
        j = _get(f"{FINNHUB}/company-news", {
            "symbol": symbol, "from": frm.isoformat(), "to": to.isoformat(), "token": token_finnhub
        })
        out: List[Dict[str, Any]] = []
        items = []
        if isinstance(j, list):
            items = j
        elif isinstance(j, dict):
            # Finnhub may return list directly; other APIs may wrap under 'news' or 'articles'
            items = j.get("news", j.get("articles", [])) or []
        for a in items[:40]:
            out.append({
                "title": a.get("headline") or a.get("title") or "",
                "url": a.get("url") or "",
                "source": a.get("source") or "",
                "publishedAt": dt.datetime.utcfromtimestamp(a.get("datetime", 0)).isoformat() if a.get("datetime") else "",
            })
        return out
    except Exception:
        return []

def general_news(query: str, news_api_key: str, limit: int = 20, domains: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    NewsAPI fallback with finance-focused query rewriting to filter out movie/entertainment noise.
    """
    if not query:
        return []
    q = f"({query}) AND (stock OR shares OR finance OR earnings OR ticker)"
    params = {"q": q, "language": "en", "sortBy": "publishedAt", "pageSize": limit, "apiKey": news_api_key}
    if domains:
        params["domains"] = domains
    try:
        j = _get("https://newsapi.org/v2/everything", params=params)
        out = []
        for a in j.get("articles", [])[:limit]:
            out.append({
                "title": a.get("title") or "",
                "url": a.get("url") or "",
                "source": (a.get("source") or {}).get("name", ""),
                "publishedAt": a.get("publishedAt") or "",
            })
        return out
    except Exception:
        return []

# ---------------------- FMP market screens ----------------------

def fmp_gainers(api_key: str) -> List[Dict[str, Any]]:
    """Top gainers today.  Endpoint: /api/v3/stock_market/gainers  :contentReference[oaicite:8]{index=8}"""
    try:
        out = _get(f"{FMP}/v3/stock_market/gainers", {"apikey": api_key})[:200]
        # Filter to US-listed tickers (best-effort): prefer tickers without exchange suffixes
        filtered = []
        for it in out:
            sym = (it.get("symbol") or "").upper()
            # Exclude symbols with non-US exchange suffixes like .L, .TO, etc.
            if "." in sym:
                continue
            # Optionally verify primary exchange via profile (cheap best-effort)
            try:
                prof = fmp_profile(sym, api_key)
                exch = (prof.get("exchange") or prof.get("exchangeShortName") or "").upper() if isinstance(prof, dict) else ""
                if exch and exch not in ("NASDAQ", "NYSE", "AMEX"):
                    continue
            except Exception:
                pass
            filtered.append(it)
        if filtered:
            return filtered[:100]
    except Exception:
        # fall through to yfinance fallback
        pass

    # Fallback: compute simple movers from a curated large-cap list using yfinance
    if yf is None:
        return []
    try:
        df = yf.download(tickers=_MAJOR_US_TICKERS, period="2d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
        rows = []
        # df may be multi-indexed by ticker
        for t in _MAJOR_US_TICKERS:
            try:
                sub = df[t] if t in df.columns.levels[0] else None
            except Exception:
                sub = None
            if sub is None:
                # try single-level frame
                try:
                    sub = df
                except Exception:
                    continue
            try:
                # find close column case-insensitively
                close_col = None
                for c in sub.columns:
                    if str(c).lower() == "close":
                        close_col = c
                        break
                if close_col is None:
                    continue
                if len(sub) < 2:
                    continue
                last = float(sub[close_col].iat[-1])
                prev = float(sub[close_col].iat[-2])
                pct = (last - prev) / prev * 100 if prev else 0.0
                rows.append({"symbol": t, "companyName": t, "price": last, "changesPercentage": pct})
            except Exception:
                continue
        # sort by pct desc
        rows = sorted(rows, key=lambda x: x.get("changesPercentage", 0), reverse=True)
        return rows[:100]
    except Exception:
        return []

def fmp_losers(api_key: str) -> List[Dict[str, Any]]:
    """Top losers today.  Endpoint: /api/v3/stock_market/losers (doc group)  :contentReference[oaicite:9]{index=9}"""
    try:
        out = _get(f"{FMP}/v3/stock_market/losers", {"apikey": api_key})[:200]
        filtered = []
        for it in out:
            sym = (it.get("symbol") or "").upper()
            if "." in sym:
                continue
            try:
                prof = fmp_profile(sym, api_key)
                exch = (prof.get("exchange") or prof.get("exchangeShortName") or "").upper() if isinstance(prof, dict) else ""
                if exch and exch not in ("NASDAQ", "NYSE", "AMEX"):
                    continue
            except Exception:
                pass
            filtered.append(it)
        if filtered:
            return filtered[:100]
    except Exception:
        # fall through to yfinance fallback
        pass

    # Fallback: compute simple losers from curated list using yfinance
    if yf is None:
        return []
    try:
        df = yf.download(tickers=_MAJOR_US_TICKERS, period="2d", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
        rows = []
        for t in _MAJOR_US_TICKERS:
            try:
                sub = df[t] if t in df.columns.levels[0] else None
            except Exception:
                sub = None
            if sub is None:
                try:
                    sub = df
                except Exception:
                    continue
            try:
                close_col = None
                for c in sub.columns:
                    if str(c).lower() == "close":
                        close_col = c
                        break
                if close_col is None:
                    continue
                if len(sub) < 2:
                    continue
                last = float(sub[close_col].iat[-1])
                prev = float(sub[close_col].iat[-2])
                pct = (last - prev) / prev * 100 if prev else 0.0
                rows.append({"symbol": t, "companyName": t, "price": last, "changesPercentage": pct})
            except Exception:
                continue
        rows = sorted(rows, key=lambda x: x.get("changesPercentage", 0))
        return rows[:100]
    except Exception:
        return []

def fmp_screen(params: Dict[str, Any], api_key: str) -> List[Dict[str, Any]]:
    """
    General screener.  Endpoint: /api/v3/stock-screener  (sector, volume, beta, etc.)  :contentReference[oaicite:10]{index=10}
    """
    q = dict(params)
    q["apikey"] = api_key
    try:
        return _get(f"{FMP}/v3/stock-screener", q)[:500]
    except Exception:
        # Fall back to a lightweight yfinance-based screener over a curated
        # list of major US tickers when the FMP endpoint is unavailable.
        if yf is None:
            return []
        out: List[Dict[str, Any]] = []
        try:
            # map expected param names to simple checks
            vol_min = float(params.get("volumeMoreThan", 0) or 0)
            mkt_min = float(params.get("marketCapMoreThan", 0) or 0)
            beta_min = float(params.get("betaMoreThan", 0) or 0)
            price_min = float(params.get("priceMoreThan", 0) or 0)
            pe_max = float(params.get("peRatioLowerThan", float("inf")) or float("inf"))
            sector_want = str(params.get("sector", "")).strip() or None

            for s in _MAJOR_US_TICKERS:
                try:
                    t = yf.Ticker(s)
                    # get a small profile/info dict if available
                    info = {}
                    try:
                        info = getattr(t, "info", {}) or {}
                    except Exception:
                        info = {}

                    # price and volume from recent history as a fallback
                    price = None
                    vol = None
                    try:
                        h = t.history(period="2d", interval="1d")
                        if h is not None and len(h) >= 1:
                            last = h.iloc[-1]
                            prev = h.iloc[-2] if len(h) > 1 else last
                            price = float(last.get("Close") if "Close" in last else last.get("close") if "close" in last else float("nan"))
                            # volume may be missing in info; get from history if present
                            vol = int(last.get("Volume") if "Volume" in last else (last.get("volume") if "volume" in last else 0))
                    except Exception:
                        price = None
                        vol = None

                    marketcap = info.get("marketCap") or info.get("marketCap") or None
                    beta = info.get("beta") or None
                    pe = info.get("trailingPE") or info.get("trailingPE") or info.get("pe") or info.get("forwardPE") or None
                    sector = info.get("sector") or info.get("industry") or None
                    name = info.get("longName") or info.get("shortName") or s

                    # normalize numeric types
                    try:
                        marketcap = float(marketcap) if marketcap is not None else None
                    except Exception:
                        marketcap = None
                    try:
                        beta = float(beta) if beta is not None else None
                    except Exception:
                        beta = None
                    try:
                        pe = float(pe) if pe is not None else None
                    except Exception:
                        pe = None

                    # apply filters
                    if price_min and (price is None or price < price_min):
                        continue
                    if vol_min and (vol is None or vol < vol_min):
                        continue
                    if mkt_min and (marketcap is None or marketcap < mkt_min):
                        continue
                    if beta_min and (beta is None or beta < beta_min):
                        continue
                    if pe_max is not None and pe is not None and pe > pe_max:
                        continue
                    if sector_want and sector and sector_want.lower() not in sector.lower():
                        continue

                    out.append({
                        "symbol": s,
                        "companyName": name,
                        "price": price,
                        "beta": beta,
                        "volume": vol,
                        "sector": sector,
                        "pe": pe,
                    })
                except Exception:
                    continue

            return out[:500]
        except Exception:
            return []

# ---------------------- Indices (UI tiles) ----------------------

def index_snapshots() -> Dict[str, Dict[str, Any]]:
    """S&P500, NASDAQ, Dow, NYSE via yfinance symbols: ^GSPC, ^IXIC, ^DJI, ^NYA.

    Use a single bulk download for all indices to reduce network overhead. If
    bulk download fails, fall back to per-ticker fetch for robustness.
    """
    symbols = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI", "NYSE": "^NYA"}
    out: Dict[str, Dict[str, Any]] = {}
    # Prefer batch download which is significantly faster than individual Ticker.history calls
    try:
        download = getattr(yf, "download", None)
        if download is None:
            return out
        # request all index symbols in one call; threads=True speeds up under the hood
        df = download(tickers=list(symbols.values()), period="1mo", interval="1d", group_by="ticker", threads=True, progress=False, auto_adjust=False)
        # df may be a MultiIndex columns (ticker, field) or single-level if only one ticker
        for name, sym in symbols.items():
            try:
                # extract close series for the symbol
                close_series = None
                if sym in getattr(df, "columns", []):
                    # columns grouped by ticker
                    sub = df[sym]
                    for c in sub.columns:
                        if str(c).lower() in ("close", "adj close"):
                            close_series = sub[c]
                            break
                else:
                    # maybe df columns are single-level with names like Close and a ticker index
                    # attempt to pick a column named 'Close' in top-level
                    for c in df.columns:
                        if str(c).lower() == "close":
                            close_series = df[c]
                            break
                if close_series is None or close_series.empty:
                    continue
                # normalize to DataFrame with date and close
                tmp = close_series.reset_index()
                # ensure two rows for prev calculation
                if len(tmp) < 1:
                    continue
                curr = float(tmp.iloc[-1, 1])
                prev = float(tmp.iloc[-2, 1]) if len(tmp) > 1 else curr
                spark = tmp.rename(columns={tmp.columns[0]: "date", tmp.columns[1]: "close"})
                out[name] = {
                    "symbol": sym,
                    "current": curr,
                    "prev": prev,
                    "delta_pct": ((curr - prev) / prev) * 100 if prev else 0.0,
                    "spark": spark,
                }
            except Exception:
                continue
        if out:
            return out
    except Exception:
        # if bulk download fails, fall back to per-ticker approach below
        pass

    # Fallback: per-ticker (original behavior)
    ticker_cls = getattr(yf, "Ticker", None)
    if ticker_cls is None:
        return out
    for name, sym in symbols.items():
        try:
            hist = ticker_cls(sym).history(period="1mo", interval="1d")
            if hist is None or getattr(hist, "empty", False):
                continue
            hist = hist.reset_index()
            # find close column case-insensitively
            close_col = None
            for c in hist.columns:
                if str(c).lower() == "close":
                    close_col = c
                    break
            if close_col is None:
                continue
            # get current and previous safely
            try:
                curr = float(hist[close_col].iat[-1])
            except Exception:
                continue
            try:
                prev = float(hist[close_col].iat[-2]) if len(hist) > 1 else curr
            except Exception:
                prev = curr
            # build spark frame with normalized column names
            date_col = None
            for c in hist.columns:
                if str(c).lower() in ("date", "index") or str(c).lower().startswith("date"):
                    date_col = c
                    break
            spark = hist[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"}) if date_col is not None else hist[[close_col]].rename(columns={close_col: "close"})
            out[name] = {
                "symbol": sym,
                "current": curr,
                "prev": prev,
                "delta_pct": ((curr - prev) / prev) * 100 if prev else 0.0,
                "spark": spark,
            }
        except Exception:
            # fail gracefully per-index
            continue
    return out

# ---------------------- Fundamentals: sector/industry P/E ----------------------

def fmp_profile(symbol: str, api_key: str) -> Dict[str, Any]:
    """Company profile includes sector/industry and exchange.

    Try FMP first (v4/v3), then fall back to yfinance when available so the UI
    can show a minimal profile even without an FMP key or when provider access
    is restricted.
    """
    # If no api_key provided, skip FMP and try yfinance directly
    if not api_key:
        try:
            if yf is not None:
                ticker_cls = getattr(yf, "Ticker", None)
                if ticker_cls is not None:
                    info = ticker_cls(symbol).info or {}
                    return {
                        "symbol": symbol,
                        "companyName": info.get("longName") or info.get("shortName") or symbol,
                        "sector": info.get("sector") or "",
                        "industry": info.get("industry") or "",
                        "price": info.get("regularMarketPrice") or info.get("previousClose"),
                        "mktCap": info.get("marketCap"),
                        "beta": info.get("beta"),
                        "description": info.get("longBusinessSummary") or info.get("businessSummary") or "",
                    }
        except Exception:
            return {}
    # Try FMP v4/v3 endpoints
    try:
        # v4 company profile endpoint
        j = _get(f"{FMP}/v4/company-profile", {"symbol": symbol, "apikey": api_key})
        # v4 may return a list or dict wrapper
        if isinstance(j, list) and j:
            d = j[0]
            prof = d.get("profile", d)
            return prof if isinstance(prof, dict) else d
        if isinstance(j, dict) and j:
            prof = j.get("profile", j)
            if isinstance(prof, dict):
                return prof
    except Exception:
        # fall through to v3 attempt
        pass
    try:
        j = _get(f"{FMP}/v3/profile/{symbol}", {"apikey": api_key})
        return j[0] if isinstance(j, list) and j else {}
    except Exception:
        # Fallback to yfinance if available
        try:
            if yf is not None:
                ticker_cls = getattr(yf, "Ticker", None)
                if ticker_cls is not None:
                    info = ticker_cls(symbol).info or {}
                    return {
                        "symbol": symbol,
                        "companyName": info.get("longName") or info.get("shortName") or symbol,
                        "sector": info.get("sector") or "",
                        "industry": info.get("industry") or "",
                        "price": info.get("regularMarketPrice") or info.get("previousClose"),
                        "mktCap": info.get("marketCap"),
                        "beta": info.get("beta"),
                        "description": info.get("longBusinessSummary") or info.get("businessSummary") or "",
                    }
        except Exception:
            return {}
    return {}

def company_pe_ttm(symbol: str, api_key: str) -> Optional[float]:
    """Prefer ratios-ttm; fallback to key-metrics-ttm or profile 'pe'.  :contentReference[oaicite:12]{index=12}"""
    for path, key in [
        (f"{FMP}/v3/ratios-ttm/{symbol}", "peRatioTTM"),
        (f"{FMP}/v3/key-metrics-ttm/{symbol}", "peRatioTTM"),
    ]:
        try:
            j = _get(path, {"apikey": api_key})
            if isinstance(j, list) and j:
                val = j[0].get(key)
                if val is not None:
                    return float(val)
        except Exception:
            pass
    # try profile 'pe' field
    try:
        prof = fmp_profile(symbol, api_key)
        pe_val = None
        if isinstance(prof, dict):
            pe_val = prof.get("pe") or prof.get("peRatio") or prof.get("peRatioTTM")
        if pe_val is not None:
            return float(pe_val)
    except Exception:
        pass

    # Fallback: try yfinance trailing PE / compute from EPS if available
    try:
        if yf is not None:
            ticker_cls = getattr(yf, "Ticker", None)
            if ticker_cls is not None:
                info = ticker_cls(symbol).info or {}
                # try common keys
                pe = info.get("trailingPE") or info.get("forwardPE") or info.get("pe")
                if pe is not None:
                    return float(pe)
                # compute from price / epsTrailingTwelveMonths
                price = info.get("regularMarketPrice") or info.get("previousClose")
                eps = info.get("epsTrailingTwelveMonths") or info.get("trailingEps")
                if price is not None and eps:
                    try:
                        return float(price) / float(eps)
                    except Exception:
                        pass
    except Exception:
        pass
    return None

def _sector_pe_table(exchange: str, api_key: str) -> Dict[str, float]:
    """
    /api/v4/sector_price_earning_ratio?exchange=NYSE
    Returns { 'Technology': 25.1, ... }  :contentReference[oaicite:13]{index=13}
    """
    try:
        j = _get(f"{FMP}/v4/sector_price_earning_ratio", {"exchange": exchange, "apikey": api_key})
        out = {}
        for it in j:
            sec = it.get("sector") or it.get("name") or it.get("Sector")
            pe = it.get("pe")
            if sec and pe is not None:
                try:
                    out[str(sec)] = float(pe)
                except Exception:
                    continue
        return out
    except Exception:
        return {}

def _industry_pe_table(exchange: str, api_key: str) -> Dict[str, float]:
    """
    /api/v4/industry_price_earning_ratio?exchange=NYSE
    Returns { 'Semiconductors': 29.3, ... }  :contentReference[oaicite:14]{index=14}
    """
    try:
        j = _get(f"{FMP}/v4/industry_price_earning_ratio", {"exchange": exchange, "apikey": api_key})
        out = {}
        for it in j:
            ind = it.get("industry")
            pe = it.get("pe")
            if ind and pe is not None:
                try:
                    out[str(ind)] = float(pe)
                except Exception:
                    continue
        return out
    except Exception:
        return {}

def sector_industry_pe_for_symbol(symbol: str, api_key: str) -> Dict[str, Any]:
    """
    Compute sector/industry P/E for a given symbol using its exchange+sector/industry from profile.
    Tries the profile exchange; falls back to NYSE/NASDAQ tables if needed.
    """
    prof = fmp_profile(symbol, api_key)
    sector = prof.get("sector") or ""
    industry = prof.get("industry") or ""
    exch = prof.get("exchangeShortName") or prof.get("exchange") or "NYSE"
    # Try the profile exchange; fallback across major exchanges
    exchanges = [exch, "NYSE", "NASDAQ", "AMEX"]
    sec_pe = ind_pe = None
    for ex in exchanges:
        if sec_pe is None:
            t = _sector_pe_table(ex, api_key)
            if sector in t:
                sec_pe = t[sector]
        if ind_pe is None:
            t2 = _industry_pe_table(ex, api_key)
            if industry in t2:
                ind_pe = t2[industry]
        if sec_pe is not None and ind_pe is not None:
            break
    comp_pe = company_pe_ttm(symbol, api_key)
    return {
        "sector": sector or None,
        "industry": industry or None,
        "sector_pe": sec_pe,
        "industry_pe": ind_pe,
        "company_pe": comp_pe,
        "exchange": exch,
    }


def get_holders(symbol: str, api_key: str = "", finnhub_key: str = "") -> Dict[str, Any]:
    """
    Best-effort retrieval of major/ institutional holders for a symbol.

    Strategy:
    - Try FMP endpoints (v3/v4) if `api_key` provided (various FMP plans expose
      holders/shareholders endpoints). We attempt a few common paths but do not
      assume any specific schema — callers should treat the output as advisory.
    - Fallback to yfinance Ticker.institutional_holders / major_holders when
      available.

    Returns a dict with keys:
      - "institutional": list of {name, pct} rows
      - "major": list of {name, pct} rows
      - "source": provider string
    """
    out: dict[str, Any] = {"institutional": [], "major": [], "source": None}
    # Try FMP (best-effort endpoints)
    if api_key:
        try:
            # v4 /v3 legacy endpoints — try a few known patterns
            paths = [
                f"{FMP}/v4/stock_holders?symbol={symbol}&apikey={api_key}",
                f"{FMP}/v3/stock-holders/{symbol}?apikey={api_key}",
                f"{FMP}/v3/stock/holders/{symbol}?apikey={api_key}",
                f"{FMP}/v3/major-holder/{symbol}?apikey={api_key}",
            ]
            for p in paths:
                try:
                    j = _get(p, {}) if isinstance(p, str) else None
                except Exception:
                    j = None
                if not j:
                    continue
                # If the payload looks like a list of holders
                if isinstance(j, list) and j:
                    # try to normalize entries with 'holder'/'name' and 'pct' or 'percentage'
                    for it in j:
                        name = it.get("holder") or it.get("name") or it.get("shareholder")
                        pct = it.get("pct") or it.get("percentage") or it.get("percentage_of_shares") or it.get("percent")
                        try:
                            pctf = float(str(pct).strip().strip('%')) if pct is not None else None
                        except Exception:
                            pctf = None
                        if name:
                            out["major"].append({"name": name, "pct": pctf})
                    out["source"] = "FMP"
                    return out
                # If dict with nested fields
                if isinstance(j, dict):
                    # some endpoints return {"holders": [...]}
                    for key in ("holders", "majorHolders", "institutionalHolders", "shareholders"):
                        if key in j and isinstance(j[key], list):
                            for it in j[key]:
                                name = it.get("holder") or it.get("name") or it.get("shareholder")
                                pct = it.get("pct") or it.get("percentage") or it.get("percent")
                                try:
                                    pctf = float(str(pct).strip().strip('%')) if pct is not None else None
                                except Exception:
                                    pctf = None
                                if name:
                                    out["major"].append({"name": name, "pct": pctf})
                            out["source"] = "FMP"
                            return out
        except Exception:
            pass

    # Fallback: yfinance (local library) for major/institutional holders
    try:
        if yf is not None:
            ticker_cls = getattr(yf, "Ticker", None)
            if ticker_cls is not None:
                t = ticker_cls(symbol)
                # major_holders returned as DataFrame-like with two columns usually
                try:
                    maj = getattr(t, "major_holders", None)
                    if maj is not None and len(maj) > 0:
                        # major_holders often is a DataFrame with [0]=Name, [1]=% or similar
                        try:
                            dfm = pd.DataFrame(maj)
                        except Exception:
                            dfm = None
                        if dfm is not None and not dfm.empty:
                            for _, row in dfm.iterrows():
                                nm = row.iloc[0] if len(row) > 0 else None
                                pct = row.iloc[1] if len(row) > 1 else None
                                try:
                                    pctf = float(str(pct).strip().strip('%')) if pct is not None else None
                                except Exception:
                                    pctf = None
                                if nm:
                                    out["major"].append({"name": str(nm), "pct": pctf})
                except Exception:
                    pass

                try:
                    inst = getattr(t, "institutional_holders", None)
                    if inst is not None and len(inst) > 0:
                        try:
                            dfi = pd.DataFrame(inst)
                        except Exception:
                            dfi = None
                        if dfi is not None and not dfi.empty:
                            # yfinance returns columns like Holder, Shares, Date Reported, % Out
                            for _, row in dfi.iterrows():
                                nm = None
                                pct = None
                                # try common columns
                                for c in ("Holder", "holder", 0):
                                    if c in dfi.columns:
                                        nm = row[c]
                                        break
                                for c in ("% Out", "%", "% of Shares Outstanding", "%Out", 3):
                                    if c in dfi.columns:
                                        pct = row[c]
                                        break
                                # fallback to position-based access
                                if nm is None:
                                    try:
                                        nm = row.iloc[0]
                                    except Exception:
                                        nm = None
                                if pct is None:
                                    try:
                                        pct = row.iloc[-1]
                                    except Exception:
                                        pct = None
                                try:
                                    pctf = float(str(pct).strip().strip('%')) if pct is not None else None
                                except Exception:
                                    pctf = None
                                if nm:
                                    out["institutional"].append({"name": str(nm), "pct": pctf})
                except Exception:
                    pass
                if out["major"] or out["institutional"]:
                    out["source"] = "yfinance"
                    return out
    except Exception:
        pass

    return out


# ---------------------- ETFs / Mutual Funds / IPOs / Major Events ----------------------

from utils.cache import timed_lru_cache
from config import get_cache_ttl


@timed_lru_cache(ttl_seconds=get_cache_ttl("list_etfs", 600), maxsize=128)
def list_etfs(api_key: str = "", finnhub_token: str = "") -> List[Dict[str, Any]]:
    """
    Return a best-effort list of US-listed ETFs. Tries FMP (common ETF endpoints),
    then Finnhub symbol search with type filter, then falls back to a curated
    yfinance-based scan when available.
    The result is a list of dicts with keys like {'symbol','name','exchange','assetType'}.
    """
    out: List[Dict[str, Any]] = []
    # Try FMP common ETF endpoints
    if api_key:
        candidates = [
            f"{FMP}/v3/etf/list",
            f"{FMP}/v4/etf/list",
            f"{FMP}/v3/etf",
            f"{FMP}/v3/etf_all",
        ]
        for p in candidates:
            try:
                j = _get(p, {"apikey": api_key})
                if isinstance(j, list) and j:
                    for it in j:
                        sym = it.get("symbol") or it.get("ticker") or it.get("code")
                        name = it.get("name") or it.get("companyName") or it.get("title")
                        exch = it.get("exchange") or it.get("exchangeShortName") or ""
                        if sym:
                            out.append({"symbol": sym, "name": name or "", "exchange": exch, "assetType": "ETF"})
                    if out:
                        return out
            except Exception:
                continue

    # Try Finnhub symbol search with 'etf' filter if available
    if finnhub_token:
        try:
            j = _get(f"{FINNHUB}/search", {"q": "ETF", "token": finnhub_token})
            for it in j.get("result", [])[:500]:
                # Finnhub sometimes includes field 'type' or 'mic' — best-effort filter
                sym = it.get("symbol") or ""
                desc = it.get("description") or ""
                if sym and ("ETF" in desc.upper() or sym.upper().endswith(".ETF") or "ETF" in it.get("type", "").upper()):
                    out.append({"symbol": sym, "name": desc, "exchange": it.get("exchange", ""), "assetType": "ETF"})
            if out:
                return out
        except Exception:
            pass

    # Fallback: use yfinance to scan a small curated ETF list if available
    if yf is None:
        return out
    try:
        # a short curated list of popular US ETFs to provide immediate value
        curated = ["SPY", "IVV", "VTI", "VOO", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLY"]
        for s in curated:
            try:
                t = yf.Ticker(s)
                info = getattr(t, "info", {}) or {}
                name = info.get("longName") or info.get("shortName") or s
                out.append({"symbol": s, "name": name, "exchange": info.get("exchange", ""), "assetType": "ETF"})
            except Exception:
                continue
        return out
    except Exception:
        return out


@timed_lru_cache(ttl_seconds=get_cache_ttl("list_mutual_funds", 600), maxsize=128)
def list_mutual_funds(api_key: str = "", finnhub_token: str = "") -> List[Dict[str, Any]]:
    """
    Best-effort list of mutual funds. Many public APIs limit mutual fund lists; we
    attempt FMP common endpoints then fallback to a small curated set via yfinance.
    """
    out: List[Dict[str, Any]] = []
    if api_key:
        candidates = [
            f"{FMP}/v3/mutual_fund/list",
            f"{FMP}/v3/mutualfunds",
            f"{FMP}/v4/mutual_fund/list",
        ]
        for p in candidates:
            try:
                j = _get(p, {"apikey": api_key})
                if isinstance(j, list) and j:
                    for it in j:
                        sym = it.get("symbol") or it.get("ticker") or it.get("code")
                        name = it.get("name") or it.get("fundName") or ""
                        if sym:
                            out.append({"symbol": sym, "name": name, "assetType": "MutualFund"})
                    if out:
                        return out
            except Exception:
                continue

    # Finnhub doesn't provide a centralized mutual fund list; fall back to curated yfinance list
    if yf is None:
        return out
    try:
        curated = ["VFINX", "FXAIX", "VTSMX", "SWPPX", "VFIAX"]
        for s in curated:
            try:
                t = yf.Ticker(s)
                info = getattr(t, "info", {}) or {}
                name = info.get("longName") or info.get("shortName") or s
                out.append({"symbol": s, "name": name, "assetType": "MutualFund"})
            except Exception:
                continue
        return out
    except Exception:
        return out


@timed_lru_cache(ttl_seconds=get_cache_ttl("upcoming_ipos", 3600), maxsize=64)
def upcoming_ipos(finnhub_token: str = "", fmp_key: str = "", days_ahead: int = 90) -> List[Dict[str, Any]]:
    """
    Return upcoming IPOs within the next `days_ahead` days. Tries Finnhub calendar IPO
    endpoint first, then FMP IPO calendar variants. Returns list of dicts with at
    least {'symbol','name','date','exchange','expectedPrice'} when available.
    """
    out: List[Dict[str, Any]] = []
    today = dt.date.today()
    to = today + dt.timedelta(days=days_ahead)

    # Finnhub: /calendar/ipo (best-effort)
    if finnhub_token:
        try:
            j = _get(f"{FINNHUB}/calendar/ipo", {"from": today.isoformat(), "to": to.isoformat(), "token": finnhub_token})
            # Finnhub may return dict with 'ipoCalendar' or list
            items = j.get("ipoCalendar") if isinstance(j, dict) else j
            if items is None:
                items = j
            if isinstance(items, list):
                for it in items:
                    out.append({
                        "symbol": it.get("symbol") or it.get("ticker") or "",
                        "name": it.get("name") or it.get("companyName") or "",
                        "date": it.get("date") or it.get("ipoDate") or it.get("startDate"),
                        "exchange": it.get("exchange") or it.get("exchangeShortName") or "",
                        "expectedPrice": it.get("expectedPrice") or it.get("price") or None,
                    })
                if out:
                    return out
        except Exception:
            pass

    # FMP attempts: try a few known patterns
    if fmp_key:
        candidates = [
            f"{FMP}/v3/ipo_calendar",
            f"{FMP}/v3/ipo",
            f"{FMP}/v4/ipo_calendar",
        ]
        for p in candidates:
            try:
                j = _get(p, {"from": today.isoformat(), "to": to.isoformat(), "apikey": fmp_key})
                if isinstance(j, list) and j:
                    for it in j:
                        out.append({
                            "symbol": it.get("symbol") or it.get("ticker") or "",
                            "name": it.get("name") or it.get("companyName") or "",
                            "date": it.get("date") or it.get("ipoDate") or None,
                            "exchange": it.get("exchange") or "",
                            "expectedPrice": it.get("price") or it.get("expectedPrice") or None,
                        })
                    if out:
                        return out
            except Exception:
                continue

    return out


@timed_lru_cache(ttl_seconds=get_cache_ttl("get_major_events", 300), maxsize=256)
def get_major_events(symbol: str, finnhub_token: str = "", fmp_key: str = "") -> Dict[str, Any]:
    """
    Aggregate major events for a symbol: upcoming earnings, recent earnings summary,
    company-specific news, recent dividends/splits and a brief indicator of recent major
    filings when available. Uses Finnhub and FMP where possible, falls back to yfinance.
    """
    out: Dict[str, Any] = {"earnings_next": None, "earnings_last": None, "news": [], "dividends": [], "splits": [], "source": None}
    # Earnings: try utils.earnings helpers if available
    try:
        from utils.earnings import get_next_earnings_info, get_last_earnings_summary

        try:
            next_info = get_next_earnings_info(symbol)
            out["earnings_next"] = next_info
        except Exception:
            out["earnings_next"] = None

        try:
            last_info = get_last_earnings_summary(symbol)
            out["earnings_last"] = last_info
        except Exception:
            out["earnings_last"] = None
    except Exception:
        pass

    # Company news via Finnhub
    try:
        if finnhub_token:
            out["news"] = company_news(symbol, finnhub_token, days=30)
            out["source"] = out.get("source") or "Finnhub"
    except Exception:
        out["news"] = []

    # Dividends and splits via yfinance fallback
    try:
        if yf is not None:
            t = yf.Ticker(symbol)
            try:
                divs = getattr(t, "dividends", None)
                if divs is not None and len(divs) > 0:
                    # convert to list of recent dividend records
                    dd = []
                    for idx, val in divs.tail(6).items():
                        dd.append({"date": str(idx.date()), "amount": float(val)})
                    out["dividends"] = dd
            except Exception:
                out["dividends"] = []
            try:
                splits = getattr(t, "splits", None)
                if splits is not None and len(splits) > 0:
                    ss = []
                    for idx, val in splits.tail(6).items():
                        ss.append({"date": str(idx.date()), "ratio": str(val)})
                    out["splits"] = ss
            except Exception:
                out["splits"] = []
            if out.get("source") is None:
                out["source"] = "yfinance"
    except Exception:
        pass

    return out

# ---------------------- Alpaca (optional portfolio sync) ----------------------

def alpaca_positions(api_key: str, secret: str) -> List[Dict[str, Any]]:
    """Paper endpoint: /v2/positions"""
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/positions",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret},
            timeout=20,
        )
        r.raise_for_status()
        rows = []
        for p in r.json():
            try:
                qty_val = p.get("qty", 0) or 0
                avg_val = p.get("avg_entry_price", 0.0) or 0.0
                rows.append({
                    "symbol": p.get("symbol"),
                    "qty": float(qty_val),
                    "avg_price": float(avg_val),
                })
            except Exception:
                # skip malformed position
                continue
        return rows
    except Exception:
        return []
