import pandas as pd

# yfinance is optional for tests. Use it when available. We avoid raising at import
# time so tests/CI that don't need live data can still import the package.
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore
from utils.cache import cache_get, cache_set

def get_ohlc(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Return OHLC DataFrame for symbol using yfinance when available.

    The returned frame will have columns: Open, High, Low, Close, Volume (if present).
    Returns empty DataFrame if data cannot be fetched or yfinance is unavailable.
    """
    # If yfinance wasn't importable at module import time, try again here and
    # raise a helpful ImportError if still unavailable. This surfaces missing
    # dependency issues to the UI instead of silently returning empty frames.
    if yf is None:
        try:
            import importlib
            globals()['yf'] = importlib.import_module('yfinance')
        except Exception:
            raise ImportError(
                "yfinance is required to fetch OHLC data. Install with 'pip install yfinance'"
            )

    # Try cache first
    cache_key = f"ohlc:{symbol}:{period}:{interval}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        yf_mod = globals().get('yf')
        assert yf_mod is not None
        ticker = yf_mod.Ticker(symbol)
        # Primary attempt: use the Ticker.history API
        df = None
        try:
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
        except Exception:
            df = None
        # If history returned nothing, try the higher-level download API as a fallback
        if df is None or getattr(df, "empty", True):
            try:
                df = yf_mod.download(tickers=symbol, period=period, interval=interval, group_by='ticker', auto_adjust=False)
            except Exception:
                df = None
        # If still empty, return an empty DataFrame (caller will handle UI/error display)
        if df is None or getattr(df, "empty", True):
            return pd.DataFrame()
        # Normalize column names to capitalized expected names
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        # Ensure required columns exist
        for c in ("Open", "High", "Low", "Close"):
            if c not in df.columns:
                raise RuntimeError(f"Fetched data missing required column: {c}")
        # Cache copy and return
        try:
            cache_set(cache_key, df.copy())
        except Exception:
            pass
        return df
    except ImportError:
        # propagate import errors for callers to surface
        raise
    except Exception:
        # Non-fatal: return empty frame to let callers decide how to handle missing OHLC
        return pd.DataFrame()


def get_latest_quote(symbol: str) -> dict:
    """Return a small dict with latest price info.

    Tries providers in order of availability: Finnhub (if configured), yfinance, then falls back
    to the last close from `get_ohlc`. Returns a dict with keys: price (float), ts (int|None),
    extended (bool), source (str). Does not raise on network issues — returns empty dict on failure.
    """
    cache_key = f"quote:{symbol}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.copy()

    out = {}
    # 1) Try Finnhub quote adapter if available (safe import)
    try:
        from data_fetchers.finnhub import get_quote_finnhub
        q = get_quote_finnhub(symbol) or {}
        # Finnhub returns keys like 'c' (current), 'o','h','l','pc','t'
        if isinstance(q, dict) and q.get("c") is not None:
            # defensive typing: ensure the returned values can be converted
            cval = q.get("c")
            tval = q.get("t")
            try:
                price_val = float(cval) if isinstance(cval, (int, float, str)) else None
            except Exception:
                price_val = None
            try:
                ts_val = int(tval) if isinstance(tval, (int, float, str)) and str(tval).isdigit() else None
            except Exception:
                ts_val = None
            if price_val is not None:
                out = {"price": price_val, "ts": ts_val, "extended": False, "source": "finnhub"}
            try:
                cache_set(cache_key, out)
            except Exception:
                pass
            return out
    except Exception:
        # ignore and try next provider
        pass

    # 2) Try yfinance for pre/post market fields if available
    try:
        try:
            import importlib
            yf_mod = globals().get('yf') or importlib.import_module('yfinance')
            globals()['yf'] = yf_mod
        except Exception:
            yf_mod = None
        if yf_mod is not None:
            try:
                t = yf_mod.Ticker(symbol)
                info = getattr(t, 'info', None) or {}
                # yfinance may expose preMarketPrice / postMarketPrice
                pre = info.get('preMarketPrice') or info.get('preMarketChange')
                post = info.get('postMarketPrice') or info.get('postMarketChange')
                reg = info.get('regularMarketPrice') or info.get('regularMarketPreviousClose') or info.get('previousClose')
                if pre is not None:
                    try:
                        price_val = float(pre) if isinstance(pre, (int, float, str)) else None
                    except Exception:
                        price_val = None
                    if price_val is not None:
                        out = {"price": price_val, "ts": None, "extended": True, "source": "yfinance_pre"}
                elif post is not None:
                    try:
                        price_val = float(post) if isinstance(post, (int, float, str)) else None
                    except Exception:
                        price_val = None
                    if price_val is not None:
                        out = {"price": price_val, "ts": None, "extended": True, "source": "yfinance_post"}
                elif reg is not None:
                    try:
                        price_val = float(reg) if isinstance(reg, (int, float, str)) else None
                    except Exception:
                        price_val = None
                    if price_val is not None:
                        out = {"price": price_val, "ts": None, "extended": False, "source": "yfinance_regular"}
                if out:
                    try:
                        cache_set(cache_key, out)
                    except Exception:
                        pass
                    return out
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fallback: last available Close from OHLC
    try:
        df = get_ohlc(symbol, period="5d", interval="1d")
        if df is not None and not df.empty:
            # Normalize column casing
            colmap = {c.lower(): c for c in df.columns}
            ccol = colmap.get('close', 'Close')
            last = df.iloc[-1]
            price = last.get(ccol) if ccol in last else None
            if price is not None:
                try:
                    price_val = float(price) if isinstance(price, (int, float, str)) else None
                except Exception:
                    price_val = None
                if price_val is not None:
                    out = {"price": price_val, "ts": None, "extended": False, "source": "ohlc_close"}
                try:
                    cache_set(cache_key, out)
                except Exception:
                    pass
                return out
    except Exception:
        pass

    return {}


def get_latest_price(symbol: str) -> float:
    """Convenience: return latest numeric price or NaN if unavailable."""
    q = get_latest_quote(symbol) or {}
    price_val = q.get('price')
    try:
        if isinstance(price_val, (int, float, str)):
            return float(price_val)
    except Exception:
        pass
    return float('nan')
