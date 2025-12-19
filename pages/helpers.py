from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
import json


def get_watchlist_from_portfolio(defaults: Optional[List[str]] = None) -> List[str]:
    defaults = defaults or ["SPY", "QQQ", "TQQQ", "AAPL", "TSLA", "AMZN"]
    try:
        p = Path("data/portfolio.json")
        if p.exists():
            j = json.loads(p.read_text(encoding="utf8"))
            syms = [s.get("symbol", "").upper() for s in j.get("positions", []) if s.get("symbol")]
            if syms:
                return list(dict.fromkeys(syms))
    except Exception:
        pass
    return defaults


def _fetch_analyst(sym: str):
    """Return analyst consensus DataFrame-like or None. Safe when no key is present."""
    try:
        # avoid import-time key resolution; adapter will handle missing keys
        from config import get_runtime_key
        fh_key = get_runtime_key("FINNHUB_API_KEY")
        if not fh_key:
            return None
        from data_fetchers.finnhub import fetch_analyst_consensus

        df = fetch_analyst_consensus(fh_key, [sym])
        return df if (hasattr(df, "empty") and not df.empty) else None
    except Exception:
        return None


def _news_headlines(sym: str, limit: int = 10):
    """Return a list of article dicts (may be empty)."""
    try:
        from data_fetchers.news import newsapi_headlines

        return newsapi_headlines(sym, limit=limit) or []
    except Exception:
        return []


def _vader_fn(text: str) -> float:
    try:
        from data_fetchers.news import vader_score

        return float(vader_score(text))
    except Exception:
        return 0.0
