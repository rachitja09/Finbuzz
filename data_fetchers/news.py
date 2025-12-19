from __future__ import annotations
import requests
import streamlit as st
import pandas as pd
from config import get_runtime_key

# vaderSentiment is optional for tests/environments without NLP deps. Provide a
# lightweight fallback polarity scorer (very simple heuristic) so the module can
# be imported and used in limited form.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sia = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover - environment dependent
    SentimentIntensityAnalyzer = None
    _sia = None


def _get_json(url: str, params: dict | None = None, timeout=30):
    from requests.exceptions import RequestException, SSLError
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except SSLError:
        return None
    except RequestException:
        return None

@st.cache_data(ttl=900)
def newsapi_headlines(query: str, limit=20) -> list[dict]:
    """Developer plan is limited and delayed ~24h; fine for research UI."""
    # Prefer Finnhub company-news when available and permitted by the user's key.
    q = query.strip()
    # If Finnhub is configured, attempt to fetch company-news first (more targeted).
    fh_key = get_runtime_key("FINNHUB_API_KEY")
    if fh_key:
        try:
            # The Finnhub company-news endpoint expects from/to dates. Use a 30d window.
            from datetime import datetime, timedelta, timezone

            # Use timezone-aware UTC datetime instead of utcnow() (deprecated)
            to_dt = datetime.now(timezone.utc).date()
            frm_dt = to_dt - timedelta(days=30)
            r = _get_json("https://finnhub.io/api/v1/company-news", params={"symbol": q, "from": frm_dt.isoformat(), "to": to_dt.isoformat(), "token": fh_key}, timeout=10)
            if r:
                # Finnhub returns a list of articles
                out = []
                for it in r[:limit]:
                    out.append({
                        "title": it.get("headline") or it.get("summary") or it.get("source"),
                        "url": it.get("url"),
                        "publishedAt": it.get("datetime"),
                        "source": it.get("source"),
                        "description": it.get("summary"),
                    })
                return out
        except Exception:
            # fall through to NewsAPI
            pass

    n_key = get_runtime_key("NEWS_API_KEY")
    if not n_key:
        return []
    # Improve query: if user provides a ticker-like token (1-5 uppercase letters),
    # expand query to company name synonyms to reduce noisy hits (e.g., movie pages).
    q = query.strip()
    # simple ticker heuristic
    if q.isupper() and 1 <= len(q) <= 5:
        # allow common company name match via yfinance if available
        try:
            import yfinance as yf
            info = yf.Ticker(q).info or {}
            cname = info.get("longName") or info.get("shortName")
            if cname:
                q = f'"{cname}" OR {q}'
        except Exception:
            q = q

    params = {"q": q, "pageSize": limit, "sortBy": "publishedAt", "language": "en", "apiKey": n_key}
    data = _get_json("https://newsapi.org/v2/everything", params=params) or {}
    return data.get("articles", [])

def vader_score(text: str) -> float:
    if _sia is not None:
        s = _sia.polarity_scores(text or "")
        return float(s.get("compound", 0))
    # Fallback: very small heuristic using sentiment words (not a replacement)
    txt = (text or "").lower()
    pos = sum(1 for w in ("good", "up", "beat", "raise", "positive", "gain") if w in txt)
    neg = sum(1 for w in ("bad", "down", "miss", "cut", "negative", "loss") if w in txt)
    if pos == neg:
        return 0.0
    return float((pos - neg) / max(1, pos + neg))

def fetch_news_with_sentiment(api_key: str, tickers: list[str]) -> pd.DataFrame:
    query = " OR ".join(tickers) if tickers else "stock market"
    try:
        resp = requests.get("https://newsapi.org/v2/everything", params={
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": api_key
        }, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
    except Exception:
        articles = []
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None

    rows = []
    for art in articles:
        title = art.get("title", "")
        if not title:
            continue
        if analyzer is not None:
            score = analyzer.polarity_scores(title)["compound"]
        else:
            score = vader_score(title)
        label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        rows.append({"headline": title, "sentiment": score, "label": label})
    return pd.DataFrame(rows)
