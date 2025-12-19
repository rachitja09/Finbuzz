from __future__ import annotations
import pandas as pd
from typing import Any, Dict, List, Optional


def _analyst_score_from_df(analyst_df: Optional[pd.DataFrame]) -> float:
    """Compute a normalized analyst consensus score in [-1,1] from a DataFrame
    with columns like strongBuy,buy,hold,sell,strongSell similar to Finnhub output."""
    if analyst_df is None:
        return 0.0
    try:
        if hasattr(analyst_df, "empty") and analyst_df.empty:
            return 0.0
        # use first row
        row = analyst_df.iloc[0] if hasattr(analyst_df, "iloc") else analyst_df
        def _to_float(val):
            try:
                # pandas Series/Index may be returned; handle gracefully
                if hasattr(val, "item"):
                    return float(val.item())
                return float(val)
            except Exception:
                try:
                    return float(str(val))
                except Exception:
                    return 0.0

        strongbuy = _to_float(row.get("strongBuy", 0) or 0)
        buy = _to_float(row.get("buy", 0) or 0)
        hold = _to_float(row.get("hold", 0) or 0)
        sell = _to_float(row.get("sell", 0) or 0)
        strongsell = _to_float(row.get("strongSell", 0) or 0)
        total = max(1.0, strongbuy + buy + hold + sell + strongsell)
        cons = ((strongbuy + buy) - (sell + strongsell)) / total
        # clamp to [-1,1]
        return max(-1.0, min(1.0, float(cons)))
    except Exception:
        return 0.0


def _news_score_from_headlines(headlines: Optional[List[Dict[str, Any]]], vader_fn) -> float:
    """Compute a small news sentiment score in [-1,1] from a list of article dicts.
    Expects `vader_fn(text)->float` to compute compound score."""
    if not headlines:
        return 0.0
    scores = []
    for a in headlines[:20]:
        title = a.get("title") or a.get("headline") or ""
        desc = a.get("description") or a.get("summary") or ""
        text = f"{title}. {desc}" if desc else title
        try:
            s = float(vader_fn(text))
        except Exception:
            s = 0.0
        scores.append(s)
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    # clamp
    return max(-1.0, min(1.0, avg))


def compute_recommendation(sym: str, df: pd.DataFrame, tech_w: float = 1.0, analyst_w: float = 0.7, consensus_w: float = 0.5, news_w: float = 0.3,
                           fetch_analyst_fn=None, news_headlines_fn=None, vader_fn=None) -> Dict[str, Any]:
    """Compute a composite recommendation dict. Fetcher functions are injectable for testing.

    Returns a dict with keys: score, tech_score, analyst_score, news_score, components, recommendation, explanation
    """
    out = {"score": 0.0, "tech_score": 0.0, "analyst_score": 0.0, "news_score": 0.0, "components": {}, "recommendation": "Hold", "explanation": []}
    if df is None or len(df) == 0:
        out["explanation"].append("No price data")
        return out

    # Technical signals
    try:
        last = df.iloc[-1]
        ema20 = float(last.get("ema20") or 0)
        ema50 = float(last.get("ema50") or 0)
        sma_sig = 0
        if ema20 and ema50:
            sma_sig = 1 if ema20 > ema50 else -1
        rsi = float(last.get("rsi") or 0)
        rsi_sig = 0
        if rsi:
            rsi_sig = 1 if rsi < 30 else (-1 if rsi > 70 else 0)
        mom_sig = 0
        if len(df) >= 6:
            prev = float(df.iloc[-6].get("close") or df.iloc[-6].get("Close") or 0)
            if prev:
                curr = float(last.get("close") or last.get("Close") or 0)
                pct = (curr - prev) / prev
                mom_sig = 1 if pct > 0 else (-1 if pct < 0 else 0)
        tech_score = (sma_sig + rsi_sig + mom_sig) / 3.0
        out["tech_score"] = float(tech_score)
        out["components"] = {"sma": int(sma_sig), "rsi": int(rsi_sig), "momentum": int(mom_sig)}
        out["explanation"].append(f"Technical: sma={sma_sig}, rsi={rsi_sig}, mom={mom_sig} -> {tech_score:+.2f}")
    except Exception as e:
        out["explanation"].append(f"Technical computation failed: {e}")

    # Analyst consensus
    analyst_score = 0.0
    if fetch_analyst_fn is not None:
        try:
            adf = fetch_analyst_fn(sym)
            analyst_score = _analyst_score_from_df(adf)
            out["analyst_score"] = analyst_score
            if analyst_score != 0.0:
                out["explanation"].append(f"Analyst consensus: {analyst_score:+.2f}")
        except Exception:
            analyst_score = 0.0

    # News sentiment
    news_score = 0.0
    if news_headlines_fn is not None and vader_fn is not None:
        try:
            heads = news_headlines_fn(sym, limit=10)
            news_score = _news_score_from_headlines(heads, vader_fn)
            out["news_score"] = news_score
            if news_score != 0.0:
                out["explanation"].append(f"News sentiment: {news_score:+.2f}")
        except Exception:
            news_score = 0.0

    # Compose final score
    score = tech_w * out.get("tech_score", 0.0) + analyst_w * analyst_score + consensus_w * analyst_score + news_w * news_score
    out["score"] = float(score)
    if score >= 0.25:
        out["recommendation"] = "Buy"
    elif score <= -0.25:
        out["recommendation"] = "Sell"
    else:
        out["recommendation"] = "Hold"
    out["explanation"].append(f"Composite score: {score:+.2f}")
    return out
