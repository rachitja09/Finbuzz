import requests
import pandas as pd
from config import get_runtime_key
from requests.exceptions import SSLError, RequestException


def get_quote_finnhub(symbol: str) -> dict:
    """Fetch latest quote for a symbol from Finnhub.

    Respects explicit-empty-env semantics via config.get_runtime_key.
    """
    key = get_runtime_key("FINNHUB_API_KEY")
    if not key:
        return {}
    url = "https://finnhub.io/api/v1/quote"
    try:
        resp = requests.get(url, params={"symbol": symbol, "token": key}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except SSLError as e:
        return {"_error": f"SSL error contacting Finnhub: {e}"}
    except RequestException as e:
        return {"_error": f"Failed contacting Finnhub: {e}"}


def fetch_analyst_consensus(api_key: str, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/recommendation",
                params={"symbol": ticker, "token": api_key}, timeout=15
            )
            resp.raise_for_status()
            recs = resp.json() or []
            if recs:
                latest = recs[0]
                rows.append({
                    "ticker": ticker,
                    "strongBuy": int(latest.get("strongBuy", 0)),
                    "buy": int(latest.get("buy", 0)),
                    "hold": int(latest.get("hold", 0)),
                    "sell": int(latest.get("sell", 0)),
                    "strongSell": int(latest.get("strongSell", 0)),
                    "targetMean": latest.get("targetMean")
                })
        except SSLError:
            # network/SSL issue -> skip but record nothing
            continue
        except RequestException:
            continue
    return pd.DataFrame(rows)
