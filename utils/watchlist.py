from typing import Iterable, Dict
import concurrent.futures
from data_fetchers.finnhub import get_quote_finnhub


def prefetch_quotes(tickers: Iterable[str], max_workers: int = 6) -> Dict[str, dict]:
    """Prefetch quotes concurrently and return a dict ticker->quote.

    If FINNHUB_API_KEY is missing, get_quote_finnhub returns empty dicts.
    """
    out: Dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_quote_finnhub, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                out[t] = fut.result()
            except Exception:
                out[t] = {}
    return out
