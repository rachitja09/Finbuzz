"""Simple supply-chain helper with a small mock mapping.

This module provides get_supply_chain(sym) -> dict with keys 'suppliers' and 'customers'.
It's intentionally lightweight and uses a local in-memory mapping. Replace or extend
with real data providers later (e.g., GEP, Bloomberg, or a scraped dataset).
"""
from __future__ import annotations
from typing import Dict, List, Any
import os
import functools
import requests

# Local mock mapping (fallback). Keep small and replaceable.
_MOCK = {
    "AAPL": {
        "suppliers": [
            {"symbol": "TSM", "name": "Taiwan Semiconductor Manufacturing Co."},
            {"symbol": "QCOM", "name": "Qualcomm Inc."},
            {"symbol": "ASML", "name": "ASML Holding"},
        ],
        "customers": [
            {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
        ],
    },
    "MSFT": {
        "suppliers": [
            {"symbol": "INTC", "name": "Intel Corporation"},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        ],
        "customers": [
            {"symbol": "ORCL", "name": "Oracle Corporation"},
            {"symbol": "SAP", "name": "SAP SE"},
        ],
    },
    "TSLA": {
        "suppliers": [
            {"symbol": "CATL", "name": "Contemporary Amperex Technology Co. Limited"},
            {"symbol": "LAC", "name": "Lithium Americas"},
        ],
        "customers": [
            {"symbol": "DHL", "name": "DHL Group"},
        ],
    },
}


def _get_key_from_env_or_secrets(name: str) -> str | None:
    """Delegate runtime key resolution to the centralized helper in `config`.

    Keeps the explicit-empty-env semantics consistent across the codebase.
    """
    try:
        from config import get_runtime_key

        return get_runtime_key(name)
    except Exception:
        # fallback: try environment only
        v = os.environ.get(name)
        if v:
            return v
    return None


@functools.lru_cache(maxsize=256)
def _fetch_finnhub_peers(sym: str, api_key: str) -> List[str]:
    """Return a list of peer symbols from Finnhub (best-effort)."""
    try:
        url = f"https://finnhub.io/api/v1/stock/peers?symbol={sym}&token={api_key}"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return [s.upper() for s in data if isinstance(s, str)]
    except Exception:
        pass
    return []


def _fetch_finnhub_company(sym: str, api_key: str) -> Dict[str, Any]:
    """Best-effort company profile from Finnhub (may include industry/peers indirectly)."""
    try:
        # avoid importing data_providers at module import-time
        from data_providers import FINNHUB
        url = f"{FINNHUB}/stock/profile2?symbol={sym}&token={api_key}"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), dict) else {}
    except Exception:
        return {}


def _fetch_fmp_related(sym: str, api_key: str) -> List[str]:
    """Best-effort: use FMP "similar" or peers endpoints if available to get related tickers."""
    try:
        # There is an endpoint /api/v3/profile/{symbol} which may include peers in some responses
        from data_providers import FMP
        j = requests.get(f"{FMP}/v3/profile/{sym}", params={"apikey": api_key}, timeout=6)
        j.raise_for_status()
        data = j.json()
        if isinstance(data, list) and data:
            obj = data[0]
        elif isinstance(data, dict):
            obj = data
        else:
            obj = {}
        peers = obj.get("peers") or obj.get("peersList") or []
        if isinstance(peers, list) and peers:
            return [p.upper() for p in peers if isinstance(p, str)]
    except Exception:
        pass

    # Fallback: try other peers endpoints
    try:
        from data_providers import FMP
        j = requests.get(f"{FMP}/v3/stock_peers/{sym}", params={"apikey": api_key}, timeout=6)
        j.raise_for_status()
        data = j.json()
        if isinstance(data, list):
            return [p.upper() for p in data if isinstance(p, str)]
    except Exception:
        pass

    return []


def get_supply_chain(sym: str, enrich_live: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Return suppliers and customers for a given symbol.

    By default returns the local mock mapping. If `enrich_live` is True and a
    FINNHUB API key is available (in env or Streamlit secrets), the function
    will attempt to fetch peer/related companies and include them as 'related'
    customers (best-effort). This is a pragmatic enrichment; true supplier/customer
    relationships require specialized datasets.
    """
    if not sym:
        return {"suppliers": [], "customers": []}
    k = sym.strip().upper()
    base = _MOCK.get(k, {"suppliers": [], "customers": []}).copy()

    if enrich_live:
        fh_key = _get_key_from_env_or_secrets("FINNHUB_API_KEY")
        if fh_key:
            peers = _fetch_finnhub_peers(k, fh_key)
            # add peers as 'customers' labeled as related peers (avoid duplicates)
            existing = {x.get("symbol") for x in base.get("customers", [])}
            for p in peers:
                if p not in existing and p != k:
                    base.setdefault("customers", []).append({"symbol": p, "name": p})
        # also try FMP-related peers and use them to populate suppliers when missing
        fmp_key = _get_key_from_env_or_secrets("FMP_API_KEY") or _get_key_from_env_or_secrets("FMP_API_KEY_V3")
        combined_peers = set(peers)
        if fmp_key:
            try:
                fmp_peers = _fetch_fmp_related(k, fmp_key)
                for p in fmp_peers:
                    combined_peers.add(p)
                    if p not in existing and p != k:
                        base.setdefault("customers", []).append({"symbol": p, "name": p})
            except Exception:
                pass
        # If the mock provider has no suppliers, populate a few peers as 'related suppliers' (best-effort)
        if not base.get("suppliers") and combined_peers:
            cnt = 0
            for p in sorted(combined_peers):
                if p == k:
                    continue
                base.setdefault("suppliers", []).append({"symbol": p, "name": p + " (related)"})
                cnt += 1
                if cnt >= 5:
                    break
    return base


@functools.lru_cache(maxsize=512)
def get_kpis(sym: str) -> Dict[str, Any]:
    """Best-effort lightweight KPIs for a symbol.

    Returns a small dict with keys: symbol, name, price, mktCap, delta_pct, pe, sector.
    Uses `data_providers` functions when available and falls back to very small local data.
    This function is cached to avoid repeated network calls when rendering the graph.
    """
    out: Dict[str, Any] = {"symbol": ""}
    if not sym:
        return out
    s = sym.strip().upper()
    out["symbol"] = s
    # resolve keys from env/secrets (best-effort)
    fh_key = _get_key_from_env_or_secrets("FINNHUB_API_KEY")
    fmp_key = _get_key_from_env_or_secrets("FMP_API_KEY") or _get_key_from_env_or_secrets("FMP_API_KEY_V3")
    try:
        # import inside function to avoid import-time network activity
        import data_providers as _dp

        prof = _dp.fmp_profile(s, fmp_key or "") or {}
        quote = _dp.quote_now(s, fh_key) if fh_key else {}
        pe = None
        try:
            pe = _dp.company_pe_ttm(s, fmp_key or "")
        except Exception:
            pe = None

        name = prof.get("companyName") or prof.get("name") or prof.get("longName") or s
        price = quote.get("c") or prof.get("price")
        mkt = prof.get("mktCap") or prof.get("mktcap")
        delta_pct = quote.get("dp")

        out.update({"name": name, "price": price, "mktCap": mkt, "delta_pct": delta_pct, "pe": pe, "sector": prof.get("sector") or prof.get("industry")})
        return out
    except Exception:
        # fallback to mock mapping name if available
        base = _MOCK.get(s)
        if base:
            out["name"] = s
            return out
    return out
