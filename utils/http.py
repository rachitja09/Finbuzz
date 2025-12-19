"""HTTP helpers: cached requests session and simple per-host rate limiting.

Provides `cached_get` as a drop-in replacement for callers that expect a
`_get(url, params)` function returning parsed JSON/text.

Uses requests-cache for on-disk caching (safe for local dev). This reduces
API usage and improves test repeatability.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional

import requests
try:
    import requests_cache
except Exception:
    requests_cache = None

# configure a cached session (disk-backed sqlite) if requests_cache is installed
if requests_cache is not None:
    session = requests_cache.CachedSession(".cache/http_cache", backend="sqlite", expire_after=600)
else:
    session = requests.Session()

# Simple per-host last-call tracker to avoid spamming the same host
_LAST_CALL: Dict[str, float] = {}
_MIN_INTERVAL_PER_HOST = 0.5  # seconds; configurable default


def _host_from_url(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc


def cached_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20, min_interval: Optional[float] = None) -> Any:
    """GET the URL with polite throttling and caching.

    Returns parsed JSON when possible, otherwise returns text.
    """
    host = _host_from_url(url)
    now = time.time()
    mi = float(min_interval) if min_interval is not None else _MIN_INTERVAL_PER_HOST
    last = _LAST_CALL.get(host, 0.0)
    dt = now - last
    if dt < mi:
        time.sleep(mi - dt)
    _LAST_CALL[host] = time.time()

    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    # try JSON, fall back to text
    try:
        return r.json()
    except Exception:
        return r.text
