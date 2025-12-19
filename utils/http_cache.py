"""Simple on-disk HTTP GET cache with TTL and retry/backoff.

Usage: from utils.http_cache import cached_get
cached_get(url, params=None, ttl=300)
"""
from __future__ import annotations
import json
import os
import time
import hashlib
from typing import Optional
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, ".http_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")

def _make_key(url: str, params: Optional[dict]) -> str:
    s = url + "|" + (json.dumps(params, sort_keys=True) if params else "")
    return hashlib.sha1(s.encode("utf8")).hexdigest()

def cached_get(url: str, params: Optional[dict] = None, ttl: int = 300, max_retries: int = 3, backoff: float = 0.5):
    key = _make_key(url, params)
    p = _cache_path(key)
    now = time.time()
    # Return cached if fresh
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf8") as f:
                rec = json.load(f)
            if now - rec.get("ts", 0) < ttl:
                return rec.get("status"), rec.get("text"), rec.get("json")
    except Exception:
        pass

    # Fetch with retries/backoff
    attempt = 0
    from requests.exceptions import RequestException, SSLError

    while attempt < max_retries:
        try:
            r = requests.get(url, params=params, timeout=10)
            try:
                j = r.json()
            except Exception:
                j = None
            # cache response
            try:
                with open(p, "w", encoding="utf8") as f:
                    json.dump({"ts": now, "status": r.status_code, "text": r.text[:2000], "json": j}, f)
            except Exception:
                pass
            return r.status_code, r.text, j
        except SSLError as e:
            # SSL certificate verification failed — don't crash; return error info
            return None, f"SSL error: {e}", None
        except RequestException:
            attempt += 1
            time.sleep(backoff * (2 ** (attempt - 1)))
    return None, None, None
