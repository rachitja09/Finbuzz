from __future__ import annotations
import threading
import time
from typing import Any, Dict, Optional


class TTLCache:
    """A tiny, thread-safe TTL cache for small signal results."""

    def __init__(self, default_ttl: float = 30.0):
        self._store: Dict[str, tuple[float, Any]] = {}
        self._ttl = float(default_ttl)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            tup = self._store.get(key)
            if not tup:
                return None
            ts, val = tup
            if now - ts > self._ttl:
                del self._store[key]
                return None
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# module-level default cache
_default_cache = TTLCache()


def cache_get(key: str) -> Optional[Any]:
    return _default_cache.get(key)


def cache_set(key: str, value: Any) -> None:
    _default_cache.set(key, value)


def cache_clear() -> None:
    _default_cache.clear()
import pickle
import hashlib
from typing import Any


def _hash(obj: Any) -> str:
    try:
        return hashlib.sha1(pickle.dumps(obj)).hexdigest()
    except Exception:
        return str(hash(obj))


def simple_cache(key_obj: Any, value: Any, store: dict, max_items: int = 128):
    """Simple session-style cache stored in a dict (e.g., st.session_state['ui']).

    key_obj: any picklable object used to compute cache key
    value: the value to store
    store: dictionary (session_state.ui)
    """
    k = _hash(key_obj)
    cache = store.setdefault("_simple_cache", {})
    if len(cache) > max_items:
        # drop oldest simplisticly
        cache.pop(next(iter(cache)))
    cache[k] = value
    return k


def simple_get(key_obj: Any, store: dict):
    k = _hash(key_obj)
    cache = store.setdefault("_simple_cache", {})
    return cache.get(k)


def timed_lru_cache(ttl_seconds: int = 300, maxsize: int = 128):
    """Return a decorator that caches function results with LRU and TTL.

    Lightweight helper used for non-Streamlit functions (works in tests).
    """
    from functools import lru_cache, wraps
    import time

    def decorator(fn):
        cached = lru_cache(maxsize=maxsize)(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # cache key based on args/kwargs; we implement TTL by storing
            # a timestamp attribute on the wrapper and clearing the cache
            now = time.time()
            last = getattr(wrapper, "_last_clear", 0)
            if now - last > ttl_seconds:
                try:
                    cached.cache_clear()
                except Exception:
                    pass
                setattr(wrapper, "_last_clear", now)
            return cached(*args, **kwargs)

        setattr(wrapper, "_last_clear", 0)
        setattr(wrapper, "cache_clear", getattr(cached, "cache_clear", lambda: None))
        return wrapper

    return decorator
