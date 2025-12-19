"""Small retry helper utilities for network calls.

Provides a requests.get wrapper with exponential backoff and jitter.
"""
from __future__ import annotations
import time
import random
from typing import Optional, Tuple, Any

import requests


def requests_get_with_retry(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 10.0,
    retries: int = 3,
    backoff_factor: float = 0.8,
    max_backoff: float = 8.0,
    allowed_exceptions: Tuple[Any, ...] = (requests.exceptions.RequestException,)
) -> Optional[requests.Response]:
    """Perform requests.get with retries on transient exceptions.

    Returns the Response object on success, or None if all retries failed.
    This function intentionally swallows exceptions and returns None on failure so
    callers can fall back gracefully.
    """
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            return r
        except allowed_exceptions:
            if attempt == retries:
                return None
            # exponential backoff with jitter
            sleep_for = min(max_backoff, backoff_factor * (2 ** attempt))
            # jitter: uniform between 0.5x and 1.5x
            sleep_for = sleep_for * random.uniform(0.5, 1.5)
            time.sleep(sleep_for)
            attempt += 1
    return None
