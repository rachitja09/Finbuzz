from __future__ import annotations
import time
from typing import Dict, Any, Optional

import requests
import pandas as pd

BASES = {
    "doc": "https://api.gdeltproject.org/api/v2/doc/doc",
    "context": "https://api.gdeltproject.org/api/v2/context/context",
    "geo": "https://api.gdeltproject.org/api/v2/geo/geo",
}


class GDELT:
    """Lightweight GDELT v2 client.

    - No API key required.
    - Includes a simple throttler to avoid hammering the public API.
    - geo_search returns a pandas.DataFrame when CSV is requested.
    """

    def __init__(self, session: Optional[requests.Session] = None, min_interval_sec: float = 1.0, max_retries: int = 2):
        self.s = session or requests.Session()
        self.min_interval = float(min_interval_sec)
        self._last = 0.0
        self.max_retries = int(max_retries)

    def _throttle(self) -> None:
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()

    def _get_json(self, base: str, params: Dict[str, Any]) -> Dict[str, Any]:
        attempts = 0
        while True:
            attempts += 1
            self._throttle()
            r = self.s.get(base, params=params, timeout=30)
            if r.status_code == 429 and attempts <= self.max_retries:
                # rate-limited, back off
                time.sleep(1.0 * attempts)
                continue
            r.raise_for_status()
            # GDELT usually returns JSON for DOC/CONTEXT; be defensive in case
            # the endpoint returns an empty body or non-JSON error page.
            try:
                return r.json()
            except ValueError:
                # empty or invalid JSON — return an empty mapping so callers
                # that expect dict.get(...) keep working rather than crashing.
                txt = (r.text or "").strip()
                if not txt:
                    return {}
                # If the response looks like HTML or plain text, return a
                # small sentinel dict with the raw text so UIs can render it.
                return {"raw": txt}

    def doc_search(self, query: str, *, startdatetime: Optional[str] = None, enddatetime: Optional[str] = None,
                   maxrecords: int = 250, mode: str = "artlist", format_: str = "json", **kwargs) -> Dict[str, Any]:
        """DOC 2.0 search (full-text, ~last 3 months). Query is a GDELT query string.

        startdatetime/enddatetime: 'YYYYMMDDHHMMSS' UTC strings if provided
        mode: artlist | timelinevol | timelinesent | ...
        format_: json|csv (we default to json)
        """
        params: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "maxrecords": int(maxrecords),
            "format": format_,
            **({"startdatetime": startdatetime} if startdatetime else {}),
            **({"enddatetime": enddatetime} if enddatetime else {}),
        }
        params.update(kwargs)
        return self._get_json(BASES["doc"], params)

    def context_search(self, query: str, *, format_: str = "json", **kwargs) -> Dict[str, Any]:
        """Context 2.0: sentence-level snippets (JSON)"""
        params = {"query": query, "format": format_, **kwargs}
        return self._get_json(BASES["context"], params)

    def geo_search(self, query: str, *, mode: str = "PointData", format_: str = "CSV", **kwargs) -> pd.DataFrame:
        """GEO 2.0: geocoded hits.

        mode examples: PointData, CountryInfo, Admin1Info
        format_: CSV | JSON | GeoJSON
        Returns: pandas.DataFrame for CSV (most convenient)
        """
        params = {"query": query, "mode": mode, "format": format_, **kwargs}
        attempts = 0
        while True:
            attempts += 1
            self._throttle()
            r = self.s.get(BASES["geo"], params=params, timeout=30)
            if r.status_code == 429 and attempts <= self.max_retries:
                time.sleep(1.0 * attempts)
                continue
            r.raise_for_status()
            if format_.upper() == "CSV":
                # Try to parse CSV into pandas safely
                from io import StringIO

                try:
                    return pd.read_csv(StringIO(r.text))
                except Exception:
                    # Last-resort: return a single-row DataFrame with raw text
                    return pd.DataFrame({"raw": [r.text]})
            # For JSON/GeoJSON return a normalized DataFrame
            try:
                return pd.json_normalize(r.json())
            except Exception:
                return pd.DataFrame({"raw": [r.text]})

    @staticmethod
    def read_events_csv(url: str) -> pd.DataFrame:
        """Read a GDELT events/GKG CSV by URL into a DataFrame.

        Use the GDELT codebooks to interpret columns. This reads tab-separated
        files in the data.gdeltproject.org feed.
        """
        return pd.read_csv(url, sep="\t", header=None, low_memory=False)
