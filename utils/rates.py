import os
import re
from typing import Dict, Optional
import requests
import streamlit as st
import sys


def _get_fred_key() -> str | None:
    """Resolve FRED_API_KEY at runtime using the standard helper.

    This mirrors the project's explicit-empty-env semantics: if the env var is present
    but empty, treat it as an explicit disable for runtime keys.
    """
    try:
        from config import get_runtime_key

        return get_runtime_key("FRED_API_KEY")
    except Exception:
        # fallback: check environment variable only
        if "FRED_API_KEY" in os.environ:
            v = os.environ.get("FRED_API_KEY")
            return v if v else None
        # no runtime key available
        return None


def _fetch_rates_impl() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"fed_funds": None, "10y": None, "sofr": None, "cpi_yoy": None, "fed_funds_target_low": None, "fed_funds_target_high": None}
    try:
        fred_key = _get_fred_key()

        if fred_key:
            base = "https://api.stlouisfed.org/fred/series/observations"
            for code, k in [("FEDFUNDS", "fed_funds"), ("DGS10", "10y"), ("SOFR", "sofr")]:
                    try:
                        try:
                            from utils.retry import requests_get_with_retry
                            r = requests_get_with_retry(base, params={"series_id": code, "api_key": fred_key, "file_type": "json", "limit": 1}, timeout=5, retries=2)
                        except Exception:
                            r = requests.get(base, params={"series_id": code, "api_key": fred_key, "file_type": "json", "limit": 1}, timeout=5)
                        if r is None:
                            out[k] = None
                        else:
                            r.raise_for_status()
                            j = r.json()
                            obs = j.get("observations") or []
                            if obs:
                                out[k] = float(obs[-1].get("value") or float("nan"))
                    except requests.exceptions.SSLError:
                        out[k] = None
                    except requests.exceptions.RequestException:
                        out[k] = None
            try:
                r = requests.get(base, params={"series_id": "CPIAUCSL", "api_key": fred_key, "file_type": "json", "limit": 24}, timeout=6)
                r.raise_for_status()
                j = r.json()
                obs = j.get("observations") or []
                if len(obs) >= 13:
                    last = float(obs[-1].get("value") or float("nan"))
                    prev = float(obs[-13].get("value") or float("nan"))
                    if prev and prev != 0.0:
                        out["cpi_yoy"] = (last - prev) / prev * 100.0
            except requests.exceptions.SSLError:
                out["cpi_yoy"] = None
            except requests.exceptions.RequestException:
                out["cpi_yoy"] = None
            return out

        try:
            try:
                from utils.retry import requests_get_with_retry
                t = requests_get_with_retry("https://home.treasury.gov/resource-library/data-chart-center/interest-rates/daily-treasury-rates.csv", timeout=5, retries=2)
            except Exception:
                t = requests.get("https://home.treasury.gov/resource-library/data-chart-center/interest-rates/daily-treasury-rates.csv", timeout=5)
            if t is not None and t.status_code == 200:
                txt = t.text.splitlines()
                hdr = txt[0].split(',') if txt else []

                def _find_10y_index(headers: list[str]) -> Optional[int]:
                    if "DGS10" in headers:
                        return headers.index("DGS10")
                    for i, h in enumerate(headers):
                        if h is None:
                            continue
                        hn = h.strip()
                        if ("10" in hn and ("Yr" in hn or "yr" in hn or "Y" in hn)) or hn in ("10 Yr", "10Y", "10y", "10"):
                            return i
                    return None

                idx = _find_10y_index(hdr)
                for ln in reversed(txt):
                    if ln.strip() and not ln.startswith("Date"):
                        parts = ln.split(',')
                        if idx is not None and idx < len(parts):
                            try:
                                out["10y"] = float(parts[idx])
                            except Exception:
                                out["10y"] = None
                        break
        except requests.exceptions.SSLError:
            out["10y"] = None
        except requests.exceptions.RequestException:
            out["10y"] = None

        try:
            try:
                from utils.retry import requests_get_with_retry
                s = requests_get_with_retry("https://www.newyorkfed.org/medialibrary/media/markets/desk-operations/sofr/sofr.csv", timeout=5, retries=2)
            except Exception:
                s = requests.get("https://www.newyorkfed.org/medialibrary/media/markets/desk-operations/sofr/sofr.csv", timeout=5)
            if s is not None and s.status_code == 200:
                lines = s.text.splitlines()
                for ln in reversed(lines):
                    if ln.strip() and not ln.startswith("Date"):
                        val = ln.split(',')[-1]
                        try:
                            out["sofr"] = float(val)
                        except Exception:
                            out["sofr"] = None
                        break
        except requests.exceptions.SSLError:
            out["sofr"] = None
        except requests.exceptions.RequestException:
            out["sofr"] = None

        try:
            try:
                from utils.retry import requests_get_with_retry
                r2 = requests_get_with_retry("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", timeout=5, retries=2)
            except Exception:
                r2 = requests.get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", timeout=5)
            if r2 is not None and r2.status_code == 200:
                text = r2.text
                # Try to extract an announced target range like '4.00 to 4.25 percent' or '4.00 – 4.25 percent'
                m = re.search(r"target range[\s\S]{0,200}?([0-9]+(?:\.[0-9]+)?)\s*(?:to|–|-|—)\s*([0-9]+(?:\.[0-9]+)?)\s*percent", text, flags=re.IGNORECASE)
                if m:
                    try:
                        low = float(m.group(1))
                        high = float(m.group(2))
                        out["fed_funds_target_low"] = low
                        out["fed_funds_target_high"] = high
                    except Exception:
                        out["fed_funds_target_low"] = None
                        out["fed_funds_target_high"] = None
                else:
                    # Fallback: try to find a single effective rate number on the release page
                    m2 = re.search(r"(\d+\.\d+)%", text)
                    if m2:
                        try:
                            out["fed_funds"] = float(m2.group(1))
                        except Exception:
                            out["fed_funds"] = None
            else:
                # fallback to older releases page if calendar page not reachable
                try:
                    from utils.retry import requests_get_with_retry
                    r3 = requests_get_with_retry("https://www.federalreserve.gov/releases/fedfunds.htm", timeout=5, retries=2)
                except Exception:
                    r3 = requests.get("https://www.federalreserve.gov/releases/fedfunds.htm", timeout=5)
                if r3 is not None and r3.status_code == 200:
                    m3 = re.search(r"(\d+\.\d+)%", r3.text)
                    if m3:
                        try:
                            out["fed_funds"] = float(m3.group(1))
                        except Exception:
                            out["fed_funds"] = None
        except requests.exceptions.SSLError:
            out["fed_funds"] = None
        except requests.exceptions.RequestException:
            out["fed_funds"] = None
    except Exception:
        pass
    return out


if "pytest" in sys.modules:
    def fetch_rates() -> Dict[str, Optional[float]]:
        return _fetch_rates_impl()
else:
    @st.cache_data(ttl=300)
    def _fetch_rates_cached() -> Dict[str, Optional[float]]:
        return _fetch_rates_impl()

    def fetch_rates() -> Dict[str, Optional[float]]:
        return _fetch_rates_cached()


@st.cache_data(ttl=600)
def fetch_rate_series(series_id: str, fred_key: str | None, points: int = 60):
    if not fred_key:
        return []
    try:
        base = "https://api.stlouisfed.org/fred/series/observations"
        try:
            from utils.retry import requests_get_with_retry
            r = requests_get_with_retry(base, params={"series_id": series_id, "api_key": fred_key, "file_type": "json", "limit": points}, timeout=6, retries=2)
        except Exception:
            r = requests.get(base, params={"series_id": series_id, "api_key": fred_key, "file_type": "json", "limit": points}, timeout=6)
        if r is None:
            return []
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations") or []
        vals = []
        for o in obs:
            try:
                v = float(o.get("value") or float("nan"))
                vals.append(v)
            except Exception:
                vals.append(float("nan"))
        return vals
    except Exception:
        return []
