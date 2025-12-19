import importlib
from types import SimpleNamespace

import pytest


def _make_obs(values):
    return {"observations": [{"value": str(v)} for v in values]}


class DummyResp:
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status_code = status
        self._json = json_data
        self._text = text_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    @property
    def text(self):
        return self._text


def test_fetch_rates_with_fred(monkeypatch):
    # import the module fresh
    mod = importlib.import_module("pages.01_Home")
    importlib.reload(mod)

    # clear any cached results
    try:
        mod.fetch_rates.clear()
    except Exception:
        pass

    # pretend we have a FRED key available via st.secrets
    class FakeSecrets(dict):
        def get(self, k, default=None):
            return "FAKE_KEY" if k == "FRED_API_KEY" else default

    monkeypatch.setattr(mod, "st", SimpleNamespace(secrets=FakeSecrets()))

    # Mock requests.get to return JSON for the various series
    def fake_get(url, params=None, timeout=None):
        sid = (params or {}).get("series_id")
        if sid == "FEDFUNDS":
            return DummyResp(json_data=_make_obs([0.0] * 23 + [5.25]))
        if sid == "DGS10":
            return DummyResp(json_data=_make_obs([0.0] * 23 + [4.00]))
        if sid == "SOFR":
            return DummyResp(json_data=_make_obs([0.0] * 23 + [5.00]))
        if sid == "CPIAUCSL":
            # produce 24 months where last=300 and 13th-back=287 -> ~4.524%
            vals = list(range(277, 301))
            return DummyResp(json_data=_make_obs(vals))
        raise RuntimeError("unexpected request")

    monkeypatch.setattr("requests.get", fake_get)

    rates = mod.fetch_rates()
    # Basic sanity checks
    assert rates.get("fed_funds") is not None
    assert abs(rates.get("fed_funds") - 5.25) < 1e-6
    assert abs(rates.get("10y") - 4.0) < 1e-6
    assert abs(rates.get("sofr") - 5.0) < 1e-6
    # CPI YoY expected based on our generated vals (fetch_rates compares last vs 13th-back)
    assert rates.get("cpi_yoy") is not None
    # our vals list(range(277, 301)) -> last=300, 13th-back is index -13 -> value 288
    assert pytest.approx(rates.get("cpi_yoy"), rel=1e-3) == (300 - 288) / 288 * 100


def test_fetch_rates_fallbacks(monkeypatch):
    mod = importlib.import_module("pages.01_Home")
    importlib.reload(mod)
    try:
        mod.fetch_rates.clear()
    except Exception:
        pass

    # Ensure no FRED key
    monkeypatch.setattr(mod, "st", SimpleNamespace(secrets={}))
    monkeypatch.setenv("FRED_API_KEY", "")

    # Provide fallback CSV and HTML responses
    def fake_get(url, params=None, timeout=None):
        if "daily-treasury-rates.csv" in url:
            # header contains DGS10 as third column; provide a last line with value 3.5
            txt = "Date,1 Mo,10 Yr\n2025-09-23,0.10,3.50\n"
            return DummyResp(status=200, text_data=txt)
        if "sofr.csv" in url:
            txt = "Date,SOFR\n2025-09-23,4.20\n"
            return DummyResp(status=200, text_data=txt)
        if "fedfunds.htm" in url:
            html = "... Effective Federal Funds Rate is 5.25% today ..."
            return DummyResp(status=200, text_data=html)
        # default
        return DummyResp(status=404, text_data="")

    monkeypatch.setattr("requests.get", fake_get)

    rates = mod.fetch_rates()
    assert rates.get("10y") is not None
    assert abs(rates.get("10y") - 3.5) < 0.01
    assert rates.get("sofr") is not None
    assert abs(rates.get("sofr") - 4.2) < 0.01
    assert rates.get("fed_funds") is not None
    assert abs(rates.get("fed_funds") - 5.25) < 0.1
