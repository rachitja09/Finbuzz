import importlib
from types import SimpleNamespace

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


def _make_obs(values):
    return {"observations": [{"value": str(v)} for v in values]}


def test_market_overview_fetch_rates_with_fred(monkeypatch):
    mod = importlib.import_module("pages.01_Home")
    importlib.reload(mod)

    # provide a fake FRED key via Streamlit secrets
    monkeypatch.setattr(mod, "st", SimpleNamespace(secrets={"FRED_API_KEY": "FAKE"}))

    def fake_get(url, params=None, timeout=None):
        sid = (params or {}).get("series_id")
        if sid == "FEDFUNDS":
            return DummyResp(json_data=_make_obs([0]*23 + [5.25]))
        if sid == "DGS10":
            return DummyResp(json_data=_make_obs([0]*23 + [4.00]))
        if sid == "SOFR":
            return DummyResp(json_data=_make_obs([0]*23 + [5.00]))
        raise RuntimeError("unexpected request")

    monkeypatch.setattr("requests.get", fake_get)

    rates = mod.fetch_rates()
    assert rates.get("fed_funds") is not None
    assert abs(rates.get("fed_funds") - 5.25) < 1e-6
    assert abs(rates.get("10y") - 4.0) < 1e-6
    assert abs(rates.get("sofr") - 5.0) < 1e-6


def test_market_overview_fetch_rates_fallbacks(monkeypatch):
    mod = importlib.import_module("pages.01_Home")
    importlib.reload(mod)

    # ensure no FRED key by providing an empty secrets mapping and an empty env var
    monkeypatch.setattr(mod, "st", SimpleNamespace(secrets={}))
    monkeypatch.setenv("FRED_API_KEY", "")

    def fake_get(url, params=None, timeout=None):
        if "daily-treasury-rates.csv" in url:
            txt = "Date,1 Mo,10 Yr\n2025-09-23,0.10,3.50\n"
            return DummyResp(status=200, text_data=txt)
        if "sofr.csv" in url:
            txt = "Date,SOFR\n2025-09-23,4.20\n"
            return DummyResp(status=200, text_data=txt)
        if "fedfunds.htm" in url:
            html = "... Effective Federal Funds Rate is 5.25% today ..."
            return DummyResp(status=200, text_data=html)
        return DummyResp(status=404, text_data="")

    monkeypatch.setattr("requests.get", fake_get)

    rates = mod.fetch_rates()
    assert rates.get("10y") is not None
    assert abs(rates.get("10y") - 3.5) < 0.01
    assert rates.get("sofr") is not None
    assert abs(rates.get("sofr") - 4.2) < 0.01
    assert rates.get("fed_funds") is not None
    assert abs(rates.get("fed_funds") - 5.25) < 0.1
