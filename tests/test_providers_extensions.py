import datetime as dt
import types

import pytest

import data_providers as dp


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_list_etfs_fmp(monkeypatch):
    # Mock _get to return a simple ETF list
    sample = [{"symbol": "SPY", "name": "SPDR S&P 500", "exchange": "NYSE"}]

    monkeypatch.setattr(dp, "_get", lambda url, params=None, timeout=20: sample)
    res = dp.list_etfs(api_key="fakekey", finnhub_token="")
    assert isinstance(res, list)
    assert any(r.get("symbol") == "SPY" for r in res)


def test_list_etfs_yfinance_fallback(monkeypatch):
    # Force FMP to fail and ensure yfinance fallback works when yf is present
    monkeypatch.setattr(dp, "_get", lambda *a, **k: (_ for _ in ()).throw(Exception("nope")))
    # monkeypatch yfinance minimal interface
    class FakeTicker:
        def __init__(self, s):
            self.s = s

        @property
        def info(self):
            return {"longName": "Fake ETF", "exchange": "ARCA"}

    monkeypatch.setattr(dp, "yf", types.SimpleNamespace(Ticker=FakeTicker, download=None))
    res = dp.list_etfs(api_key="", finnhub_token="")
    assert isinstance(res, list)
    assert any(isinstance(it.get("symbol"), str) for it in res)


def test_upcoming_ipos_finnhub(monkeypatch):
    today = dt.date.today().isoformat()
    sample = [{"symbol": "NEWC", "name": "NewCo", "date": today, "exchange": "NASDAQ", "expectedPrice": 10}]
    monkeypatch.setattr(dp, "_get", lambda url, params=None, timeout=20: sample)
    res = dp.upcoming_ipos(finnhub_token="fake", fmp_key="", days_ahead=30)
    assert isinstance(res, list)
    assert res and res[0].get("symbol") == "NEWC"


def test_get_major_events_uses_yfinance(monkeypatch):
    # Mock utils.earnings functions to return known values
    monkeypatch.setitem(__import__("sys").modules, 'utils.earnings', types.SimpleNamespace(get_next_earnings_info=lambda s: {"date":"2025-01-01"}, get_last_earnings_summary=lambda s: {"eps":1.0}))
    # Mock company_news
    monkeypatch.setattr(dp, "company_news", lambda symbol, token, days=30: [{"title":"Test"}])
    class FakeTicker2:
        def __init__(self, s):
            pass

        @property
        def dividends(self):
            return []

        @property
        def splits(self):
            return []

    monkeypatch.setattr(dp, "yf", types.SimpleNamespace(Ticker=FakeTicker2))
    res = dp.get_major_events("AAPL", finnhub_token="", fmp_key="")
    assert isinstance(res, dict)
    assert "earnings_next" in res and res["earnings_next"] is not None