import importlib
from types import SimpleNamespace

class DummyResp:
    def __init__(self, status=200, json_data=None, text_data=''):
        self.status_code = status
        self._json = json_data
        self._text = text_data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception('HTTP')
    def json(self):
        return self._json
    @property
    def text(self):
        return self._text


def test_market_overview_fetch_rates_fallback(monkeypatch):
    mod = importlib.import_module('ui.market_overview')
    importlib.reload(mod)

    # Ensure the cached wrapper (if present) is cleared
    try:
        mod.fetch_rates.clear()
    except Exception:
        pass

    # Force no runtime FRED key via st.secrets and explicit empty env var
    monkeypatch.setattr(mod, 'st', SimpleNamespace(secrets={}))
    monkeypatch.setenv('FRED_API_KEY', '')

    def fake_get(url, params=None, timeout=None):
        if 'daily-treasury-rates.csv' in url:
            txt = 'Date,1 Mo,10 Yr\n2025-09-23,0.10,3.50\n'
            return DummyResp(status=200, text_data=txt)
        if 'sofr.csv' in url:
            txt = 'Date,SOFR\n2025-09-23,4.20\n'
            return DummyResp(status=200, text_data=txt)
        if 'fedfunds.htm' in url:
            html = '... Effective Federal Funds Rate is 5.25% today ...'
            return DummyResp(status=200, text_data=html)
        return DummyResp(status=404, text_data='')

    monkeypatch.setattr('requests.get', fake_get)

    rates = mod.fetch_rates()
    assert rates.get('10y') is not None
    assert abs(rates.get('10y') - 3.5) < 1e-6
    assert rates.get('sofr') is not None
    assert rates.get('fed_funds') is not None
