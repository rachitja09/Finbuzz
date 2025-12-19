# test for Fed target-range parsing


def _fake_response(text: str, status_code: int = 200):
    class R:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise Exception(f"HTTP {self.status_code}")

    return R(text, status_code)


def test_fed_target_range_parsing(monkeypatch):
    import utils.rates as rates_mod

    # Create HTML that contains a target range phrase
    sample_html = "<html><body><p>The Federal Open Market Committee has set a target range of 4.00 – 4.25 percent for the federal funds rate.</p></body></html>"

    def fake_get(url, *args, **kwargs):
        # When the calendar or release page is requested, return our sample_html
        if "fomccalendars" in url or "fedfunds" in url:
            return _fake_response(sample_html)
        # fallback for other requests
        return _fake_response("", 404)

    monkeypatch.setattr(rates_mod.requests, "get", fake_get)
    # Ensure runtime FRED key is explicitly disabled so fallback parsing runs
    monkeypatch.setenv("FRED_API_KEY", "")

    out = rates_mod._fetch_rates_impl()
    assert out.get("fed_funds_target_low") == 4.0
    assert out.get("fed_funds_target_high") == 4.25
