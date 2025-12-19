"""Verify whether a provided FMP key matches the one loaded by the app and
perform a lightweight, non-sensitive probe against the profile endpoint.

Usage (PowerShell):
  $env:PROBE_KEY='...'; python scripts/verify_fmp_key.py

The script will never print the key itself — only True/False and HTTP status/snippets.
"""
import os
import requests


def _get_fmp_key() -> str | None:
    try:
        from config import get_runtime_key

        return get_runtime_key("FMP_API_KEY")
    except Exception:
        # fallback to env var only
        v = os.environ.get("FMP_API_KEY")
        return v if v else None


probe = os.getenv("PROBE_KEY")
config_key = _get_fmp_key()
print("Config FMP present:", bool(config_key))
print("Provided PROBE_KEY present:", bool(probe))
if probe is not None:
    print("Matches config:", probe == config_key)
    try:
        r = requests.get(
            "https://financialmodelingprep.com/api/v3/profile/AAPL",
            params={"apikey": probe},
            timeout=8,
        )
        print("Probe status:", r.status_code)
        # show a short snippet of the response body without revealing keys
        print("Probe snippet:", (r.text or "")[:240])
    except Exception as e:
        print("Probe failed:", e)
else:
    print("Set PROBE_KEY environment variable to run the probe.")
