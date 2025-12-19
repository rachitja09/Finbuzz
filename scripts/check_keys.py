"""Report presence of API keys defined in config.py without making network calls.

This prints SET/MISSING for the common keys used by the dashboard.
"""
import importlib
import config

importlib.reload(config)

keys = [
    "ALPHA_VANTAGE_API_KEY",
    "NEWS_API_KEY",
    "FINNHUB_API_KEY",
    "FMP_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "OPEN_AI_KEY",
    "FRED_API_KEY",
]

for k in keys:
    v = getattr(config, k, None)
    print(f"{k}: SET" if v else f"{k}: MISSING")
