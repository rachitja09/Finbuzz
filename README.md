# Stock Dashboard

Quick local run and development notes.

Secrets
1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and fill your API keys. Do NOT commit `.streamlit/secrets.toml`.

Run tests (recommended)
1. Create a virtualenv and install test/dev deps from `requirements-ci.txt`.
2. Run `pytest -q` to execute the test suite.

Editor diagnostics
If your editor flags missing imports for optional packages (plotly, yfinance, vaderSentiment), the project includes a `pyrightconfig.json` that disables missing-import diagnostics for the repo.

Notes
- Heavy dependencies (yfinance, plotly, vaderSentiment, pandas_ta) are optional and guarded so the project can be imported and tested in CI without installing them all.
- To enable full UI, install the optional packages listed in `requirements.txt`.
