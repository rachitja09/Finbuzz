Minimal deployment checklist (what you need to do)

Local prep
- [ ] Run the local test (see DEPLOY_STREAMLIT.md) and validate pages: Home, Markets, News, Backtest.
- [ ] Confirm `requirements.txt` is up-to-date and includes optional dependencies you want (vaderSentiment, yfinance, plotly).

GitHub & Streamlit
- [ ] Create a private GitHub repo and push the code to `main`.
- [ ] Create a Streamlit Cloud account and connect your GitHub account.
- [ ] Create a new app on Streamlit Cloud pointing at your repo/branch and `app.py`.
- [ ] Add secrets in Streamlit Cloud (FMP_API_KEY, FINNHUB_API_KEY, NEWS_API_KEY, FRED_API_KEY).
- [ ] Deploy and smoke-test the app.

Access & Security
- [ ] Decide access model: Streamlit private app (if supported) or Cloudflare Access.
- [ ] Add invited users with the chosen method.
- [ ] Set provider billing alerts and usage thresholds.

Operational
- [ ] Monitor logs for the first 48–72 hours.
- [ ] Revisit cache TTLs after real usage to balance freshness vs cost.
- [ ] Rotate keys on a schedule and have a revocation plan.

Legal / privacy
- [ ] Update the app with a short notice linking to `PRIVACY_TERMS.md` and obtain user consent.

If you want, I can open a PR that adds a small banner/snippet with the short privacy notice to `app.py` or `pages/01_Home.py` so invited users see the consent prompt on first visit.