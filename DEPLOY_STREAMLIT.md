Streamlit Cloud deployment instructions (tailored for this repo)

This guide shows the exact steps to deploy this Streamlit app to Streamlit Cloud (share.streamlit.io). It assumes you want to use your own API keys (FMP, Finnhub, NewsAPI, FRED) stored as secrets in Streamlit Cloud and invite a small set of trusted users.

1) Quick local test (PowerShell)
- Install dependencies and run locally with your keys set as environment variables:

```powershell
$env:FMP_API_KEY = 'your_fmp_key_here'
$env:FINNHUB_API_KEY = 'your_finnhub_key_here'
$env:NEWS_API_KEY = 'your_news_api_key_here'
$env:FRED_API_KEY = 'your_fred_key_here'
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

2) Create a private GitHub repo and push
- Create a private repo at GitHub. Push the project (main branch recommended):

```powershell
git init
git add .
git commit -m "deploy-ready"
git branch -M main
git remote add origin git@github.com:yourname/yourrepo.git
git push -u origin main
```

3) Connect Streamlit Cloud
- Go to https://share.streamlit.io and sign in with GitHub (use the account that can access the private repo).
- Click "New app" -> Select repository -> Choose branch `main` -> Set the main file path to `app.py`.
- Start command (Streamlit Cloud usually detects automatically):

```
streamlit run app.py --server.port $PORT --server.headless true
```

4) Add secrets (Streamlit Cloud)
- On the app page: Settings → Secrets (or `Advanced` → `Secrets`). Add these keys (replace with your actual values):

```
FMP_API_KEY = "..."
FINNHUB_API_KEY = "..."
NEWS_API_KEY = "..."
FRED_API_KEY = "..."
```

- Save secrets. They will appear as environment variables to the running app.

5) Deploy & verify
- Trigger a deploy (or push to `main` for auto-deploy). Visit the app URL and run through:
  - Home page loads and charts render
  - News and analyst widgets show data
  - Markets/ETFs page populates
  - Backtest runs (Quick SMA Backtest)

6) Make the app invite-only
- Streamlit Cloud private app controls vary by plan. If your account supports private apps/collaborators:
  - Use the app's Access/Share settings to invite specific GitHub users.
- If not, protect the app with Cloudflare Access (recommended) or use Render/Fly.io instead for easier invite-only flows.

7) Operational safeguards (do these now)
- Ensure caching TTLs in `config.py` are set conservatively to avoid heavy API usage.
- Set up provider billing alerts (FMP, Finnhub, NewsAPI) so you know if usage spikes.
- Add logging/monitoring (Sentry or Streamlit Cloud logs) to capture errors and usage.

8) Post-deploy testing
- Invite 1 test user, have them interact, and watch logs for errors and API call volume.

Notes and Fall-back
- If Streamlit Cloud plan doesn't support private apps for your repo, consider Render or Fly.io and put Cloudflare Access in front for invite-only sign-in.
- Do NOT commit API keys to source. Always use the Streamlit Secrets UI or your host's secret store.

If you want, I can prepare a short README snippet and a Terms/Privacy notice to show in-app. I already added a privacy/terms file in the repo; customize names/dates before showing to users.