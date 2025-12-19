Quick deploy notes (summary)

- Streamlit Cloud:
  - Repository: your-github-username/stock-dashboard
  - Branch: main (or the branch you prefer)
  - Main file path: app.py
  - Start command: streamlit run app.py --server.port $PORT --server.headless true
  - Secrets: add FMP_API_KEY, FINNHUB_API_KEY, NEWS_API_KEY, FRED_API_KEY in the Streamlit Secrets UI

- Keep auto-deploy enabled so pushes to your branch redeploy the app.

- If you need private, invite-only access and Streamlit Cloud plan doesn't support it, use Cloudflare Access or Render + Cloudflare.

See `DEPLOY_STREAMLIT.md` for full instructions.