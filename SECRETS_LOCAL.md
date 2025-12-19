This project supports convenient local secrets for development. DO NOT COMMIT or share these files.

Where to put secrets locally

- Streamlit secrets (preferred for Streamlit runs):
  - `.streamlit/secrets.toml` (already created when you chose to store keys)
  - Keys are accessible in the app via `st.secrets["KEY_NAME"]`.

- Environment variables / .env (fallback):
  - `.env` is loaded by `config._load_dotenv_file()` at import-time when running locally.
  - Keys set here are read as environment variables and used by `config.py`.

Security notes

- These files are ignored by `.gitignore` by default. Verify `.gitignore` contains `.streamlit/secrets.toml` and `*.env`.
- For CI (GitHub Actions), add secrets in the repository's Settings -> Secrets and reference them in the workflow as environment variables.
- For production, use a secrets manager (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager) instead of committing secrets to disk.

Keys currently provided locally (for convenience): FRED_API_KEY, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, FINNHUB_API_KEY, FMP_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, OPEN_AI_KEY.
