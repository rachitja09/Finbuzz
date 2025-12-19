Secrets and local development
=============================

This project uses several API keys/secrets at runtime and in CI. Do NOT commit real secrets into the repository.

Required secret names (used by `.github/workflows/ci.yml` and some runtime code):

- NEWS_API_KEY
- ALPHA_VANTAGE_API_KEY
- FINNHUB_API_KEY
- FMP_API_KEY
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- OPEN_AI_KEY

Recommended setup
-----------------

1. Add secrets to GitHub (recommended for CI):

   - GitHub → Repository → Settings → Secrets and variables → Actions → New repository secret
   - Add each secret name above with its value.

2. Local development (safe, do not commit):

   - Copy the template `.secrets.example` to `.secrets` or create a `.env` file and keep it out of source control:

     PowerShell example:

     ```powershell
     Copy-Item .secrets.example .secrets
     # edit .secrets with your keys (do not commit)
     Notepad .secrets
     ```

   - Or set environment variables in your shell before running the app:

     ```powershell
     $env:NEWS_API_KEY = 'your-news-key'
     $env:ALPACA_API_KEY = 'your-alpaca-key'
     # then run streamlit or tests
     python -m streamlit run app.py
     ```

Using GitHub CLI to add secrets
------------------------------

If you prefer automation, use the GitHub CLI from your development machine:

```powershell
gh secret set NEWS_API_KEY --body 'your-news-key' --repo YOUR_ORG/YOUR_REPO
gh secret set ALPACA_API_KEY --body 'your-alpaca-key' --repo YOUR_ORG/YOUR_REPO
```

Local testing with act
----------------------

If you run Actions locally with `act`, pass secrets with `-s` or use a local secrets file:

```powershell
act -s NEWS_API_KEY='your-news-key' -s ALPACA_API_KEY='your-alpaca-key'
```

Security notes
--------------

- Never store plain text keys in the repository. Use GitHub Secrets or an external secrets manager.
- If a secret is accidentally committed, rotate it immediately.
