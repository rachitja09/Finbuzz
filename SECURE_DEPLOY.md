# Secure public deployment guide

This document explains how to publish the dashboard publicly while keeping provider
API keys and user data secure. Follow these steps to deploy safely and avoid
committing secrets into the repository.

1. Overview
-----------
- The application is designed to keep all provider API keys server-side and read
  them at runtime via `config.get_runtime_key(name)`. For public deployments,
  set these secrets in the hosting platform's secret manager rather than checking
  them into version control.

2. Recommended hosting options
-----------------------------
- Streamlit Cloud: has built-in secrets management (recommended for simplicity).
- Heroku / Railway / Fly.io: use config vars or secrets management.
- Docker / Kubernetes: inject secrets via environment variables or sealed secrets.

3. Required secrets (set these on the deployment platform)
---------------------------------------------------------
- FRED_API_KEY
- FMP_API_KEY
- FINNHUB_API_KEY
- NEWS_API_KEY
- ALPACA_API_KEY
- ALPACA_SECRET_KEY
- OPEN_AI_KEY (if using OpenAI features)

Optional:
- DASHBOARD_ACCESS_TOKEN — a shared token to restrict access to the dashboard UI.
  If set, users must enter this token in the sidebar to unlock the app. This keeps
  provider keys secret on the server (recommended for small group sharing).

4. Steps for Streamlit Cloud (example)
--------------------------------------
- Push the repository to GitHub.
- On Streamlit Cloud, create a new app and point it at your repo/branch.
- In the "Secrets" panel, add the keys listed above (name -> value).
- (Optional) Set `DASHBOARD_ACCESS_TOKEN` to a random strong string and share
  that token with the people you want to give access to.
- Deploy the app. When visitors open the dashboard, they'll be prompted for the
  access token if `DASHBOARD_ACCESS_TOKEN` is set.

5. Security hygiene
-------------------
- Rotate API keys periodically.
- Use per-user or per-app keys where possible (avoid a single shared key across
  many users; if you must, combine with `DASHBOARD_ACCESS_TOKEN`).
- Enable logging and monitoring for the deployment (Sentry, logs) to detect abuse.

6. Removing secrets before publishing this repo (optional but recommended)
---------------------------------------------------------------------------
- Ensure `.streamlit/secrets.toml` and `.env` only contain placeholders. The
  repository contains placeholder files; real values should be set on the
  hosting platform.
- Use `scripts/verify_no_secrets.py` in CI to prevent accidental commits.

7. User privacy
---------------
- The app does not collect personal data by default. If you add user-specific
  functionality, ensure it is opt-in and secure the storage (encrypt sensitive
  fields at rest) and remove or anonymize logs that may contain PII.

8. Shared access workflow suggestion
-----------------------------------
- Use `DASHBOARD_ACCESS_TOKEN` for small-group sharing.
- For larger audiences, put the dashboard behind an authenticated gateway (OAuth
  or SSO) and populate the token store per user.

---

If you want, I can:
- Add an admin page to rotate the `DASHBOARD_ACCESS_TOKEN` (server-side only).
- Add GitHub Actions to automatically redact secrets before publishing a public
  release branch (dangerous if misconfigured; approach carefully).

