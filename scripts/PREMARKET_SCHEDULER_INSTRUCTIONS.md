Premarket alert scheduling — Windows Task Scheduler and cloud examples

Overview

This document shows safe, minimal examples to run `scripts/premarket_alert.py` each trading day 15 minutes before US market open (09:15 ET). The script defaults to dry-run — it prints the digest. Add `--send` to actually deliver notifications. Never store secrets in the repo; use Task Scheduler secure mechanisms or secrets in your cloud CI.

1) Local PowerShell wrapper (recommended)

Create a small wrapper PowerShell script that (a) activates the virtualenv, (b) sets required environment variables in-process, and (c) runs the script.

Example file: `run_premarket_alert.ps1`

```powershell
# Absolute paths recommended
$projectRoot = 'C:\Users\rachi\OneDrive - UW\Desktop\stock dashboard'
$venvScripts = Join-Path $projectRoot '.venv-1\Scripts'
# Prepend venv to PATH for this process
$env:PATH = "$venvScripts;" + $env:PATH
# Set runtime secrets in-process (do NOT hardcode in repo in production)
# Configure runtime secrets in your scheduler or secret manager. This example omits
# SendGrid/Slack since those integrations are intentionally disabled in the repo.
# Run the script (dry-run by default). Remove --send while testing.
python "$projectRoot\scripts\premarket_alert.py" --send
```

Notes:
- Keep the wrapper on a secure machine and protect its contents. For production, prefer sourcing secrets from a secure store instead of embedding in the file.
- If your environment requires activation (some venvs require it), you can call `& "$venvScripts\Activate.ps1"` first.

2) Create a Windows Task Scheduler task

- Open Task Scheduler.
- Create Task -> General: give it a name (e.g., `PremarketAlert`), set "Run whether user is logged on or not" if desired.
- Triggers -> New:
  - Begin the task: On a schedule
  - Settings: Weekly (Mon-Fri)
  - Start: pick a date/time and set Recur every 1 week, select Mon-Fri.
  - Advanced settings: Repeat? No. Delay task for: 0 minutes.
- Actions -> New:
  - Program/script: powershell
  - Add arguments: -NoProfile -ExecutionPolicy Bypass -File "C:\path\to\run_premarket_alert.ps1"
  - Start in (optional): C:\path\to\project
- Conditions/Settings: tweak based on your host (wake to run, stop if runs longer than, etc.)

3) GitHub Actions (example)

If you prefer cloud scheduling and can store secrets in the Actions secrets store, create a workflow like this (replace UTC time appropriately; GitHub Actions cron uses UTC):

`.github/workflows/premarket_alert.yml`

```yaml
name: Premarket Alert
on:
  schedule:
    # NOTE: GitHub Actions 'cron' is UTC. Convert 09:15 ET to UTC for your target dates.
    # During Eastern DST (approx Mar-Nov) 09:15 ET == 13:15 UTC. Outside DST 09:15 ET == 14:15 UTC.
    - cron: '15 13 * * 1-5' # example runs at 13:15 UTC Mon-Fri
jobs:
  notify:
    runs-on: ubuntu-latest
    # Configure any runtime secrets using the Actions secrets store if you re-enable delivery adapters.
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run premarket alert
        run: |
          python scripts/premarket_alert.py --send
```

4) Notes on timezone and scheduling

- ET to UTC conversion varies with DST. If you need robust handling, prefer a scheduler that supports IANA timezones (some cloud schedulers do) or write a small wrapper to check local ET time and run only when the local time is 09:15 ET.
- Consider running additional checks (market holiday calendar) — the script currently runs every scheduled day. Add a holiday filter if you don't want alerts on market holidays.

5) Safety and secrets

- Never commit secrets (API keys, webhook URLs) into the repository.
- Use OS-level secret stores or the cloud provider's secret manager (GitHub Secrets, Azure Key Vault, AWS Secrets Manager).
- The script uses `config.get_runtime_key()` so it will find env vars or Streamlit secrets automatically.

6) Testing locally

- Run dry-run:

```powershell
python scripts/premarket_alert.py
```

- To actually send (after verifying keys are correct):

```powershell
python scripts/premarket_alert.py --send
```

7) Next steps (optional)

- Wire in richer pre/post-market quotes from a provider that offers extended-hours pricing and premarket trades.
- Add HTML email formatting and link to a dashboard snapshot.
- Add retry/backoff to delivery functions or use a delivery queue for resilience.

---

If you'd like, I can also:
- Create the `run_premarket_alert.ps1` wrapper in `scripts/` (I won't put secrets in it — I'll leave placeholders), and
- Add a small GitHub Actions workflow file with placeholders for secrets.

Tell me if you want those added now.
