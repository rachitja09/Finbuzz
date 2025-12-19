# PowerShell wrapper for running premarket_alert.py
# WARNING: Do NOT store real secrets in this file in production. Replace placeholders
# with secure retrieval (Key Vault, encrypted store) or set env vars in the scheduler.

$projectRoot = 'C:\Users\rachi\OneDrive - UW\Desktop\stock dashboard'
$venvScripts = Join-Path $projectRoot '.venv-1\Scripts'
# Prepend virtualenv Scripts to PATH for the process
$env:PATH = "$venvScripts;" + $env:PATH

# Optionally, load secrets from an encrypted local file or Windows credential manager.
# Note: SendGrid/Slack delivery adapters are disabled in source. If you implement a
# secure delivery adapter, set the required env vars here or use a secret manager.

# Run the premarket alert script. Remove --send until you've verified.
python "$projectRoot\scripts\premarket_alert.py" --send
