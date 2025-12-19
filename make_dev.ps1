# PowerShell helper to create a venv and install dev deps
param(
    [string]$venvPath = ".venv-2"
)
python -m venv $venvPath
& "$PWD\$venvPath\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-extras.txt
python -m pip install -r requirements-ci.txt
Write-Host "Dev environment ready. Activate with: & \"$PWD\$venvPath\Scripts\Activate.ps1\""
