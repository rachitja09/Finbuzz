# Safely remove the backups directory from repo root
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$backups = Join-Path $root "..\backups"
if (Test-Path $backups) {
    Write-Host "Removing backups at $backups"
    Remove-Item -LiteralPath $backups -Recurse -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $backups)) { Write-Host "backups removed" } else { Write-Host "Failed to remove backups" }
} else {
    Write-Host "No backups directory found"
}
