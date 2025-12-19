Reminder: run a workspace-wide syntax check before running the app (PowerShell):

python -m compileall -q .; if ($LASTEXITCODE -eq 0) { echo 'Syntax check passed' } else { echo 'Syntax check failed' }

(You asked to be reminded to run this later.)
