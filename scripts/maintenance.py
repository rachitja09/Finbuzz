"""Maintenance helper: run tests, lint, and optional dependency updates.

This script is intended for local/dev usage. It does not modify secrets. It runs
pytest, ruff (lint), and optionally runs `pip list --outdated` to indicate
available updates. Use with the project's venv activated.
"""
import subprocess
import sys

cmds = [
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-m", "ruff", "check", "."],
]

for c in cmds:
    print("Running:", " ".join(c))
    rc = subprocess.call(c)
    if rc != 0:
        print("Command failed:", c)
        sys.exit(rc)

print("All checks passed locally.")

# optionally list outdated packages
if "--outdated" in sys.argv:
    subprocess.call([sys.executable, "-m", "pip", "list", "--outdated"])