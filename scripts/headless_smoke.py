"""Headless smoke import script for CI / local checks.

This script inserts a fake 'pytest' module into sys.modules so the app/pages
skip network-heavy UI rendering paths (the codebase checks for 'pytest' in
sys.modules to avoid doing network work at import time). It then imports
the key modules and reports success or any import errors.

Run locally with:
    python scripts/headless_smoke.py
"""
from __future__ import annotations
import sys
import importlib
import types
from pathlib import Path

# Ensure project root is on sys.path so `import app` and `pages.*` work when
# running this script directly from the repository.
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

MODULES_TO_CHECK = [
    'app',
    'pages.01_Home',
    'pages.03_Portfolio',
    'pages.07_Diagnostics',
    'utils.macro',
    'utils.rates',
]


def main():
    # Insert a dummy pytest module so files skip live network UI paths
    sys.modules['pytest'] = types.ModuleType('pytest')
    successes = []
    failures = []
    for m in MODULES_TO_CHECK:
        try:
            importlib.invalidate_caches()
            mod = importlib.import_module(m)
            successes.append(m)
            print(f"OK: imported {m}")
        except Exception as e:
            failures.append((m, e))
            print(f"ERR: failed to import {m}: {type(e).__name__}: {e}")

    print("\nSummary:")
    print(f"  Successes: {len(successes)}")
    for s in successes:
        print(f"    - {s}")
    print(f"  Failures: {len(failures)}")
    for f, e in failures:
        print(f"    - {f}: {type(e).__name__}: {e}")

    if failures:
        raise SystemExit(2)
    print("Headless smoke import OK")


if __name__ == '__main__':
    main()
