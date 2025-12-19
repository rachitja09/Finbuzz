"""CI helper: fail if secrets or .env are committed to the repository.

Usage: python scripts/verify_no_secrets.py
This checks `git ls-files` for common secret files and exits non-zero if found.
"""
import subprocess

FILES_TO_BLOCK = [".streamlit/secrets.toml", ".env"]


def main() -> int:
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
    except Exception as e:
        print("Not a git repo or git not available; skipping check.", e)
        return 0

    tracked = set(line.strip() for line in out.splitlines() if line.strip())
    found = [f for f in FILES_TO_BLOCK if f in tracked]
    if found:
        print("ERROR: The following secret files are tracked in git (remove them and add to .gitignore):")
        for f in found:
            print(" - ", f)
        return 2
    print("No tracked secret files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
