"""Minimal config stub for tests and import-time checks.

This file provides default values for API keys and other configuration
that other modules import at import-time. Values are intentionally
empty so code can import without performing network calls.
"""

import os
from typing import Optional
import streamlit as st
from pathlib import Path

# External API keys (left empty for tests/CI)
FINNHUB_API_KEY = ""
FMP_API_KEY = ""
NEWS_API_KEY = ""

# Application flags
DEBUG = False

__all__ = [
    "FINNHUB_API_KEY",
    "FMP_API_KEY",
    "NEWS_API_KEY",
    "DEBUG",
]

# Cache TTLs (seconds) used across the app for providers and expensive calls.
# Tune these values in-memory or via environment for different deployments.
CACHE_TTLS = {
    # short-lived per-symbol items (events/quotes)
    "get_major_events": 300,  # 5 minutes
    # listings that change occasionally
    "list_etfs": 600,        # 10 minutes
    "list_mutual_funds": 600,
    # IPO calendar is relatively stable within a day
    "upcoming_ipos": 3600,   # 1 hour
}


def get_cache_ttl(key: str, default: int) -> int:
    """Return configured TTL for caches; falls back to default if not set."""
    try:
        return int(CACHE_TTLS.get(key, default))
    except Exception:
        return default


def _get_secret(key: str) -> Optional[str]:
    """
    Try to fetch from Streamlit secrets (if available),
    then fall back to environment variables.
    """
    try:
        # Accessing st.secrets can raise when Streamlit isn't running (tests/CI)
        return st.secrets[key]
    except Exception:
        # Catch all exceptions including StreamlitSecretNotFoundError
        return os.getenv(key)


def get_runtime_key(name: str) -> Optional[str]:
    """
    Resolve a runtime API key with the following precedence and semantics:

    - If an environment variable with the given name exists:
        - If it's non-empty, return its value.
        - If it's present but empty (""), treat this as an explicit disable and
          return None (do NOT fall back to Streamlit secrets).
    - Otherwise, if Streamlit secrets are available and contain the key, return that.
    - Otherwise return the module-level constant (if any) or None.

    This enforces the convention that setting an env var to an empty string
    intentionally disables runtime secrets (useful for deterministic tests).
    """
    # If the env var exists (even if empty) we take that as authoritative
    if name in os.environ:
        v = os.environ.get(name)
        # present but empty disables runtime key
        if v is None or v == "":
            return None
        return v

    # If no env var, prefer Streamlit secrets when available
    try:
        # avoid importing streamlit at module import time in tests unless needed
        import streamlit as _st

        try:
            s = _st.secrets.get(name)
            if s:
                return s
        except Exception:
            pass
    except Exception:
        pass

    # Fall back to module-level constant if set
    val = globals().get(name)
    if isinstance(val, str) and val:
        return val
    return None


def require(key: str) -> str:
    """
    Return a non-None string for critical secrets, or raise.
    """
    val = _get_secret(key)
    if not val:
        raise RuntimeError(f"Missing required secret or env variable: {key}")
    return val


# Expose required keys as module-level constants so imports are simple and
# Pylance/pyright see non-Optional str types.
# To keep test/CI imports from failing when secrets are not provided, we try
# to `require(...)` but fall back to an empty string and set a flag. This
# preserves the developer "fail fast" behavior when desired but avoids
# breaking pytest/CI where secrets are intentionally absent.
_MISSING_SECRETS = False
try:
    ALPHA_VANTAGE_API_KEY = require("ALPHA_VANTAGE_API_KEY")
    NEWS_API_KEY = require("NEWS_API_KEY")
    FINNHUB_API_KEY = require("FINNHUB_API_KEY")
    FMP_API_KEY = require("FMP_API_KEY")
    ALPACA_API_KEY = require("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = require("ALPACA_SECRET_KEY")
    OPEN_AI_KEY = require("OPEN_AI_KEY")
    FRED_API_KEY = require("FRED_API_KEY")
except RuntimeError:
    # Tests/CI often run without secrets; keep imports working but mark that
    # secrets are missing so callers can handle it explicitly.
    _MISSING_SECRETS = True
    ALPHA_VANTAGE_API_KEY = ""
    NEWS_API_KEY = ""
    FINNHUB_API_KEY = ""
    FMP_API_KEY = ""
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    OPEN_AI_KEY = ""
    FRED_API_KEY = ""


def _load_dotenv_file(path: str = ".env") -> None:
    """Attempt to load environment variables from a .env file.

    Tries to use python-dotenv if installed, otherwise performs a simple
    parse. This helps the app auto-load keys during local development so you
    don't have to re-type keys in the UI each run.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        # prefer python-dotenv if available
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(p))
        return
    except Exception:
        pass

    try:
        # fallback: simple parser
        for ln in p.read_text(encoding="utf8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and os.environ.get(k) is None:
                os.environ[k] = v
    except Exception:
        return


def write_local_secrets(mapping: dict[str, str], dest: str | Path = ".streamlit/secrets.toml") -> None:
    """Persist secrets to a local `.streamlit/secrets.toml` file.

    WARNING: writing secrets to disk may be a security risk. This helper
    is intended for local development convenience only. The function will
    create `.streamlit/` if needed and write basic TOML entries. Existing
    file will be overwritten.
    """
    dest_p = Path(dest)
    dest_dir = dest_p.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for k, v in mapping.items():
        # ensure string value and basic escaping
        sv = str(v).replace('"', '\\"')
        lines.append(f'{k} = "{sv}"')
    dest_p.write_text("\n".join(lines) + "\n", encoding="utf8")


# If keys were missing earlier, attempt to auto-load from a .env file and
# then re-hydrate module-level constants from environment variables so the
# app can run without manual typing each time.
if _MISSING_SECRETS:
    _load_dotenv_file()
    # Re-read from environment
    ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEY or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY = NEWS_API_KEY or os.getenv("NEWS_API_KEY", "")
    FINNHUB_API_KEY = FINNHUB_API_KEY or os.getenv("FINNHUB_API_KEY", "")
    FMP_API_KEY = FMP_API_KEY or os.getenv("FMP_API_KEY", "")
    ALPACA_API_KEY = ALPACA_API_KEY or os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = ALPACA_SECRET_KEY or os.getenv("ALPACA_SECRET_KEY", "")
    OPEN_AI_KEY = OPEN_AI_KEY or os.getenv("OPEN_AI_KEY", "")
    FRED_API_KEY = FRED_API_KEY or os.getenv("FRED_API_KEY", "")


__all__ = [
    "ALPHA_VANTAGE_API_KEY",
    "NEWS_API_KEY",
    "FINNHUB_API_KEY",
    "FMP_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "OPEN_AI_KEY",
    "FRED_API_KEY",
    "require",
]
