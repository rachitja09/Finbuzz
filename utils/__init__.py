# utils/__init__.py
# Makes this a package and exports small helpers used by pages.
from typing import Optional
import os


def env_or_secret(secrets_obj: dict, name: str, default: Optional[str] = "") -> Optional[str]:
	"""Return value from environment variable or Streamlit secrets dict.

	Backwards-compatible helper. Prefer importing explicit constants from
	`config.py` (for example: `from config import FMP_API_KEY`) so static
	analyzers see non-Optional `str` values.
	"""
	# Prefer centralized config constants when present
	try:
		import config

		cfg_val = getattr(config, name, None)
		if cfg_val:
			return cfg_val
	except Exception:
		pass

	val = os.environ.get(name)
	if val:
		return val
	try:
		# secrets_obj may be a mapping-like object (st.secrets)
		return secrets_obj.get(name, default)  # type: ignore[attr-defined]
	except Exception:
		return default

