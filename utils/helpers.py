import math
import numpy as np
import pandas as pd
from typing import Any


def millify(n):
    try:
        n = float(n)
    except Exception:
        return "—"
    if math.isnan(n):
        return "—"
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:,.0f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


def millify_short(n: float | int | None, places: int = 2) -> str:
    """Return a compact human-friendly string for large numbers with suffixes.

    Examples:
      3740000000 -> "3.74 B"
      3740000    -> "3.74 M"
      1234       -> "1.23 K"
      None/NaN   -> "—"
    """
    try:
        if n is None:
            return "—"
        v = float(n)
        if math.isnan(v) or not math.isfinite(v):
            return "—"
        abs_v = abs(v)
        if abs_v >= 1_000_000_000_000:
            return f"{v / 1_000_000_000_000:.{places}f} T"
        if abs_v >= 1_000_000_000:
            return f"{v / 1_000_000_000:.{places}f} B"
        if abs_v >= 1_000_000:
            return f"{v / 1_000_000:.{places}f} M"
        if abs_v >= 1_000:
            return f"{v / 1_000:.{places}f} K"
        return f"{v:.{places}f}"
    except Exception:
        return "—"


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _to_scalar(val: Any) -> Any:
    """If val is a pandas Series with one element, return that element, else return val."""
    if isinstance(val, pd.Series):
        if len(val) == 0:
            return np.nan
        return val.iloc[0]
    return val


def _safe_float(val) -> float:
    """Coerce val to a float safely. Returns np.nan on failure or if value is not finite."""
    v = _to_scalar(val)
    try:
        num = pd.to_numeric(v, errors="coerce")
        arr = np.asarray(num)
        if arr.size == 0:
            return np.nan
        scalar = arr.item() if arr.size == 1 else arr.flat[0]
        if pd.isna(scalar):
            return np.nan
        return float(scalar)
    except Exception:
        return np.nan


def fmt_number(val: float | None, places: int = 2) -> str:
    """Format numeric values with commas and a fixed number of decimal places.

    Returns a placeholder string for missing or non-finite values.
    """
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        return f"{val:,.{places}f}"
    except Exception:
        return str(val or "")


def fmt_money(val: float | None, places: int = 2, prefix: str = "$") -> str:
    """Format a monetary value with currency prefix and commas. Handles None/NaN by returning em dash."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        return f"{prefix}{val:,.{places}f}"
    except Exception:
        return str(val or "")


def format_percent(val: float | None, places: int = 2, show_sign: bool = True) -> str:
    """Format a float as a percentage string (e.g., +1.23%)."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "—"
        # Treat input as already in percent units (e.g., 1.23 -> '1.23%').
        rounded = round(float(val), places)
        fmt = f"{{:{'+' if show_sign else ''}.{places}f}}%"
        return fmt.format(rounded)
    except Exception:
        return str(val or "")


def badge_text(label: str, flag: bool | None) -> str:
    """Return a short caption text with emoji representing positive/neutral/negative."""
    if flag is True:
        return f"✅ {label} favorable"
    if flag is False:
        return f"❌ {label} unfavorable"
    return f"⚪ {label} neutral"
