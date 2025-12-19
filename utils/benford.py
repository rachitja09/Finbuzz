"""Benford's Law helpers for data-integrity checks.

This module provides small, dependency-free utilities to compute the leading-digit
distribution of numeric series, compare it to Benford's expected distribution,
and return a compact score/diagnostic useful for flagging anomalies.

Design goals:
- No heavy dependencies (avoid scipy). Return chi2-like statistic and an RMS error
  'benford_score' that is easy to interpret and test.
- Robust: ignore zeros, negatives are allowed (leading digit from absolute value),
  ignore NaN/non-numeric values.

Typical usage:
>>> from utils.benford import benford_report
>>> report = benford_report(pandas_series_of_numeric_values)
>>> report['benford_score']  # lower is closer to Benford
"""
from __future__ import annotations

from typing import Iterable, List, Dict, Optional, Tuple
import math

BENFORD_PROBS: List[float] = [math.log10(1 + 1 / d) for d in range(1, 10)]


def leading_digit_of_number(x: float) -> Optional[int]:
    """Return the first non-zero decimal digit of |x|, or None for invalid/zero.

    Examples:
        123.4 -> 1
        0.00456 -> 4
        -250 -> 2
        0 -> None
    """
    try:
        if x is None:
            return None
        # Convert to float; if this fails, treat as non-numeric
        v = float(x)
    except Exception:
        return None
    v = abs(v)
    if v == 0 or math.isnan(v) or math.isinf(v):
        return None
    # Normalize v to be >=1 by scaling
    # Use scientific notation: get the mantissa part
    # Convert to string via repr to avoid locale issues
    try:
        s = f"{v:.16e}"  # scientific: d.ddddde+/-NN
        mant, _exp = s.split("e")
        # mant like '1.2345678901234567'
        for ch in mant:
            if ch.isdigit() and ch != '0':
                return int(ch)
        return None
    except Exception:
        # Fallback: iterative scaling
        try:
            while v < 1:
                v *= 10
            while v >= 10:
                v /= 10
            first = int(str(v)[0])
            return first if first != 0 else None
        except Exception:
            return None


def digit_counts(values: Iterable[float]) -> List[int]:
    """Return counts of leading digits 1..9 for the iterable of numeric values."""
    counts = [0] * 9
    total = 0
    for x in values:
        d = leading_digit_of_number(x)
        if d is None:
            continue
        counts[d - 1] += 1
        total += 1
    return counts


def digit_distribution_percent(values: Iterable[float]) -> List[float]:
    """Return the percentage distribution (0..100) of leading digits 1..9."""
    counts = digit_counts(values)
    total = sum(counts)
    if total == 0:
        return [0.0] * 9
    return [100.0 * c / total for c in counts]


def expected_benford_percent() -> List[float]:
    """Return Benford expected distribution as percentages 1..9."""
    return [100.0 * p for p in BENFORD_PROBS]


def chi2_statistic(values: Iterable[float]) -> Tuple[float, List[float], List[int]]:
    """Compute a chi-squared style statistic comparing observed counts to Benford expected counts.

    Returns: (chi2_stat, expected_counts_list, observed_counts_list)
    Note: This function returns the raw chi2 statistic but not a p-value (no scipy).
    """
    obs = digit_counts(values)
    total = sum(obs)
    if total == 0:
        return (0.0, [0.0] * 9, obs)
    expected = [total * p for p in BENFORD_PROBS]
    chi2 = 0.0
    for o, e in zip(obs, expected):
        if e <= 0:
            continue
        chi2 += (o - e) ** 2 / e
    return (chi2, expected, obs)


def benford_score(values: Iterable[float]) -> float:
    """Return a compact positive score representing deviation from Benford.

    Implementation: RMS between observed percent distribution and expected Benford percent.
    Lower is better; 0.0 is a perfect match.
    """
    obsp = digit_distribution_percent(values)
    exp = expected_benford_percent()
    if sum(obsp) == 0:
        return float('nan')
    # RMS
    s = 0.0
    for o, e in zip(obsp, exp):
        diff = o - e
        s += diff * diff
    rms = math.sqrt(s / len(obsp))
    return rms


def benford_report(values: Iterable[float]) -> Dict[str, object]:
    """Return a diagnostic report for the iterable of numeric values.

    Report includes:
    - "score": RMS deviation (lower is closer)
    - "obs_percent": list[float] percentages per digit
    - "exp_percent": list[float] Benford expected percentages
    - "chi2": chi-squared statistic
    - "counts": observed counts per digit
    - "total": total counted values
    """
    obs_counts = digit_counts(values)
    total = sum(obs_counts)
    obs_pct = [100.0 * c / total if total > 0 else 0.0 for c in obs_counts]
    exp_pct = expected_benford_percent()
    chi2, exp_counts, obs_counts = chi2_statistic(values)
    score = benford_score(values)
    return {
        "score": score,
        "obs_percent": obs_pct,
        "exp_percent": exp_pct,
        "chi2": chi2,
        "expected_counts": exp_counts,
        "counts": obs_counts,
        "total": total,
    }
