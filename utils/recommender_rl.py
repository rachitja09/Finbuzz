"""Lightweight adaptive weight updater for recommendation signals.

This module implements a tiny, safe online update rule for the strategy weights
used by the recommendation engine. It's intentionally simple: no external ML
dependencies, persist weights in prefs, and apply small multiplicative updates
based on observed reward (positive P/L => reinforce weights of features used).

The intent is to provide an explainable, low-risk adaptation mechanism (not a
full RL training loop). Use `record_outcome` to report realized reward for a
previous suggestion; the updater adjusts weights slightly and persists them.
"""
from __future__ import annotations
from typing import Dict
import math
from .prefs import get_strategy_weights, set_strategy_weights


def get_weights() -> Dict[str, float]:
    """Return current weights (tech_w, analyst_w, consensus_w, news_w).

Weights are positive floats; if prefs absent, defaults are used.
"""
    return get_strategy_weights()


def _normalize(ws: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in ws.values())
    if s <= 0:
        # fallback defaults
        return {"tech_w": 1.0, "analyst_w": 0.7, "consensus_w": 0.5, "news_w": 0.3}
    return {k: float(max(0.0, v) / s * len(ws)) for k, v in ws.items()}


def record_outcome(feature_vector: Dict[str, float], reward: float, lr: float = 0.05) -> Dict[str, float]:
    """Update weights given a feature vector and observed scalar reward.

    - feature_vector: mapping of feature name to contribution magnitude (non-negative)
    - reward: realized reward (e.g., signed P/L). Positive reward reinforces features.
    - lr: learning rate controlling update magnitude.

    Returns the new normalized weights.
    """
    # Load current weights
    ws = get_strategy_weights()
    # feature keys we care about
    keys = ["tech_w", "analyst_w", "consensus_w", "news_w"]
    # compute multiplicative update: w_i *= exp(lr * reward * feature_i)
    for k in keys:
        f = float(feature_vector.get(k, 0.0) or 0.0)
        try:
            ws[k] = float(ws.get(k, 1.0) * math.exp(lr * float(reward) * f))
        except Exception:
            ws[k] = float(ws.get(k, 1.0))

    # normalize to keep magnitudes stable
    n = _normalize(ws)
    set_strategy_weights(n)
    return n
