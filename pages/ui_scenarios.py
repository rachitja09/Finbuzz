import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import fmt_number


def _coerce_prob(val: float, default: float) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) and v >= 0 else default
    except Exception:
        return default


def _normalize_probs(p):
    total = sum(p)
    return [x / total if total > 0 else 0 for x in p]


def compute_scenarios(close: pd.Series, last: float | None, vol: float | None, mom_3m: float | None, probs: list[float]) -> list[dict]:
    drift = (mom_3m if np.isfinite(mom_3m) else 0.0) / 0.25  # annualize simple drift proxy
    dv = vol if vol is not None and np.isfinite(vol) else 0.25
    bull_ret = drift + dv
    base_ret = drift * 0.5
    bear_ret = drift - dv

    scenarios = [
        {"label": "Bull", "prob": probs[0], "ret": bull_ret, "price": last * (1 + bull_ret) if last is not None and np.isfinite(last) else np.nan},
        {"label": "Base", "prob": probs[1], "ret": base_ret, "price": last * (1 + base_ret) if last is not None and np.isfinite(last) else np.nan},
        {"label": "Bear", "prob": probs[2], "ret": bear_ret, "price": last * (1 + bear_ret) if last is not None and np.isfinite(last) else np.nan},
    ]
    return scenarios


def render_scenarios(df: pd.DataFrame, symbol: str | None = None) -> None:
    """Render a bull/base/bear scenario block with probability weights and impact estimates.

    This is intentionally lightweight: it uses historical vol and momentum to
    size moves, with user-tunable probabilities. The goal is to provide a clear
    framework rather than hard predictions.
    """
    close = None
    try:
        colmap = {c.lower(): c for c in df.columns}
        c = colmap.get("close") or colmap.get("adjclose") or colmap.get("adj_close")
        if c:
            close = pd.Series(df[c], dtype=float).dropna()
    except Exception:
        close = None

    if close is None or close.empty:
        return

    last = float(close.iloc[-1]) if len(close) else np.nan
    vol = float(close.pct_change().dropna().std() * np.sqrt(252)) if len(close) > 20 else np.nan
    mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else np.nan

    with st.expander("Bull / Base / Bear scenarios", expanded=False):
        colp1, colp2, colp3 = st.columns(3)
        p_bull = _coerce_prob(colp1.number_input("Bull %", value=30.0, min_value=0.0, max_value=100.0, step=1.0), 30.0)
        p_base = _coerce_prob(colp2.number_input("Base %", value=50.0, min_value=0.0, max_value=100.0, step=1.0), 50.0)
        p_bear = _coerce_prob(colp3.number_input("Bear %", value=20.0, min_value=0.0, max_value=100.0, step=1.0), 20.0)
        probs = _normalize_probs([p_bull, p_base, p_bear])

        scenarios = compute_scenarios(close, last, vol, mom_3m, probs)

        cols = st.columns(3)
        for i, sc in enumerate(scenarios):
            cols[i].metric(
                f"{sc['label']} ({int(sc['prob']*100)}%)",
                f"{fmt_number(sc['price'], 2) if np.isfinite(sc['price']) else '—'}",
                f"{sc['ret']*100:+.1f}%" if np.isfinite(sc['ret']) else "—",
            )

        # Probability-weighted expected price
        try:
            exp_price = sum([(sc["price"] or 0) * sc["prob"] for sc in scenarios]) if np.isfinite(last) else np.nan
            st.caption(f"Probability-weighted 1y heuristic: {fmt_number(exp_price, 2) if np.isfinite(exp_price) else '—'} (not advice)")
        except Exception:
            pass
