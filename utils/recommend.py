from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from utils.helpers import _safe_float
from utils.indicator_analysis import analyze_indicators
from utils.benford import benford_report
from utils.prefs import get_benford_prefs
from utils import signals
from utils import macro


def make_recommendation_simple(latest_row: pd.Series, profile: dict | None = None, boll_pct: float | None = None) -> Tuple[str, str]:
    """Backward-compatible simple recommendation (keeps prior behavior)."""
    score = 0
    reasons: list[str] = []
    close = float(_safe_float(latest_row.get("close")))
    ema20 = float(_safe_float(latest_row.get("ema20")))
    ema50 = float(_safe_float(latest_row.get("ema50")))
    rsi_val = float(_safe_float(latest_row.get("rsi")))

    # trend
    if np.isfinite(close) and np.isfinite(ema20) and np.isfinite(ema50):
        if close > ema20 > ema50:
            score += 1
            reasons.append("Uptrend: price > EMA20 > EMA50")
        elif close < ema20 < ema50:
            score -= 1
            reasons.append("Downtrend: price < EMA20 < EMA50")

    # RSI
    if np.isfinite(rsi_val):
        if rsi_val < 30:
            score += 1
            reasons.append(f"RSI oversold ({rsi_val:.0f})")
        elif rsi_val > 70:
            score -= 1
            reasons.append(f"RSI overbought ({rsi_val:.0f})")

    # Bollinger
    if boll_pct is not None and np.isfinite(boll_pct):
        if boll_pct < 0.1:
            score += 1
            reasons.append(f"Price below lower Bollinger band (%B={boll_pct:.2f})")
        elif boll_pct > 0.9:
            score -= 1
            reasons.append(f"Price above upper Bollinger band (%B={boll_pct:.2f})")

    # fundamentals (optional)
    if profile:
        try:
            pe = float(profile.get("pe", profile.get("peRatioTTM", float("nan"))))
            sector_pe = float(profile.get("sector_pe", float("nan"))) if profile.get("sector_pe") is not None else float("nan")
            if np.isfinite(pe) and np.isfinite(sector_pe):
                if pe > sector_pe * 1.5:
                    score -= 1
                    reasons.append("P/E elevated vs sector")
        except Exception:
            pass

    verdict = "Hold"
    if score >= 2:
        verdict = "Buy"
    elif score <= -2:
        verdict = "Sell"

    reason_text = "; ".join(reasons) if reasons else "No strong signals"
    return verdict, reason_text


def make_recommendation_enhanced(df: pd.DataFrame, latest_row: pd.Series, profile: dict | None = None, analyst: dict | None = None, news_suggestion: str | None = None, earnings_days: int | None = None) -> Tuple[str, str]:
    """Enhanced recommendation combining technical indicators, news, earnings, fundamentals and analyst signals.

    Returns (verdict, reason_text). The function is deterministic and designed for UI display.
    """
    indicators = analyze_indicators(df)
    score = 0.0
    # Strategy weights (can be adapted by RL)
    try:
        from utils.prefs import get_strategy_weights
        w = get_strategy_weights()
    except Exception:
        w = {"tech_w": 1.0, "analyst_w": 0.7, "consensus_w": 0.5, "news_w": 0.3}
    # Adjust weights conservatively based on macro regime
    try:
        regime = macro.get_macro_regime()
        rname = regime.get('regime')
        if rname in ('Macro-Defensive', 'Valuation-Rich', 'Inversion-Warn'):
            # scale down technical conviction and news weight
            w['tech_w'] = float(w.get('tech_w', 1.0)) * 0.7
            w['news_w'] = float(w.get('news_w', 0.3)) * 0.7
        elif rname == 'Opportunistic':
            # slightly increase technical weight
            w['tech_w'] = float(w.get('tech_w', 1.0)) * 1.1
    except Exception:
        pass
    reasons: list[str] = []

    # Compute core signals first (trend, rsi, bollinger, volume, analysts, news, fundamentals)

    # --- Statistical / risk signals from price series ---
    try:
        # prefer 'close' series from df if available
        if isinstance(df, pd.DataFrame) and 'close' in df.columns:
            close_series = df['close'].astype(float).dropna()
            # Sharpe and Sortino provide risk-adjusted sense of recent performance
            sh = signals.sharpe_ratio(close_series)
            so = signals.sortino_ratio(close_series)
            vol = signals.volatility(close_series)
            mom = signals.momentum(close_series, periods=63)  # quarter-ish momentum
            rsi_val = signals.rsi_from_series(close_series)
            # ATR needs high/low/close
            atr_val = None
            if {'high', 'low'}.issubset(df.columns):
                atr_val = signals.atr(df['high'], df['low'], df['close'])

            # Incorporate modestly into score
            if np.isfinite(sh):
                # positive sharpe increases conviction, negative reduces
                score += 0.5 * float(w.get('tech_w', 1.0)) * max(-1.0, min(1.0, sh / 3.0))
                reasons.append(f"Sharpe {sh:.2f}")
            if np.isfinite(so):
                score += 0.4 * float(w.get('tech_w', 1.0)) * max(-1.0, min(1.0, so / 3.0))
                reasons.append(f"Sortino {so:.2f}")
            if np.isfinite(vol):
                # higher vol reduces conviction slightly (risk-aware)
                score -= 0.2 * float(w.get('tech_w', 1.0)) * min(2.0, vol / 0.4)
                reasons.append(f"Volatility {vol:.2f}")
            if isinstance(mom, float) and np.isfinite(mom):
                score += 0.6 * float(w.get('tech_w', 1.0)) * max(-1.0, min(1.0, mom))
                reasons.append(f"Momentum {mom:.2%}")
            if isinstance(rsi_val, (int, float)) and np.isfinite(rsi_val):
                if rsi_val < 30:
                    score += 0.5 * float(w.get('tech_w', 1.0))
                    reasons.append(f"RSI oversold ({rsi_val:.0f})")
                elif rsi_val > 70:
                    score -= 0.5 * float(w.get('tech_w', 1.0))
                    reasons.append(f"RSI overbought ({rsi_val:.0f})")
            if atr_val is not None and isinstance(atr_val, (int, float)) and np.isfinite(atr_val):
                reasons.append(f"ATR {atr_val:.2f}")
    except Exception:
        # tolerate calculation errors
        pass

    # Trend weighting
    # trend influenced by technical weight
    if indicators.get("trend") == "uptrend":
        score += 1.5 * float(w.get("tech_w", 1.0))
        reasons.append("Trend: uptrend (EMA20>EMA50)")
    elif indicators.get("trend") == "downtrend":
        score -= 1.5 * float(w.get("tech_w", 1.0))
        reasons.append("Trend: downtrend (EMA20<EMA50)")

    # RSI
    rsi_sig = indicators.get("rsi")
    if rsi_sig == "oversold":
        score += 0.8 * float(w.get("tech_w", 1.0))
        reasons.append("RSI indicates oversold")
    elif rsi_sig == "overbought":
        score -= 0.8 * float(w.get("tech_w", 1.0))
        reasons.append("RSI indicates overbought")

    # Bollinger
    bb = indicators.get("bollinger") or {}
    if bb.get("label") == "below_lower":
        score += 0.7 * float(w.get("tech_w", 1.0))
        reasons.append("Price below lower Bollinger band")
    elif bb.get("label") == "above_upper":
        score -= 0.7 * float(w.get("tech_w", 1.0))
        reasons.append("Price above upper Bollinger band")

    # Volume spikes strengthen signal
    vol_sig = indicators.get("volume")
    if vol_sig == "spike":
        score += 0.5 * float(w.get("tech_w", 1.0))
        reasons.append("Volume spike detected")
    elif vol_sig == "low":
        score -= 0.2 * float(w.get("tech_w", 1.0))
        reasons.append("Low volume")

    # Earnings proximity penalizes or rewards caution: if earnings within 2 days, reduce conviction
    if earnings_days is not None and earnings_days <= 2:
        score *= 0.5
        reasons.append("Earnings soon: reduced conviction")

    # Analyst signals
    if isinstance(analyst, dict):
        # simple approach: positive/negative consensus fields
        try:
            strongbuy = float(analyst.get("strongBuy", 0) or 0)
            buy = float(analyst.get("buy", 0) or 0)
            hold = float(analyst.get("hold", 0) or 0)
            sell = float(analyst.get("sell", 0) or 0)
            strongsell = float(analyst.get("strongSell", 0) or 0)
            total = max(1.0, strongbuy + buy + hold + sell + strongsell)
            cons = ((strongbuy + buy) - (sell + strongsell)) / total
            score += float(cons) * float(w.get("analyst_w", 0.7))
            reasons.append(f"Analyst consensus score {cons:.2f}")
        except Exception:
            pass

    # News suggestion: simple mapping
    if news_suggestion == "Buy":
        score += 0.8 * float(w.get("news_w", 0.3))
        reasons.append("News sentiment: positive")
    elif news_suggestion == "Sell":
        score -= 0.8 * float(w.get("news_w", 0.3))
        reasons.append("News sentiment: negative")

    # Fundamentals: penalize very high P/E vs sector
    if profile:
        try:
            pe = float(profile.get("pe", profile.get("peRatioTTM", float("nan"))))
            sector_pe = float(profile.get("sector_pe", float("nan"))) if profile.get("sector_pe") is not None else float("nan")
            if np.isfinite(pe) and np.isfinite(sector_pe):
                if pe > sector_pe * 1.5:
                    score -= 0.9 * float(w.get("consensus_w", 0.5))
                    reasons.append("P/E elevated vs sector")
        except Exception:
            pass

    # Map score to verdict thresholds
    final = "Hold"
    if score >= 2.0:
        final = "Buy"
    elif score <= -2.0:
        final = "Sell"

    # Append macro regime reason for transparency
    try:
        if 'regime' in locals():
            final_reason = regime.get('reason')
            if final_reason:
                reasons.append(f"Macro: {final_reason}")
    except Exception:
        pass

    # --- Apply Benford data-integrity action (post-hoc) ---
    try:
        bconf = get_benford_prefs()
        if bconf.get("enabled", True) and isinstance(df, pd.DataFrame) and "volume" in df.columns:
            min_samples = int(bconf.get("min_samples", 50))
            threshold = float(bconf.get("threshold", 12.0))
            mode = str(bconf.get("mode", "penalize"))
            penalty = float(bconf.get("penalty", 0.8))
            vol_series = df["volume"].dropna()
            if len(vol_series) >= min_samples:
                br = benford_report(vol_series)
                bscore_obj = br.get("score")
                if isinstance(bscore_obj, (int, float)):
                    bscore = float(bscore_obj)
                elif isinstance(bscore_obj, str):
                    try:
                        bscore = float(bscore_obj)
                    except Exception:
                        bscore = float("nan")
                else:
                    bscore = float("nan")

                if np.isfinite(bscore) and bscore > threshold:
                    # Warn-only: append a reason but don't change verdict
                    if mode == "warn":
                        reasons.append(f"Data-integrity (warn): volume deviates from Benford (score={bscore:.2f})")
                    # Penalize: reduce conviction magnitude
                    elif mode == "penalize":
                        # scale score towards 0 by penalty multiplier
                        score *= penalty
                        # recompute final verdict conservatively
                        final = "Hold"
                        if score >= 2.0:
                            final = "Buy"
                        elif score <= -2.0:
                            final = "Sell"
                        reasons.append(f"Data-integrity: volume deviates from Benford (score={bscore:.2f}) — conviction reduced")
                    # Block: override and return Hold with reason
                    elif mode == "block":
                        final = "Hold"
                        reasons.append(f"Data-integrity (blocked): volume deviates from Benford (score={bscore:.2f}) — suggestion suppressed")
    except Exception:
        # Do not let diagnostics break the recommendation
        pass

    return final, ("; ".join(reasons) if reasons else "No strong signals")


# Backwards-compatible default name
def make_recommendation(latest_row: pd.Series, profile: dict | None = None, boll_pct: float | None = None) -> Tuple[str, str]:
    # Use enhanced recommendation when a full dataframe is not available; fall back to simple
    try:
        # best-effort: if latest_row has a `.name` that is an index into a DataFrame, we can't reconstruct the df here
        return make_recommendation_simple(latest_row, profile, boll_pct)
    except Exception:
        return make_recommendation_simple(latest_row, profile, boll_pct)
