import streamlit as st
import pandas as pd
from typing import Any

from utils.ui_safe_display import run_safe, display_df
from utils.backtest import run_sma_backtest
from utils.helpers import fmt_number
from pages.helpers import _fetch_analyst, _news_headlines, _vader_fn


def render_recommendation(symbol: str, df: pd.DataFrame) -> None:
    """Render the explainable recommendation panel used on the Home page.

    This function is defensive: it imports the strategy implementation at runtime
    and falls back to a neutral recommendation if unavailable. It uses helpers
    from pages.helpers for analyst/news lookup so imports remain light at module
    import time.
    """
    try:
        from utils.strategy import compute_recommendation as compute_recommendation_impl
    except Exception:
        compute_recommendation_impl = None

    try:
        with st.container():
            st.markdown("### Explainable Strategy")
            if compute_recommendation_impl is not None:
                rec = compute_recommendation_impl(
                    symbol,
                    df,
                    tech_w=st.session_state.ui.get("tech_w", 1.0),
                    analyst_w=st.session_state.ui.get("analyst_w", 0.7),
                    consensus_w=st.session_state.ui.get("consensus_w", 0.5),
                    news_w=st.session_state.ui.get("news_w", 0.3),
                    fetch_analyst_fn=_fetch_analyst,
                    news_headlines_fn=_news_headlines,
                    vader_fn=_vader_fn,
                )
            else:
                rec = {
                    "score": 0.0,
                    "tech_score": 0.0,
                    "components": {},
                    "recommendation": "Hold",
                    "explanation": ["No strategy implementation available."],
                }

            col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
            score_pct = int(rec.get("score", 0.0) * 100)
            col_r1.metric("Composite", f"{score_pct:+d}%", rec.get("recommendation", "Hold"))
            col_r2.metric("Tech", f"{int(rec.get('tech_score',0)*100):+d}%")
            col_r3.metric("Components", len(rec.get("components", {})))

            with st.expander("Why this recommendation?", expanded=False):
                for ln in rec.get("explanation", []):
                    st.write(ln)
                try:
                    comp = pd.Series(rec.get("components") or {}).rename("value").to_frame()
                    display_df(comp, use_container_width=True)
                except Exception:
                    st.write(rec.get("components"))

            # Small backtest summary (SMA strategy) — run safely to avoid blocking
            def _small_backtest():
                try:
                    price_series = pd.Series(df["close"]) if "close" in df.columns else pd.Series(dtype=float)
                    bt = run_sma_backtest(price_series, short=20, long=50)
                    if bt:
                        st.subheader("Backtest (SMA strategy)")
                        if isinstance(bt, dict):
                            summary = {k: bt.get(k) for k in ("final_value", "returns_pct", "trades", "wins")}
                            st.write(summary)
                except Exception:
                    st.info("Backtest unavailable")

            run_safe(_small_backtest)
    except Exception:
        # Non-fatal: keep the Home page robust even if recommendation rendering fails
        return
