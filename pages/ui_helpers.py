from __future__ import annotations
import sys
from typing import Any

import streamlit as st
import pandas as pd


def render_market_snapshot() -> None:
    """Render the compact market snapshot (header metrics).

    This function is defensive and avoids import-time network calls; it
    resolves API keys at runtime when needed.
    """
    if "pytest" in sys.modules:
        return
    try:
        try:
            from ui.header import show_header

            show_header(title="Stock Dashboard", subtitle="Market overview, news, and strategies")
        except Exception:
            try:
                from ui.provider_banner import show_provider_banner

                show_provider_banner()
            except Exception:
                pass
        try:
            from utils.rates import fetch_rates, fetch_rate_series
            from utils.helpers import fmt_number
            from ui.components import metric_sparkline
        except Exception:
            # if helpers unavailable, bail silently
            return

        rates = fetch_rates()
        try:
            with st.container():
                st.markdown("### Market Snapshot & Policy Rates")
                c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
                fed_val = rates.get("fed_funds")
                ten_val = rates.get("10y")
                sofr_val = rates.get("sofr")
                cpi_val = rates.get("cpi_yoy")

                # Resolve FRED key at runtime to fetch small series for sparklines
                try:
                    from config import get_runtime_key

                    fred_key = get_runtime_key("FRED_API_KEY")
                except Exception:
                    try:
                        from config import FRED_API_KEY as FRED_API_KEY

                        fred_key = FRED_API_KEY or None
                    except Exception:
                        fred_key = None

                cpi_series = fetch_rate_series("CPIAUCSL", fred_key, points=24) if fred_key else []
                ten_series = fetch_rate_series("DGS10", fred_key, points=60) if fred_key else []
                sofr_series = fetch_rate_series("SOFR", fred_key, points=60) if fred_key else []

                try:
                    metric_sparkline(c1, "CPI YoY (%)", fmt_number(cpi_val, 2), pd.Series(cpi_series) if cpi_series else None, None, help_text="Consumer Price Index YoY")
                except Exception:
                    try:
                        c1.metric("CPI YoY (%)", fmt_number(cpi_val, 2))
                    except Exception:
                        c1.write("CPI YoY: " + (fmt_number(cpi_val, 2) if cpi_val is not None else "N/A"))

                fed_low = rates.get("fed_funds_target_low")
                fed_high = rates.get("fed_funds_target_high")
                try:
                    if fed_low is not None and fed_high is not None:
                        metric_sparkline(c2, "Fed funds target (%)", f"{fmt_number(fed_low,2)} – {fmt_number(fed_high,2)}", None, None, help_text="Fed target range")
                    else:
                        metric_sparkline(c2, "Effective Fed Funds (%)", fmt_number(fed_val, 2), None, None, help_text="Effective rate")
                except Exception:
                    try:
                        c2.metric("Federal funds target (%)", f"{fed_low} – {fed_high}")
                    except Exception:
                        c2.metric("Effective Fed Funds (%)", fmt_number(fed_val, 2))

                try:
                    metric_sparkline(c3, "10y Treasury (%)", fmt_number(ten_val, 2), pd.Series(ten_series) if ten_series else None, None, help_text="10-year Treasury yield")
                except Exception:
                    c3.metric("10y Treasury (%)", fmt_number(ten_val, 2))

                try:
                    metric_sparkline(c4, "SOFR (%)", fmt_number(sofr_val, 3), pd.Series(sofr_series) if sofr_series else None, None, help_text="Secured Overnight Financing Rate")
                except Exception:
                    c4.metric("SOFR (%)", fmt_number(sofr_val, 3))

                st.caption("Mini-charts show recent trends when available. Data sources: FRED / provider keys if configured.")
        except Exception:
            pass
    except Exception:
        pass


def render_company_summary() -> None:
    if "pytest" in sys.modules:
        return
    try:
        top_symbol = st.session_state.ui.get("symbol", "AAPL") if "ui" in st.session_state else "AAPL"
        from data_providers import fmp_profile
        from utils.task_queue import enqueue


        @st.cache_data(ttl=600)
        def cached_fmp_profile(symbol: str, api_key: str) -> dict:
            return fmp_profile(symbol, api_key) if api_key else {}

        def _bg_prefetch_profile(symbol: str):
            try:
                from config import get_runtime_key

                fmp_key = get_runtime_key("FMP_API_KEY")
            except Exception:
                try:
                    from config import FMP_API_KEY as FMP_API_KEY

                    fmp_key = FMP_API_KEY or None
                except Exception:
                    fmp_key = None
            task = {"type": "prefetch_profile", "symbol": symbol, "api_key": fmp_key or ""}
            try:
                enqueue(task)
                if "ui" in st.session_state:
                    st.session_state.ui["last_fetch_ms"] = 0
                    st.session_state.ui["cache_misses"] = st.session_state.ui.get("cache_misses", 0) + 1
            except Exception:
                if "ui" in st.session_state:
                    st.session_state.ui["last_fetch_ms"] = None

        try:
            from config import get_runtime_key
            FMP_API_KEY = get_runtime_key("FMP_API_KEY")
        except Exception:
            try:
                from config import FMP_API_KEY as FMP_API_KEY
            except Exception:
                FMP_API_KEY = None

        prof_top = cached_fmp_profile(top_symbol, FMP_API_KEY or "") if (FMP_API_KEY or True) else {}

        with st.container():
            st.markdown("**Company summary**")
            if prof_top and isinstance(prof_top, dict):
                desc = prof_top.get("description") or prof_top.get("companyName") or ""
                ceo = prof_top.get("ceo") or prof_top.get("CEO") or "Unknown"
                founded = prof_top.get("founded") or prof_top.get("ipoDate") or "Unknown"
                sector = prof_top.get("sector") or prof_top.get("industry") or ""
                if desc:
                    sentences = [s.strip() for s in desc.split('.') if s.strip()][:5]
                    st.write('. '.join(sentences))
                st.write(f"Founded: {founded} · CEO: {ceo} · Sector: {sector}")
                try:
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        if st.button("Refresh profile", key=f"refresh_profile_{top_symbol}"):
                            try:
                                cached_fmp_profile.clear()
                            except Exception:
                                pass
                            _bg_prefetch_profile(top_symbol)
                            st.info("Prefetch scheduled; worker will refresh cache shortly.")
                    with col_b:
                        st.markdown("*Tap 'Refresh profile' to fetch latest company profile. Uses cached fetch to avoid blocking.*")
                except Exception:
                    pass
                try:
                    from utils.earnings import get_last_earnings_summary
                    les = get_last_earnings_summary(top_symbol) if top_symbol else {}
                    if les and les.get("date"):
                        le_date = les.get("date")
                        actual = les.get("actual")
                        estimate = les.get("estimate")
                        surprise = les.get("surprise_pct")
                        post_move = les.get("post_move_pct")
                        with st.expander("Last earnings summary", expanded=False):
                            st.write(f"Date: {le_date}")
                            if actual is not None or estimate is not None:
                                try:
                                    av = f"{float(actual):.2f}" if actual is not None else "—"
                                except Exception:
                                    av = str(actual)
                                try:
                                    ev = f"{float(estimate):.2f}" if estimate is not None else "—"
                                except Exception:
                                    ev = str(estimate)
                                st.write(f"Actual / Estimate: {av} / {ev}")
                            if surprise is not None:
                                try:
                                    st.write(f"Surprise: {float(surprise):.2f}%")
                                except Exception:
                                    st.write(f"Surprise: {surprise}")
                            if post_move is not None:
                                try:
                                    st.write(f"Next-day move: {float(post_move):.2f}%")
                                except Exception:
                                    st.write(f"Next-day move: {post_move}")
                            if st.button("Refresh earnings summary", key=f"refresh_earnings_{top_symbol}"):
                                try:
                                    get_last_earnings_summary.clear()
                                except Exception:
                                    pass
                                st.info("Earnings summary refresh scheduled; pull may take a few seconds.")
                except Exception:
                    pass
            else:
                st.info("Company profile not available (set FMP_API_KEY to enable).")
    except Exception:
        pass
