from __future__ import annotations
import streamlit as st
from typing import List
import os


def _resolve_providers() -> List[str]:
    providers = []
    try:
        from config import get_runtime_key

        if get_runtime_key("FMP_API_KEY"):
            providers.append("FMP")
        if get_runtime_key("FINNHUB_API_KEY"):
            providers.append("Finnhub")
        if get_runtime_key("FRED_API_KEY"):
            providers.append("FRED")
        if get_runtime_key("NEWS_API_KEY"):
            providers.append("NewsAPI")
    except Exception:
        # fallback to env only
        if os.environ.get("FMP_API_KEY"):
            providers.append("FMP")
        if os.environ.get("FINNHUB_API_KEY"):
            providers.append("Finnhub")
        if os.environ.get("FRED_API_KEY"):
            providers.append("FRED")
        if os.environ.get("NEWS_API_KEY"):
            providers.append("NewsAPI")

    # yfinance availability (non-sensitive)
    try:
        import yfinance as _yf  # type: ignore

        providers.append("yfinance")
    except Exception:
        pass

    return providers


def show_provider_banner():
    try:
        providers = _resolve_providers()
        # Build a small set of badges with colors for each provider
        if providers:
            cols = st.columns(len(providers))
            for i, p in enumerate(providers):
                try:
                    # color map for familiar providers
                    cmap = {
                        "FMP": "#1f77b4",
                        "Finnhub": "#2ca02c",
                        "FRED": "#9467bd",
                        "NewsAPI": "#ff7f0e",
                        "yfinance": "#17becf",
                    }
                    color = cmap.get(p, "#6c757d")
                    # inline SVG icons (small, lightweight) for visual clarity
                    svg_map = {
                        "FMP": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" fill="white" opacity="0.15"/><path d="M4 12h16" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
                        "Finnhub": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v20" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
                        "FRED": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" fill="white" opacity="0.12"/></svg>',
                        "NewsAPI": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 6h16M4 12h16M4 18h10" stroke="white" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
                        "yfinance": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="white" stroke-width="1.2"/></svg>',
                    }
                    icon = svg_map.get(p)
                    try:
                        from ui.components import badge

                        cols[i].markdown(badge(p, color=color, icon_svg=icon), unsafe_allow_html=True)
                    except Exception:
                        cols[i].markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:{color};color:#fff;font-weight:600'>{p}</div>", unsafe_allow_html=True)
                except Exception:
                    cols[i].write(p)
            st.caption("External data providers detected — some features may be enhanced when keys are configured.")
        else:
            st.info("No external data providers configured; app will use local fallbacks where possible.")
    except Exception:
        # non-critical
        pass
