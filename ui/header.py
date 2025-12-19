import streamlit as st
from typing import Optional
from pathlib import Path

from ui.provider_banner import _resolve_providers
from ui.components import badge
from utils.helpers import fmt_number
from utils.rates import fetch_rates


def _read_version():
    try:
        p = Path(__file__).parents[1] / "version.txt"
        if p.exists():
            return p.read_text(encoding="utf8").strip()
    except Exception:
        pass
    return None


def show_header(title: str = "Stock Dashboard", subtitle: Optional[str] = None):
    """Render a compact header with title, version, and provider badges."""
    try:
        try:
            # apply minimal theme tweaks for consistent appearance
            from ui.theme import apply_theme

            apply_theme()
        except Exception:
            pass
        ver = _read_version()
        cols = st.columns([3, 1])
        with cols[0]:
            try:
                st.markdown(f"<h2 style='margin:0;padding:0'>{title}</h2>", unsafe_allow_html=True)
                if subtitle:
                    st.caption(subtitle)
            except Exception:
                st.write(title)
        with cols[1]:
            # provider badges compact
            providers = _resolve_providers()
            if providers:
                html = ""
                cmap = {"FMP": "#1f77b4", "Finnhub": "#2ca02c", "FRED": "#9467bd", "NewsAPI": "#ff7f0e", "yfinance": "#17becf"}
                for p in providers:
                    color = cmap.get(p, "#6c757d")
                    # use badge helper; safe fallback if not available
                    try:
                        from ui.components import badge as _badge

                        html += _badge(p, color=color, icon_svg=None)
                    except Exception:
                        html += f"<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:{color};color:#fff;margin-right:6px'>{p}</div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("No external providers configured")
            if ver:
                st.caption(f"v{ver}")
    except Exception:
        # non-fatal: fall back to simple title
        try:
            st.title(title)
        except Exception:
            pass
