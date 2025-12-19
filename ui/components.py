import streamlit as st
import pandas as pd
from typing import Optional


def metric_sparkline(column, label: str, value: str, series: Optional[pd.Series] = None, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Render a small metric with an optional sparkline and a short help caption.

    help_text: optional 1-2 line explanation about what the metric shows (units, source).
    """
    column.metric(label, value, delta)
    if series is not None and len(series) > 1:
        try:
            with column:
                st.line_chart(series.tail(32), height=60, use_container_width=True)
        except Exception:
            pass
    if help_text:
        # render a short, muted caption under the metric to act as an inline tooltip/legend
        try:
            column.caption(help_text)
        except Exception:
            # fallback: place caption in the parent container
            try:
                st.caption(help_text)
            except Exception:
                pass


def render_debug_panel():
    if "ui" not in st.session_state:
        return
    ui = st.session_state.ui
    fetch_ms = ui.get("last_fetch_ms")
    cache_hits = ui.get("cache_hits", 0)
    cache_misses = ui.get("cache_misses", 0)
    with st.expander("Debug / Performance", expanded=False):
        st.write(f"Last fetch time: {fetch_ms:.0f} ms" if fetch_ms is not None else "Last fetch time: —")
        st.write(f"Cache hits: {cache_hits} · Cache misses: {cache_misses}")
        if st.button("Clear cached chart"):
            ui["last_chart_key"] = None
            ui["last_chart_fig"] = None
            st.write("Cache cleared. Refresh the page to re-render.")


def compact_columns(n: int, widths: Optional[list[float]] = None):
    if widths is None:
        return st.columns(n)
    return st.columns(widths)


def badge(label: str, color: str = "#6c757d", icon_svg: str | None = None) -> str:
    """Return HTML for a small badge with optional inline SVG icon.

    The returned string is safe to render with st.markdown(..., unsafe_allow_html=True).
    Keep styling compact so badges fit in one header line.
    """
    svg_html = f"<span style='display:inline-block;vertical-align:middle;margin-right:6px'>{icon_svg}</span>" if icon_svg else ""
    html = f"<div style='display:inline-block;padding:6px 10px;border-radius:6px;background:{color};color:#fff;font-weight:600;margin-right:6px'>{svg_html}<span style='vertical-align:middle'>{label}</span></div>"
    return html
