import streamlit as st
import pandas as pd
from typing import Optional, Sequence


def metric_sparkline(column, label: str, value: str, series: Optional[pd.Series] = None, delta: Optional[str] = None):
    """Render a compact metric with a small sparkline below it using streamlit native charts.

    column: one of st.columns() entries
    label: metric label
    value: displayed as primary text
    series: optional pd.Series for sparkline (index should be time)
    delta: optional delta text shown under the value
    """
    column.metric(label, value, delta)
    if series is not None and len(series) > 1:
        try:
            # Use streamlit's native line_chart for a tiny sparkline
            # Keep it small and lightweight
            with column:
                st.line_chart(series.tail(32), height=60, use_container_width=True)
        except Exception:
            # Non-critical: skip sparkline on failure
            pass


def render_debug_panel():
    """Render a small debug / diagnostics panel showing fetch time and cache stats from session_state.ui."""
    if "ui" not in st.session_state:
        return
    ui = st.session_state.ui
    fetch_ms = ui.get("last_fetch_ms")
    cache_hits = ui.get("cache_hits", 0)
    cache_misses = ui.get("cache_misses", 0)
    with st.expander("Debug / Performance", expanded=False):
        st.write(f"Last fetch time: {fetch_ms:.0f} ms" if fetch_ms is not None else "Last fetch time: —")
        st.write(f"Cache hits: {cache_hits} · Cache misses: {cache_misses}")
        # lightweight toggles useful for debugging
        if st.button("Clear cached chart"):
            ui["last_chart_key"] = None
            ui["last_chart_fig"] = None
            st.write("Cache cleared. Refresh the page to re-render.")


def compact_columns(n: int, widths: Optional[Sequence[float]] = None):
    """Helper to create columns with compact spacing (reduce gaps by using width ratios)."""
    if widths is None:
        return st.columns(n)
    return st.columns(widths)
