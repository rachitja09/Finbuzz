import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

from utils.charts import candle_with_rsi_bbands
from utils.ui_safe_display import safe_plotly_chart


def render_candlestick(symbol: str, df: pd.DataFrame, period: str, interval: str, show_rsi: bool, chart_points: int = 800) -> None:
    """Render the main candlestick chart with RSI and Bollinger Bands.

    This mirrors the behavior previously in pages/01_Home.py but is isolated to
    keep that file smaller and easier for static analyzers to process.
    """
    # Only build charts when not running under pytest
    import sys

    if "pytest" in sys.modules:
        return

    # If Plotly is unavailable, skip gracefully
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        go = None

    if go is None:
        return

    # explicit type check of df so pyright does not treat it as NDFrame in boolean context
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    try:
        max_points = int(chart_points or 1000)
        plot_df = df if len(df) <= max_points else df.tail(max_points)

        # simple session_state-backed fig cache keyed by symbol/period/interval/len
        chart_key = (symbol, period, interval, show_rsi, len(plot_df))
        cached_key = st.session_state.ui.get("last_chart_key")
        if cached_key == chart_key and st.session_state.ui.get("last_chart_fig") is not None:
            fig = st.session_state.ui.get("last_chart_fig")
        else:
            with st.spinner("Rendering chart…"):
                # utils.charts expects capitalized column names in some places
                fig = candle_with_rsi_bbands(plot_df.rename(columns={c: c.capitalize() for c in plot_df.columns}), title="Price with EMA20/EMA50 & Bollinger Bands")
                st.session_state.ui["last_chart_key"] = chart_key
                st.session_state.ui["last_chart_fig"] = fig

        safe_plotly_chart(fig, use_container_width=True)
    except Exception:
        # fallback to a minimal manual candlestick if the high-level chart builder fails
        try:
            xcol = df["datetime"] if "datetime" in df.columns else (df["date"] if "date" in df.columns else df.index)
            x = np.asarray(pd.to_datetime(xcol))
            x_list = [pd.to_datetime(v).to_pydatetime() for v in x]
            fig = go.Figure()
            colmap = {c.lower(): c for c in df.columns}
            ocol = colmap.get("open", "open")
            hcol = colmap.get("high", "high")
            lcol = colmap.get("low", "low")
            ccol = colmap.get("close", "close")
            Candlestick = getattr(go, "Candlestick", None)
            if Candlestick is not None:
                o_arr = pd.Series(df[ocol], index=df.index, dtype=float).to_numpy().tolist()
                h_arr = pd.Series(df[hcol], index=df.index, dtype=float).to_numpy().tolist()
                l_arr = pd.Series(df[lcol], index=df.index, dtype=float).to_numpy().tolist()
                c_arr = pd.Series(df[ccol], index=df.index, dtype=float).to_numpy().tolist()
                x_plot = [pd.to_datetime(v).to_pydatetime() for v in x_list]
                fig.add_trace(Candlestick(x=x_plot, open=o_arr, high=h_arr, low=l_arr, close=c_arr, name="Price"))
            try:
                from ui.plotting import apply_plotly_theme

                if fig is not None:
                    fig = apply_plotly_theme(fig, title="Price with EMA20/EMA50 & Bollinger Bands", x_title="Date", y_title="Price (USD)", dark=True)
            except Exception:
                pass
            safe_plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Unable to render chart for this data.")
