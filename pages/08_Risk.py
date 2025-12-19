"""Risk analytics page: lightweight VaR/CVaR and exposures."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.risk import portfolio_var, portfolio_returns_from_prices
from data_fetchers.prices import get_ohlc
from utils.ui_safe_display import display_df, safe_plotly_chart


def page() -> None:
    st.title("Risk Analytics")
    st.write("Compute simple portfolio VaR and CVaR from recent historical prices.")

    symbols = st.text_input("Comma-separated tickers", value="AAPL,MSFT,TSLA")
    if not symbols:
        st.info("Enter tickers to analyze")
        return
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    period = st.selectbox("Price history period", ["1mo", "3mo", "6mo", "1y"], index=1)
    confidence = st.slider("VaR confidence", 90, 99, 95) / 100.0
    method = st.radio("VaR method", ["historical", "parametric"], index=0)
    # mypy/pyright help: cast to the exact literal type expected by portfolio_var
    method = method  # type: ignore

    if st.button("Compute"):
        # Fetch prices
        price_frames = []
        for s in syms:
            try:
                df = get_ohlc(s, period=period, interval="1d")
                price_frames.append(df["Close"].rename(s))
            except Exception as e:
                st.warning(f"Could not fetch prices for {s}: {e}")
        if not price_frames:
            st.error("No price data available.")
            return
        price_df = pd.concat(price_frames, axis=1).dropna()

        display_df(price_df.tail(), use_container_width=True)

        var, cvar = portfolio_var(price_df, confidence=confidence, method=method)  # type: ignore
        st.metric("Portfolio VaR (loss fraction)", f"{var:.4f}")
        st.metric("Portfolio CVaR (loss fraction)", f"{cvar:.4f}")

        # show returns top-level stats
        port_ret = portfolio_returns_from_prices(price_df)
        # use plotly via safe wrapper for more robust rendering
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(y=port_ret.cumsum(), mode="lines", name="Cumulative Returns")
            )
            safe_plotly_chart(fig, use_container_width=True)
        except Exception:
            try:
                st.line_chart(port_ret.cumsum())
            except Exception:
                st.write("(Unable to render returns chart)")


if __name__ == "__main__":
    page()
