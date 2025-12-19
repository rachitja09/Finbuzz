from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

try:
    from config import get_runtime_key

    FMP_KEY = get_runtime_key("FMP_API_KEY")
except Exception:
    try:
        from config import FMP_API_KEY as FMP_KEY
    except Exception:
        FMP_KEY = None

from data_providers import fmp_gainers, fmp_losers, fmp_screen, price_history
from indicators import ema, rsi
from utils import prefs


st.set_page_config(page_title="Strategy", page_icon="🧭", layout="wide")
try:
    st.title("🧭 Strategy Screener")
except UnicodeEncodeError:
    try:
        st.title("Strategy Screener")
    except Exception:
        pass


# Only build interactive UI when Streamlit is executing the page (avoid import-time calls)
if "pytest" not in sys.modules:
    tab_day, tab_swing, tab_short, tab_long = st.tabs(
        ["Day Trading", "Swing Trading", "Short Term", "Long Term"]
    )

    # Extract tab bodies into small functions so we can run them via run_safe()
    # This prevents an exception in one tab from terminating the whole Streamlit app.
    def _day_tab():
        # --- Day Trading (gainers/losers)
        st.caption("High intraday movers with strong volume.")

        # Show if developer mock mode is enabled which forces local fallbacks
        try:
            if prefs.get_dev_mock_pref():
                st.info("Developer Mock mode is enabled — data shown below may be local fallbacks, not live provider data.")
        except Exception:
            pass

        try:
            g = pd.DataFrame(fmp_gainers(FMP_KEY or ""))
        except Exception:
            g = pd.DataFrame()

        try:
            losers = pd.DataFrame(fmp_losers(FMP_KEY or ""))
        except Exception:
            losers = pd.DataFrame()

        df = pd.concat([g, losers], ignore_index=True) if (not g.empty or not losers.empty) else pd.DataFrame()

        if df.empty:
            # Indicate whether a live provider is configured; if not, note fallback behavior
            try:
                from config import get_runtime_key

                has_fmp = bool(get_runtime_key("FMP_API_KEY"))
            except Exception:
                import os

                has_fmp = bool(os.environ.get("FMP_API_KEY"))

            if not has_fmp:
                st.info("No live FMP data available — showing local fallbacks when possible.")
            else:
                st.info("No data right now.")
        else:
            df = df.rename(columns={"ticker": "symbol", "changesPercentage": "changePct", "price": "price"})

            # coerce changePct to numeric when possible
            if "changePct" in df.columns:
                try:
                    df["changePct"] = (
                        df["changePct"].astype(str).str.replace("%", "", regex=False).astype(float)
                    )
                except Exception:
                    # leave as-is if conversion fails
                    pass

            df = (
                df.sort_values("changePct", key=abs, ascending=False).head(30)
                if "changePct" in df.columns
                else df.head(30)
            )

            # Ensure sensible column names exist before selecting the view
            if "companyName" not in df.columns:
                for alt in ("company", "company_name", "name"):
                    if alt in df.columns:
                        df["companyName"] = df[alt]
                        break
                else:
                    df["companyName"] = ""

            if "price" not in df.columns:
                for alt in ("close", "last", "c"):
                    if alt in df.columns:
                        df["price"] = df[alt]
                        break
                else:
                    df["price"] = float("nan")

            # format price and change, color up/down and add units
            def fmt_num(x):
                try:
                    return f"{float(x):,.2f}"
                except Exception:
                    return x


            def colored_pct(x):
                try:
                    v = float(x)
                    col = "#2ca02c" if v > 0 else ("#d62728" if v < 0 else "#6c757d")
                    arrow = "▲" if v > 0 else ("▼" if v < 0 else "–")
                    return f"<div style='color:{col};font-weight:600'>{arrow} {v:,.2f}%</div>"
                except Exception:
                    return x

            rows = []
            for _, r in df.head(30).iterrows():
                rows.append(
                    """
                    <tr>
                        <td style='font-weight:700'>{symbol}</td>
                        <td>{name}</td>
                        <td style='text-align:right'>{price}</td>
                        <td style='text-align:right'>{raw_change}</td>
                        <td style='text-align:right'>{pretty_change}</td>
                    </tr>
                    """.format(
                        symbol=r.get("symbol", ""),
                        name=r.get("companyName", ""),
                        price=fmt_num(r.get("price")),
                        raw_change=(fmt_num(r.get("changePct")) if r.get("changePct") is not None else ""),
                        pretty_change=(colored_pct(r.get("changePct")) if "changePct" in r else ""),
                    )
                )

            table = (
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr><th>Symbol</th><th>Name</th><th style='text-align:right'>Price (USD)</th>"
                "<th style='text-align:right'>Change</th><th style='text-align:right'>Change %</th></tr></thead>"
                "<tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )

            # Render as HTML using Streamlit components to avoid literal-HTML rendering in some environments
            try:
                import streamlit.components.v1 as components

                components.html(table, height=220)
            except Exception:
                # Fallback to markdown if components unavailable
                st.markdown(table, unsafe_allow_html=True)

            # Local fallback: compute simple intraday movers from a small curated list
            try:
                st.caption("Local fallback: scanning a small curated list")
                cur_list = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META", "NFLX"]
                rows = []
                from data_fetchers.prices import get_ohlc

                for s in cur_list:
                    try:
                        h = get_ohlc(s, period="5d", interval="1d")
                        if h is None or h.empty:
                            continue
                        last = h.iloc[-1]
                        prev = h.iloc[-2] if len(h) > 1 else last
                        price = float(last.get("close") or last.get("Close") or 0)
                        prevp = float(prev.get("close") or prev.get("Close") or price)
                        pct = (price - prevp) / prevp * 100 if prevp and prevp != 0 else 0
                        rows.append({"symbol": s, "price": price, "changePct": pct, "companyName": s})
                    except Exception:
                        continue

                if rows:
                    df_local = pd.DataFrame(rows).sort_values("changePct", key=abs, ascending=False)
                    from utils.ui_safe_display import display_df
                    display_df(
                        df_local.loc[:, ["symbol", "price", "changePct", "companyName"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Local fallback produced no results.")
            except Exception:
                pass

    def _swing_tab():
        # --- Swing Trading (EMA20>EMA50 and RSI pullback)
        st.caption("Uptrend + mild pullback (EMA20>EMA50; RSI ~35–55).")

        sector = st.selectbox(
            "Sector (optional)",
            ["", "Technology", "Financial Services", "Healthcare", "Consumer Cyclical", "Industrials"],
        )

        params: dict[str, object] = {"volumeMoreThan": 1_000_000, "marketCapMoreThan": 2_000_000_000}
        if sector:
            params["sector"] = sector

        try:
            base = pd.DataFrame(fmp_screen(params, FMP_KEY or ""))
        except Exception:
            base = pd.DataFrame()

        picks = []
        for _, row in base.head(80).iterrows():
            sym = row.get("symbol")
            try:
                sym_str = str(sym) if sym is not None else ""
                hist = price_history(sym_str, period="3mo", interval="1d")
                if hist.empty:
                    continue
                close_s = pd.Series(hist["Close"], index=hist.index, dtype=float)
                r = float(rsi(close_s, 14).iloc[-1])
                e20 = float(ema(close_s, 20).iloc[-1])
                e50 = float(ema(close_s, 50).iloc[-1])
                if e20 > e50 and 35 <= r <= 55:
                    picks.append({"symbol": sym, "rsi14": r, "price": float(hist["Close"].iloc[-1])})
            except Exception:
                continue

        out = pd.DataFrame(picks).sort_values("rsi14") if picks else pd.DataFrame()
        if not out.empty:
            from utils.ui_safe_display import display_df
            display_df(out, use_container_width=True, hide_index=True)
        else:
            st.info("No matches this minute.")

    def _short_tab():
        # --- Short Term (high beta + high volume)
        st.caption("High beta + heavy volume.")
        params = {"betaMoreThan": 1.3, "volumeMoreThan": 2_000_000, "marketCapMoreThan": 2_000_000_000}

        try:
            base = pd.DataFrame(fmp_screen(params, FMP_KEY or ""))
        except Exception:
            base = pd.DataFrame()

        if not base.empty:
            cols = [c for c in ["symbol", "companyName", "price", "beta", "volume", "sector"] if c in base.columns]
            from utils.ui_safe_display import display_df
            display_df(base[cols].head(40), use_container_width=True, hide_index=True)
        else:
            st.info("No results.")

    def _long_tab():
        # --- Long Term (quality tilt)
        st.caption("Large-cap & reasonable P/E.")
        params = {"marketCapMoreThan": 10_000_000_000, "priceMoreThan": 5, "peRatioLowerThan": 35}

        try:
            base = pd.DataFrame(fmp_screen(params, FMP_KEY or ""))
        except Exception:
            base = pd.DataFrame()

        cols = ["symbol", "companyName", "price", "sector", "pe"]
        out = base.loc[:, [c for c in cols if c in base.columns]].head(60) if not base.empty else pd.DataFrame()
        if not out.empty:
            from utils.ui_safe_display import display_df
            display_df(out, use_container_width=True, hide_index=True)
        else:
            st.info("No results.")

    # --- Day Trading (gainers/losers)
    with tab_day:
        from utils.ui_safe_display import run_safe

        run_safe(_day_tab)

    # --- Swing Trading (EMA20>EMA50 and RSI pullback)
    with tab_swing:
        from utils.ui_safe_display import run_safe

        run_safe(_swing_tab)

    # --- Short Term (high beta + high volume)
    with tab_short:
        from utils.ui_safe_display import run_safe

        run_safe(_short_tab)

    # --- Long Term (quality tilt)
    with tab_long:
        from utils.ui_safe_display import run_safe

        run_safe(_long_tab)
