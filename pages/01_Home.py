import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from typing import Any
from utils.frames import sanitize_for_arrow
from utils.charts import candle_with_rsi_bbands
from ui.components import metric_sparkline, compact_columns
from utils.ui_safe_display import safe_plotly_chart, run_safe, display_df
from utils.backtest import run_sma_backtest
from utils.watchlist import prefetch_quotes
from data_fetchers.prices import get_ohlc
from utils.rates import fetch_rates, fetch_rate_series
from utils.helpers import fmt_number, _safe_float, fmt_money, format_percent, badge_text
from pages.helpers import get_watchlist_from_portfolio, _fetch_analyst, _news_headlines, _vader_fn


def render_etf_analysis(etf_input: str):
    """Render ETF analysis for comma-separated tickers. Extracted to reduce file complexity."""
    try:
        with st.expander("ETF Analysis & Suggestions", expanded=False):
            st.markdown("Enter ETF tickers (comma-separated) to compute the same explainable recommendations we show for stocks.")
            etf_input_val = etf_input or "SPY,QQQ,IVV,IWM"
            etf_input_box = st.text_input("ETFs", value=etf_input_val)
            if etf_input_box:
                etfs = [s.strip().upper() for s in etf_input_box.split(",") if s.strip()]
            else:
                etfs = []

            etf_rows = []
            for t in etfs:
                try:
                    # reuse the cached loader to fetch OHLC and compute indicators; fallback to yfinance for ETFs
                    try:
                        tdf = load_ohlc(t, "3mo", "1d") if "pytest" not in sys.modules else pd.DataFrame()
                        tdf = pd.DataFrame(tdf)
                    except Exception:
                        tdf = pd.DataFrame()
                    if (tdf is None or tdf.empty) and 'yf' in globals() and yf is not None:
                        try:
                            raw = yf.Ticker(t).history(period="3mo", interval="1d")
                            if not raw.empty:
                                raw = raw.reset_index()
                                raw = raw.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Close": "close", "Volume": "volume"})
                                tdf = raw
                        except Exception:
                            tdf = pd.DataFrame()

                    # If yfinance returned price rows but no precomputed indicators, compute them here
                    try:
                        if tdf is not None and not tdf.empty:
                            # normalise column names
                            tdf = tdf.rename(columns={c: c.lower() for c in tdf.columns})
                            cols = {c.lower() for c in tdf.columns}
                            if 'close' in cols and not {'ema20', 'ema50', 'rsi'}.issubset(cols):
                                close = pd.Series(tdf['close'], index=tdf.index, dtype=float)
                                tdf['ema20'] = close.ewm(span=20, adjust=False).mean()
                                tdf['ema50'] = close.ewm(span=50, adjust=False).mean()
                                win = 20
                                mid = close.rolling(win).mean()
                                std = close.rolling(win).std()
                                tdf['bb_mid'] = mid
                                tdf['bb_up'] = mid + 2 * std
                                tdf['bb_low'] = mid - 2 * std
                                delta = close.diff()
                                gain = delta.clip(lower=0.0)
                                loss = -delta.clip(upper=0.0)
                                avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
                                avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
                                rs = avg_gain / avg_loss.replace(0, np.nan)
                                tdf['rsi'] = pd.Series(100 - 100 / (1 + rs), index=close.index).fillna(0)
                                tr_vals = np.maximum((pd.Series(tdf['high']) - pd.Series(tdf['low'])).to_numpy(), np.abs((pd.Series(tdf['high']) - pd.Series(close.shift())).to_numpy()), np.abs((pd.Series(tdf['low']) - pd.Series(close.shift())).to_numpy()))
                                tr = pd.Series(tr_vals, index=close.index)
                                tdf['atr'] = tr.rolling(14).mean()
                    except Exception:
                        # non-fatal: leave tdf as-is if indicator computation fails
                        pass

                    if tdf is None or tdf.empty:
                        etf_rows.append({"Symbol": t, "Last": "N/A", "Recommendation": "N/A", "Score": None, "Reason": "No data"})
                        continue
                    # ensure columns have lowercase names used by strategy
                    if not {'ema20', 'ema50', 'rsi'}.issubset([c.lower() for c in tdf.columns]):
                        # recompute indicators via load_ohlc path if possible
                        try:
                            tdf = load_ohlc(t, "3mo", "1d") if "pytest" not in sys.modules else pd.DataFrame(tdf)
                            tdf = pd.DataFrame(tdf)
                        except Exception:
                            pass

                    last_price = float(tdf.iloc[-1].get("close") or tdf.iloc[-1].get("Close") or float("nan"))
                    if compute_recommendation_impl is not None:
                        rec = compute_recommendation_impl(t, tdf, tech_w=st.session_state.ui.get("tech_w", 1.0), analyst_w=st.session_state.ui.get("analyst_w", 0.7), consensus_w=st.session_state.ui.get("consensus_w", 0.5), news_w=st.session_state.ui.get("news_w", 0.3), fetch_analyst_fn=_fetch_analyst, news_headlines_fn=_news_headlines, vader_fn=_vader_fn)
                    else:
                        try:
                            verdict, reason = make_recommendation(tdf.iloc[-1], profile=None)
                            rec = {"recommendation": verdict, "score": 0.0, "explanation": [reason], "tech_score": 0.0, "components": {}}
                        except Exception:
                            rec = {"recommendation": "Hold", "score": 0.0, "explanation": ["No strategy available"]}
                    etf_rows.append({"Symbol": t, "Last": f"${last_price:,.2f}" if pd.notna(last_price) else "N/A", "Recommendation": rec.get("recommendation", "N/A"), "Score": f"{rec.get('score',0):+.2f}", "Reason": ("; ".join(rec.get("explanation", []))[:200])})
                except Exception as e:
                    etf_rows.append({"Symbol": t, "Last": "err", "Recommendation": "err", "Score": None, "Reason": str(e)})

            if etf_rows:
                try:
                    etf_df = pd.DataFrame(etf_rows)
                    display_df(etf_df, use_container_width=True, hide_index=True)
                except Exception:
                    st.table(etf_rows)
    except Exception:
        # non-fatal: ETF analysis optional
        pass


# Resolve provider API keys at runtime to avoid import-time side-effects and support
# explicit-empty-env semantics used by tests/CI.
try:
    from config import get_runtime_key
    FINNHUB_API_KEY = get_runtime_key("FINNHUB_API_KEY")
    FMP_API_KEY = get_runtime_key("FMP_API_KEY")
    NEWS_API_KEY = get_runtime_key("NEWS_API_KEY")
    FRED_API_KEY = get_runtime_key("FRED_API_KEY")
except Exception:
    try:
        from config import FINNHUB_API_KEY, FMP_API_KEY, NEWS_API_KEY, FRED_API_KEY
    except Exception:
        FINNHUB_API_KEY = None
        FMP_API_KEY = None
        NEWS_API_KEY = None
        FRED_API_KEY = None

# fetcher adapters moved to pages.helpers to reduce file complexity

# Optional heavy deps
try:
    import pandas_ta as _ta
    ta = _ta
except Exception:
    ta = None

yf: Any = None
try:
    import yfinance as _yf  # type: ignore
    yf = _yf
except Exception:
    pass
go: Any = None
try:
    import plotly.graph_objects as _go
    go = _go
except Exception:
    go = None


# fetch_rates and fetch_rate_series are provided by utils.rates to centralize rate logic


# Formatting and numeric helpers are centralized in utils.helpers


from pages.ui_helpers import render_market_snapshot, render_company_summary

# Render header and company summary via helper module to reduce file complexity
render_market_snapshot()
render_company_summary()

# Persistent UI state
if "ui" not in st.session_state:
    from utils.prefs import get_strategy_weights
    w = get_strategy_weights()
    st.session_state.ui = {"symbol": "AAPL", "period": "3mo", "interval": "1d", "tech_w": w.get("tech_w", 1.0), "analyst_w": w.get("analyst_w", 0.7), "consensus_w": w.get("consensus_w", 0.5), "news_w": w.get("news_w", 0.3), "last_chart_key": None, "last_chart_fig": None}

# Controls
colL, colR = st.columns([3, 1])
with colR:
    symbol = (st.text_input("Symbol", st.session_state.ui.get("symbol", "AAPL")) or "").strip().upper()
    period = st.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"].index(st.session_state.ui.get("period", "3mo")))
    interval = st.selectbox("Interval", ["1d", "60m", "30m", "15m", "5m", "1m"], index=["1d", "60m", "30m", "15m", "5m", "1m"].index(st.session_state.ui.get("interval", "1d")))
    show_rsi = st.checkbox("Show RSI(14)", value=True)
    with st.expander("Recommendation model weights (tunable)"):
        tech_w = st.slider("Technical weight", 0.0, 2.0, st.session_state.ui.get("tech_w", 1.0), 0.1)
        analyst_w = st.slider("Analyst target weight", 0.0, 2.0, st.session_state.ui.get("analyst_w", 0.7), 0.1)
        consensus_w = st.slider("Analyst consensus weight", 0.0, 2.0, st.session_state.ui.get("consensus_w", 0.5), 0.1)
    news_w = st.slider("News weight", 0.0, 2.0, st.session_state.ui.get("news_w", 0.3), 0.1)
    # Chart downsampling points (controls responsiveness vs detail)
    chart_points = st.slider("Max chart points", 200, 5000, st.session_state.ui.get("chart_points", 800), step=50,
                             help="Maximum number of points to render on charts; lower values improve responsiveness for long histories.")
    st.session_state.ui.update({"symbol": symbol, "period": period, "interval": interval, "tech_w": tech_w, "analyst_w": analyst_w, "consensus_w": consensus_w, "news_w": news_w})
    st.session_state.ui.update({"chart_points": chart_points})

if not symbol:
    st.stop()

# Consent / privacy acknowledgement for invited users (skip under pytest)
if "pytest" not in sys.modules:
    try:
        # show the short privacy/terms notice and require acceptance before proceeding
        if not st.session_state.ui.get("consent_accepted", False):
            with st.expander("Privacy & Terms — please read and accept to continue", expanded=True):
                st.markdown("""
                **Short privacy & terms**

                - This app fetches third-party market and news data using the owner's API keys.
                - Your requests may be proxied and minimally logged for monitoring and cost control.
                - Data and recommendations are for informational purposes only; not financial advice.

                By clicking **Accept** you acknowledge you have read and agree to these terms.
                """)
                if st.button("Accept and continue", key="accept_privacy"):
                    st.session_state.ui["consent_accepted"] = True
                    # experimental_rerun may not exist in older/newer Streamlit versions; call safely
                    getattr(st, "experimental_rerun", lambda: None)()
            # if not accepted, stop rendering further content
            if not st.session_state.ui.get("consent_accepted", False):
                st.stop()
    except Exception:
        # non-fatal: if UI rendering of consent fails, allow access (safer than blocking in error)
        pass

# Quick watchlist scanner: on app open, compute a light-weight "top pick" across a watchlist
def _get_watchlist_from_portfolio(defaults=None):
    # wrapper to maintain existing API while delegating to helpers
    return get_watchlist_from_portfolio(defaults)

def _estimate_intraday_return(df: pd.DataFrame, score: float) -> float:
    """Rudimentary estimate of intraday expected pct move using recent daily volatility and composite score.
    This is heuristic: expected_pct = score * recent_daily_std * 100. """
    try:
        if df is None or df.empty:
            return 0.0
        cols = {c.lower(): c for c in df.columns}
        close_col = cols.get("close") or cols.get("Close")
        if close_col is None:
            return 0.0
        series = pd.Series(df[close_col]).dropna().astype(float)
        if len(series) < 5:
            return 0.0
        daily_ret = series.pct_change().dropna()
        vol = float(daily_ret.tail(20).std())
        return float(score * vol * 100.0)
    except Exception:
        return 0.0

# Display a lightweight top-pick banner (non-blocking)
try:
    from utils.strategy import compute_recommendation as _compute_recommendation
except Exception:
    _compute_recommendation = None

if _compute_recommendation is not None:
    try:
        watch = _get_watchlist_from_portfolio()[:12]
        best = None
        for s in watch:
            try:
                tdf = load_ohlc(s, "3mo", "1d") if "pytest" not in sys.modules else pd.DataFrame()
                if tdf is None or tdf.empty:
                    continue
                rec = _compute_recommendation(s, tdf, fetch_analyst_fn=_fetch_analyst, news_headlines_fn=_news_headlines, vader_fn=_vader_fn)
                score = rec.get("score", 0.0)
                # prefer Buy with highest positive score, or lowest (most negative) Sell
                if best is None:
                    best = (s, score, rec, tdf)
                else:
                    # choose highest absolute recommendation magnitude
                    if abs(score) > abs(best[1]):
                        best = (s, score, rec, tdf)
            except Exception:
                continue

        if best is not None:
            bp_sym, bp_score, bp_rec, bp_df = best
            est = _estimate_intraday_return(bp_df, bp_rec.get("score", 0.0))
            sentiment = bp_rec.get("news_score") if isinstance(bp_rec.get("news_score"), float) else None
            # render compact banner
            b1, b2, b3 = st.columns([3, 2, 2])
            b1.metric("Top pick", f"{bp_sym} — {bp_rec.get('recommendation')}", f"Score {bp_score:+.2f}")
            b2.metric("Est intraday %", f"{est:+.2f}%")
            # compute simple risk metric: recent daily vol pct
            try:
                cols = {c.lower(): c for c in bp_df.columns}
                close_col = cols.get("close") or cols.get("Close")
                rr = 0.0
                if close_col:
                    sers = pd.Series(bp_df[close_col]).astype(float).pct_change().dropna()
                    rr = float(sers.tail(20).std() * 100.0)
                b3.metric("Risk (daily vol %)", f"{rr:.2f}%")
            except Exception:
                b3.metric("Risk (daily vol %)", "N/A")
            with st.expander("Why this pick?", expanded=False):
                st.write(bp_rec.get("explanation", []))
    except Exception:
        # non-fatal UI enhancement
        pass

@st.cache_data(ttl=300, show_spinner=False)
def load_ohlc(sym, per, iv):
    df = get_ohlc(sym, period=per, interval=iv)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "close" in df.columns:
        close = pd.Series(df["close"], index=df.index, dtype=float)
        high = pd.Series(df["high"], index=df.index, dtype=float)
        low = pd.Series(df["low"], index=df.index, dtype=float)
        df["ema20"] = close.ewm(span=20, adjust=False).mean()
        df["ema50"] = close.ewm(span=50, adjust=False).mean()
        win = 20
        mid = close.rolling(win).mean()
        std = close.rolling(win).std()
        df["bb_mid"] = mid
        df["bb_up"] = mid + 2 * std
        df["bb_low"] = mid - 2 * std
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = pd.Series(100 - 100 / (1 + rs), index=close.index).fillna(0)
        tr_vals = np.maximum((high - low).to_numpy(), np.abs((high - close.shift()).to_numpy()), np.abs((low - close.shift()).to_numpy()))
        tr = pd.Series(tr_vals, index=close.index)
        df["atr"] = tr.rolling(14).mean()
    return sanitize_for_arrow(df)

if "pytest" not in sys.modules:
    try:
        df = load_ohlc(symbol, period, interval)
    except Exception as e:
        st.error(f"Failed to fetch OHLC data: {e}")
        st.stop()
    df: pd.DataFrame = pd.DataFrame(df)
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        st.error("No data returned for this symbol/period/interval.")
        st.stop()

    # Plot
    try:
        # Downsample long series for plotly rendering to avoid heavy charts
        from utils.frames import downsample_time_index
        plot_df = df.tail(2000)
        max_pts = int(st.session_state.ui.get("chart_points", 800))
        plot_df = downsample_time_index(plot_df, max_points=max_pts)
        chart_key = (symbol, period, interval, show_rsi, len(plot_df))
        if st.session_state.ui.get("last_chart_key") == chart_key and st.session_state.ui.get("last_chart_fig") is not None:
            fig = st.session_state.ui.get("last_chart_fig")
        else:
            fig = candle_with_rsi_bbands(plot_df.rename(columns={c: c.capitalize() for c in plot_df.columns}), title="Price with EMA20/EMA50 & Bollinger Bands")
            st.session_state.ui["last_chart_key"] = chart_key
            st.session_state.ui["last_chart_fig"] = fig

        # render chart safely
        safe_plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Unable to render chart for this data.")

    # Company profile and earnings display moved earlier to avoid duplication

    # Snapshot metrics (compact)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    # prev_close should come from `prev`, last_close_val from `last` (was swapped)
    prev_close = float(prev.get("close") or np.nan)
    last_close_val = float(last.get("close") or np.nan)
    chg = last_close_val - prev_close
    pct = (chg / prev_close * 100.0) if (np.isfinite(prev_close) and prev_close != 0.0) else 0.0
    m1, m2, m3, m4, m5, m6 = compact_columns(6, widths=[1, 1, 1, 1, 1, 1])
    # Add short help texts explaining the metric units and source to improve clarity
    price_help = "Latest trade price (USD). Sparkline shows recent closes. Percent is change vs previous row."
    try:
        metric_sparkline(m1, "Price", fmt_number(last_close_val), pd.Series(df["close"]) if "close" in df.columns else None, f"{pct:+.2f}%", help_text=price_help)
    except Exception:
        m1.metric("Price", fmt_number(last_close_val), f"{pct:+.2f}%")
        try:
            m1.caption(price_help)
        except Exception:
            pass

    # Reuse canonical strategy implementation from utils.strategy to keep logic centralized
    try:
        from utils.strategy import compute_recommendation as compute_recommendation_impl
    except Exception:
        compute_recommendation_impl = None

    # Render the recommendation in a side box (delegated to helper to reduce file size)
    try:
        from pages.ui_recommendation import render_recommendation

        render_recommendation(symbol, df)
    except Exception:
        # non-fatal: if the helper fails, continue gracefully
        pass

    # --- ETF Analysis & Suggestions ---
    try:
        render_etf_analysis("SPY,QQQ,IVV,IWM")
    except Exception:
        # non-fatal: ETF analysis optional
        pass

    # Simulator removed: trading simulation was moved to a dedicated Portfolio page.
    # Under pytest we avoid network calls; provide an empty DataFrame for simulator data
    # but DO NOT overwrite the OHLC `df` used by charts and KPIs. Use a separate
    # variable to avoid accidentally clearing price/history data.
    simulator_df = pd.DataFrame()

    # --- prepare x axis for charts ---
    # produce a numpy array even under pytest so the x variable has consistent type
    if "datetime" in df.columns:
        xcol = df["datetime"]
        x = pd.to_datetime(xcol).to_numpy() if not pd.api.types.is_datetime64_any_dtype(xcol) else xcol.to_numpy()
    elif "date" in df.columns:
        xcol = df["date"]
        x = pd.to_datetime(xcol).to_numpy() if not pd.api.types.is_datetime64_any_dtype(xcol) else xcol.to_numpy()
    else:
        x = df.index.to_numpy()

    # ensure pytest path also defines a native Python x_list for plotting
    try:
        x_list: list[Any] = [pd.to_datetime(v).to_pydatetime() for v in x]
    except Exception:
        x_list = list(x.tolist() if hasattr(x, "tolist") else list(x))

    # --- Candlestick & UI rendering ---
if "pytest" not in sys.modules:
    # Only build charts and UI when not running under pytest (avoids heavy deps/network calls)
    if go is not None:  # type: ignore[reportGeneralTypeIssues]
        # explicit type check of df so pyright does not treat it as NDFrame in boolean context
        if not isinstance(df, pd.DataFrame):
            # nothing to render
            pass
        elif len(df) == 0:
            # no data
            pass
        else:
            # delegate chart rendering to helper module (keeps this file smaller)
            try:
                from pages.ui_charts import render_candlestick

                render_candlestick(symbol, df, period, interval, show_rsi, chart_points=st.session_state.ui.get("chart_points", 800))
            except Exception:
                # non-fatal: if chart helper fails, continue
                try:
                    st.warning("Unable to render chart for this data.")
                except Exception:
                    pass

    # --- Snapshot metrics ---
# helpers: coerce single-item Series -> scalar and safely to float
# _to_scalar and _safe_float are imported from utils.helpers above


def _badge_html(label: str, positive: bool | None) -> str:
    """Return a small HTML span used as colored badge (green/red/gray)."""
    color = "#2ca02c" if positive is True else ("#d62728" if positive is False else "#6c757d")
    return f"<span style='display:inline-block;padding:4px 8px;border-radius:6px;background:{color};color:#fff;font-weight:600'>{label}</span>"

if "pytest" not in sys.modules:
    # Defensive: ensure DataFrame has at least one row before indexing with iloc
    if not isinstance(df, pd.DataFrame) or df.empty:
        # No data — show placeholders and initialize safe defaults so downstream
        # code can continue without special-casing `None` values.
        m1, m2, m3, m4, m5, m6 = compact_columns(6, widths=[1,1,1,1,1,1])
        try:
            m1.metric("Price", "—")
        except Exception:
            pass
        m2.metric("Change", "—")
        m3.metric("Vol vs 20d", "—")
        m4.metric("ATR(14)", "—")
        m5.metric("52w High", "—")
        m6.metric("52w Low", "—")
        # Provide empty pandas Series for `last`/`prev` so functions expecting
        # a Series receive the right type (avoids pyright Optional/None issues).
        last = pd.Series(dtype=object)
        prev = last
        # Safe numeric placeholders used later
        prev_close = float(np.nan)
        last_close_val = float(np.nan)
        chg = 0.0
        pct = 0.0
        v20 = float(np.nan)
        last_vol = float(np.nan)
        atr = float(np.nan)
        high_52w = float(np.nan)
        low_52w = float(np.nan)
        # leave df as-is for downstream code paths
    else:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        # coerce values to plain floats to avoid NDFrame/Series typing issues
        prev_close = float(_safe_float(prev.get("close", np.nan)))
        last_close_val = float(_safe_float(last.get("close", np.nan)))
        chg = last_close_val - prev_close
        pct = (chg / prev_close * 100.0) if (np.isfinite(prev_close) and prev_close != 0.0) else 0.0

        # Use compact columns helper for efficient space usage
        m1, m2, m3, m4, m5, m6 = compact_columns(6, widths=[1,1,1,1,1,1])
        if len(df) >= 20:
            # ensure rolling mean yields a Series for iloc
            vol_series = pd.Series(df["volume"], index=df.index, dtype=float)
            v20_series = pd.Series(vol_series.rolling(20).mean(), index=vol_series.index, dtype=float)
            v20 = float(_safe_float(v20_series.iloc[-1]))
        else:
            v20 = np.nan
        last_vol = float(_safe_float(last.get("volume", np.nan)))
        # Render compact metrics with small sparklines where applicable
        try:
            metric_sparkline(m1, "Price", "$" + fmt_number(_safe_float(last.get('close', np.nan))), pd.Series(df['close']) if 'close' in df.columns else None, format_percent(pct/100.0, show_sign=True))
        except Exception:
            m1.metric("Price", "$" + fmt_number(_safe_float(last.get('close', np.nan))), format_percent(pct/100.0, show_sign=True))
        m2.metric("Change (USD)", "$" + fmt_number(chg))
        m3.metric("Vol vs 20d", (fmt_number(last_vol / v20) + "×") if np.isfinite(v20) and v20 > 0 else "—")

        atr = float(_safe_float(last.get("atr", np.nan)))
        m4.metric("ATR(14) (USD)", fmt_number(atr))

        win_52w = min(252, len(df))
        close_series = pd.Series(df["close"], index=df.index, dtype=float)
        high_series = pd.Series(close_series.rolling(win_52w).max(), index=close_series.index, dtype=float)
        low_series = pd.Series(close_series.rolling(win_52w).min(), index=close_series.index, dtype=float)
        high_52w = float(_safe_float(high_series.iloc[-1]))
        low_52w = float(_safe_float(low_series.iloc[-1]))
        m5.metric("52w High (USD)", fmt_number(high_52w) if np.isfinite(high_52w) else "—")
        m6.metric("52w Low (USD)",  fmt_number(low_52w) if np.isfinite(low_52w) else "—")

    # KPI row (company P/E, sector P/E, Vol vs 20d, 52w HL, Bollinger %B)
    try:
        from data_providers import sector_industry_pe_for_symbol
        pe_info = sector_industry_pe_for_symbol(symbol, FMP_API_KEY) if FMP_API_KEY else {}
    except Exception:
        pe_info = {}

    # Bollinger %B: (Close - Lower) / (Upper - Lower)
    boll_pct = None
    try:
        close_val = float(_safe_float(last.get("close", np.nan)))
        low_band = float(_safe_float(last.get("bb_low", np.nan)))
        up_band = float(_safe_float(last.get("bb_up", np.nan)))
        if np.isfinite(low_band) and np.isfinite(up_band) and (up_band - low_band) != 0:
            boll_pct = (close_val - low_band) / (up_band - low_band)
    except Exception:
        boll_pct = None

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    company_pe = pe_info.get("company_pe") if isinstance(pe_info, dict) else None
    sector_pe = pe_info.get("sector_pe") if isinstance(pe_info, dict) else None
    try:
        from utils.helpers import fmt_money, format_percent, badge_text
    except Exception:
        # fallback to previously available helpers
        from utils.helpers import fmt_number as fmt_money, fmt_number as format_percent, badge_text

    try:
        k1.metric("Last Close", fmt_money(last_close_val))
    except Exception:
        k1.metric("Last Close", fmt_money(_safe_float(last.get('close', np.nan))))
    k2.metric("Company P/E", fmt_number(company_pe, 1) if company_pe else "—")
    k3.metric("Sector P/E", fmt_number(sector_pe, 1) if sector_pe else "—")
    try:
        vol_text = f"{(last_vol / v20):.2f}×" if np.isfinite(v20) and v20 > 0 else "—"
    except Exception:
        vol_text = "—"
    k4.metric("Vol vs 20d", vol_text)
    k5.metric("52w High/Low", (fmt_number(high_52w, 0) + "/" + fmt_number(low_52w, 0)) if np.isfinite(high_52w) and np.isfinite(low_52w) else "—")
    k6.metric("Bollinger %B", fmt_number(boll_pct) if boll_pct is not None and np.isfinite(boll_pct) else "—")

    # KPI badges: Streamlit-native, small captions with emoji (no HTML)
    try:
        # use centralized badge_text from utils.helpers

        # Company P/E vs Sector P/E: green if cheaper than sector, red if >1.5x sector
        pe_badge: bool | None = None
        if company_pe and sector_pe:
            try:
                cpe = float(company_pe)
                spe = float(sector_pe)
                if np.isfinite(cpe) and np.isfinite(spe):
                    if cpe > spe * 1.5:
                        pe_badge = False
                    elif cpe < spe:
                        pe_badge = True
                    else:
                        pe_badge = None
            except Exception:
                pe_badge = None
        k2.caption(badge_text("P/E", pe_badge))

        # Vol vs 20d: high (>1.2) -> green (momentum), low (<0.6) -> red
        vol_badge: bool | None = None
        try:
            if np.isfinite(v20) and v20 > 0:
                ratio = last_vol / v20
                if ratio > 1.2:
                    vol_badge = True
                elif ratio < 0.6:
                    vol_badge = False
                else:
                    vol_badge = None
        except Exception:
            vol_badge = None
        k4.caption(badge_text("Vol", vol_badge))

        # 52w position: closer to low -> green (bargain), closer to high -> red
        range_badge: bool | None = None
        try:
            if np.isfinite(high_52w) and np.isfinite(low_52w) and (high_52w - low_52w) > 0:
                pct_pos = (last_close_val - low_52w) / (high_52w - low_52w)
                if pct_pos < 0.33:
                    range_badge = True
                elif pct_pos > 0.66:
                    range_badge = False
                else:
                    range_badge = None
        except Exception:
            range_badge = None
        k5.caption(badge_text("52w", range_badge))

        # Bollinger %B badge
        bb_badge: bool | None = None
        try:
            if boll_pct is not None and np.isfinite(boll_pct):
                if boll_pct < 0.1:
                    bb_badge = True
                elif boll_pct > 0.9:
                    bb_badge = False
                else:
                    bb_badge = None
        except Exception:
            bb_badge = None
        k6.caption(badge_text("%B", bb_badge))
    except Exception:
        # Non-critical: if badge rendering fails, ignore
        pass

    # Risk KPIs (Sharpe/Sortino/Vol/MaxDD/Momentum)
    try:
        from pages.ui_metrics import render_kpis

        render_kpis(symbol, df)
    except Exception:
        try:
            st.info("Risk KPIs unavailable for this dataset.")
        except Exception:
            pass

    # Long-horizon heuristic trajectory (not advice)
    try:
        from pages.ui_forecast import render_forecast

        render_forecast(symbol, df, horizon_years=10)
    except Exception:
        # non-fatal, keep dashboard responsive
        try:
            st.caption("Trajectory unavailable for this dataset.")
        except Exception:
            pass

    # Bull / Base / Bear scenario grid (probability-weighted, heuristic)
    try:
        from pages.ui_scenarios import render_scenarios

        render_scenarios(df, symbol=symbol)
    except Exception:
        try:
            st.caption("Scenarios unavailable for this dataset.")
        except Exception:
            pass

    # Debug panel and simulated trading UI removed per user request.

    # Bollinger %B already shown in KPI row above

    # Recommendation: combine technicals and optional fundamentals
    def make_recommendation(latest_row: pd.Series, profile: dict | None = None) -> tuple[str, str]:
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
        # fundamentals (optional): penalize very high P/E vs sector if profile provided
        if profile:
            try:
                pe = float(profile.get("pe", profile.get("peRatioTTM", np.nan)))
                sector_pe = float(profile.get("sector_pe", np.nan)) if profile.get("sector_pe") is not None else np.nan
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

    # Try to fetch a lightweight profile and analyst consensus if available (non-blocking)
    profile = None
    analyst_info = None
    try:
        from data_fetchers.finnhub import get_quote_finnhub, fetch_analyst_consensus
        q = get_quote_finnhub(symbol)
        # copy a few fields into profile for optional checks
        profile = {"pe": q.get("pe") if isinstance(q.get("pe"), (int,float)) else None}
        # fetch analyst target/consensus (may return None or dict)
        if FINNHUB_API_KEY:
            analyst_info = fetch_analyst_consensus(FINNHUB_API_KEY, [symbol])
            # fetch_analyst_consensus returns a DataFrame in this codebase; convert
            # to a simple dict for downstream logic if rows exist.
            try:
                if hasattr(analyst_info, "empty") and not analyst_info.empty:
                    # take the first (most recent) row
                    row = analyst_info.iloc[0]
                    analyst_info = dict(row.dropna().to_dict())
                else:
                    analyst_info = None
            except Exception:
                analyst_info = None
    except Exception:
        profile = None
        analyst_info = None

    # Compute analyst-derived signals
    analyst_score = 0.0
    analyst_reasons: list[str] = []
    cons_score = 0.0
    # placeholder for news-derived suggestion (populated later if NEWS_API_KEY available)
    news_suggestion: str | None = None
    # Analyst summary values we'll show in UI
    analyst_target: float | None = None
    analyst_rec_summary: str | None = None
    if isinstance(analyst_info, dict):
        try:
            # Safely coerce potential numpy/pandas values into Python floats
            raw_tgt = analyst_info.get("targetMedian") or analyst_info.get("targetMean") or analyst_info.get("targetMean")
            tgt = _safe_float(raw_tgt)
            analyst_target = float(tgt) if np.isfinite(tgt) else None
            last_close = _safe_float(last.get("close", np.nan)) if hasattr(last, 'index') else np.nan
            if np.isfinite(tgt) and np.isfinite(last_close):
                # positive if target > price
                delta_pct = (tgt - last_close) / last_close if last_close != 0 else 0.0
                analyst_score += float(delta_pct) * 10.0  # scale to comparable range
                analyst_reasons.append(f"Analyst target {tgt:.2f} ({delta_pct*100:+.1f}%)")
            # consensus: buy/hold/sell breakdown -> compute a simple score
            # Finnhub returns count-like fields (strongBuy,buy,hold,sell,strongSell)
            sb = _safe_float(analyst_info.get("strongBuy", 0))
            b = _safe_float(analyst_info.get("buy", 0))
            h = _safe_float(analyst_info.get("hold", 0))
            s = _safe_float(analyst_info.get("sell", 0))
            ss = _safe_float(analyst_info.get("strongSell", 0))
            total = max(1.0, sb + b + h + s + ss)
            cons_score = ((sb + b) - (s + ss)) / total
            analyst_score += cons_score * 2.0
            try:
                analyst_rec_summary = f"B/H/S={int(sb+b)}/{int(h)}/{int(s+ss)}"
                analyst_reasons.append(f"Analyst consensus {analyst_rec_summary}")
            except Exception:
                pass
        except Exception:
            pass

    # Indicator analysis and enhanced recommendation
    try:
        # ensure we have an `edays` variable for downstream recommendation logic
        edays = None
        try:
            from utils.earnings import days_until_next_earnings
            if FINNHUB_API_KEY:
                try:
                    edays = days_until_next_earnings(symbol)
                except Exception:
                    edays = None
            if edays is None and yf is not None:
                try:
                    t = yf.Ticker(symbol)
                    cal = getattr(t, "calendar", None) or {}
                    if isinstance(cal, dict):
                        for v in cal.values():
                            if hasattr(v, "date"):
                                edays = str(v)
                                break
                except Exception:
                    edays = None
        except Exception:
            edays = None
        from utils.indicator_analysis import analyze_indicators
        from utils.recommend import make_recommendation_enhanced
        ind_summary = analyze_indicators(df)
        # Ensure earnings_days is int or None (it may be a string like 'on 2025-08-01')
        edays_val: int | None = None
        if 'edays' in locals():
            try:
                if isinstance(edays, int):
                    edays_val = edays
                elif isinstance(edays, str) and edays.isdigit():
                    edays_val = int(edays)
            except Exception:
                edays_val = None
        tech_verdict, tech_reason = make_recommendation_enhanced(df, last, profile=profile, analyst=analyst_info, news_suggestion=news_suggestion, earnings_days=edays_val)
        # map tech verdict to numeric value for display breakdown
        tech_num = 2.0 if tech_verdict == "Buy" else (-2.0 if tech_verdict == "Sell" else 0.0)
    except Exception:
        # fallback to legacy behavior
        try:
            from utils.recommend import make_recommendation as _make_recommend
            tech_verdict, tech_reason = _make_recommend(last, profile, boll_pct=boll_pct)
            tech_num = 2.0 if tech_verdict == "Buy" else (-2.0 if tech_verdict == "Sell" else 0.0)
            ind_summary = {"trend": None, "rsi": None, "bollinger": {}, "volume": None}
        except Exception:
            tech_verdict, tech_reason = make_recommendation(last, profile)
            tech_num = 0.0
            ind_summary = {"trend": None, "rsi": None, "bollinger": {}, "volume": None}

    # Composite score using tunable weights; include optional news weight
    # Ensure weights are numeric and have safe defaults
    try:
        tech_w = float(tech_w)
    except Exception:
        tech_w = float(st.session_state.ui.get("tech_w", 1.0))
    try:
        analyst_w = float(analyst_w)
    except Exception:
        analyst_w = float(st.session_state.ui.get("analyst_w", 0.7))
    try:
        consensus_w = float(consensus_w)
    except Exception:
        consensus_w = float(st.session_state.ui.get("consensus_w", 0.5))
    try:
        news_w = float(news_w)
    except Exception:
        news_w = float(st.session_state.ui.get("news_w", 0.3))

    composite = float(tech_num) * tech_w + float(analyst_score) * analyst_w
    # incorporate consensus weight as a modifier if we have consensus
    if isinstance(analyst_info, dict) and "recommendation" in analyst_info:
        composite = composite * (1.0 + float(consensus_w) * 0.1)

    # Map composite back to verdict
    if composite >= 2.0:
        final = "Buy"
    elif composite <= -2.0:
        final = "Sell"
    else:
        final = "Hold"

    # Incorporate news weight into composite if available
    news_val_for_composite = 0.0
    if news_suggestion is not None:
        news_val_for_composite = 1.0 if news_suggestion == "Buy" else (-1.0 if news_suggestion == "Sell" else 0.0)
    composite = composite + news_val_for_composite * float(news_w)

    # Enhanced suggestion UI: composite score + breakdown + explainable reasons
    left, right = st.columns([1, 2])
    with left:
        if final == "Buy":
            st.success(f"Suggestion: {final}")
        elif final == "Sell":
            st.error(f"Suggestion: {final}")
        else:
            st.info(f"Suggestion: {final}")

    # show numeric composite and component scores
    with left:
        try:
            st.metric("Composite score", f"{composite:.2f}")
        except Exception:
            st.write(f"Composite score: {composite}")
        st.write(f"Tech: {tech_num:.2f} · Analyst: {analyst_score:.2f} · ConsensusMod: {cons_score:.2f} · News: {news_val_for_composite:.1f} (w={news_w:.1f})")

        # show analyst target & summary if available
        if analyst_target is not None or analyst_rec_summary is not None:
            tgt_text = f"Target: ${analyst_target:.2f}" if analyst_target is not None and np.isfinite(analyst_target) else "Target: —"
            rec_text = f"Analyst: {analyst_rec_summary}" if analyst_rec_summary else "Analyst: —"
            st.caption(f"{tgt_text} · {rec_text}")

    with right:
        st.subheader("Why this suggestion")
        reason_lines = [tech_reason] if tech_reason else []
        # include indicator summary as readable bullets
        try:
            if ind_summary:
                if ind_summary.get("trend"):
                    reason_lines.append(f"Trend detection: {ind_summary.get('trend')}")
                if ind_summary.get("rsi"):
                    reason_lines.append(f"RSI signal: {ind_summary.get('rsi')}")
                bb = ind_summary.get("bollinger") or {}
                if bb.get("label"):
                    reason_lines.append(f"Bollinger: {bb.get('label')} (pct={bb.get('pct'):.2f})" if bb.get("pct") is not None else f"Bollinger: {bb.get('label')}")
                if ind_summary.get("volume"):
                    reason_lines.append(f"Volume signal: {ind_summary.get('volume')}")
        except Exception:
            pass
        reason_lines += analyst_reasons
        if reason_lines:
            with st.expander("Details", expanded=True):
                for r in reason_lines:
                    if r:
                        st.markdown(f"- {r}")
        else:
            st.write("No strong signals detected.")

    # Add a simple component breakdown chart (tech / analyst / news)
    try:
        comp_labels = ["Technical", "Analyst", "News"]
        comp_vals = [float(tech_num), float(analyst_score), float(news_val_for_composite) * float(news_w)]
        # If all component values are essentially zero, skip plotting to avoid a blank chart
        if all((not np.isfinite(v)) or float(v) == 0.0 for v in comp_vals):
            # show a subtle placeholder instead of an empty chart
            st.info("No strong component signals to display.")
        else:
            # normalize to small range for display
            import plotly.graph_objects as _pg
            # color positive contributions green, negative red
            colors = ["#2ca02c" if v > 0 else ("#d62728" if v < 0 else "#6c757d") for v in comp_vals]
            fig_break = _pg.Figure(_pg.Bar(x=comp_labels, y=comp_vals, marker_color=colors))
            fig_break.update_layout(title_text="Signal breakdown", height=220, margin=dict(l=10, r=10, t=30, b=10))
            try:
                from ui.plotting import apply_plotly_theme

                fig_break = apply_plotly_theme(fig_break, title="Signal breakdown", x_title='Component', y_title='Value', dark=True)
            except Exception:
                pass
            safe_plotly_chart(fig_break, use_container_width=True)
    except Exception:
        pass

    # Integrated strategy: combine technicals, analyst targets, news, and holders
    try:
        from data_providers import get_holders

        def integrated_strategy(symbol: str, tech_val: float, analyst_score: float, analyst_target: float | None, news_sugg: str | None, df_prices: pd.DataFrame) -> dict:
            """Return an explainable dict: action, timeframe_days, expected_return_pct, confidence (0-1), reasons[]"""
            reasons: list[str] = []
            # base bias from technicals
            bias = 0.0
            if tech_val > 1:
                bias += 0.5
                reasons.append("Strong technical Buy signal")
            elif tech_val < -1:
                bias -= 0.5
                reasons.append("Strong technical Sell signal")

            # analyst influence
            if analyst_score:
                bias += float(analyst_score) * 0.05
                reasons.append(f"Analyst momentum ({analyst_score:.2f})")

            # news influence
            if news_sugg == "Buy":
                bias += 0.2
                reasons.append("Positive aggregated news")
            elif news_sugg == "Sell":
                bias -= 0.2
                reasons.append("Negative aggregated news")

            # incorporate target gap
            expected_return = None
            if analyst_target and df_prices is not None and not df_prices.empty:
                try:
                    lastp = float(_safe_float(df_prices.iloc[-1].get("close", df_prices.iloc[-1].get("Close", np.nan))))
                    if np.isfinite(lastp) and analyst_target and analyst_target > 0:
                        expected_return = (analyst_target - lastp) / lastp * 100.0
                        reasons.append(f"Analyst target gap {expected_return:+.1f}%")
                except Exception:
                    expected_return = None

            # holders: if institutional ownership high (>40%) that increases confidence for longer horizon
            holders = get_holders(symbol, FMP_API_KEY or "", FINNHUB_API_KEY or "") if (symbol and (FMP_API_KEY or FINNHUB_API_KEY)) else {"institutional": [], "major": [], "source": None}
            inst_pct = 0.0
            try:
                inst_list = holders.get("institutional") if isinstance(holders, dict) else None
                if inst_list:
                    for h in inst_list:
                        if h and h.get("pct"):
                            inst_pct += float(h.get("pct") or 0.0)
                # clamp
                inst_pct = min(inst_pct, 100.0)
                if inst_pct > 40:
                    reasons.append(f"High institutional ownership ({inst_pct:.0f}%) — increases medium-term confidence")
                elif inst_pct < 10 and inst_pct > 0:
                    reasons.append(f"Low institutional ownership ({inst_pct:.0f}%) — may be retail-driven")
            except Exception:
                inst_pct = 0.0

            # volatility: use ATR / price as proxy
            vol_factor = 0.0
            try:
                last_atr = float(_safe_float(df_prices.iloc[-1].get("atr", np.nan))) if df_prices is not None and not df_prices.empty else np.nan
                last_price = float(_safe_float(df_prices.iloc[-1].get("close", np.nan))) if df_prices is not None and not df_prices.empty else np.nan
                if np.isfinite(last_atr) and np.isfinite(last_price) and last_price > 0:
                    vol_pct = last_atr / last_price
                    vol_factor = float(vol_pct)
                    if vol_factor > 0.05:
                        reasons.append(f"High short-term volatility (~{vol_factor*100:.1f}% ATR)")
                else:
                    vol_factor = 0.0
            except Exception:
                vol_factor = 0.0

            # compute confidence as function of data richness and agreement
            conf = 0.2
            # more sources -> higher base
            if analyst_target is not None:
                conf += 0.25
            if holders.get("institutional"):
                conf += 0.15
            if news_sugg is not None:
                conf += 0.1
            # adjust for volatility (high vol reduces recommended position size/confidence)
            conf = max(0.05, min(0.95, conf * (1.0 - min(0.5, vol_factor*2.0))))

            # final action decision
            action = "Hold"
            timeframe_days = 7
            if bias >= 0.6:
                action = "Buy"
                timeframe_days = 30 if inst_pct > 30 else 7
            elif bias <= -0.6:
                action = "Sell"
                timeframe_days = 3

            # expected return: use analyst target if available, otherwise heuristic from bias
            if expected_return is None:
                expected_return = bias * 10.0  # heuristic: bias 0.5 -> 5% expected

            return {
                "action": action,
                "timeframe_days": int(timeframe_days),
                "expected_return_pct": float(expected_return) if expected_return is not None else None,
                "confidence": float(conf),
                "reasons": reasons,
                "holders": holders,
            }

        strat = integrated_strategy(symbol, tech_num, analyst_score, analyst_target, news_suggestion, df)
        # render the integrated strategy
        try:
            st.header("Integrated strategy")
            leftc, rightc = st.columns([2, 3])
            with leftc:
                if strat["action"] == "Buy":
                    st.success(f"Action: {strat['action']} — {strat['expected_return_pct']:+.1f}% expected")
                elif strat["action"] == "Sell":
                    st.error(f"Action: {strat['action']} — expected {strat['expected_return_pct']:+.1f}%")
                else:
                    st.info(f"Action: {strat['action']} — expected {strat['expected_return_pct']:+.1f}%")
                st.metric("Confidence", f"{strat['confidence']*100:.0f}%")
                st.caption(f"Suggested horizon: {strat['timeframe_days']} days")
            with rightc:
                st.subheader("Why")
                for r in strat["reasons"]:
                    st.markdown(f"- {r}")
                # show top holders if present
                if strat.get("holders") and (strat["holders"].get("major") or strat["holders"].get("institutional")):
                    st.subheader("Top holders (best-effort)")
                    rows = []
                    for h in strat["holders"].get("major", [])[:6]:
                        rows.append(f"{h.get('name')} — {h.get('pct') if h.get('pct') is not None else '—'}%")
                    for h in strat["holders"].get("institutional", [])[:6]:
                        rows.append(f"{h.get('name')} — {h.get('pct') if h.get('pct') is not None else '—'}%")
                    if rows:
                        st.write("\n".join(rows))
        except Exception:
            pass
    except Exception:
        pass

    # --- RSI subplot (optional) ---
    if show_rsi and go is not None:
        rfig = go.Figure()
        # ensure numeric arrays to satisfy plotly/pyright
        r_y = np.asarray(df["rsi"]) if "rsi" in df.columns else np.asarray([])
        # produce a plain Python list for Plotly
        r_y_list = r_y.tolist() if hasattr(r_y, "tolist") else list(r_y)
        # ensure x_list exists (fallback to index-derived list)
        if 'x_list' not in locals():
            try:
                xcol = df["datetime"] if "datetime" in df.columns else (df["date"] if "date" in df.columns else df.index)
                x_arr = np.asarray(pd.to_datetime(xcol))
                x_list = [pd.to_datetime(v).to_pydatetime() for v in x_arr]
            except Exception:
                x_list = list(df.index.to_numpy())

        # use the precomputed native Python x_list for Plotly x values
        # normalize into plain Python lists to satisfy plotly and pyright
        from typing import cast as _cast, Any as _Any
        x_plot = list(x_list)
        r_y_plot = list(r_y_list)
        rfig.add_trace(go.Scatter(x=_cast(list[_Any], x_plot), y=_cast(list[_Any], r_y_plot), name="RSI(14)", mode="lines"))  # type: ignore[arg-type]
        rfig.add_hline(y=30, line_dash="dot")
        rfig.add_hline(y=70, line_dash="dot")
        rfig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10),
                   legend=dict(orientation="h", x=0.0, y=1.1),
                   xaxis_title="Date/Time", yaxis_title="RSI")
    try:
        from ui.plotting import apply_plotly_theme

        if rfig is not None:
            rfig = apply_plotly_theme(rfig, title="RSI(14)", x_title="Date/Time", y_title="RSI", dark=True)
    except Exception:
        pass
    safe_plotly_chart(rfig, use_container_width=True)

    # --- Transparent action hint (explainable) ---
    # compute scalar ema values to avoid ambiguous pandas types in comparisons
    ema20_val = float(_safe_float(last.get("ema20", np.nan)))
    ema50_val = float(_safe_float(last.get("ema50", np.nan)))
    bull = (last_close_val > ema20_val > ema50_val)
    bear = (last_close_val < ema20_val < ema50_val)
    if bull:
        st.success("Trend up: price > EMA20 > EMA50 → **Bias: Buy / Swing‑Long**")
    elif bear:
        st.error("Trend down: price < EMA20 < EMA50 → **Bias: Avoid / Short‑biased**")
    else:
        st.info("Mixed signals → **Bias: Neutral / Hold**")

    st.caption("Note: enable Alpaca/Finnhub streaming for intraday updates when you’re ready.")

    # --- News sentiment aggregation and suggestion ---
    try:
        from data_fetchers.news import fetch_news_with_sentiment
        news_df = fetch_news_with_sentiment(NEWS_API_KEY, [symbol]) if NEWS_API_KEY else None
        if news_df is not None and not getattr(news_df, "empty", True):
            # compute mean compound score
            avg = float(news_df["sentiment"].mean())
            pos_pct = float((news_df["sentiment"] > 0.05).sum()) / max(1.0, len(news_df))
            if avg > 0.05 or pos_pct > 0.5:
                news_suggestion = "Buy"
            elif avg < -0.05:
                news_suggestion = "Sell"
            else:
                news_suggestion = "Hold"
            st.subheader("News-based suggestion")
            st.write(f"Aggregate sentiment: {avg:.3f} · Positive headlines: {pos_pct*100:.0f}%")
            st.info(f"Suggestion from news: {news_suggestion}")
        else:
            st.info("No recent news available (set NEWS_API_KEY to enable).")
    except Exception:
        st.info("Unable to fetch news sentiment.")

    # Watchlist prefetch and compact display
    try:
        with st.expander("Watchlist (prefetch quotes)", expanded=False):
            wl = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,GOOG,AMZN")
            tickers = [t.strip().upper() for t in (wl or "").split(',') if t.strip()]
            if tickers:
                quotes = prefetch_quotes(tickers)
                cols = st.columns(len(tickers))
                for i, t in enumerate(tickers):
                    q = quotes.get(t, {}) or {}
                    price = q.get('c') or q.get('current') or '—'
                    cols[i].metric(t, f"{price}")
    except Exception:
        pass

    # SMA backtest quick expander
    try:
        with st.expander("Quick SMA backtest", expanded=False):
            short = st.number_input("Short SMA", value=20, min_value=1)
            long = st.number_input("Long SMA", value=50, min_value=1)
            if st.button("Run backtest"):
                price_series = pd.Series(df['close']) if 'close' in df.columns else pd.Series(dtype=float)
                res = run_sma_backtest(price_series, short=short, long=long)
                st.write(f"Final value: ${res['final_value']:.2f} · Returns: {res['returns_pct']:.2f}% · Trades: {res['trades']} · Wins: {res['wins']}")
                st.line_chart(res['equity_curve'].fillna(method='ffill').tail(200))
                # Optional: record backtest outcome into RL weight updater (opt-in)
                try:
                    from utils.recommender_rl import record_outcome, get_weights
                    from utils.prefs import get_rl_prefs
                    rl_prefs = get_rl_prefs()
                    if rl_prefs.get("enabled", False):
                        if st.checkbox("Apply RL learning from this backtest (update strategy weights)"):
                            # For SMA backtest, the feature vector is technical-only
                            feature_vector = {"tech_w": 1.0, "analyst_w": 0.0, "consensus_w": 0.0, "news_w": 0.0}
                            # reward: use returns_pct as fraction
                            reward = float(res.get("returns_pct", 0.0)) / 100.0
                            new_weights = record_outcome(feature_vector, reward, lr=rl_prefs.get("lr", 0.05))
                            st.success("Updated strategy weights from backtest")
                            st.json(new_weights)
                except Exception:
                    # Non-fatal: continue without RL update
                    pass
                # If the backtest returned per-trade details, show them and allow CSV download
                try:
                    trades = res.get('trades_list') if isinstance(res, dict) else None
                    if trades and isinstance(trades, list):
                        tdf = pd.DataFrame(trades)
                        if not tdf.empty:
                            st.subheader('Trades (entry/exit)')
                            try:
                                display_df(tdf, use_container_width=True)
                            except Exception:
                                st.write(tdf)
                            # CSV download
                            try:
                                csv_bytes = tdf.to_csv(index=False).encode('utf-8')
                                st.download_button('Download trades CSV', data=csv_bytes, file_name=f'{symbol}_sma_trades.csv', mime='text/csv')
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception:
        pass
