"""Market overview UI used by app.py (extracted from pages/05_Streaming.py).

This module provides a callable `render()` function so the top-level app can
include the Market Overview without exposing a separate page under `pages/`.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import os
import sys
import requests
import re
from typing import Dict, Any, Optional
from data_providers import index_snapshots, fmp_gainers, fmp_losers


# Rates fetching implementation is placed at module-level so tests can import
# and exercise fetch_rates() directly. It prefers a runtime FRED API key from
# Streamlit secrets, then a non-empty environment variable. An explicit empty
# env var disables runtime usage and forces the fallback parsing paths.
def _fetch_rates_impl() -> dict:
    out: dict = {"fed_funds": None, "10y": None, "sofr": None}
    try:
        # Determine runtime fred key: prefer st.secrets, then env if non-empty,
        # else fall back to module-level constant (unless runtime was explicitly disabled).
        fred_key: Optional[str] = None
        try:
            fred_key = st.secrets.get("FRED_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            fred_key = None
        disabled_runtime_key = False
        if not fred_key:
            if "FRED_API_KEY" in os.environ:
                env_val = os.environ.get("FRED_API_KEY")
                if env_val:
                    fred_key = env_val
                else:
                    fred_key = None
                    disabled_runtime_key = True
            else:
                fred_key = None
        if not fred_key and not disabled_runtime_key:
            try:
                from config import get_runtime_key
                fred_key = get_runtime_key("FRED_API_KEY")
            except Exception:
                try:
                    from config import FRED_API_KEY as FRED_API_KEY
                    fred_key = FRED_API_KEY or None
                except Exception:
                    fred_key = None

        if fred_key:
            base = "https://api.stlouisfed.org/fred/series/observations"
            for code, k in [("FEDFUNDS", "fed_funds"), ("DGS10", "10y"), ("SOFR", "sofr")]:
                try:
                    r = requests.get(base, params={"series_id": code, "api_key": fred_key, "file_type": "json", "limit": 1}, timeout=5)
                    r.raise_for_status()
                    j = r.json()
                    obs = j.get("observations") or []
                    if obs:
                        out[k] = float(obs[-1].get("value") or float("nan"))
                except Exception:
                    out[k] = None
            return out

        # Fallbacks: Treasury CSV for 10y, NY Fed SOFR CSV, and Fed releases page for fed funds
        try:
            t = requests.get("https://home.treasury.gov/resource-library/data-chart-center/interest-rates/daily-treasury-rates.csv", timeout=5)
            if t.status_code == 200:
                txt = t.text.splitlines()
                hdr = txt[0].split(',') if txt else []

                def _find_10y_index(headers: list[str]) -> Optional[int]:
                    if "DGS10" in headers:
                        return headers.index("DGS10")
                    for i, h in enumerate(headers):
                        if not h:
                            continue
                        hn = h.strip()
                        if ("10" in hn and ("Yr" in hn or "yr" in hn or "Y" in hn)) or hn in ("10 Yr", "10Y", "10y", "10"):
                            return i
                    return None

                idx = _find_10y_index(hdr)
                for ln in reversed(txt):
                    if ln.strip() and not ln.startswith("Date"):
                        parts = ln.split(',')
                        if idx is not None and idx < len(parts):
                            try:
                                out["10y"] = float(parts[idx])
                            except Exception:
                                out["10y"] = None
                        break
        except Exception:
            out["10y"] = None

        try:
            s = requests.get("https://www.newyorkfed.org/medialibrary/media/markets/desk-operations/sofr/sofr.csv", timeout=5)
            if s.status_code == 200:
                lines = s.text.splitlines()
                for ln in reversed(lines):
                    if ln.strip() and not ln.startswith("Date"):
                        val = ln.split(',')[-1]
                        try:
                            out["sofr"] = float(val)
                        except Exception:
                            out["sofr"] = None
                        break
        except Exception:
            out["sofr"] = None

        try:
            r2 = requests.get("https://www.federalreserve.gov/releases/fedfunds.htm", timeout=5)
            if r2.status_code == 200:
                m = re.search(r"(\d+\.\d+)%", r2.text)
                if m:
                    out["fed_funds"] = float(m.group(1))
        except Exception:
            out["fed_funds"] = None
    except Exception:
        pass
    return out


if "pytest" in sys.modules:
    def fetch_rates() -> dict:
        return _fetch_rates_impl()
else:
    @st.cache_data(ttl=300)
    def _fetch_rates_cached() -> dict:
        return _fetch_rates_impl()

    def fetch_rates() -> dict:
        return _fetch_rates_cached()


@st.cache_data(ttl=60)
def load_index_snapshots() -> Dict[str, Dict[str, Any]]:
    try:
        return index_snapshots()
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_movers(api_key: Optional[str]) -> Dict[str, Any]:
    out = {"gainers": [], "losers": []}
    try:
        if api_key:
            out["gainers"] = fmp_gainers(api_key) or []
            out["losers"] = fmp_losers(api_key) or []
        else:
            # Fast fallback: use yfinance to compute movers from a curated US ticker list
            try:
                import yfinance as yf_local
                cur = [
                    "AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","META","NFLX",
                    "AMD","INTC","QCOM","ORCL","ADBE","CRM","PYPL","IBM"
                ]
                rows = []
                for s in cur:
                    try:
                        t = yf_local.Ticker(s)
                        hist = t.history(period="2d", interval="1d")
                        if hist is None or hist.empty:
                            continue
                        last = hist.iloc[-1]
                        prev = hist.iloc[-2] if len(hist) > 1 else last
                        price = float(last.get("Close") or last.get("close") or 0)
                        prevp = float(prev.get("Close") or prev.get("close") or price)
                        pct = (price - prevp) / prevp * 100 if prevp and prevp != 0 else 0
                        rows.append({"symbol": s, "companyName": s, "price": price, "changesPercentage": pct})
                    except Exception:
                        continue
                if rows:
                    dfm = pd.DataFrame(rows).sort_values("changesPercentage", ascending=False)
                    out["gainers"] = dfm.head(10).to_dict("records")
                    out["losers"] = dfm.tail(10).to_dict("records")
            except Exception:
                pass
    except Exception:
        pass
    return out


def small_card(col, name: str, snap: Dict[str, Any]):
    import plotly.express as px
    curr = snap.get("current")
    prev = snap.get("prev")
    delta_val = curr - prev if (curr is not None and prev is not None) else 0.0
    delta_pct = snap.get("delta_pct", ((delta_val / prev) * 100 if prev else 0.0))
    with col:
        st.markdown(f"### {name}")
        # colored delta badge: green for positive, red for negative
        positive = None
        try:
            positive = True if delta_val > 0 else (False if delta_val < 0 else None)
        except Exception:
            positive = None
        from utils.helpers import fmt_number as fmt_number_local

        delta_str = f"{fmt_number_local(delta_val)} ({fmt_number_local(delta_pct)}%)" if curr is not None else None
        if delta_str is not None:
            color = "#2ca02c" if positive is True else ("#d62728" if positive is False else "#6c757d")
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px'><div><strong>{fmt_number_local(curr)}</strong></div><div style='padding:6px 10px;border-radius:6px;background:{color};color:#fff;font-weight:600'>{delta_str}</div></div>", unsafe_allow_html=True)
        else:
            st.metric(label="Price", value=f"{curr:,.2f}" if curr is not None else "", delta=delta_str)
        spark = snap.get("spark")
        if isinstance(spark, pd.DataFrame) and not spark.empty:
            try:
                fig = px.line(spark, x="date", y="close", height=80)
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
                from utils.ui_safe_display import safe_plotly_chart
                safe_plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(spark.set_index("date")["close"])


def build_indices_chart(snaps: Dict[str, Dict[str, Any]]):
    frames = []
    for name, s in snaps.items():
        df = s.get("spark")
        if not (isinstance(df, pd.DataFrame) and not df.empty):
            continue
        # identify a numeric price-like column (common names: close, Close, price)
        price_col = None
        for cand in ("close", "Close", "price", "last", "close_price"):
            if cand in df.columns:
                # quick numeric check: try coercion
                coerced = pd.to_numeric(df[cand], errors="coerce")
                if coerced.notna().any():
                    price_col = cand
                    break
        if price_col is None:
            # no numeric price column found; skip this frame to avoid serialization issues
            continue
        # build a two-column frame with date and numeric price
        try:
            tmp = df[["date", price_col]].copy() if "date" in df.columns else df[[price_col]].copy()
            # rename columns to a consistent shape
            if "date" in tmp.columns:
                tmp.columns = ["date", name]
            else:
                # create a synthetic date index if missing
                tmp = tmp.reset_index()
                if "index" in tmp.columns:
                    tmp = tmp.rename(columns={"index": "date"})
                    tmp.columns = ["date", name]
                else:
                    # final fallback: skip
                    continue
            # ensure the price column is numeric
            tmp[name] = pd.to_numeric(tmp[name], errors="coerce")
            if tmp[name].notna().sum() == 0:
                continue
            frames.append(tmp)
        except Exception:
            # be conservative: skip frames that cause any issue
            continue
    if not frames:
        return None
    out = frames[0]
    for other in frames[1:]:
        out = out.merge(other, on="date", how="outer")
    out = out.sort_values("date").set_index("date")
    return out


def render_movers_table(movers: list, title: str):
    if not movers:
        st.info(f"{title} unavailable (no data or API access).")
        return
    rows = []
    for it in movers[:25]:
        sym = it.get("ticker") or it.get("symbol") or it.get("tickerSymbol") or it.get("symbolInput") or ""
        name = it.get("companyName") or it.get("name") or it.get("company") or ""
        price = it.get("price") or it.get("lastPrice") or it.get("previousClose") or it.get("priceChanges") or None
        change = it.get("changes") or it.get("change") or it.get("difference") or None
        pct = it.get("changesPercentage") or it.get("changePercent") or None
        rows.append({"symbol": sym, "name": name, "price": price, "change": change, "pct": pct})
    df = pd.DataFrame(rows)
    def fmt(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return x
    if not df.empty:
        if "price" in df.columns:
            df["price_raw"] = df["price"].copy()
            df["price"] = df["price"].apply(fmt)
        if "change" in df.columns:
            df["change_raw"] = df["change"].copy()
            df["change"] = df["change"].apply(fmt)
        # Format pct and add colored HTML for up/down
        def color_pct(val):
            try:
                f = float(val)
                col = "#2ca02c" if f > 0 else ("#d62728" if f < 0 else "#6c757d")
                arrow = "▲" if f > 0 else ("▼" if f < 0 else "–")
                return f"<div style='color:{col};font-weight:600'>{arrow} {f:,.2f}%</div>"
            except Exception:
                return val

        if "pct" in df.columns:
            df["pct_display"] = df["pct"].apply(lambda x: color_pct(x if x is not None else 0))

        # Present as a compact HTML table so colors render inline with the text
        html_rows = []
        for _, r in df.iterrows():
            html_rows.append(f"<tr><td style='padding:6px 12px;font-weight:700'>{r.get('symbol','')}</td><td style='padding:6px 12px'>{r.get('name','')}</td><td style='padding:6px 12px;text-align:right'>{r.get('price','')}</td><td style='padding:6px 12px;text-align:right'>{r.get('change','')}</td><td style='padding:6px 12px;text-align:right'>{r.get('pct_display','')}</td></tr>")
        table = "<table style='width:100%;border-collapse:collapse'>"
        table += "<thead><tr><th style='text-align:left'>Symbol</th><th style='text-align:left'>Name</th><th style='text-align:right'>Price (USD)</th><th style='text-align:right'>Change</th><th style='text-align:right'>Change %</th></tr></thead>"
        table += "<tbody>" + "".join(html_rows) + "</tbody></table>"
        st.markdown(table, unsafe_allow_html=True)
        st.caption("Note: Prices quoted in USD when available. Filtered to US-listed securities.")


def render():
    st.set_page_config(page_title="Market Overview", layout="wide")
    st.title("Market Overview 📈")
    st.write("Live indices and daily market movers. Data refreshes every minute (cached).")

    # Resolve provider API keys at runtime (avoid import-time constants)
    try:
        from config import get_runtime_key
        fmp_key = get_runtime_key("FMP_API_KEY")
        fh_key = get_runtime_key("FINNHUB_API_KEY")
    except Exception:
        # fallback: try env or None
        import os

        fmp_key = os.environ.get("FMP_API_KEY") or None
        fh_key = os.environ.get("FINNHUB_API_KEY") or None
    # Keep local prefs but don't expose a selection to the user.

    # Provider presence banner (non-sensitive): show which external providers will be used
    try:
        providers = []
        if fmp_key:
            providers.append("FMP")
        if fh_key:
            providers.append("Finnhub")
        # yfinance availability
        try:
            import yfinance as _yft
            providers.append("yfinance")
        except Exception:
            pass

        if providers:
            st.info("Data providers enabled: " + ", ".join(providers))
        else:
            st.info("No external data providers configured; app will use local fallbacks where possible.")
    except Exception:
        # Non-critical: don't block rendering
        pass

    # Refresh control
    if st.button("Refresh now", key="market_refresh"):
        # clear our cached data functions
        try:
            load_index_snapshots.clear()
        except Exception:
            pass
        try:
            load_movers.clear()
        except Exception:
            pass
        # attempt to rerun the app to reflect cleared cache
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):
            try:
                rerun()
            except Exception:
                pass

    # Add quick rates panel: FED funds (effective), 10y Treasury, and SOFR.
    rates = fetch_rates()

    rates = fetch_rates()

    snaps = load_index_snapshots()
    if not snaps:
        st.warning("Index snapshots are unavailable — ensure `yfinance` is installed and internet access is available.")
        return

    cols = st.columns(len(snaps))
    for c, (name, snap) in zip(cols, snaps.items()):
        small_card(c, name, snap)

    st.markdown("---")
    # Provider status banners (non-sensitive). Probe FMP & Finnhub quickly to surface 401/403 for movers/news.
    if fmp_key:
        try:
            r = requests.get(f"https://financialmodelingprep.com/api/v3/profile/AAPL?apikey={fmp_key}", timeout=5)
            if r.status_code != 200:
                st.warning(f"FMP probe: {r.status_code} — {r.text[:200]}")
        except Exception as _e:
            st.warning(f"FMP probe failed: {_e}")
    else:
        st.info("FMP key not configured — movers will be unavailable unless Mock mode or keys are set.")

    # Finnhub probe
    if fh_key:
        try:
            fr = requests.get(f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={fh_key}", timeout=5)
            if fr.status_code != 200:
                st.warning(f"Finnhub probe: {fr.status_code}  {fr.text[:200]}")
        except Exception as _e:
            st.warning(f"Finnhub probe failed: {_e}")
    else:
        st.info("Finnhub key not configured — news/earnings features may be limited.")
    # Compact rates / macro panel
    try:
        with st.container():
            st.markdown("### Market Snapshot & Policy Rates")
            c1, c2, c3, c4 = st.columns([2,1,1,1])
            c1.write("\n")
            fed_val = rates.get("fed_funds")
            ten_val = rates.get("10y")
            sofr_val = rates.get("sofr")
            from utils.helpers import fmt_number as fmt_number_local

            fed_low = rates.get("fed_funds_target_low")
            fed_high = rates.get("fed_funds_target_high")
            if fed_low is not None and fed_high is not None:
                try:
                    c2.metric("Federal funds target (%)", f"{fmt_number_local(fed_low,2)} – {fmt_number_local(fed_high,2)}")
                except Exception:
                    c2.metric("Federal funds target (%)", f"{fed_low} – {fed_high}")
            else:
                c2.metric("Effective Fed Funds (%)", fmt_number_local(fed_val, 2))
            c3.metric("10y Treasury (%)", fmt_number_local(ten_val, 2))
            c4.metric("SOFR (%)", fmt_number_local(sofr_val, 3))
    except Exception:
        pass

    # Indices chart removed: users found it not useful and it added load time.
    # We keep index snapshots available as small cards above and skip the heavy
    # combined chart to improve responsiveness.

    st.markdown("---")
    st.subheader("Daily market movers")
    movers = load_movers(fmp_key)
    left, right = st.columns(2)
    with left:
        st.markdown("### Top Gainers")
        render_movers_table(movers.get("gainers", []), "Top gainers")
    with right:
        st.markdown("### Top Losers")
        render_movers_table(movers.get("losers", []), "Top losers")
