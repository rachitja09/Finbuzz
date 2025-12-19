# pages/04_Compare.py
from __future__ import annotations
import requests
import pandas as pd
import numpy as np
from typing import Optional
from utils.frames import sanitize_for_arrow
from utils.ui_safe_display import display_df, run_safe, safe_plotly_chart
from utils.charts import candle_with_rsi_bbands, analyst_target_summary
from utils.helpers import _safe_float, fmt_money, millify_short
import sys
from utils.prefs import get_data_source_pref
import streamlit as st
try:
    from config import get_runtime_key
    FMP_KEY = get_runtime_key("FMP_API_KEY")
    FINNHUB_KEY = get_runtime_key("FINNHUB_API_KEY")
except Exception:
    try:
        from config import FMP_API_KEY as FMP_KEY, FINNHUB_API_KEY as FINNHUB_KEY
    except Exception:
        FMP_KEY = None
        FINNHUB_KEY = None
import os as _os
_DEV_MOCK = bool(_os.getenv("DEV_MOCK", "0") == "1")

# Lazy import mock providers only when needed
def _use_mock(): 
    return _DEV_MOCK

# If API keys are not provided via config/st.secrets, allow the user to enter
# them interactively in an expander. Inputs are stored in Streamlit session
# state for the duration of the session and assigned to the module-level
# names so the cached fetchers below can use them.
if not (FMP_KEY and FINNHUB_KEY) and "pytest" not in sys.modules:
    # local import to avoid module-level side effects in tests
    from config import write_local_secrets

    with st.expander("Enter API keys (optional)", expanded=False):
        fmp_input = st.text_input("FinancialModelingPrep API key", value=FMP_KEY or "", type="password", key="fmp_api_key")
        finnhub_input = st.text_input("Finnhub API key", value=FINNHUB_KEY or "", type="password", key="finnhub_api_key")
        save = st.button("Save keys locally (.streamlit/secrets.toml)", key="save_local_keys")
        # If the user typed values, assign them to the module globals so
        # the cached fetchers (which read FMP_KEY/FINNHUB_KEY at call time)
        # will use the runtime keys.
        if fmp_input:
            FMP_KEY = fmp_input
        if finnhub_input:
            FINNHUB_KEY = finnhub_input
        if save:
            data = {}
            if fmp_input:
                data["FMP_API_KEY"] = fmp_input
            if finnhub_input:
                data["FINNHUB_API_KEY"] = finnhub_input
            if data:
                try:
                    write_local_secrets(data)
                    st.success("Saved keys to .streamlit/secrets.toml (local)")
                except Exception as e:
                    st.error(f"Failed to save secrets: {e}")
# Optional imports: these are heavy and not required for test-time imports. Use
# guarded imports and provide lightweight fallbacks so the module can be imported
# during CI/tests without the full visualization stack.
try:
    import yfinance as yf
except Exception:  # pragma: no cover - environment dependent
    yf = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - environment dependent
    go = None
    from typing import Any
    def make_subplots(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("plotly is required to build charts")

# Optional auto-refresh every 60s
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, limit=None, key="compare_autorefresh")
except Exception:
    pass

# Data source selector: Auto prefers live providers but falls back to free sources/mocks
_pref_mode = get_data_source_pref()
data_mode = st.radio("Data source", options=["Auto (live first)", "Live (require keys)", "Mock only"], index=["Auto (live first)", "Live (require keys)", "Mock only"].index(_pref_mode), horizontal=True,
                    help="Auto: try live APIs first, then free/yfinance, then mock; Live: require keys; Mock: use synthetic data")

st.set_page_config(page_title="Compare Fundamentals", layout="wide")
st.title("🧭 Compare Fundamentals")
# Non-sensitive debug banner: show whether required API keys are present (booleans only)
try:
    from ui.provider_banner import show_provider_banner
    show_provider_banner()
except Exception:
    # If Streamlit not running in normal mode, skip the banner
    pass
    
if "pytest" not in sys.modules:
    # If keys are present, perform a tiny, non-sensitive probe to show HTTP status
    # (this helps surface 401/403 from providers instead of silently returning empty data)
    try:
        import requests as _req

        def _show_provider_issue(name: str, status: int, short_msg: str):
            if status == 403:
                st.warning(f"{name} probe: 403 — Access denied. Your API key appears to lack the required permissions or is for a legacy/retired plan.\nSuggested actions: add a current key in the Expander above, confirm your plan includes this endpoint, or switch Data source to 'Mock only'.")
            elif status == 401:
                st.warning(f"{name} probe: 401 — Unauthorized. Check that your key is correct and not expired.")
            else:
                st.warning(f"{name} probe: {status} — {short_msg}")

        if FMP_KEY:
            try:
                # Probe a modern v4 endpoint (company-profile v4) instead of legacy v3 endpoints
                r = _req.get(f"https://financialmodelingprep.com/api/v4/company-profile?symbol=AAPL&apikey={FMP_KEY}", timeout=6)
                if r.status_code != 200:
                    _show_provider_issue("FMP", r.status_code, r.text[:200])
            except Exception as _e:  # pragma: no cover - network
                st.warning(f"FMP probe failed: {_e}")

        if FINNHUB_KEY:
            try:
                # Use the lightweight quote endpoint for a benign permission check.
                # Some Finnhub endpoints (e.g. price-target) require elevated access
                # and will return 403 for users without that plan; prefer a simpler
                # probe and give actionable guidance when 403 is returned.
                r2 = _req.get(f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={FINNHUB_KEY}", timeout=6)
                if r2.status_code != 200:
                    _show_provider_issue("Finnhub", r2.status_code, r2.text[:200])
            except Exception as _e:  # pragma: no cover - network
                st.warning(f"Finnhub probe failed: {_e}")
    except Exception:
        # If requests isn't available or Streamlit isn't context, skip probes
        pass

# -------- Utils ----------------------------------------------------------------

def fmt2(x):
    return "—" if not isinstance(x, (int, float)) or not np.isfinite(x) else f"{x:.2f}"

def get_json(url: str, timeout=30):
    try:
        from utils.http_cache import cached_get
        status, text, j = cached_get(url, params=None, ttl=300)
        if status and int(status) == 200:
            return j
        return None
    except Exception:
        # fallback to plain requests if http_cache fails
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

# -------- Cached fetchers ------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fmp_profile(sym: str) -> dict:
    if not FMP_KEY:
        return {}
    # Try v4 company-profile endpoint first
    url_v4 = f"https://financialmodelingprep.com/api/v4/company-profile?symbol={sym}&apikey={FMP_KEY}"
    data = get_json(url_v4) or []
    if isinstance(data, list) and data:
        d = data[0]
        # v4 returns a nested dict under 'profile'
        prof = d.get('profile', d)
        return {
            "symbol": prof.get("symbol", sym),
            "companyName": prof.get("companyName") or prof.get("name") or sym,
            "sector": prof.get("sector", ""),
            "industry": prof.get("industry", ""),
            "price": prof.get("price"),
            "mktCap": prof.get("mktCap") or prof.get("marketCap"),
            "beta": prof.get("beta"),
        }
    # Fallback to v3
    url_v3 = f"https://financialmodelingprep.com/api/v3/profile/{sym}?apikey={FMP_KEY}"
    data = get_json(url_v3) or []
    if isinstance(data, list) and data:
        return data[0]
    # Fallback to yfinance
    try:
        if yf is not None:
            ticker_cls = getattr(yf, "Ticker", None)
            if ticker_cls is not None:
                info = ticker_cls(sym).info or {}
                return {
                    "symbol": sym,
                    "companyName": info.get("longName") or info.get("shortName") or sym,
                    "sector": info.get("sector") or "",
                    "industry": info.get("industry") or "",
                    "price": info.get("regularMarketPrice") or info.get("previousClose"),
                    "mktCap": info.get("marketCap"),
                    "beta": info.get("beta"),
                }
    except Exception:
        pass
    return {}

@st.cache_data(ttl=900, show_spinner=False)
def fmp_ratios_ttm(sym: str) -> dict:
    if not FMP_KEY:
        return {}
    # Try v4 key-metrics-ttm endpoint first
    url_v4 = f"https://financialmodelingprep.com/api/v4/key-metrics-ttm?symbol={sym}&apikey={FMP_KEY}"
    data = get_json(url_v4) or []
    if isinstance(data, list) and data:
        d = data[0]
        # v4 returns a nested dict under 'metrics'
        metrics = d.get('metrics', d)
        # Map v4 fields to v3 names for compatibility
        return {
            "peRatioTTM": metrics.get("peRatioTTM") or metrics.get("peRatio"),
            "priceToBookRatioTTM": metrics.get("priceToBookRatioTTM") or metrics.get("pbRatio"),
            "netProfitMarginTTM": metrics.get("netProfitMarginTTM") or metrics.get("netProfitMargin"),
            "returnOnEquityTTM": metrics.get("returnOnEquityTTM") or metrics.get("roe"),
            "returnOnAssetsTTM": metrics.get("returnOnAssetsTTM") or metrics.get("roa"),
            "currentRatioTTM": metrics.get("currentRatioTTM") or metrics.get("currentRatio"),
            "debtEquityRatioTTM": metrics.get("debtEquityRatioTTM") or metrics.get("debtToEquity"),
            "freeCashFlowPerShareTTM": metrics.get("freeCashFlowPerShareTTM") or metrics.get("fcfPerShare"),
        }
    # Fallback to v3
    url_v3 = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{sym}?apikey={FMP_KEY}"
    data = get_json(url_v3) or []
    return data[0] if isinstance(data, list) and data else {}

@st.cache_data(ttl=3600, show_spinner=False)
def fmp_sector_pe() -> list[dict]:
    if not FMP_KEY:
        return []
    # Use documented endpoint name: sector_price_earning_ratio (no trailing 's')
    url = f"https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?apikey={FMP_KEY}"
    data = get_json(url)
    return data if isinstance(data, list) else []

@st.cache_data(ttl=3600, show_spinner=False)
def fmp_industry_pe() -> list[dict]:
    if not FMP_KEY:
        return []
    url = f"https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?apikey={FMP_KEY}"
    data = get_json(url)
    return data if isinstance(data, list) else []

@st.cache_data(ttl=900, show_spinner=False)
def finnhub_price_target(sym: str) -> dict:
    if not FINNHUB_KEY:
        return {}
    url = f"https://finnhub.io/api/v1/stock/price-target?symbol={sym}&token={FINNHUB_KEY}"
    data = get_json(url)
    return data if isinstance(data, dict) else {}

@st.cache_data(ttl=900, show_spinner=False)
def finnhub_reco(sym: str) -> dict:
    if not FINNHUB_KEY:
        return {}
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={sym}&token={FINNHUB_KEY}"
    data = get_json(url) or []
    return data[0] if isinstance(data, list) and data else {}

@st.cache_data(ttl=900, show_spinner=False)
def hist_ohlc(sym: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        ticker_cls = getattr(yf, "Ticker", None)
        if ticker_cls is None:
            return pd.DataFrame()
        df = ticker_cls(sym).history(period=period, interval=interval, auto_adjust=False)
        return sanitize_for_arrow(df) if df is not None and not getattr(df, "empty", False) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# Helper to prefer mocks when live fetchers return no data or DEV_MOCK=1
def _maybe_mock_profile(sym: str, live: dict) -> dict:
    if live:
        return live
    try:
        if _use_mock():
            from utils.mock_providers import profile as _mprof
            return _mprof(sym)
    except Exception:
        pass
    return {}

def _maybe_mock_ratios(sym: str, live: dict) -> dict:
    if live:
        return live
    try:
        if _use_mock():
            from utils.mock_providers import ratios_ttm as _mr
            return _mr(sym)
    except Exception:
        pass
    return {}

def _maybe_mock_ohlc(sym: str, live_df) -> pd.DataFrame:
    if live_df is not None and not getattr(live_df, "empty", True):
        return live_df
    try:
        if _use_mock():
            from utils.mock_providers import ohlc as _mohlc
            return _mohlc(sym)
    except Exception:
        pass
    return pd.DataFrame()

def _maybe_mock_target(sym: str, live: dict) -> dict:
    if live:
        return live
    try:
        if _use_mock():
            from utils.mock_providers import finnhub_price_target as _mt
            return _mt(sym)
    except Exception:
        pass
    return {}

def _maybe_mock_reco(sym: str, live: dict) -> dict:
    if live:
        return live
    try:
        if _use_mock():
            from utils.mock_providers import finnhub_reco as _mr
            return _mr(sym)
    except Exception:
        pass
    return {}

# -------- Indicators & charts --------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return pd.Series(out, index=series.index, dtype=float)

def tiny_candle_with_rsi(df: pd.DataFrame, title: str):
    # guard heavy optional import; return None when plotly isn't available so callers can
    # render a graceful textual fallback instead of crashing at import-time
    try:
        from plotly.subplots import make_subplots as _make_subplots
    except Exception:
        return None

    fig = _make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3])
    # add candlestick using simple numpy arrays to avoid ExtensionArray typing issues
    fig.add_candlestick(x=df.index.to_numpy(), open=df["Open"].to_numpy(), high=df["High"].to_numpy(), low=df["Low"].to_numpy(), close=df["Close"].to_numpy(),
                        row=1, col=1, name="Price")
    close_s = pd.Series(df["Close"], index=df.index, dtype=float)
    r = rsi(close_s)
    fig.add_scatter(x=df.index.to_numpy(), y=r.to_numpy(), mode="lines", name="RSI(14)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot")
    fig.add_hline(y=70, line_dash="dot")
    fig.update_layout(
        title=title, height=420, margin=dict(l=10, r=10, t=40, b=10), showlegend=True,
        xaxis_title="Date", yaxis_title="Price", yaxis2_title="RSI"
    )
    return fig

# -------- UI -------------------------------------------------------------------
left, right = st.columns(2)
sym1 = left.text_input("Ticker 1", "AAPL", key="compare_sym1").strip().upper()
sym2 = right.text_input("Ticker 2", "MSFT", key="compare_sym2").strip().upper()
go_btn = st.button("Compare", type="primary", key="compare_go")

def kpi_block(col, sym: str, prof: dict):
    price = _safe_float(prof.get("price"))
    beta = _safe_float(prof.get("beta"))
    mcap = _safe_float(prof.get("mktCap"))
    name = prof.get("companyName") or sym
    with col:
        st.caption(f"**{sym} — {name}**")
        st.metric("Price", fmt_money(price))
        c1, c2 = st.columns(2)
        # present market cap in human-friendly compact form with suffix
        try:
            cap_text = ("$" + millify_short(mcap, places=2)) if mcap and mcap == mcap else "—"
        except Exception:
            cap_text = "—"
        c1.metric("Market Cap", cap_text)
        c2.metric("Beta", f"{beta:.3f}" if np.isfinite(beta) else "—")

def ratio_frame(r1: dict, r2: dict, sym1: str, sym2: str) -> pd.DataFrame:
    fields = {
        "peRatioTTM": "P/E (TTM)",
        "priceToBookRatioTTM": "P/B",
        "netProfitMarginTTM": "Net Margin",
        "returnOnEquityTTM": "ROE",
        "returnOnAssetsTTM": "ROA",
        "currentRatioTTM": "Current Ratio",
        "debtEquityRatioTTM": "Debt/Equity",
        "freeCashFlowPerShareTTM": "FCF/Share",
    }
    rows = []
    for k, label in fields.items():
        rows.append({"Metric": label, sym1: _safe_float(r1.get(k)), sym2: _safe_float(r2.get(k))})
    df = pd.DataFrame(rows)
    # Relative score per row
    lower_better = {"P/E (TTM)", "P/B", "Debt/Equity"}
    for col in [sym1, sym2]:
        vals = pd.to_numeric(df[col], errors="coerce")
        # ensure we have a pandas Series so attribute methods exist for the type checker
        vals_s = pd.Series(vals, index=df.index, dtype=float)
        arr = vals_s.to_numpy()
        # use nan-aware extrema and guard zero-range
        lo = float(np.nanmin(arr)) if arr.size else float("nan")
        hi = float(np.nanmax(arr)) if arr.size else float("nan")
        rng = (hi - lo) if np.isfinite(hi - lo) and (hi - lo) != 0 else 1.0
        normv = (vals_s - lo) / rng
        df[f"{col}_score"] = [1 - x if m in lower_better else x for x, m in zip(normv.fillna(0).to_numpy(), df["Metric"]) ]
    return df

def resolve_benchmarks(prof: dict, sector_pe: list[dict], industry_pe: list[dict]) -> tuple[float, float]:
    sec_name = norm(prof.get("sector"))
    ind_name = norm(prof.get("industry"))
    sec_val = next((_safe_float(x.get("pe")) for x in sector_pe if norm(x.get("sector")) == sec_name), float("nan"))
    ind_val = next((_safe_float(x.get("pe")) for x in industry_pe if norm(x.get("industry")) == ind_name), float("nan"))
    return (sec_val, ind_val)

if go_btn and sym1 and sym2:
    # Enforce Live mode when selected
    if data_mode == "Live (require keys)" and not (FMP_KEY and FINNHUB_KEY):
        st.error("🔑 Live data mode selected — add API keys to `.streamlit/secrets.toml` (FMP + Finnhub).")
        st.stop()

    with st.spinner("Fetching fundamentals, benchmarks, and history…"):
        # Live fetchers (may return empty dicts/DataFrames on error)
        prof1_live, prof2_live = fmp_profile(sym1), fmp_profile(sym2)
        rat1_live, rat2_live = fmp_ratios_ttm(sym1), fmp_ratios_ttm(sym2)
        sector_pe_list, industry_pe_list = fmp_sector_pe(), fmp_industry_pe()
        tgt1_live, tgt2_live = finnhub_price_target(sym1), finnhub_price_target(sym2)
        rec1_live, rec2_live = finnhub_reco(sym1), finnhub_reco(sym2)
        ohlc1_live, ohlc2_live = hist_ohlc(sym1), hist_ohlc(sym2)

        # Resolve according to data_mode
        if data_mode == "Mock only":
                from utils.mock_providers import profile as _mp
                from utils.mock_providers import ratios_ttm as _mr
                from utils.mock_providers import ohlc as _mh
                from utils.mock_providers import finnhub_price_target as _mt
                from utils.mock_providers import finnhub_reco as _mrec
                prof1 = _mp(sym1)
                prof2 = _mp(sym2)
                rat1 = _mr(sym1)
                rat2 = _mr(sym2)
                tgt1 = _mt(sym1)
                tgt2 = _mt(sym2)
                rec1 = _mrec(sym1)
                rec2 = _mrec(sym2)
                ohlc1 = _mh(sym1)
                ohlc2 = _mh(sym2)
        else:
            # Auto: prefer live, then yfinance fallback, then mock when enabled
            prof1 = _maybe_mock_profile(sym1, prof1_live)
            prof2 = _maybe_mock_profile(sym2, prof2_live)
            rat1 = _maybe_mock_ratios(sym1, rat1_live)
            rat2 = _maybe_mock_ratios(sym2, rat2_live)
            tgt1 = _maybe_mock_target(sym1, tgt1_live)
            tgt2 = _maybe_mock_target(sym2, tgt2_live)
            rec1 = _maybe_mock_reco(sym1, rec1_live)
            rec2 = _maybe_mock_reco(sym2, rec2_live)
            ohlc1 = _maybe_mock_ohlc(sym1, ohlc1_live)
            ohlc2 = _maybe_mock_ohlc(sym2, ohlc2_live)

    # Benchmarks & deltas
    pe1 = _safe_float(rat1.get("peRatioTTM", prof1.get("pe")))
    pe2 = _safe_float(rat2.get("peRatioTTM", prof2.get("pe")))
    sec1, ind1 = resolve_benchmarks(prof1, sector_pe_list, industry_pe_list)
    sec2, ind2 = resolve_benchmarks(prof2, sector_pe_list, industry_pe_list)
    delta1_sec, delta1_ind = pe1 - sec1, pe1 - ind1
    delta2_sec, delta2_ind = pe2 - sec2, pe2 - ind2

    # ---- Snapshot KPIs
    st.subheader("Snapshot")
    k1, k2 = st.columns(2)
    kpi_block(k1, sym1, prof1)
    with k1:
        st.metric("P/E vs Sector", f"{delta1_sec:+.2f}" if np.isfinite(delta1_sec) else "—", help=f"Sector P/E: {fmt2(sec1)}")
        st.metric("P/E vs Industry", f"{delta1_ind:+.2f}" if np.isfinite(delta1_ind) else "—", help=f"Industry P/E: {fmt2(ind1)}")
    kpi_block(k2, sym2, prof2)
    with k2:
        st.metric("P/E vs Sector", f"{delta2_sec:+.2f}" if np.isfinite(delta2_sec) else "—", help=f"Sector P/E: {fmt2(sec2)}")
        st.metric("P/E vs Industry", f"{delta2_ind:+.2f}" if np.isfinite(delta2_ind) else "—", help=f"Industry P/E: {fmt2(ind2)}")

    # ---- Tabs
    tab_rat, tab_trg, tab_ch, tab_prof = st.tabs(["📊 Ratios", "🎯 Targets & Ratings", "📈 Chart", "ℹ️ Profile"])

    with tab_rat:
        df = ratio_frame(rat1, rat2, sym1, sym2)
        extra = pd.DataFrame([
            {"Metric": "Sector P/E",        sym1: sec1,        sym2: sec2},
            {"Metric": "Industry P/E",      sym1: ind1,        sym2: ind2},
            {"Metric": "Δ P/E vs Sector",   sym1: delta1_sec,  sym2: delta2_sec},
            {"Metric": "Δ P/E vs Industry", sym1: delta1_ind,  sym2: delta2_ind},
        ])
        df = pd.concat([df, extra], ignore_index=True)

        # Ensure numeric columns are numeric to avoid pyarrow conversion issues
        for c in [sym1, sym2, f"{sym1}_score", f"{sym2}_score"]:
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass
        display_df(
            df[["Metric", sym1, sym2, f"{sym1}_score", f"{sym2}_score"]],
            hide_index=True, use_container_width=True,
            column_config={
                sym1: st.column_config.NumberColumn(format="%.3f"),
                sym2: st.column_config.NumberColumn(format="%.3f"),
                f"{sym1}_score": st.column_config.ProgressColumn(f"{sym1} Rel.", min_value=0.0, max_value=1.0, format="%.0f%%"),
                f"{sym2}_score": st.column_config.ProgressColumn(f"{sym2} Rel.", min_value=0.0, max_value=1.0, format="%.0f%%"),
            },
        )
        st.caption("Row‑wise relative bars (higher = better). P/E, P/B, D/E are inverted (lower is better).")

    with tab_trg:
        cA, cB = st.columns(2)

        def target_card(col, sym, tgt, cur_price):
            cur = _safe_float(cur_price)
            mean_t = _safe_float(tgt.get("targetMean"))
            high_t = _safe_float(tgt.get("targetHigh"))
            low_t  = _safe_float(tgt.get("targetLow"))

            # Reasonable gauge bounds
            lo = low_t if np.isfinite(low_t) else (0.7 * cur if np.isfinite(cur) else 0.0)
            hi = high_t if np.isfinite(high_t) else (1.3 * cur if np.isfinite(cur) else 1.0)

            # Avoid Infinity% when reference is 0/NaN
            delta_kwargs = None
            if np.isfinite(mean_t) and mean_t != 0:
                delta_kwargs = dict(reference=mean_t, relative=True, valueformat=".2%")
            # else: omit delta entirely

            Indicator = getattr(go, "Indicator", None)
            Figure = getattr(go, "Figure", None)
            if Indicator is not None and Figure is not None:
                ind = Indicator(
                    mode="gauge+number" + ("+delta" if delta_kwargs else ""),
                    value=cur if np.isfinite(cur) else 0,
                    number={"prefix": "$"},
                    gauge={"axis": {"range": [lo, hi]}},
                    title={"text": f"{sym} — Current vs Mean Target"},
                    delta=delta_kwargs
                )
                fig = Figure(ind)
                col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                # fallback textual summary when plotly not available
                if np.isfinite(mean_t):
                    col.write(f"Mean: **${mean_t:.2f}** • High: **{fmt2(high_t)}** • Low: **{fmt2(low_t)}**")
                else:
                    col.info("No target data")
            if np.isfinite(mean_t):
                col.write(f"Mean: **${mean_t:.2f}** • High: **{fmt2(high_t)}** • Low: **{fmt2(low_t)}**")
            else:
                col.info("No target data")

    price1 = _safe_float(prof1.get("price"))
    price2 = _safe_float(prof2.get("price"))
    target_card(cA, sym1, tgt1, price1)
    target_card(cB, sym2, tgt2, price2)

    # analyst suggestion summary below cards
    s1 = analyst_target_summary(tgt1 or {}, current_price=price1)
    s2 = analyst_target_summary(tgt2 or {}, current_price=price2)
    scol1, scol2 = st.columns(2)
    with scol1:
        st.metric(f"{sym1} Suggestion", s1.get("verdict", "—"))
    with scol2:
        st.metric(f"{sym2} Suggestion", s2.get("verdict", "—"))

    def reco_frame(reco: dict, label: str):
        sb = int(reco.get("strongBuy", 0))
        b = int(reco.get("buy", 0))
        h = int(reco.get("hold", 0))
        s = int(reco.get("sell", 0))
        ss = int(reco.get("strongSell", 0))
        total = max(sb + b + h + s + ss, 1)
        # ProgressColumn wants 0..100 if you want percent labels
        return pd.DataFrame({
            "Bucket": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
            label: [100 * sb / total, 100 * b / total, 100 * h / total, 100 * s / total, 100 * ss / total],
        })

    rec_df = reco_frame(rec1, sym1).merge(reco_frame(rec2, sym2), on="Bucket")
    display_df(
        rec_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            sym1: st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%"),
            sym2: st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%"),
        },
    )

    with tab_ch:
        cS1, cS2 = st.columns(2)
        from utils.frames import downsample_time_index
        for sym, dfc, col in [(sym1, ohlc1, cS1), (sym2, ohlc2, cS2)]:
            if not dfc.empty:
                try:
                    plot_df = downsample_time_index(dfc.tail(2000), max_points=800)
                    fig = candle_with_rsi_bbands(plot_df.rename(columns={c: c.capitalize() for c in plot_df.columns}), f"{sym} · 3M")
                    safe_plotly_chart(fig, target_container=col, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    fig = tiny_candle_with_rsi(dfc, f"{sym} · 3M")
                    if fig is not None:
                        safe_plotly_chart(fig, target_container=col, use_container_width=True, config={"displayModeBar": False})
                    else:
                        col.info("Plotly is not installed; install plotly to see charts, or use the textual summary below.")
            else:
                col.info(f"No OHLC for {sym}")

    with tab_prof:
        def profile_table(sym, prof):
            fields = ["companyName","symbol","sector","industry","price","mktCap","beta","volAvg","range","exchangeShortName"]
            return pd.DataFrame([{"Metric": f, sym: prof.get(f)} for f in fields])

    merged = pd.merge(profile_table(sym1, prof1), profile_table(sym2, prof2), on="Metric", how="outer")
    # For profile display, cast everything to strings and fill missing values so
    # Streamlit/pyarrow doesn't attempt to coerce mixed-type columns to numeric
    # types (which causes Arrow conversion errors when e.g. a cell contains
    # 'Apple Inc.' but other rows are numeric).
    df_to_show = merged.fillna("—").astype(str)

    display_df(df_to_show, hide_index=True, use_container_width=True)

else:
    st.info("Enter two tickers and press **Compare**.")
