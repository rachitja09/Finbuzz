import streamlit as st
import sys
import pandas as pd
import numpy as np
from utils.helpers import _safe_float
from utils.storage import save_portfolio, load_portfolio
from utils.earnings import days_until_next_earnings, get_vix
from utils.style import theme_toggle

st.set_page_config(page_title="Portfolio", layout="wide")
try:
    st.title("📁 Portfolio")
except UnicodeEncodeError:
    try:
        st.title("Portfolio")
    except Exception:
        pass
except Exception:
    try:
        st.title("Portfolio")
    except Exception:
        pass
st.caption("Enter positions; quotes auto‑update each minute on page refresh.")

try:
    from ui.provider_banner import show_provider_banner
    show_provider_banner()
except Exception:
    pass

# Theme toggle (reads/writes prefs)
theme_toggle()

# Load persisted portfolio from disk into session state
if "positions" not in st.session_state:
    st.session_state["positions"] = load_portfolio()

with st.expander("➕ Add / Update position"):
    sym = st.text_input("Symbol", "").upper().strip()
    qty = st.number_input("Quantity", min_value=0.0, step=0.01)
    avg = st.number_input("Avg Price", min_value=0.0, step=0.01)
    if st.button("Add/Update"):
        if sym:
            df = st.session_state["positions"].copy()
            mask = df["symbol"] == sym
            if mask.any():
                df.loc[mask, ["qty", "avg_price"]] = [qty, avg]
            else:
                df.loc[len(df)] = [sym, qty, avg]
            st.session_state["positions"] = df
            save_portfolio(df)
            st.success("Saved.")

st.write("---")

# Keyboard palette (quick actions)
with st.expander("⌨️ Command palette"):
    kp = st.text_input("Jump to symbol / add: (e.g. 'AAPL' or 'add MSFT 10 200')")
    if kp:
        parts = kp.split()
        if parts[0].lower() == "add" and len(parts) >= 4:
            s = parts[1].upper()
            q = float(parts[2])
            a = float(parts[3])
            df = st.session_state["positions"].copy()
            df.loc[len(df)] = [s, q, a]
            st.session_state["positions"] = df
            save_portfolio(df)
            st.success(f"Added {s}")
        elif len(parts) == 1:
            # quick jump - prefill symbol input and show compare link
            st.experimental_set_query_params(symbol=parts[0].upper())
            st.info(f"Jumped to {parts[0].upper()} (use Compare page)")

pos = st.session_state["positions"].copy()

# Only run live/networking UI when not under pytest (keeps imports/tests hermetic)
if "pytest" not in sys.modules:
    vix = get_vix()
    regime = "Unknown"
    if vix is not None:
        if vix < 18:
            regime = "Low volatility"
        elif vix < 28:
            regime = "Moderate volatility"
        else:
            regime = "High volatility"
    st.info(f"Market regime: {regime} (VIX ~ {vix if vix is not None else '—'})")

    # Macro regime banner (conservative; uses utils.macro when available)
    try:
        from utils import macro
        mr = macro.get_macro_regime()
        mreg = mr.get('regime', 'Unknown')
        mreason = mr.get('reason', '')
        mcape = mr.get('cape')
        mcape_pct = mr.get('cape_pct')
        myields = mr.get('yields') or {}
        spread_2y_10 = myields.get('2y_10y') if isinstance(myields, dict) else None
        spread_3m_10 = myields.get('3m_10y') if isinstance(myields, dict) else None

        col1, col2 = st.columns([1, 4])
        with col1:
            color = 'gray'
            if mreg in ('Macro-Defensive', 'Valuation-Rich', 'Inversion-Warn'):
                color = 'red'
            elif mreg == 'Opportunistic':
                color = 'green'
            st.markdown(f"**Macro:** <span style='color:{color};font-weight:600'>{mreg}</span>", unsafe_allow_html=True)
        with col2:
            if mcape is not None and mcape_pct is not None:
                st.write(f"CAPE: {mcape:.2f} (pct {mcape_pct:.1f}%)")
            elif mcape is not None:
                st.write(f"CAPE: {mcape:.2f}")
            else:
                st.write("CAPE: N/A")
            st.write(f"Spreads: 2y-10y: {spread_2y_10 if spread_2y_10 is not None else 'N/A'} | 3m-10y: {spread_3m_10 if spread_3m_10 is not None else 'N/A'}")
            if mreason:
                st.caption(mreason)
    except Exception:
        # graceful fallback if macro helper unavailable
        pass

    # Display portfolio with live quotes and earnings-aware filter
    if pos.empty:
        st.info("Add positions above.")
    else:
        rows = []
        # import finnhub fetcher lazily to avoid network at import-time
        try:
            from data_fetchers.finnhub import get_quote_finnhub
        except Exception:
            get_quote_finnhub = lambda s: {}

        for idx, r in pos.iterrows():
            sym = r["symbol"]
            # Skip entries with earnings in next 2 days if user prefers
            days = days_until_next_earnings(sym)
            warn_earnings = days is not None and days <= 2
            try:
                q = get_quote_finnhub(sym)
            except Exception:
                q = {}
            price = _safe_float(q.get("c"))
            pnl = (price - r["avg_price"]) * r["qty"] if np.isfinite(price) else float("nan")
            rows.append({"Symbol": sym, "Qty": r["qty"], "Avg Price": r["avg_price"], "Last": price, "P/L": pnl, "EarningsSoon": warn_earnings})
    out = pd.DataFrame(rows)
    denom = (out["Avg Price"] * out["Qty"]).replace(0, np.nan)
    out["P/L %"] = (out["P/L"] / denom) * 100.0

    # Friendly formatting columns for display
    display_table = out.copy()
    # Format numeric columns: currency and percent
    def _fmt_currency(x):
        try:
            if np.isfinite(x):
                return f"${x:,.2f}"
        except Exception:
            pass
        return "—"

    def _fmt_percent(x):
        try:
            if np.isfinite(x):
                return f"{x:+.2f}%"
        except Exception:
            pass
        return "—"

    display_table["Avg Price"] = display_table["Avg Price"].apply(_fmt_currency)
    display_table["Last"] = display_table["Last"].apply(_fmt_currency)
    display_table["P/L"] = display_table["P/L"].apply(_fmt_currency)
    display_table["P/L %"] = display_table["P/L %"].apply(_fmt_percent)

    # Editable table controls
    sel = st.multiselect("Select rows to delete/edit", options=out["Symbol"].tolist())
    if st.button("Delete selected") and sel:
        df = pos[~pos["symbol"].isin(sel)].reset_index(drop=True)
        st.session_state["positions"] = df
        save_portfolio(df)
        st.success("Deleted selected rows")

    from utils.ui_safe_display import display_df
    # show a caption and the formatted table
    st.caption("Portfolio summary — Last prices and P/L. Values updated on page refresh.")
    display_df(display_table, use_container_width=True, hide_index=True, column_config={"Symbol": getattr(st.column_config, 'TextColumn', lambda **k: None)(label='Symbol')})

    # Add quick visuals: position weight pie and P/L bar chart
    try:
        total_value = (out["Avg Price"] * out["Qty"]).fillna(0)
        vals = total_value.where(total_value > 0, 0)
        weights = (vals / vals.sum()).fillna(0)
        import plotly.graph_objects as go
        if not out.empty and vals.sum() > 0:
            fig_pie = go.Figure(go.Pie(labels=out["Symbol"].tolist(), values=vals.tolist(), hole=0.4, sort=False))
            from ui.plotting import apply_plotly_theme
            fig_pie.update_layout(title_text="Position Weights", legend_title_text="Symbol")
            apply_plotly_theme(fig_pie, title="Position Weights", dark=(st.session_state.get('theme','dark')=='dark'))
            st.plotly_chart(fig_pie, use_container_width=True)

        # P/L bar chart
        pl_vals = out.set_index("Symbol")["P/L"].fillna(0).to_dict()
        if pl_vals:
            fig_bar = go.Figure(go.Bar(x=list(pl_vals.keys()), y=list(pl_vals.values()), marker_color=["green" if v >= 0 else "crimson" for v in pl_vals.values()], text=[f"${v:,.2f}" for v in pl_vals.values()], textposition="auto"))
            from ui.plotting import apply_plotly_theme
            fig_bar.update_layout(title_text="P/L by Symbol", yaxis_title="P/L (USD)")
            apply_plotly_theme(fig_bar, title="P/L by Symbol", y_title="P/L (USD)", dark=(st.session_state.get('theme','dark')=='dark'))
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        pass

    # --- Automated suggestions (simple EMA crossover) ---
    suggestions = []
    try:
        from data_fetchers.prices import get_ohlc
        for idx, row in pos.iterrows():
            sym = row["symbol"]
            try:
                hist = get_ohlc(sym, period="3mo", interval="1d")
                if isinstance(hist, pd.DataFrame) and not hist.empty and "close" in hist.columns:
                    close = pd.Series(hist["close"]).astype(float)
                    if len(close) >= 50:
                        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
                        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
                        if ema20 > ema50:
                            suggestions.append({"symbol": sym, "action": "Consider BUY (EMA20>EMA50)", "ema20": float(ema20), "ema50": float(ema50)})
                        else:
                            suggestions.append({"symbol": sym, "action": "Consider SELL/HOLD (EMA20<=EMA50)", "ema20": float(ema20), "ema50": float(ema50)})
            except Exception:
                continue
    except Exception:
        suggestions = []

    if suggestions:
        st.markdown("---")
        st.subheader("Automated suggestions")
        # present suggestions as a small table with color-coded actions
        sug_df = pd.DataFrame(suggestions)
        # friendly formatting
        sug_df["ema20"] = sug_df["ema20"].apply(lambda x: f"{x:.2f}")
        sug_df["ema50"] = sug_df["ema50"].apply(lambda x: f"{x:.2f}")
        from utils.ui_safe_display import display_df as _display_df
        _display_df(sug_df.rename(columns={"symbol": "Symbol", "action": "Action", "ema20": "EMA20", "ema50": "EMA50"}), use_container_width=True, hide_index=True)

    # --- Quick trade (Alpaca) integration (guarded) ---
    try:
        from config import get_runtime_key
        ALPACA_API_KEY = get_runtime_key("ALPACA_API_KEY")
        ALPACA_SECRET_KEY = get_runtime_key("ALPACA_SECRET_KEY")
        alpaca_enabled = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
    except Exception:
        try:
            from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
            alpaca_enabled = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
        except Exception:
            alpaca_enabled = False

    if alpaca_enabled and "pytest" not in sys.modules:
        with st.expander("Quick Trade (live via Alpaca)"):
            q_sym = st.text_input("Symbol to trade", "")
            q_qty = st.number_input("Quantity", min_value=0.0, step=1.0)
            side = st.selectbox("Side", ["buy", "sell"])
            if st.button("Submit Market Order"):
                if q_sym and q_qty > 0:
                    try:
                        import requests
                        base = "https://paper-api.alpaca.markets"
                        headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
                        payload = {"symbol": q_sym, "qty": q_qty, "side": side, "type": "market", "time_in_force": "day"}
                        r = requests.post(base + "/v2/orders", json=payload, headers=headers, timeout=10)
                        if r.status_code in (200, 201):
                            st.success(f"Order submitted: {q_sym} {side} {q_qty}")
                        else:
                            st.error(f"Order failed ({r.status_code}): {r.text[:200]}")
                    except Exception as e:
                        st.error(f"Failed to submit order: {e}")
    else:
        with st.expander("Quick Trade"):
            st.info("Quick trade disabled — Alpaca keys not configured or running under test.")
