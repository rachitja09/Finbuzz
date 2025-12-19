import streamlit as st
import pandas as pd
import os
from data_fetchers.prices import get_ohlc
from utils import signals
from utils import macro
from config import get_runtime_key
from utils.rates import fetch_rates


st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("⚙️ Diagnostics & Macro Signals")

st.markdown("This page shows data source resolution, OHLC diagnostics and macro signals (CAPE and yield spreads).")

# Key resolution section (non-sensitive booleans only)
keys = ["NEWS_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY", "FRED_API_KEY"]
rows = []
for k in keys:
    env_exists = k in os.environ
    env_val = os.environ.get(k)
    runtime = get_runtime_key(k)
    secrets_present = False
    try:
        import streamlit as _st

        try:
            s = _st.secrets.get(k)
            if s:
                secrets_present = True
        except Exception:
            secrets_present = False
    except Exception:
        secrets_present = False

    if env_exists:
        if env_val is None or env_val == "":
            source = "env (explicitly disabled)"
        else:
            source = "env"
    elif secrets_present:
        source = "secrets"
    elif runtime is not None:
        source = "module"
    else:
        source = "none"

    rows.append({"key": k, "source": source})

st.subheader("API key resolution (booleans only)")
st.table(rows)

st.markdown("---")

# Macro section
st.subheader("Macro signals (Shiller CAPE & Yield Curve)")
reg = macro.get_macro_regime()
cape = reg.get('cape')
cape_pct = reg.get('cape_pct')
yields = reg.get('yields') or {}
pill_col1, pill_col2 = st.columns([1, 3])
with pill_col1:
    # Regime pill
    rname = reg.get('regime', 'Unknown')
    color = 'gray'
    if rname in ('Macro-Defensive', 'Valuation-Rich', 'Inversion-Warn'):
        color = 'red'
    elif rname == 'Opportunistic':
        color = 'green'
    st.markdown(f"**Regime:** <span style='color:{color};font-weight:600'>{rname}</span>", unsafe_allow_html=True)

with pill_col2:
    st.write(reg.get('reason', ''))

st.write("CAPE latest: ", f"{cape:.2f}" if cape is not None else "N/A", " — percentile:", f"{cape_pct:.1f}%" if cape_pct is not None else "N/A")

try:
    # Show time series if available
    sh_df = macro.fetch_shiller_series()
    if not sh_df.empty:
        cape_series = macro.compute_cape(sh_df)
        if not cape_series.empty:
            # prefer Plotly for interactive charting
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cape_series.index, y=cape_series.values, mode='lines', name='CAPE', hovertemplate='Date: %{x}<br>CAPE: %{y:.2f}<extra></extra>'))
                fig.update_layout(title='Shiller CAPE', xaxis_title='Date', yaxis_title='CAPE')
                fig.update_yaxes(tickformat='.2f')
                from ui.plotting import apply_plotly_theme
                apply_plotly_theme(fig, title='Shiller CAPE', dark=(st.session_state.get('theme','dark')=='dark'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(cape_series)
except Exception:
    pass

y_2y_10y = None
y_3m_10y = None
if isinstance(yields, dict):
    y_2y_10y = yields.get('2y_10y')
    y_3m_10y = yields.get('3m_10y')
else:
    y_2y_10y = None
    y_3m_10y = None

st.write("Yield spreads (latest): 2y-10y:", y_2y_10y, "3m-10y:", y_3m_10y)
try:
    # try to show time series for yields if available via rates.fetch_rate_series
    fred_key = None
    try:
        fred_key = getattr(reg.get('yields'), 'fred_key', None)
    except Exception:
        fred_key = None
    # attempt to fetch recent series for 10y/2y/3m
    try:
        ten = macro.get_yield_spreads().get('10y')
    except Exception:
        ten = None
    # Show the numeric status
    st.json(macro.get_yield_spreads())
    # try Plotly time series via rates.fetch_rate_series if available
    try:
        from utils import rates as _rates
        seq10 = _rates.fetch_rate_series('DGS10', fred_key, points=180) or []
        seq2 = _rates.fetch_rate_series('DGS2', fred_key, points=180) or []
        seq3 = _rates.fetch_rate_series('DGS3MO', fred_key, points=180) or []
        if seq10 or seq2 or seq3:
            # build DataFrame for plotting
            import pandas as _pd
            max_len = max(len(seq10), len(seq2), len(seq3))
            dates = pd.date_range(end=pd.Timestamp.today(), periods=max_len, freq='D')
            dfy = _pd.DataFrame({'date': dates})
            if seq10:
                dfy['10y'] = list(reversed(list(seq10)))[:max_len]
            if seq2:
                dfy['2y'] = list(reversed(list(seq2)))[:max_len]
            if seq3:
                dfy['3m'] = list(reversed(list(seq3)))[:max_len]
            dfy = dfy.set_index('date')
            try:
                import plotly.graph_objects as go
                fig2 = go.Figure()
                for col in ['10y', '2y', '3m']:
                    if col in dfy.columns:
                        fig2.add_trace(go.Scatter(x=dfy.index, y=dfy[col], mode='lines', name=col, hovertemplate='Date: %{x}<br>%{y:.2f}%%<extra></extra>'))
                fig2.update_layout(title='Yield Series (recent)', xaxis_title='Date', yaxis_title='Yield (%)')
                fig2.update_yaxes(tickformat='.2f')
                from ui.plotting import apply_plotly_theme
                apply_plotly_theme(fig2, title='Yield Series', dark=(st.session_state.get('theme','dark')=='dark'))
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.line_chart(dfy)
    except Exception:
        pass
except Exception:
    pass

st.markdown("---")

# OHLC and signals section
st.subheader("OHLC & Signal Diagnostics")
symbol = st.text_input("Symbol", value="AAPL")
period = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
if st.button("Run diagnostics"):
    try:
        df = get_ohlc(symbol, period=period, interval=interval)
        st.success(f"Fetched OHLC for {symbol} — {len(df)} rows")
        st.dataframe(df.tail(5))

        sh = signals.sharpe_ratio(df['Close'], symbol=symbol)
        so = signals.sortino_ratio(df['Close'], symbol=symbol)
        vol = signals.volatility(df['Close'], symbol=symbol)
        mom = signals.momentum(df['Close'], symbol=symbol)
        rsi = signals.rsi_from_series(df['Close'], symbol=symbol)
        atrv = None
        if {'High', 'Low'}.issubset(df.columns):
            atrv = signals.atr(df['High'], df['Low'], df['Close'], symbol=symbol)

        st.markdown("### Signals")
        cols = st.columns(3)
        cols[0].metric("Sharpe", f"{sh:.2f}" if pd.notna(sh) else "N/A")
        cols[1].metric("Sortino", f"{so:.2f}" if pd.notna(so) else "N/A")
        cols[2].metric("Vol (ann)", f"{vol:.2f}" if pd.notna(vol) else "N/A")

        cols2 = st.columns(3)
        cols2[0].metric("Momentum(63)", f"{mom:.2%}" if pd.notna(mom) else "N/A")
        cols2[1].metric("RSI", f"{rsi:.0f}" if rsi is not None else "N/A")
        cols2[2].metric("ATR", f"{atrv:.2f}" if atrv is not None else "N/A")
    except Exception as e:
        st.error(f"Diagnostics failed: {type(e).__name__}: {e}")
