import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
from math import sqrt

st.set_page_config(page_title="Portfolio Simulator", layout="wide")

try:
    from ui.header import show_header

    show_header(title="Simulator", subtitle="Forward-looking Monte Carlo paths & scenario analysis")
except Exception:
    st.title("Stock Simulator")
    st.markdown("""
    Estimate forward price paths and ROI for a single US-listed stock using historical
    returns and a conservative Monte Carlo geometric Brownian motion model. The
    simulator is designed to be robust when optional data providers (yfinance, news)
    are missing — it will show helpful errors when live data isn't available.
    """)

from data_fetchers.prices import get_ohlc
from config import get_runtime_key
import data_providers
from utils import earnings as earnings_utils
import data_fetchers.finnhub as finnhub_fetchers


def safe_get_ohlc(symbol: str, lookback_period: str = "1y") -> pd.DataFrame:
    try:
        return get_ohlc(symbol, period=lookback_period, interval="1d")
    except Exception as e:
        st.error(f"Failed to fetch price history for {symbol}: {e}")
        return pd.DataFrame()


def gbm_simulate(S0: float, mu: float, sigma: float, days: int, paths: int = 500, rng: Optional[np.random.Generator] = None):
    rng = rng or np.random.default_rng(42)
    dt = 1/252
    prices = np.zeros((days+1, paths), dtype=float)
    prices[0, :] = S0
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * sqrt(dt)
    for t in range(1, days+1):
        z = rng.standard_normal(paths)
        prices[t, :] = prices[t-1, :] * np.exp(drift + vol * z)
    return prices


def summarize_sim(prices: np.ndarray):
    # prices: (T+1, N)
    final = prices[-1, :]
    mean = float(np.mean(final))
    median = float(np.median(final))
    p05 = float(np.percentile(final, 5))
    p95 = float(np.percentile(final, 95))
    return {"mean": mean, "median": median, "p05": p05, "p95": p95}


with st.sidebar:
    st.header("Simulation inputs")
    symbol = st.text_input("Ticker (US exchange)", value="AAPL")
    buy_mode = st.radio("Purchase by", ("Shares", "Dollars"))
    shares = st.number_input("Shares purchased", min_value=0.0, value=10.0, step=1.0)
    dollars = st.number_input("Dollars invested", min_value=0.0, value=1000.0, step=100.0)
    cost_price = st.number_input("Purchase price per share (leave 0 to use current price)", min_value=0.0, value=0.0, step=0.01)
    lookback = st.selectbox("Historic lookback for return estimates", ("3mo", "6mo", "1y", "3y"), index=2)
    horizon_type = st.selectbox("Horizon unit", ("Days", "Months", "Years"), index=0)
    # Allow manual numeric input for horizon so users can type 1,2,3 etc. Keep the unit selector.
    horizon_value = st.number_input("Horizon (in selected unit)", min_value=1, max_value=10000, value=30, step=1,
                                    help="Enter the horizon in the selected unit (Days / Months / Years). For example, choose unit=Days and enter 1 for a one-day horizon.")
    num_paths = st.slider("Monte Carlo paths", 50, 2000, 500, step=50)
    st.markdown("---")
    st.subheader("Model tuning")
    scenario = st.selectbox("Scenario preset", ("Base", "Bull", "Bear", "Custom"), index=0)
    analyst_w = st.slider("Analyst influence", 0.0, 2.0, 0.7, step=0.05)
    holders_w = st.slider("Holders influence", 0.0, 2.0, 0.5, step=0.05)
    sentiment_w = st.slider("News sentiment weight", 0.0, 2.0, 0.3, step=0.05)
    signal_influence = st.slider("Max signal influence (conservative blend)", 0.0, 1.0, 0.5, step=0.05,
                                 help="Cap how much analyst/holders/sentiment signals can adjust historical drift.")
    if scenario == "Custom":
        mu_mult = st.slider("Mu multiplier", 0.5, 2.0, 1.0, step=0.05)
        sigma_mult = st.slider("Sigma multiplier", 0.5, 2.0, 1.0, step=0.05)
    else:
        mu_mult = 1.0
        sigma_mult = 1.0
    run_button = st.button("Run simulation")

if not symbol:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

if run_button:
    df = safe_get_ohlc(symbol, lookback_period=lookback)
    if df.empty:
        st.stop()

    # determine purchase price and shares
    current_price = float(df['Close'].iloc[-1])
    if cost_price <= 0:
        cost_price = current_price
    if buy_mode == "Shares":
        n_shares = float(shares)
        invested = n_shares * cost_price
    else:
        invested = float(dollars)
        n_shares = invested / cost_price if cost_price > 0 else 0.0

    # compute daily returns, drift (mu) and volatility (sigma)
    returns = df['Close'].pct_change().dropna()
    if returns.empty:
        st.error("Not enough price history to compute returns.")
        st.stop()

    mu_daily = float(returns.mean())
    sigma_daily = float(returns.std())
    mu_annual = mu_daily * 252
    sigma_annual = sigma_daily * sqrt(252)

    # horizon conversion
    if horizon_type == "Days":
        days = int(horizon_value)
    elif horizon_type == "Months":
        days = int(horizon_value * 21)
    else:
        days = int(horizon_value * 252)

    st.subheader(f"Simulating {symbol} for {days} trading days")
    st.write(f"(Selected horizon: {horizon_value} {horizon_type.lower()} → {days} trading days)")
    st.write(f"Current price: ${current_price:,.2f} — using purchase price: ${cost_price:,.2f}")

    # Optional: try to get a lightweight sentiment adjustment from news module if available
    sentiment_adj = 1.0
    s = 0.0
    sentiment_effect = 0.0
    try:
        from data_fetchers.news import newsapi_headlines, vader_score
        arts = newsapi_headlines(symbol, limit=6) if callable(newsapi_headlines) else []
        if arts:
            scores = []
            for a in arts:
                txt = (a.get('title','') or '') + ' ' + (a.get('description','') or '')
                try:
                    sc = vader_score(txt)
                except Exception:
                    sc = 0.0
                scores.append(float(sc))
            s = float(np.mean(scores)) if scores else 0.0
        else:
            s = 0.0
        # Map sentiment (-1..1) to a small annual drift adjustment (-2%..+2%)
        st.write(f"News sentiment (mean headline): {s:.3f}")
        # sentiment effect in annualized drift terms
        sentiment_effect = float(s) * 0.02 * float(sentiment_w)  # +/- up to ~2% scaled by weight
        sentiment_adj += sentiment_effect
        st.write(f"Sentiment effect on drift: {sentiment_effect*100:+.2f}% (weight {sentiment_w})")
    except Exception:
        # Not fatal; continue without sentiment
        s = 0.0
        sentiment_adj = 1.0
        sentiment_effect = 0.0

    # ---- Analyst consensus (Finnhub) and holders influence ----
    analyst_target = None
    finnhub_key = get_runtime_key("FINNHUB_API_KEY")
    try:
        if finnhub_key:
            acdf = finnhub_fetchers.fetch_analyst_consensus(finnhub_key, [symbol])
            if hasattr(acdf, 'empty') and not acdf.empty:
                row = acdf.iloc[0]
                analyst_target = row.get('targetMean') or row.get('targetMedian') or None
                try:
                    analyst_target = float(analyst_target) if analyst_target is not None else None
                except Exception:
                    analyst_target = None
                if analyst_target:
                    gap_pct = (analyst_target - current_price) / current_price
                    st.write(f"Analyst consensus target: ${analyst_target:,.2f} ({gap_pct*100:+.1f}% vs current)")
    except Exception:
        analyst_target = None

    holders = None
    inst_avg = None
    try:
        holders = data_providers.get_holders(symbol, api_key=get_runtime_key('FMP_API_KEY') or "", finnhub_key=(finnhub_key or ""))
        if holders and isinstance(holders, dict):
            inst = holders.get('institutional') or []
            if inst:
                try:
                    inst_pcts = [float(x.get('pct') or 0.0) for x in inst if x.get('pct') is not None]
                    inst_avg = sum(inst_pcts) / len(inst_pcts) if inst_pcts else None
                except Exception:
                    inst_avg = None
                if inst_avg:
                    st.write(f"Institutional holders: avg % out = {inst_avg:.1f}% (source: {holders.get('source')})")
    except Exception:
        holders = None

    # compute analyst and holders effects in annual drift terms
    analyst_effect = 0.0
    if analyst_target is not None:
        gap_pct = (analyst_target - current_price) / current_price
        # scale the gap to an annual drift contribution; conservative scaling factor
        analyst_effect = float(gap_pct) * 0.2 * float(analyst_w)
        st.write(f"Analyst drift contribution: {analyst_effect*100:+.2f}% (weight {analyst_w})")

    holders_effect = 0.0
    if inst_avg is not None:
        try:
            if inst_avg >= 40.0:
                holders_effect = 0.01 * float(holders_w)
            elif inst_avg < 10.0:
                holders_effect = -0.005 * float(holders_w)
            else:
                holders_effect = 0.0
            st.write(f"Holders contribution: {holders_effect*100:+.2f}% (avg {inst_avg:.1f}%, weight {holders_w})")
        except Exception:
            holders_effect = 0.0

    # Adjusted drift: conservative blend between historical return estimate and signal-driven adjustments
    total_signal_weight = float(analyst_w) + float(holders_w) + float(sentiment_w)
    # map total weights to a raw alpha in (0,1), then cap by user-specified signal_influence for conservatism
    raw_alpha = (total_signal_weight / (total_signal_weight + 2.0)) if total_signal_weight > 0 else 0.0
    alpha = min(float(signal_influence), float(raw_alpha))
    # signal-driven annual drift (historical mu scaled by mu_mult plus explicit signal contributions)
    signal_mu = mu_annual * float(mu_mult) + analyst_effect + holders_effect + (sentiment_effect if 'sentiment_effect' in locals() else 0.0)
    # blend conservatively: most weight remains on historical mu unless alpha is large
    adj_mu = (mu_annual * float(mu_mult)) * (1.0 - alpha) + signal_mu * alpha
    st.write(f"Signal blend alpha: {alpha:.2f} (raw {raw_alpha:.2f}, cap {signal_influence})")
    sigma_annual = sigma_annual * float(sigma_mult)

    # ---- Analyst consensus (Finnhub) ----
    analyst_target = None
    finnhub_key = get_runtime_key("FINNHUB_API_KEY")
    try:
        if finnhub_key:
            acdf = finnhub_fetchers.fetch_analyst_consensus(finnhub_key, [symbol])
            if hasattr(acdf, 'empty') and not acdf.empty:
                row = acdf.iloc[0]
                analyst_target = row.get('targetMean') or row.get('targetMedian') or None
                try:
                    analyst_target = float(analyst_target) if analyst_target is not None else None
                except Exception:
                    analyst_target = None
                if analyst_target:
                    gap_pct = (analyst_target - current_price) / current_price * 100.0
                    st.write(f"Analyst consensus target: ${analyst_target:,.2f} ({gap_pct:+.1f}% vs current)")
    except Exception:
        analyst_target = None

    # ---- Institutional / major holders ----
    holders = None
    inst_avg = None
    try:
        # FMP or Finnhub keys (either may unlock holder endpoints)
        holders = data_providers.get_holders(symbol, api_key=get_runtime_key('FMP_API_KEY') or "", finnhub_key=(finnhub_key or ""))
        if holders and isinstance(holders, dict):
            inst = holders.get('institutional') or []
            if inst:
                # compute simple institutional ownership pct when available
                try:
                    inst_pcts = [float(x.get('pct') or 0.0) for x in inst if x.get('pct') is not None]
                    inst_avg = sum(inst_pcts) / len(inst_pcts) if inst_pcts else None
                except Exception:
                    inst_avg = None
                if inst_avg:
                    st.write(f"Institutional holders: avg % out = {inst_avg:.1f}% (source: {holders.get('source')})")
    except Exception:
        holders = None

    # ---- Earnings calendar / event risk ----
    try:
        edays = earnings_utils.days_until_next_earnings(symbol)
        if edays is not None:
            st.write(f"Days until next earnings: {edays}")
            st.write(earnings_utils.earnings_impact_text(symbol, edays))
        else:
            en = earnings_utils.get_next_earnings_info(symbol)
            if en and en.get('date'):
                st.write(f"Next earnings (source={en.get('source')}): {en.get('date')}")
    except Exception:
        # not fatal
        pass

    # ---- Benchmark (S&P 500) for context ----
    bench_label = "S&P 500 (^GSPC)"
    bench_df = None
    try:
        bench_df = safe_get_ohlc("^GSPC", lookback_period=lookback)
        if bench_df is not None and not bench_df.empty:
            bench_returns = bench_df['Close'].pct_change().dropna()
            bench_mu = float(bench_returns.mean()) * 252
            bench_sigma = float(bench_returns.std()) * sqrt(252)
            st.write(f"Benchmark ({bench_label}) annualized mean: {bench_mu*100:.2f}%, vol: {bench_sigma*100:.2f}%")
    except Exception:
        bench_df = None

    # Run Monte Carlo GBM
    prices = gbm_simulate(S0=current_price, mu=adj_mu, sigma=sigma_annual, days=days, paths=num_paths)

    stats = summarize_sim(prices)

    st.subheader("Simulation summary (final price distribution)")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        st.metric("Mean final price (USD)", f"${stats['mean']:,.2f}")
    with c2:
        st.metric("Median final price (USD)", f"${stats['median']:,.2f}")
    with c3:
        st.metric("5% low (USD)", f"${stats['p05']:,.2f}")
    with c4:
        st.metric("95% high (USD)", f"${stats['p95']:,.2f}")

    # Show a small sample of simulated paths
    import plotly.graph_objects as go

    # compute percentiles along time for shaded bands
    p10 = np.percentile(prices, 10, axis=1)
    p25 = np.percentile(prices, 25, axis=1)
    p50 = np.percentile(prices, 50, axis=1)
    p75 = np.percentile(prices, 75, axis=1)
    p90 = np.percentile(prices, 90, axis=1)

    # Build a calendar date index for the x-axis using the historical OHLC index as the anchor
    try:
        last_dates = df.index.to_numpy()
        last_date = pd.Timestamp(last_dates[-1]) if len(last_dates) > 0 else pd.Timestamp.today()
        date_index = [last_date + pd.tseries.offsets.BDay(i) for i in range(prices.shape[0])]
        x = date_index
    except Exception:
        x = list(range(prices.shape[0]))
    fig = go.Figure()
    # color palette (pleasant blue tones)
    band10_color = 'rgba(29,131,201,0.12)'
    band25_color = 'rgba(29,131,201,0.20)'
    median_color = 'rgba(0,200,255,1)'

    # 10-90 band (fill between p90 and p10)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p90) + list(p10[::-1]),
        fill='toself',
        fillcolor=band10_color,
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='10-90% band'
    ))
    # 25-75 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p75) + list(p25[::-1]),
        fill='toself',
        fillcolor=band25_color,
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='25-75% band'
    ))

    # median line
    fig.add_trace(go.Scatter(
        x=x,
        y=p50,
        mode='lines',
        line=dict(color=median_color, width=3),
        name='Median',
        hovertemplate='Day %{x}<br>Median price: $%{y:.2f}<extra></extra>'
    ))

    # optional: overlay a small sample of individual simulated paths (faint)
    try:
        sample_n = min(30, prices.shape[1])
        rng = np.random.default_rng(123)
        idxs = rng.choice(prices.shape[1], size=sample_n, replace=False)
        for i in idxs:
            fig.add_trace(go.Scatter(x=x, y=prices[:, i], mode='lines', line=dict(color='rgba(255,255,255,0.06)', width=1), hoverinfo='skip', showlegend=False))
    except Exception:
        pass

    from ui.plotting import apply_plotly_theme

    fig = apply_plotly_theme(fig, title=f"Simulated distribution for {symbol}", x_title='Date', y_title='Price (USD)', dark=True)

    # tighten x ticks for long horizons
    if isinstance(x[0], (pd.Timestamp,)) and len(x) <= 50:
        fig.update_xaxes(dtick='D1', tickformat='%b %d')
    st.plotly_chart(fig, use_container_width=True)

    # CSV exports: median path and percentiles
    try:
        out_df = pd.DataFrame({
            'date': x,
            'median': p50,
            'p25': p25,
            'p75': p75,
            'p10': p10,
            'p90': p90,
        })
        out_df['date'] = pd.to_datetime(out_df['date'])
        csv = out_df.to_csv(index=False)
        st.download_button("Download median path & percentile CSV", csv, file_name=f"{symbol}_sim_path.csv")
    except Exception:
        pass

    # ROI and CAGR calculations based on median final price
    median_final = stats['median']
    final_value = median_final * n_shares
    roi = (final_value - invested) / invested if invested > 0 else 0.0
    years = days / 252
    cagr = (median_final / cost_price) ** (1 / years) - 1 if years > 0 and cost_price > 0 else 0.0

    st.subheader("Estimated outcomes")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write(f"Invested: ${invested:,.2f}")
        st.write(f"Shares: {n_shares:,.4f}")
        st.write(f"Median final portfolio value: ${final_value:,.2f}")
    with col_b:
        st.metric("Estimated ROI (median)", f"{roi*100:.2f}%")
        st.metric("Estimated CAGR (median)", f"{cagr*100:.2f}% over {years:.2f} years")

    # Quick compute of recent drawdown for sizing guidance (safe fallback)
    try:
        recent_drawdown = (df['Close'].cummax() - df['Close']).iloc[-1] / df['Close'].iloc[-1] if ('Close' in df.columns and len(df) > 0) else 0.0
    except Exception:
        recent_drawdown = 0.0

    # Compare expected CAGR to benchmark if available and provide rebalancing guidance
    try:
        if bench_df is not None and not bench_df.empty:
            bench_returns = bench_df['Close'].pct_change().dropna()
            bench_cagr = (1 + float(np.mean(bench_returns))) ** 252 - 1
            with st.expander("Benchmark comparison & suggested actions", expanded=True):
                st.write(f"Benchmark ({bench_label}) historic CAGR: {bench_cagr*100:.2f}%")
                st.write(f"Simulated expected CAGR (median): {cagr*100:.2f}%")
                # Simple decision rules for rebalancing suggestions
                rebalance_msgs = []
                if cagr > bench_cagr + 0.05:
                    rebalance_msgs.append("Outperformance vs benchmark — consider increasing allocation (if conviction and risk tolerance allow).")
                elif cagr < bench_cagr - 0.05:
                    rebalance_msgs.append("Underperformance vs benchmark — consider trimming position and reallocating to diversified ETFs.")
                else:
                    rebalance_msgs.append("Performance roughly in line with benchmark — consider maintaining position or dollar-cost averaging.")

                # Volatility-adjusted sizing suggestion
                if sigma_annual > 0.6:
                    rebalance_msgs.append("High historic volatility — reduce position size or use staggered entry (DCA).")
                if recent_drawdown > 0.2:
                    rebalance_msgs.append("Significant recent drawdown — use reduced sizing and confirm stabilization before adding.")

                for m in rebalance_msgs:
                    st.write(f"- {m}")
    except Exception:
        pass

    st.subheader("Actionable suggestions")
    suggestions = []
    # compare to benchmark CAGR if possible
    try:
        if bench_df is not None and not bench_df.empty:
            bench_returns = bench_df['Close'].pct_change().dropna()
            bench_cagr = (1 + float(np.mean(bench_returns))) ** 252 - 1
            st.write(f"Benchmark CAGR (historic): {bench_cagr*100:.2f}%")
            if cagr > bench_cagr:
                suggestions.append("Expected CAGR exceeds benchmark — consider increasing allocation if conviction and risk tolerance allow.")
            else:
                suggestions.append("Expected CAGR underperforms benchmark — consider diversification or hedging.")
    except Exception:
        pass

    if cagr < 0:
        suggestions.append("Consider trimming position or placing a stop-loss. Review fundamentals and upcoming earnings.")
    elif cagr < 0.05:
        suggestions.append("Low expected CAGR vs long-term benchmarks — consider diversification into ETFs or adding dollar-cost averaging.")
    else:
        suggestions.append("Expected CAGR looks attractive; validate with upcoming earnings and sentiment before increasing allocation.")

    # Add quick checks based on volatility and recent drawdowns
    recent_drawdown = (df['Close'].cummax() - df['Close']).iloc[-1] / df['Close'].iloc[-1]
    if sigma_annual > 0.6:
        suggestions.append("High historical volatility — position sizing should be conservative.")
    if recent_drawdown > 0.15:
        suggestions.append("Recent drawdown >15% — wait for stabilization or average in smaller tranches.")

    for s in suggestions:
        st.write(f"- {s}")

    # Export simulated final price distribution and sample paths
    if st.button("Download median final distribution CSV"):
        out = pd.DataFrame({"final_price": prices[-1, :]})
        st.download_button("Download CSV", out.to_csv(index=False), file_name=f"{symbol}_sim_final.csv")
