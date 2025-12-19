from __future__ import annotations

import sys
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Markets & Products", layout="wide")
try:
    st.title("Markets & Products — ETFs, Mutual Funds, IPOs, Events")
except Exception:
    try:
        st.title("Markets & Products")
    except Exception:
        pass

# Avoid import-time network calls when running tests
if "pytest" not in sys.modules:
    from data_providers import list_etfs, list_mutual_funds, upcoming_ipos, get_major_events
    from config import get_runtime_key
    from utils.ui_safe_display import display_df, run_safe

    fh = None
    fmp = None
    try:
        fh = get_runtime_key("FINNHUB_API_KEY")
    except Exception:
        try:
            from config import FINNHUB_API_KEY as fh
        except Exception:
            fh = None
    try:
        fmp = get_runtime_key("FMP_API_KEY")
    except Exception:
        try:
            from config import FMP_API_KEY as fmp
        except Exception:
            fmp = None

    st.header("ETFs")
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        q = st.text_input("Search ETFs / symbol / issuer", value="")
    with col2:
        asset_filter = st.selectbox("Asset class", options=["All", "ETF", "MutualFund"], index=0)
    with col3:
        per_page = st.selectbox("Per page", options=[10, 25, 50, 100], index=2)

    if st.button("Refresh ETF list"):
        st.info("Refreshing list — providers will be re-probed on next render.")
    try:
        etfs = list_etfs(fmp or "", fh or "")
        df = pd.DataFrame(etfs) if etfs else pd.DataFrame()

        # Basic filtering: search and asset class
        if not df.empty:
            # normalize columns
            cols = [c for c in ["symbol", "name", "issuer", "expenseRatio"] if c in df.columns]
            search_mask = pd.Series(True, index=df.index)
            if q:
                ql = str(q).lower()
                search_mask = df.apply(lambda row: any(ql in str(row.get(c, "")).lower() for c in cols), axis=1)

            if asset_filter != "All":
                class_col = "assetType" if "assetType" in df.columns else "type" if "type" in df.columns else None
                if class_col:
                    df = df[search_mask & (df[class_col].fillna("") == asset_filter)]
                else:
                    df = df[search_mask]
            else:
                df = df[search_mask]

            total = len(df)
            page = st.session_state.get("markets_etf_page", 0)
            max_page = max(0, (total - 1) // per_page)
            colp1, colp2, colp3 = st.columns([1, 2, 1])
            with colp1:
                if st.button("Prev", key="etf_prev"):
                    st.session_state["markets_etf_page"] = max(0, page - 1)
            with colp2:
                st.write(f"Page {page + 1} / {max_page + 1} — {total} items")
            with colp3:
                if st.button("Next", key="etf_next"):
                    st.session_state["markets_etf_page"] = min(max_page, page + 1)

            start = page * per_page
            end = start + per_page
            page_df = df.iloc[start:end]

            # Render table and per-row detail expander
            display_df(page_df.reset_index(drop=True), use_container_width=True)
            for _, row in page_df.reset_index(drop=True).iterrows():
                sym = str(row.get("symbol") or row.get("ticker") or "").strip()
                with st.expander(f"{sym} details", expanded=False):
                    st.write({
                        "Name": row.get("name") or row.get("companyName"),
                        "Issuer": row.get("issuer"),
                        "Expense Ratio": row.get("expenseRatio"),
                        "AUM": row.get("aum") or row.get("totalAssets"),
                    })
                    # fetch live events small panel
                    try:
                        ev = get_major_events(sym, fh or "", fmp or "")
                        st.write("Earnings next:")
                        st.json(ev.get("earnings_next") or {})
                        st.write("Last earnings summary:")
                        st.json(ev.get("earnings_last") or {})
                    except Exception:
                        st.info("Live events unavailable for this symbol")
        else:
            st.info("No ETF data available from configured providers — try enabling yfinance or providing API keys in Settings.")
    except Exception as e:
        st.error(f"Failed to fetch ETFs: {e}")

    st.header("Mutual Funds")
    try:
        mfs = list_mutual_funds(fmp or "", fh or "")
        if mfs:
            mf_df = pd.DataFrame(mfs)
            # Reuse same search box for mutual funds when query is present
            if "q" in locals() and q:
                ql = str(q).lower()
                mf_df = mf_df[mf_df.apply(lambda r: ql in str(r.get("name", "")).lower() or ql in str(r.get("symbol", "")).lower(), axis=1)]
            display_df(mf_df.head(200), use_container_width=True)
        else:
            st.info("No mutual fund data available from configured providers.")
    except Exception as e:
        st.error(f"Failed to fetch mutual funds: {e}")

    st.header("Upcoming IPOs")
    try:
        ipos = upcoming_ipos(fh or "", fmp or "", days_ahead=180)
        if ipos:
            ipodf = pd.DataFrame(ipos)
            # basic date filter
            if "date" in ipodf.columns:
                ipodf["date"] = pd.to_datetime(ipodf["date"], errors="coerce")
                days = st.number_input("Days ahead", min_value=7, max_value=365, value=180)
                cutoff = pd.Timestamp.now() + pd.Timedelta(days=int(days))
                ipodf = ipodf[ipodf["date"] <= cutoff]
            display_df(ipodf.sort_values(by="date").reset_index(drop=True), use_container_width=True)
        else:
            st.info("No upcoming IPOs found or provider access restricted.")
    except Exception as e:
        st.error(f"Failed to fetch IPOs: {e}")

    st.header("Major Events for a Symbol")
    sym = st.text_input("Symbol", value="AAPL")
    if sym:
        def _show_events():
            ev = get_major_events(sym, fh or "", fmp or "")
            st.subheader("Earnings — next")
            st.json(ev.get("earnings_next") or {})
            st.subheader("Earnings — last summary")
            st.json(ev.get("earnings_last") or {})
            st.subheader("Recent news")
            display_df(pd.DataFrame(ev.get("news") or []).head(50), use_container_width=True)
            st.subheader("Dividends & Splits")
            st.write(ev.get("dividends") or [])
            st.write(ev.get("splits") or [])

        run_safe(_show_events)