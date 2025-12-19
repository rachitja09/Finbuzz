from dotenv import load_dotenv



def main():
    """Run the Streamlit UI. Kept inside a function so importing `app` in tests
    doesn't trigger Streamlit side-effects.
    """
    import streamlit as st
    from utils.watchlist import prefetch_quotes
    try:
        from ui.market_overview import render as render_market_overview
    except Exception:
        render_market_overview = None

    load_dotenv()
    # Server-side access token: keep provider keys secret on the server and
    # require visitors to enter a shared dashboard token. Set DASHBOARD_ACCESS_TOKEN
    # in the server's environment or Streamlit secrets to enable access control.
    try:
        from config import get_runtime_key

        DASH_TOKEN = get_runtime_key("DASHBOARD_ACCESS_TOKEN")
    except Exception:
        import os

        DASH_TOKEN = os.environ.get("DASHBOARD_ACCESS_TOKEN")

    if DASH_TOKEN:
        try:
            import hmac

            if "auth_ok" not in st.session_state:
                st.session_state.auth_ok = False

            if not st.session_state.auth_ok:
                with st.sidebar.form("login_form"):
                    st.write(
                        "This dashboard is access restricted. Enter the access token to continue."
                    )
                    attempt = st.text_input("Access token", type="password")
                    submitted = st.form_submit_button("Unlock")
                    if submitted:
                        try:
                            if attempt and hmac.compare_digest(attempt, str(DASH_TOKEN)):
                                st.session_state.auth_ok = True
                                # experimental_rerun may not exist in all Streamlit versions; call safely
                                getattr(st, "experimental_rerun", lambda: None)()
                            else:
                                st.error("Invalid token")
                        except Exception:
                            st.error("Authentication failed")
                return
        except Exception:
            # If auth subsystem errors, fail open so dashboard isn't locked out
            pass
    # Resolve FINNHUB API key at runtime (avoids import-time side-effects)
    try:
        from config import get_runtime_key
        FINNHUB_API_KEY = get_runtime_key("FINNHUB_API_KEY")
    except Exception:
        try:
            from config import FINNHUB_API_KEY as FINNHUB_API_KEY
        except Exception:
            FINNHUB_API_KEY = None

    st.set_page_config(page_title="Quant Dashboard", layout="wide")
    st.title("Quant Dashboard")
    st.write("A concise, data-driven market intelligence dashboard. Use the sidebar to navigate pages.")

    # Global macro pill (non-blocking): shows current macro regime briefly under title
    try:
        from utils import macro
        mr = macro.get_macro_regime()
        mreg = mr.get('regime', None)
        mcape = mr.get('cape')
        mcape_pct = mr.get('cape_pct')
        if mreg:
            col1, col2 = st.columns([1, 6])
            with col1:
                color = 'gray'
                if mreg in ('Macro-Defensive', 'Valuation-Rich', 'Inversion-Warn'):
                    color = 'red'
                elif mreg == 'Opportunistic':
                    color = 'green'
                st.markdown(f"**Macro:** <span style='color:{color};font-weight:600'>{mreg}</span>", unsafe_allow_html=True)
            with col2:
                if mcape is not None and mcape_pct is not None:
                    st.caption(f"CAPE {mcape:.1f} ({mcape_pct:.0f}pct)")
                elif mcape is not None:
                    st.caption(f"CAPE {mcape:.1f}")
    except Exception:
        pass

    # Warm-up: prefetch a tiny watchlist on app load to improve perceived performance
    try:
        if FINNHUB_API_KEY:
            _ = prefetch_quotes(["AAPL", "MSFT", "GOOG"], max_workers=3)
    except Exception:
        pass

    # Render the market overview inline on the landing page (non-duplicate)
    if render_market_overview is not None:
        try:
            render_market_overview()
        except Exception:
            # Keep import-time safety for test/CI
            pass


if __name__ == "__main__":
    main()
