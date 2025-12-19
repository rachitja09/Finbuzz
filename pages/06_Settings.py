import streamlit as st
from utils import prefs
import sys
import os

st.set_page_config(page_title="Settings", layout="centered")
try:
    st.title("Settings")
except UnicodeEncodeError:
    try:
        st.title("Settings")
    except Exception:
        pass

# Only show interactive settings and diagnostics when running the Streamlit app
# (avoid import-time UI during tests)
if "pytest" not in sys.modules:
    tab_pref, tab_diag = st.tabs(["Data & Preferences", "Diagnostics"])

    with tab_pref:
        st.header("Data & Developer Preferences")
        mode = prefs.get_data_source_pref()
        new_mode = st.selectbox("Default data source for pages", ["Auto (live first)", "Live (require keys)", "Mock only"], index=["Auto (live first)", "Live (require keys)", "Mock only"].index(mode))
        if new_mode != mode:
            prefs.set_data_source_pref(new_mode)
            st.success(f"Saved default data source: {new_mode}")

        dev_mock = prefs.get_dev_mock_pref()
        dm = st.checkbox("Enable developer mock mode (forces mock providers)", value=dev_mock)
        if dm != dev_mock:
            prefs.set_dev_mock_pref(dm)
            st.success("Saved dev mock preference")

        # Strategy model default weights
        st.markdown("---")
        st.subheader("Recommendation model default weights")
        curr = prefs.get_strategy_weights()
        col1, col2 = st.columns(2)
        with col1:
            tech_w = st.number_input("Technical weight (tech_w)", min_value=0.0, max_value=5.0, value=curr.get("tech_w", 1.0), step=0.1)
            analyst_w = st.number_input("Analyst weight (analyst_w)", min_value=0.0, max_value=5.0, value=curr.get("analyst_w", 0.7), step=0.1)
        with col2:
            consensus_w = st.number_input("Consensus weight (consensus_w)", min_value=0.0, max_value=5.0, value=curr.get("consensus_w", 0.5), step=0.1)
            news_w = st.number_input("News weight (news_w)", min_value=0.0, max_value=5.0, value=curr.get("news_w", 0.3), step=0.1)

        if st.button("Save recommendation defaults"):
            prefs.set_strategy_weights({"tech_w": tech_w, "analyst_w": analyst_w, "consensus_w": consensus_w, "news_w": news_w})
            st.success("Saved recommendation defaults.")

        # Benford data-integrity preferences
        st.markdown("---")
        st.subheader("Benford data-integrity check (optional)")
        b = prefs.get_benford_prefs()
        ben_enabled = st.checkbox("Enable Benford integrity check on volume", value=b.get("enabled", True))
        col_a, col_b = st.columns(2)
        with col_a:
            ben_min = st.number_input("Min samples to run check", min_value=10, max_value=10000, value=b.get("min_samples", 50), step=10)
        with col_b:
            ben_thresh = st.number_input("RMS % threshold to flag (higher = more tolerant)", min_value=0.0, max_value=100.0, value=b.get("threshold", 12.0), step=0.5)
        col_c, col_d = st.columns(2)
        with col_c:
            ben_mode = st.selectbox("Benford action mode", ["warn", "penalize", "block"], index=["warn", "penalize", "block"].index(b.get("mode", "penalize")))
        with col_d:
            ben_penalty = st.number_input("Penalty multiplier (when mode=penalize)", min_value=0.0, max_value=1.0, value=b.get("penalty", 0.8), step=0.05)

        if st.button("Save Benford preferences"):
            prefs.set_benford_prefs({"enabled": bool(ben_enabled), "min_samples": int(ben_min), "threshold": float(ben_thresh), "mode": str(ben_mode), "penalty": float(ben_penalty)})
            st.success("Saved Benford preferences.")

    with tab_diag:
        st.header("⚙️ Diagnostics & Runtime Keys (non-sensitive)")
        st.markdown("This tab helps debug which API keys are active and where they were resolved from.")

        # Environment checks: yfinance installed and basic connectivity to Yahoo endpoints
        try:
            from utils.env_checks import check_yfinance_installed, check_internet_connectivity, certifi_and_ssl_info, probe_endpoints
            y_ok, y_msg = check_yfinance_installed()
            net_ok, net_msg = check_internet_connectivity()
            st.subheader("Environment checks")
            col1, col2 = st.columns(2)
            with col1:
                if y_ok:
                    st.success(y_msg)
                else:
                    st.error(y_msg)
            with col2:
                if net_ok:
                    st.success(net_msg)
                else:
                    st.error(net_msg)
            if not y_ok or not net_ok:
                st.info("Index snapshots / live fetches may be unavailable. Ensure 'yfinance' is installed and your server can reach Yahoo finance endpoints.")
                # Provide quick troubleshooting steps for common SSL issues
                st.markdown("**Troubleshooting (Windows / PowerShell)**")
                md = """
                If you see SSL certificate verification errors when connecting to providers, try the following steps:

                1. Upgrade certifi (Python's CA bundle): `pip install --upgrade certifi`
                2. Locate certifi CA bundle path in Python and set the REQUESTS_CA_BUNDLE env var if needed.
                   Run the following commands in PowerShell to find the certifi CA path and (optionally) set the env var:

                   ```powershell
                   python -c "import certifi;print(certifi.where())"
                   # Copy the printed path (e.g. C:\\...\\cacert.pem) and then:
                   # setx REQUESTS_CA_BUNDLE "C:\\path\\to\\cacert.pem"
                   ```

                3. If your network uses a TLS-inspecting proxy, ask your admin for the proxy CA and add it to your system trust store.
                """
                st.markdown(md)
                # Provide a copyable PowerShell snippet as a code block
                ps_block = (
                    "pip install --upgrade certifi\n"
                    "python -c \"import certifi;print(certifi.where())\"\n"
                    "# If certifi path printed above is e.g. C:\\path\\to\\cacert.pem then run:\n"
                    "# setx REQUESTS_CA_BUNDLE \"C:\\path\\to\\cacert.pem\"\n"
                )
                st.code(ps_block, language="powershell")
                # Interactive environment probe
                st.markdown("---")
                st.subheader("Run environment check")
                st.write("This will probe common provider endpoints and report TLS/errors. Use the insecure probe only for local debugging.")
                insecure = st.checkbox("Insecure probe (disable TLS verification) — dev only", value=False)
                probe_symbol = st.text_input("Symbol to probe (for sample quotes)", value="AAPL")
                if st.button("Run environment check"):
                    try:
                        info = certifi_and_ssl_info()
                        st.markdown("**Runtime certifi / OpenSSL info**")
                        st.json(info)
                    except Exception as e:
                        st.warning(f"Failed to get certifi/OpenSSL info: {e}")

                    st.markdown("**Endpoint probes**")
                    try:
                        results = probe_endpoints(symbol=probe_symbol, verify=not insecure)
                        # Display each endpoint result clearly
                        for k, v in results.items():
                            if isinstance(v, dict) and v.get("ok"):
                                st.success(f"{k}: OK ({v.get('status', '')})")
                            else:
                                # show detailed error
                                err = v.get("error") if isinstance(v, dict) else str(v)
                                st.error(f"{k}: failed — {err}")
                                # if TLS error, show remediation snippet
                                if err and "CERTIFICATE_VERIFY_FAILED" in err or (err and "certificate verify failed" in err.lower()):
                                    st.markdown("**Detected TLS certificate verification failures. Try:**")
                                    st.code(ps_block, language="powershell")
                    except Exception as e:
                        st.error(f"Failed to probe endpoints: {e}")
        except Exception:
            # Do not fail diagnostics if env checks are unavailable
            pass

        # Build a small resolution table without revealing secret values.
        from config import get_runtime_key
        from utils.rates import fetch_rates

        keys = ["NEWS_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY", "FRED_API_KEY"]

        rows = []
        for k in keys:
            env_exists = k in os.environ
            env_val = os.environ.get(k)
            runtime = get_runtime_key(k)
            # detect streamlit secrets presence without revealing values
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

            rows.append({"key": k, "env_exists": env_exists, "env_empty": env_val == "" if env_exists else False, "secrets": secrets_present, "runtime_resolved": runtime is not None, "source": source})

        st.subheader("API key resolution (booleans only)")
        st.table(rows)

        st.subheader("Latest fetched macro rates")
        try:
            rates = fetch_rates()
            st.json(rates)
        except Exception as e:
            st.error(f"Failed to fetch rates: {e}")

        st.markdown("---")
        st.caption("Note: This tab intentionally avoids printing any secret values. 'module' means a module-level constant or fallback was used.")

        st.markdown("---")
        st.subheader("Benford diagnostic (probe a symbol)")
        try:
            sym = st.text_input("Symbol to probe for Benford diagnostic", value="AAPL")
            from data_fetchers.prices import get_ohlc
            from utils.benford import benford_report, expected_benford_percent

            if st.button("Run Benford diagnostic"):
                try:
                    df = get_ohlc(sym, period="6mo", interval="1d")
                except Exception as e:
                    st.error(f"Failed to fetch price data for {sym}: {e}")
                    df = None

                if df is not None:
                    # Normalize column names (get_ohlc already capitalizes)
                    if "Volume" in df.columns:
                        vols = df["Volume"].dropna().astype(float)
                        if len(vols) == 0:
                            st.info("No volume data available for this symbol.")
                        else:
                            report = benford_report(vols)
                            st.json({"score": report.get("score"), "total": report.get("total"), "chi2": report.get("chi2")})
                            obs_obj = report.get("obs_percent", [])
                            exp_obj = expected_benford_percent()
                            obs = list(obs_obj) if isinstance(obs_obj, (list, tuple)) else []
                            exp = list(exp_obj) if isinstance(exp_obj, (list, tuple)) else []
                            rows = []
                            for i in range(9):
                                rows.append({"digit": i + 1, "observed_pct": obs[i] if i < len(obs) else 0.0, "benford_pct": exp[i]})
                            st.table(rows)
                            # add a small bar chart (observed vs expected)
                            try:
                                import plotly.graph_objects as go
                                digits = [r["digit"] for r in rows]
                                obs_vals = [r["observed_pct"] for r in rows]
                                exp_vals = [r["benford_pct"] for r in rows]
                                fig = go.Figure()
                                fig.add_bar(x=digits, y=obs_vals, name="Observed %")
                                fig.add_bar(x=digits, y=exp_vals, name="Benford %")
                                fig.update_layout(barmode="group", title_text=f"Benford observed vs expected for {sym}")
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
                    else:
                        st.info("Volume column not available in fetched OHLC data for this symbol.")
        except Exception:
            st.write("Benford diagnostic not available in this environment.")

        # Developer helper: allow writing local secrets for convenience (local dev only)
        st.markdown("---")
        with st.expander("Developer: add API keys / write local secrets (local only)"):
            st.warning("Persisting secrets to disk is a security risk. Only use this on a trusted machine. This will write to .streamlit/secrets.toml.")
            try:
                import config as _config
                cols = st.columns(2)
                with cols[0]:
                    in_finnhub = st.text_input("FINNHUB_API_KEY", value=_config.FINNHUB_API_KEY or "")
                    in_fmp = st.text_input("FMP_API_KEY", value=_config.FMP_API_KEY or "")
                    in_news = st.text_input("NEWS_API_KEY", value=_config.NEWS_API_KEY or "")
                with cols[1]:
                    in_fred = st.text_input("FRED_API_KEY", value=_config.FRED_API_KEY or "")
                    in_alpha = st.text_input("ALPHA_VANTAGE_API_KEY", value=_config.ALPHA_VANTAGE_API_KEY or "")
                    in_openai = st.text_input("OPEN_AI_KEY", value=_config.OPEN_AI_KEY or "")

                if st.button("Save keys to local secrets (overwrite .streamlit/secrets.toml)"):
                    confirm = st.checkbox("I understand the security implications and want to write secrets to disk")
                    if confirm:
                        mapping = {
                            "FINNHUB_API_KEY": in_finnhub,
                            "FMP_API_KEY": in_fmp,
                            "NEWS_API_KEY": in_news,
                            "FRED_API_KEY": in_fred,
                            "ALPHA_VANTAGE_API_KEY": in_alpha,
                            "OPEN_AI_KEY": in_openai,
                        }
                        try:
                            _config.write_local_secrets(mapping)
                            st.success("Wrote secrets to .streamlit/secrets.toml")
                        except Exception as e:
                            st.error(f"Failed to write local secrets: {e}")
            except Exception:
                st.info("Config helper not available in this environment.")
