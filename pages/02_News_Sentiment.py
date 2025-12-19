import sys
import streamlit as st
import pandas as pd
from data_fetchers.news import newsapi_headlines, vader_score

st.set_page_config(page_title="News & Sentiment", layout="wide")
try:
    st.title("🗞️ News & Sentiment")
except UnicodeEncodeError:
    try:
        st.title("News & Sentiment")
    except Exception:
        pass

# Consolidate two news sources into tabs: Headlines (NewsAPI) and Global (GDELT)
if "pytest" not in sys.modules:
    try:
        from ui.provider_banner import show_provider_banner
        show_provider_banner()
    except Exception:
        pass
    tab_head, tab_gdelt = st.tabs(["Headlines & Sentiment", "Global (GDELT)"])

    # --- Headlines tab (existing) ---
    with tab_head:
        query = st.text_input("Query / Ticker", "AAPL").strip()
        limit = st.slider("Headlines to analyze", 5, 50, 20, 5)

        arts = newsapi_headlines(query, limit=limit)

        if not arts:
            st.info("No NewsAPI headlines (or key missing). Add NEWS_API_KEY to secrets.")
        else:
            # Build a safe list rendering for titles/links to avoid sanitizer dropping string cols
            items = []
            rows = []
            for a in arts:
                if not isinstance(a, dict):
                    continue
                title = (a.get("title") or "").strip()
                desc = (a.get("description") or "").strip()
                url = (a.get("url") or "").strip()
                score = 0.0
                try:
                    score = float(vader_score(f"{title}. {desc}"))
                except Exception:
                    score = 0.0
                src = a.get("source", {}) or {}
                src_name = src.get("name") if isinstance(src, dict) else (a.get("source") or "")
                # markdown link line
                link_md = f"[{title}]({url})" if url else title
                items.append((score, src_name, link_md, title, url, desc))
                rows.append({"title": title, "source": src_name, "url": url, "sentiment": score, "description": desc})

            # sort by sentiment descending (most positive first)
            try:
                items.sort(key=lambda x: x[0], reverse=True)
            except Exception:
                pass

            st.subheader(f"Top {min(len(items), limit)} headlines")
            for score, src_name, link_md, title, url, desc in items:
                try:
                    st.markdown(f"- {link_md}  ")
                    st.caption(f"Source: {src_name} · Sentiment: {score:+.3f}")
                    # optionally show description
                    if desc:
                        st.write(desc)
                except Exception:
                    # fallback: write plain text
                    st.write(title)

            # provide CSV download of headlines
            try:
                tdf = pd.DataFrame(rows)
                if not tdf.empty:
                    csv_bytes = tdf.to_csv(index=False).encode("utf-8")
                    st.download_button("Download headlines CSV", data=csv_bytes, file_name=f"{query}_headlines.csv", mime="text/csv")
            except Exception:
                pass

            st.caption("VADER scores headline+description. (NewsAPI dev plan may have ~24h delay.)")

    # --- GDELT tab (moved from News_GDELT.py) ---
    with tab_gdelt:
        st.caption("Global News (GDELT) — cached ~15 minutes. Be a good citizen and avoid high-frequency polling.")
        try:
            from datetime import datetime, timedelta, timezone
            from gdelt_client import GDELT

            @st.cache_data(ttl=900)
            def cached_doc_search(query: str, start: str | None, end: str | None, maxrecords: int, mode: str):
                g = GDELT(min_interval_sec=1.0)
                return g.doc_search(query, startdatetime=start, enddatetime=end, maxrecords=maxrecords, mode=mode)

            @st.cache_data(ttl=900)
            def cached_context_search(query: str, maxrecords: int = 100):
                g = GDELT(min_interval_sec=1.0)
                return g.context_search(query, maxrecords=maxrecords)

            @st.cache_data(ttl=900)
            def cached_geo_search(query: str, mode: str = "PointData", format_: str = "CSV"):
                g = GDELT(min_interval_sec=1.0)
                return g.geo_search(query, mode=mode, format_=format_)

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=7)
            start_str = start_dt.strftime("%Y%m%d%H%M%S")
            end_str = end_dt.strftime("%Y%m%d%H%M%S")

            col1, col2 = st.columns([3, 1])
            with col1:
                use_context = st.checkbox("Use Context 2.0 (sentence snippets)", value=False)
                q = st.text_input("GDELT Query", value="NVIDIA OR NVDA AND earnings")
                mode = st.selectbox("DOC mode", ["artlist", "timelinevol", "timelinesent"], index=0)
                maxrecs = st.slider("Max records", 10, 250, 50)

                if not q:
                    st.info("Enter a query to search GDELT.")
                else:
                    if use_context:
                        try:
                            res = cached_context_search(q, maxrecords=maxrecs)
                            arts = res.get("articles", [])
                            if not arts:
                                st.info("No context matches")
                            else:
                                st.header("Context snippets")
                                for a in arts[:maxrecs]:
                                    ctx = a.get("context") or a.get("snippet") or ""
                                    st.write(f"- {ctx}  ")
                        except Exception as e:
                            st.error(f"Context search failed: {e}")
                    else:
                        try:
                            res = cached_doc_search(q, start_str, end_str, maxrecs, mode)
                            arts = res.get("articles", [])
                            df = pd.DataFrame([
                                {
                                    "title": a.get("title"),
                                    "source": a.get("source"),
                                    "url": a.get("url"),
                                    "time": a.get("seendate"),
                                    "lang": a.get("language"),
                                }
                                for a in arts
                            ])
                            st.header("Articles")
                            if df.empty:
                                st.info("No articles found")
                            else:
                                from utils.ui_safe_display import display_df
                                display_df(df, use_container_width=True)
                        except Exception as e:
                            st.error(f"DOC search failed: {e}")

            with col2:
                try:
                    geo_mode = st.selectbox("GEO mode", ["PointData", "CountryInfo", "Admin1Info"], index=0)
                    df_geo = cached_geo_search(q, mode=geo_mode, format_="CSV")
                    st.header("Geo / Map")
                    if isinstance(df_geo, pd.DataFrame) and not df_geo.empty:
                        if "country" in df_geo.columns and "count" in df_geo.columns:
                            st.bar_chart(df_geo.set_index("country")["count"])
                        else:
                            from utils.ui_safe_display import display_df
                            display_df(df_geo.head(50), use_container_width=True)
                    else:
                        st.info("No geo data returned")
                except Exception as e:
                    st.error(f"GEO search failed: {e}")
        except Exception:
            st.info("GDELT client not available in this environment.")