import streamlit as st

from utils.supply_chain import get_supply_chain, get_kpis
try:
    from config import get_runtime_key
    FINNHUB_API_KEY = get_runtime_key("FINNHUB_API_KEY")
except Exception:
    try:
        from config import FINNHUB_API_KEY
    except Exception:
        FINNHUB_API_KEY = None
from utils.helpers import fmt_number, _safe_float, fmt_money, format_percent, millify_short

st.set_page_config(page_title="Supply Chain", layout="wide")
st.title("🔗 Supply Chain Explorer")

try:
    from ui.provider_banner import show_provider_banner
    show_provider_banner()
except Exception:
    pass


def select_symbol_from_state() -> str:
    # Prefer UI state symbol if present (from Home page)
    try:
        return st.session_state.ui.get("symbol", "AAPL")
    except Exception:
        return "AAPL"


sym = (st.text_input("Symbol", value=select_symbol_from_state()).strip() or "AAPL").upper()

# Optionally enrich with live peers (Finnhub) when available
enrich_live = st.checkbox("Enrich with Finnhub peers (best-effort)", value=False)
data = get_supply_chain(sym, enrich_live=enrich_live)
if enrich_live:
    # check if key is available
    fh = FINNHUB_API_KEY or None
    if not fh:
        st.info("Finnhub enrichment enabled but no FINNHUB_API_KEY configured; enrichment will be best-effort.")
colL, colC, colR = st.columns([2, 3, 2])

with colL:
    st.subheader("Suppliers")
    suppliers = data.get("suppliers", [])
    if not suppliers:
        st.info("No supplier data available for this symbol (mock provider).")
    else:
        for i, s in enumerate(suppliers):
            sym_s = (s.get('symbol') or '').upper()
            label = f"{sym_s} — {s.get('name')}"
            # include the loop index in the key to ensure uniqueness
            if st.button(label, key=f"sup_{sym}_{sym_s}_{i}"):
                # Set global UI symbol and rerun so other pages can pick it up
                try:
                    if "ui" not in st.session_state:
                        st.session_state.ui = {"symbol": sym_s}
                    else:
                        st.session_state.ui["symbol"] = sym_s
                except Exception:
                    st.session_state["symbol"] = sym_s
                getattr(st, "experimental_rerun", lambda: None)()

with colR:
    st.subheader("Customers")
    customers = data.get("customers", [])
    if not customers:
        st.info("No customer data available for this symbol (mock provider).")
    else:
        for i, c in enumerate(customers):
            sym_c = (c.get('symbol') or '').upper()
            label = f"{sym_c} — {c.get('name')}"
            # include the index to avoid duplicate element keys
            if st.button(label, key=f"cust_{sym}_{sym_c}_{i}"):
                try:
                    if "ui" not in st.session_state:
                        st.session_state.ui = {"symbol": sym_c}
                    else:
                        st.session_state.ui["symbol"] = sym_c
                except Exception:
                    st.session_state["symbol"] = sym_c
                getattr(st, "experimental_rerun", lambda: None)()

with colC:
    st.subheader(f"{sym} — Supply Chain Overview")
    if not suppliers and not customers:
        st.write("No supply-chain data for this symbol. You can extend `utils/supply_chain.py` to use a real provider.")
    else:
        # Build a simple network visualization using networkx if available
        try:
            # import networkx dynamically to avoid static analysis errors when networkx
            # is not installed in lightweight test/CI environments
            import importlib

            nx = importlib.import_module("networkx")
            import plotly.graph_objects as go

            G = nx.DiGraph()
            # center node
            G.add_node(sym, label=sym)
            for s in suppliers:
                G.add_node(s.get('symbol'))
                G.add_edge(s.get('symbol'), sym)
            for c in customers:
                G.add_node(c.get('symbol'))
                G.add_edge(sym, c.get('symbol'))

            # layout
            pos = nx.spring_layout(G, seed=42)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = []
            node_y = []
            hover_texts = []
            nodes = list(G.nodes())
            for n in nodes:
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                # fetch lightweight KPIs (cached) and build hover text
                try:
                    k = get_kpis(n)
                    name = k.get("name") or n
                    price = k.get("price")
                    mkt = k.get("mktCap")
                    pe = k.get("pe")
                    dp = k.get("delta_pct")
                    parts = [f"{n}: {name}"]
                    if price is not None:
                        parts.append(f"Price: {fmt_money(price)}")
                    if dp is not None:
                        try:
                            parts.append(f"Δ%: {format_percent(dp/100 if abs(dp)>1 else dp)}")
                        except Exception:
                            parts.append(f"Δ%: {dp}")
                    if mkt is not None:
                        try:
                            mval = _safe_float(mkt)
                            parts.append(f"MktCap: ${millify_short(mval, places=2)}")
                        except Exception:
                            try:
                                parts.append(f"MktCap: ${millify_short(float(mkt), places=2)}")
                            except Exception:
                                parts.append(f"MktCap: {fmt_number(mkt)}")
                    if pe is not None:
                        parts.append(f"PE: {fmt_number(pe)}")
                    hover_texts.append("\n".join(parts))
                except Exception:
                    hover_texts.append(n)

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo='none', mode='lines')
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=nodes, textposition='bottom center', hoverinfo='text', hovertext=hover_texts, marker=dict(size=20, color=['#2b8cbe' if n==sym else '#a6bddb' for n in nodes]))
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)

            # simple expand control: let user choose a related node to expand
            st.markdown("**Expand a related company**")
            choice = st.selectbox("Show supply chain for:", options=[sym] + nodes, index=0, key="supply_expand")
            if choice and choice != sym:
                # fetch and replace current lists with the chosen symbol's supply chain
                new = get_supply_chain(choice, enrich_live=enrich_live)
                # update main columns by writing directly to session state symbol and rerunning
                try:
                    if "ui" not in st.session_state:
                        st.session_state.ui = {"symbol": choice}
                    else:
                        st.session_state.ui["symbol"] = choice
                except Exception:
                    st.session_state["symbol"] = choice
                # show expanded results below
                st.write("---")
                st.subheader(f"Expanded: {choice}")
                st.markdown("**Suppliers:**")
                for s in new.get("suppliers", []):
                    st.write(f"- {s.get('symbol')} — {s.get('name')}")
                st.markdown("**Customers:**")
                for c in new.get("customers", []):
                    st.write(f"- {c.get('symbol')} — {c.get('name')}")
        except Exception:
            # fallback textual list
            st.markdown("**Suppliers:**")
            for s in suppliers:
                st.write(f"- {s.get('symbol')} — {s.get('name')}")
            st.markdown("**Customers:**")
            for c in customers:
                st.write(f"- {c.get('symbol')} — {c.get('name')}")

    st.caption("Data is from the local mock provider. To enable live/up-to-date data, implement a real provider in `utils/supply_chain.py` (see code comments).\nClick a related company to browse its supply chain.")
