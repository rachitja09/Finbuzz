"""
Simple premarket alert scaffold.

This script computes overnight/top movers from a watchlist and can deliver
notifications via SendGrid (email) or Slack (webhook). It's intentionally
lightweight and safe: it reads runtime keys via the app's `config.get_runtime_key`
helper and will not write secrets. Fill in or wire delivery functions as needed.

Usage (local):
  python scripts/premarket_alert.py

To schedule: use Windows Task Scheduler (PowerShell) or a cloud scheduler. The
script prints the digest to stdout by default so it's safe to run in dry-run mode.
"""
from __future__ import annotations

import json
import sys
from typing import List, Dict, Optional
from pathlib import Path

import pandas as pd

from data_fetchers.prices import get_ohlc, get_latest_quote
import config


def get_watchlist(defaults: Optional[List[str]] = None) -> List[str]:
    """Load a watchlist; fallback to a small default list.

    The app stores a portfolio in data/portfolio.json — if present we'll use
    its symbols. Otherwise return a sensible default list.
    """
    defaults = defaults or ["SPY", "QQQ", "TQQQ", "AAPL", "TSLA", "AMZN"]
    p = Path("data/portfolio.json")
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf8"))
            syms = [s.get("symbol", "").upper() for s in j.get("positions", []) if s.get("symbol")]
            if syms:
                return list(dict.fromkeys(syms))
        except Exception:
            pass
    return defaults


def compute_overnight_changes(symbols: List[str], period: str = "5d", interval: str = "1d") -> List[Dict]:
    """Return list of dicts: {symbol, pct_change, last_close, prev_close} sorted by abs(pct_change).

    Note: This is a simple heuristic using daily closes. For true premarket
    pricing, wire into a provider with pre/post-market quotes.
    """
    rows = []
    for s in symbols:
        try:
            # fetch recent OHLC to obtain previous close values
            df = get_ohlc(s, period=period, interval=interval)
            prev_close = None
            last_close = None
            if df is not None and not df.empty:
                cols = {c.lower(): c for c in df.columns}
                close_col = cols.get("close") or cols.get("Close") or None
                if close_col:
                    closes = pd.Series(df[close_col]).dropna()
                    if len(closes) >= 2:
                        last_close = float(closes.iloc[-1])
                        prev_close = float(closes.iloc[-2])

            # Try to get an extended-hours / latest quote (may prefer pre/post market price)
            latest = None
            try:
                latest = get_latest_quote(s) or None
            except Exception:
                latest = None

            if latest and latest.get("price") is not None and prev_close is not None:
                # Use latest quote (could be extended-hours) to compute pct vs prev_close
                lp = latest.get("price")
                latest_price = None
                try:
                    if isinstance(lp, (int, float, str)):
                        latest_price = float(lp)
                except Exception:
                    latest_price = None
                if latest_price is None:
                    # cannot parse latest price; skip to next symbol
                    continue
                pct = (latest_price - prev_close) / prev_close * 100.0 if prev_close != 0 else 0.0
                rows.append({
                    "symbol": s,
                    "pct_change": pct,
                    "last_close": latest_price,
                    "prev_close": prev_close,
                    "latest_ts": latest.get("ts"),
                    "extended": bool(latest.get("extended", False)),
                    "source": latest.get("source", "latest")
                })
            elif last_close is not None and prev_close is not None:
                # fallback to daily close change
                pct = (last_close - prev_close) / prev_close * 100.0 if prev_close != 0 else 0.0
                rows.append({"symbol": s, "pct_change": pct, "last_close": last_close, "prev_close": prev_close, "latest_ts": None, "extended": False, "source": "ohlc_close"})
            else:
                # insufficient data; skip
                continue
        except Exception:
            # don't fail the whole run on a single symbol
            continue
    # sort by absolute change descending
    rows = sorted(rows, key=lambda r: abs(r.get("pct_change", 0.0)), reverse=True)
    return rows


def build_digest(movers: List[Dict], top_n: int = 5) -> str:
    """Return a short text digest for the top movers."""
    top = movers[:top_n]
    lines = [f"Premarket movers (top {len(top)}):"]
    for r in top:
        sym = r["symbol"]
        pct = r["pct_change"]
        lines.append(f"- {sym}: {pct:+.2f}% (last {r['last_close']:.2f} prev {r['prev_close']:.2f})")
    if not top:
        lines.append("No movers found or no data available.")
    return "\n".join(lines)


def send_via_print(body: str) -> None:
    print(body)



# External delivery (Slack/SendGrid) removed by request. This script now prints the
# digest to stdout and returns; to add delivery, implement a secure adapter that
# reads runtime keys from environment/Streamlit secrets (do NOT hardcode keys).


def main(dry_run: bool = True) -> int:
    wl = get_watchlist()
    movers = compute_overnight_changes(wl, period="5d", interval="1d")
    digest = build_digest(movers, top_n=5)

    # Print digest to stdout. External delivery integrations were removed to keep
    # the script minimal and self-contained. If you want email or webhook delivery
    # re-added, implement secure adapters that load keys from the environment or
    # Streamlit secrets and validate them (do not paste keys into source).
    send_via_print(digest)
    if dry_run:
        print("Dry run mode — not sending external notifications.")
        return 0

    # In non-dry runs we still print and return success; delivery is intentionally
    # disabled unless a secure adapter is implemented.
    print("Notification delivery disabled (no external adapters configured).")
    return 0


if __name__ == "__main__":
    # Simple CLI: pass --send to actually deliver (be careful!)
    dry = True
    if len(sys.argv) > 1 and sys.argv[1] in ("--send", "-s"):
        dry = False
    rc = main(dry_run=dry)
    sys.exit(rc)
