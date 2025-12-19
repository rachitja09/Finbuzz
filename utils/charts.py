import numpy as np
import pandas as pd
from typing import Tuple, Optional, TYPE_CHECKING, Any, cast

# Plotly is optional at import-time; import types only for type checking so
# static checkers know the symbols while runtime gracefully handles missing plotly.
if TYPE_CHECKING:  # pragma: no cover - static-only
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    _PLOTLY_AVAILABLE = False

# For static type checkers, cast the runtime symbols to Any so usages like
# go.Candlestick / make_subplots are accepted by Pylance even when the module
# is optional at runtime.
go_any = cast(Any, go)
make_subplots_any = cast(Any, make_subplots)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return pd.Series(out, index=series.index, dtype=float)


def bollinger_bands(series: pd.Series, window: int = 20, dev: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + dev * sd
    lower = ma - dev * sd
    return pd.Series(upper, index=series.index, dtype=float), pd.Series(lower, index=series.index, dtype=float)


def candle_with_rsi_bbands(
    ohlc: pd.DataFrame,
    title: str,
    height: int = 520,
    show_bbands: bool = True,
    show_volume: bool = True,
) -> Optional[Any]:
    """Create a candlestick chart with RSI and optional Bollinger Bands + volume.

    ohlc: DataFrame with columns Open, High, Low, Close, optionally Volume.
    """
    if not _PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is not available in this environment")
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.25, 0.2] if show_volume else [0.75, 0.25]
    fig = make_subplots_any(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    # Candlestick
    Candlestick = getattr(go_any, "Candlestick", None)
    if Candlestick is not None:
        # cleaner hover template with date and OHLC values
        hover_tmpl = "Date: %{x|%Y-%m-%d}<br>Open: $%{customdata[0]:,.2f}<br>High: $%{customdata[1]:,.2f}<br>Low: $%{customdata[2]:,.2f}<br>Close: $%{customdata[3]:,.2f}<extra></extra>"
        open_arr = pd.Series(ohlc["Open"], index=ohlc.index, dtype=float).to_numpy()
        high_arr = pd.Series(ohlc["High"], index=ohlc.index, dtype=float).to_numpy()
        low_arr = pd.Series(ohlc["Low"], index=ohlc.index, dtype=float).to_numpy()
        close_arr = pd.Series(ohlc["Close"], index=ohlc.index, dtype=float).to_numpy()
        fig.add_trace(
            Candlestick(
                x=ohlc.index.to_numpy(),
                open=open_arr,
                high=high_arr,
                low=low_arr,
                close=close_arr,
                customdata=np.stack([open_arr, high_arr, low_arr, close_arr], axis=1),
                name="Price",
                increasing_line_color="#1f77b4",
                decreasing_line_color="#d62728",
                showlegend=True,
                hovertemplate=hover_tmpl,
            ),
            row=1,
            col=1,
        )

    # Bollinger bands
    if show_bbands:
        try:
            close_series = pd.Series(ohlc["Close"], index=ohlc.index, dtype=float)
            upper, lower = bollinger_bands(close_series)
            Scatter = getattr(go_any, "Scatter", None)
            if Scatter is not None:
                fig.add_trace(Scatter(x=ohlc.index.to_numpy(), y=upper.to_numpy(), mode="lines", line=dict(width=1), name="BB Upper", marker=dict(color="rgba(31,119,180,0.2)"), hovertemplate="Date: %{x|%Y-%m-%d}<br>BB Upper: $%{y:,.2f}<extra></extra>"), row=1, col=1)
                fig.add_trace(Scatter(x=ohlc.index.to_numpy(), y=lower.to_numpy(), mode="lines", line=dict(width=1), name="BB Lower", marker=dict(color="rgba(31,119,180,0.2)"), hovertemplate="Date: %{x|%Y-%m-%d}<br>BB Lower: $%{y:,.2f}<extra></extra>"), row=1, col=1)
        except Exception:
            pass

    # RSI
    try:
        # ensure Series float type for rsi helper
        close_series = pd.Series(ohlc["Close"], index=ohlc.index, dtype=float)
        r = rsi(close_series)
        Scatter = getattr(go_any, "Scatter", None)
        if Scatter is not None:
            fig.add_trace(Scatter(x=ohlc.index.to_numpy(), y=r.to_numpy(), mode="lines", name="RSI(14)", line=dict(color="#ff7f0e"), hovertemplate="Date: %{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>"), row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#888", row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#888", row=2, col=1)
    except Exception:
        pass

    # Volume
    if show_volume and "Volume" in ohlc.columns:
        try:
            Bar = getattr(go_any, "Bar", None)
            if Bar is not None:
                fig.add_trace(Bar(x=ohlc.index.to_numpy(), y=pd.Series(ohlc["Volume"], index=ohlc.index, dtype=float).to_numpy(), name="Volume", marker_color="rgba(100,100,100,0.6)", hovertemplate="Date: %{x|%Y-%m-%d}<br>Volume: %{y:,}<extra></extra>"), row=3, col=1)
        except Exception:
            pass

    # layout polish
    # layout polish (use shared theme helper if available)
    try:
        from ui.plotting import apply_plotly_theme

        fig = apply_plotly_theme(fig, title=title, x_title='Date', y_title='Price (USD)', dark=True)
        fig.update_layout(height=height)
    except Exception:
        fig.update_layout(
            title=title,
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )

    # Format x axis as dates with reasonable tick spacing
    fig.update_xaxes(title_text="Date", rangeslider_visible=False, tickformat="%b %d\n%Y")
    # Price axis currency formatting
    fig.update_yaxes(title_text="Price", row=1, col=1, tickprefix="$", separatethousands=True)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    if show_volume and rows == 3:
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    return fig


def analyst_target_summary(target: dict, current_price: Optional[float] = None) -> dict:
    """Return a small summary of analyst targets and a suggestion.

    target: dict with keys targetMean, targetHigh, targetLow
    """
    def _tofloat(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    mean = _tofloat(target.get("targetMean"))
    high = _tofloat(target.get("targetHigh"))
    low = _tofloat(target.get("targetLow"))

    verdict = "No opinion"
    try:
        if mean is not None and current_price is not None:
            if mean > current_price * 1.05:
                verdict = "Buy"
            elif mean < current_price * 0.95:
                verdict = "Sell"
            else:
                verdict = "Hold"
    except Exception:
        verdict = "No opinion"

    return {"mean": mean, "high": high, "low": low, "verdict": verdict}

