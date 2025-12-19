import plotly.graph_objects as go
from typing import Optional


def apply_plotly_theme(fig: go.Figure, title: Optional[str] = None, x_title: Optional[str] = None, y_title: Optional[str] = None, dark: bool = True) -> go.Figure:
    """Apply a consistent, readable theme to Plotly figures used across the app.

    - title: optional chart title; if omitted, attempts to read from figure layout.
    - x_title / y_title: axis titles applied when present.
    - dark: when True use 'plotly_dark' template, otherwise 'plotly_white'.

    This implementation is defensive to avoid passing None into Plotly where a
    string is expected (helps static type checkers and avoids attribute access
    errors on incomplete Figure objects).
    """
    try:
        template = 'plotly_dark' if dark else 'plotly_white'

        # Determine a safe title text (never pass None)
        title_text: str = ""
        if isinstance(title, str) and title:
            title_text = title
        else:
            try:
                lt = getattr(fig.layout, "title", None)
                if lt is not None:
                    tt = getattr(lt, "text", None)
                    if isinstance(tt, str) and tt:
                        title_text = tt
            except Exception:
                title_text = title or ""

        fig.update_layout(
            template=template,
            title={"text": title_text, "x": 0.01, "xanchor": "left"},
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="v", x=0.98, y=0.95),
            hovermode="x unified",
            font=dict(family="Segoe UI, Roboto, Arial", size=12),
        )

        if x_title:
            fig.update_xaxes(title_text=x_title)
        if y_title:
            fig.update_yaxes(title_text=y_title)
    except Exception:
        # Keep the function safe for UI render even if Plotly internals change
        pass
    return fig
