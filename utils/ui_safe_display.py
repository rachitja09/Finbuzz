import traceback
from typing import Any
import pandas as pd
import streamlit as st

from utils.frames import sanitize_for_arrow


def display_df(df: pd.DataFrame, **st_kwargs: Any) -> None:
    """Safely display a DataFrame in Streamlit.

    Strategy:
    - Try to sanitize numeric columns with sanitize_for_arrow and call st.dataframe.
    - If st.dataframe raises (pyarrow/serialization), fall back to casting values to strings
      and call st.dataframe again.
    - As a last resort, call st.write to ensure the page keeps running.
    """
    if df is None:
        st.write("No data")
        return

    # Keep original small and avoid mutating caller's df
    try:
        safe = sanitize_for_arrow(df)
    except Exception:
        safe = df.copy()

    try:
        st.dataframe(safe, **st_kwargs)
        return
    except Exception:
        # Try a string-cast fallback
        try:
            display_df = df.fillna("—").astype(str)
            st.dataframe(display_df, **st_kwargs)
            return
        except Exception:
            # Final fallback: write as plain table/text to avoid crashing Streamlit
            try:
                st.write(df)
            except Exception:
                # Last-ditch: log the traceback to Streamlit UI
                st.text("Failed to render DataFrame. See server logs for details.")
                st.text(traceback.format_exc())


def safe_plotly_chart(fig: Any, target_container=None, **kwargs: Any) -> None:
    """Safely render a Plotly figure into Streamlit or a given container.

    If plotting fails, the function writes a friendly message instead of raising.
    """
    try:
        if target_container is None:
            st.plotly_chart(fig, **kwargs)
        else:
            target_container.plotly_chart(fig, **kwargs)
    except Exception:
        try:
            if target_container is None:
                st.warning("Unable to render Plotly chart for this dataset.")
            else:
                target_container.warning("Unable to render Plotly chart for this dataset.")
        except Exception:
            st.text("Chart rendering failed.")


def run_safe(callable_fn, *args, **kwargs):
    """Run a page or function safely: capture exceptions and show them in the UI instead of crashing."""
    try:
        return callable_fn(*args, **kwargs)
    except Exception as e:
        st.error(f"Page error: {e}")
        st.text(traceback.format_exc())
        return None
