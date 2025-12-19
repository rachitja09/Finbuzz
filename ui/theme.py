import streamlit as st


def apply_theme(accent="#1167b1"):
    """Inject a minimal, safe CSS theme to improve typography and card spacing.

    Keep styles conservative to avoid breaking Streamlit internal layout.
    """
    try:
        css = f"""
        <style>
        :root {{ --accent: {accent}; }}
        /* Improve base font and spacing */
        html, body, .stApp {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; font-size: 14px; }}
        h1, h2, h3, h4 {{ color: #0b3d91; }}
        /* Tighter metric cards */
        .stMetric {{ padding: 6px 8px !important; }}
        /* Muted captions */
        .stCaption, .css-1v0mbdj-egHxvX {{ color: #606f7b !important; font-size:12px; }}
        /* Make badges look a bit nicer when rendered as HTML */
        .provider-badge {{ display:inline-block; padding:6px 10px; border-radius:6px; color:#fff; font-weight:600; margin-right:6px }}
        /* Buttons: use accent */
        button[role="button"] {{ background: var(--accent) !important; color: #fff !important; }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass
