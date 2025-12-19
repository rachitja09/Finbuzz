import streamlit as st

def apply_dark_theme():
    st.markdown("""
    <style>
        .css-18e3th9 {background-color: #0E1117;}
        .css-1d391kg {background-color: #1E2230;}
        .st-bx {background-color: #1E2230;}
        .st-bx .st-bx {background-color: #1E2230;}
    </style>
    """, unsafe_allow_html=True)


def theme_toggle():
    """Show a small theme toggle and persist preference in storage/prefs."""
    from utils.prefs import get_theme_pref, set_theme_pref
    cur = get_theme_pref()
    col1, col2 = st.columns([1, 6])
    with col1:
        t = st.selectbox("Theme", ["dark", "light"], index=0 if cur == "dark" else 1)
        if t != cur:
            set_theme_pref(t)
    # keep existing CSS for dark only
    if cur == "dark":
        apply_dark_theme()
