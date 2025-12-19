import pandas as pd

NUMERIC_HINTS = {"open","high","low","close","adj_close","volume","vwap","rsi","macd","bb_up","bb_mid","bb_low"}

def sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if isinstance(col, str):
            if col.lower() in {"company","name","long_name","sector","industry","symbol"}:
                continue
            if col.lower() in NUMERIC_HINTS or pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = [c for c in df.columns if isinstance(c, str) and df[c].dtype == "O" and c.lower() not in {"company","name","sector","industry","symbol"}]
    if bad:
        df = df.drop(columns=bad)
    return df


def downsample_time_index(df: pd.DataFrame, max_points: int = 600) -> pd.DataFrame:
    """Downsample a time-indexed DataFrame by taking approximately evenly spaced rows
    (preserves first and last). Useful for plotting long series with Plotly to reduce
    memory/rendering time. If the DataFrame has fewer rows than max_points, returns as-is.
    """
    if df is None or getattr(df, "empty", True):
        return df
    n = len(df)
    if n <= max_points:
        return df
    # Always include first and last, and sample intermediate positions evenly
    step = max(1, int((n - 2) / (max_points - 2)))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return df.iloc[indices]
