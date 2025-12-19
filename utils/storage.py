import json
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.json")


def save_portfolio(df: pd.DataFrame) -> None:
    try:
        p = _path("portfolio")
        records = df.to_dict(orient="records")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception:
        pass


def load_portfolio() -> pd.DataFrame:
    p = _path("portfolio")
    if not os.path.exists(p):
        # explicit columns to keep DataFrame shape predictable
        return pd.DataFrame.from_records([], columns=["symbol", "qty", "avg_price"])
    try:
        with open(p, "r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame.from_records(records)
    except Exception:
        return pd.DataFrame.from_records([], columns=["symbol", "qty", "avg_price"]) 


def save_prefs(d: dict, name: str = "prefs") -> None:
    try:
        p = _path(name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def load_prefs(name: str = "prefs") -> dict:
    p = _path(name)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
