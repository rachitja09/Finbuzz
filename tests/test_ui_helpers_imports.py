import os
import pandas as pd

# Ensure Streamlit runs headless during tests
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "1")


def _sample_close(n=120):
    return pd.DataFrame({"close": pd.Series([100 + i * 0.5 for i in range(n)], dtype=float)})


def test_ui_metrics_compute_kpis_imports():
    from pages.ui_metrics import compute_kpis

    df = _sample_close()
    k = compute_kpis(df, symbol="TEST")
    assert set(k.keys()) == {"sharpe", "sortino", "vol", "mdd", "mom_3m", "mom_6m"}


def test_ui_forecast_compute_projection():
    from pages.ui_forecast import compute_projection

    df = _sample_close()
    proj = compute_projection(df, horizon_years=5, symbol="TEST")
    assert set(proj.keys()) == {"expected", "growth", "upper", "lower"}


def test_ui_scenarios_compute_scenarios():
    from pages.ui_scenarios import compute_scenarios

    close = _sample_close()["close"]
    last = float(close.iloc[-1])
    vol = float(close.pct_change().dropna().std() * (252 ** 0.5))
    mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1)
    scenarios = compute_scenarios(close, last, vol, mom_3m, [0.3, 0.5, 0.2])
    assert len(scenarios) == 3
    assert all("price" in s for s in scenarios)
