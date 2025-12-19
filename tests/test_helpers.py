import math
from utils import helpers


def test_fmt_number_and_money():
    assert helpers.fmt_number(1234.5678) == "1,234.57"
    assert helpers.fmt_number(None) == "—"
    assert helpers.fmt_money(1234.5) == "$1,234.50"
    assert helpers.fmt_money(None) == "—"


def test_safe_float_and_to_scalar():
    import pandas as pd
    s = pd.Series([42.0])
    assert helpers._to_scalar(s) == 42.0
    assert math.isfinite(helpers._safe_float(s))
    assert math.isnan(helpers._safe_float(None))


def test_format_percent_and_badge():
    res = helpers.format_percent(1.234, places=1)
    # accept either '+1.2%' or '1.2%'
    assert res.lstrip('+') == "1.2%"
    assert helpers.format_percent(None) == "—"
    assert "favorable" in helpers.badge_text("P/E", True)
    assert "unfavorable" in helpers.badge_text("P/E", False)
        # Removed stray end patch comment
