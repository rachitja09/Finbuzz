from .storage import load_prefs, save_prefs


def get_theme_pref():
    prefs = load_prefs()
    return prefs.get("theme", "dark")


def set_theme_pref(theme: str):
    prefs = load_prefs()
    prefs["theme"] = theme
    save_prefs(prefs)


def get_keyboard_shortcuts():
    prefs = load_prefs()
    return prefs.get("shortcuts", {})


def set_keyboard_shortcuts(d: dict):
    prefs = load_prefs()
    prefs["shortcuts"] = d
    save_prefs(prefs)


def get_data_source_pref():
    prefs = load_prefs()
    # values: "Auto (live first)", "Live (require keys)", "Mock only"
    return prefs.get("data_source", "Auto (live first)")


def set_data_source_pref(mode: str):
    prefs = load_prefs()
    prefs["data_source"] = mode
    save_prefs(prefs)


def get_dev_mock_pref() -> bool:
    prefs = load_prefs()
    return bool(prefs.get("dev_mock", False))


def set_dev_mock_pref(v: bool):
    prefs = load_prefs()
    prefs["dev_mock"] = bool(v)
    save_prefs(prefs)


def get_strategy_weights():
    """Return saved default weights for the recommendation model.

    Returns a dict with keys: tech_w, analyst_w, consensus_w, news_w
    """
    prefs = load_prefs()
    w = prefs.get("strategy_weights", {}) or {}
    return {
        "tech_w": float(w.get("tech_w", 1.0)),
        "analyst_w": float(w.get("analyst_w", 0.7)),
        "consensus_w": float(w.get("consensus_w", 0.5)),
        "news_w": float(w.get("news_w", 0.3)),
    }


def set_strategy_weights(weights: dict):
    """Persist default strategy weights.

    Expects a dict with numeric values for tech_w, analyst_w, consensus_w, news_w.
    """
    prefs = load_prefs()
    prefs["strategy_weights"] = {
        "tech_w": float(weights.get("tech_w", 1.0)),
        "analyst_w": float(weights.get("analyst_w", 0.7)),
        "consensus_w": float(weights.get("consensus_w", 0.5)),
        "news_w": float(weights.get("news_w", 0.3)),
    }
    save_prefs(prefs)


def get_benford_prefs():
    """Return Benford-related preferences.

    Returns dict with keys:
      - enabled: bool
      - min_samples: int
      - threshold: float
    """
    prefs = load_prefs()
    b = prefs.get("benford", {}) or {}
    return {
        "enabled": bool(b.get("enabled", True)),
        "min_samples": int(b.get("min_samples", 50)),
        "threshold": float(b.get("threshold", 12.0)),
        # mode: 'warn' | 'penalize' | 'block'
        "mode": str(b.get("mode", "penalize")),
        # penalty multiplier when mode == 'penalize'
        "penalty": float(b.get("penalty", 0.8)),
    }


def set_benford_prefs(d: dict):
    prefs = load_prefs()
    prefs["benford"] = {
        "enabled": bool(d.get("enabled", True)),
        "min_samples": int(d.get("min_samples", 50)),
        "threshold": float(d.get("threshold", 12.0)),
        "mode": str(d.get("mode", "penalize")),
        "penalty": float(d.get("penalty", 0.8)),
    }
    save_prefs(prefs)


def get_rl_prefs():
    prefs = load_prefs()
    r = prefs.get("rl", {}) or {}
    return {"enabled": bool(r.get("enabled", False)), "lr": float(r.get("lr", 0.05))}


def set_rl_prefs(d: dict):
    prefs = load_prefs()
    prefs["rl"] = {"enabled": bool(d.get("enabled", False)), "lr": float(d.get("lr", 0.05))}
    save_prefs(prefs)
