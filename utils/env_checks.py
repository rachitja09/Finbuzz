import importlib
import socket
from typing import Tuple
import ssl
import requests
import certifi
from typing import Dict, Any


def check_yfinance_installed() -> Tuple[bool, str]:
    try:
        import yfinance  # type: ignore
        return True, "yfinance installed"
    except Exception:
        # attempt to import via importlib to provide a clearer message
        try:
            importlib.import_module("yfinance")
            return True, "yfinance installed"
        except Exception:
            return False, "yfinance not importable; install with: pip install yfinance"


def check_internet_connectivity(host: str = "query1.finance.yahoo.com", port: int = 443, timeout: float = 3.0) -> Tuple[bool, str]:
    """Quick socket check to see if the app can reach a Yahoo finance endpoint.

    This is not a guarantee of full API availability but is a fast heuristic.
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True, f"Can reach {host}:{port}"
    except Exception as e:
        return False, f"Unable to reach {host}:{port} ({e})"


def certifi_and_ssl_info() -> Dict[str, str]:
    """Return certifi CA path and OpenSSL version used by this Python runtime."""
    info: Dict[str, str] = {}
    try:
        info["certifi"] = certifi.where()
    except Exception as e:  # pragma: no cover - environment-dependent
        info["certifi"] = f"error: {e}"
    try:
        info["openssl"] = ssl.OPENSSL_VERSION
    except Exception as e:  # pragma: no cover - environment-dependent
        info["openssl"] = f"error: {e}"
    return info


def probe_endpoints(symbol: str = "AAPL", verify: bool = True, timeout: float = 8.0) -> Dict[str, Any]:
    """Probe a small set of provider endpoints and return status and error messages.

    - verify: whether to verify TLS certificates (set False for dev/insecure testing)
    """
    results: Dict[str, Any] = {}
    # Yahoo / yfinance sample URL (simple socket check already done by check_internet_connectivity)
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}", timeout=timeout, verify=verify)
        results["yahoo_quote"] = {"status": r.status_code, "ok": r.ok}
    except Exception as e:
        results["yahoo_quote"] = {"ok": False, "error": str(e)}

    # Finnhub (requires key) — probe only if key present
    try:
        from config import get_runtime_key
        fh_key = get_runtime_key("FINNHUB_API_KEY")
    except Exception:
        fh_key = None
    if not fh_key:
        results["finnhub"] = {"ok": False, "error": "no key configured"}
    else:
        try:
            r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": symbol, "token": fh_key}, timeout=timeout, verify=verify)
            results["finnhub"] = {"status": r.status_code, "ok": r.ok}
        except Exception as e:
            results["finnhub"] = {"ok": False, "error": str(e)}

    # FinancialModelingPrep company profile (requires key)
    try:
        from config import get_runtime_key
        fmp_key = get_runtime_key("FMP_API_KEY")
    except Exception:
        fmp_key = None
    if not fmp_key:
        results["fmp"] = {"ok": False, "error": "no key configured"}
    else:
        try:
            r = requests.get("https://financialmodelingprep.com/api/v4/company-profile", params={"symbol": symbol, "apikey": fmp_key}, timeout=timeout, verify=verify)
            results["fmp"] = {"status": r.status_code, "ok": r.ok}
        except Exception as e:
            results["fmp"] = {"ok": False, "error": str(e)}

    # FRED sample probe
    try:
        from config import get_runtime_key
        fred_key = get_runtime_key("FRED_API_KEY")
    except Exception:
        fred_key = None
    if not fred_key:
        # still probe public treasury URL as fallback
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": "DGS10", "file_type": "json", "limit": 1}, timeout=timeout, verify=verify)
            results["fred"] = {"status": r.status_code, "ok": r.ok}
        except Exception as e:
            results["fred"] = {"ok": False, "error": str(e)}
    else:
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": "DGS10", "api_key": fred_key, "file_type": "json", "limit": 1}, timeout=timeout, verify=verify)
            results["fred"] = {"status": r.status_code, "ok": r.ok}
        except Exception as e:
            results["fred"] = {"ok": False, "error": str(e)}

    return results