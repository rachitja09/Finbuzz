import pandas as pd


def sma_crossover_signals(price: pd.Series, short: int = 20, long: int = 50) -> pd.Series:
    """Return a Series of signals: 1 when short SMA crosses above long SMA (buy), -1 when crosses below (sell), 0 otherwise.

    The returned Series aligns with the input price index.
    """
    if price is None or len(price) == 0:
        return pd.Series(dtype=float)
    # Ensure price is a pandas Series
    price = pd.Series(price, dtype=float)
    short_sma = pd.Series(price.rolling(short).mean(), index=price.index, dtype=float)
    long_sma = pd.Series(price.rolling(long).mean(), index=price.index, dtype=float)
    # signal when short crosses long using pandas Series methods
    prev_short = short_sma.shift(1)
    prev_long = long_sma.shift(1)
    cross = pd.Series((short_sma > long_sma) & (prev_short <= prev_long), index=price.index)
    cross_sell = pd.Series((short_sma < long_sma) & (prev_short >= prev_long), index=price.index)
    sig = pd.Series(0, index=price.index, dtype=int)
    sig.loc[cross.fillna(False)] = 1
    sig.loc[cross_sell.fillna(False)] = -1
    return sig


def run_sma_backtest(price: pd.Series, short: int = 20, long: int = 50, initial_cash: float = 10000.0) -> dict:
    """Run a simple SMA crossover backtest: enter at next-open price on signal, exit on opposite signal.

    Returns a summary dict with final value, total returns, number of trades, win rate, and equity curve.
    This is a deterministic, no-leverage simulation for demonstration only.
    """
    out = {
        "final_value": initial_cash,
        "returns_pct": 0.0,
        "trades": 0,
        "wins": 0,
        "equity_curve": pd.Series(dtype=float),
        "trades_list": [],  # list of trade dicts: entry_date, exit_date, entry_price, exit_price, profit
        "summary": {},
    }
    if price is None or len(price) == 0:
        return out

    price = pd.Series(price, dtype=float).dropna()
    signals = sma_crossover_signals(price, short=short, long=long)
    cash = initial_cash
    position = 0.0
    equity = []
    entry_price = 0.0
    wins = 0
    trades = 0
    entry_idx = None
    entries = []
    for idx in price.index:
        sig = int(signals.loc[idx]) if idx in signals.index else 0
        p_val = price.loc[idx]
        if p_val is None or (isinstance(p_val, float) and pd.isna(p_val)):
            continue
        p = float(p_val)
        # Entry
        if sig == 1 and position == 0:
            # buy as many shares as possible at price p
            shares = cash // p
            if shares > 0:
                position = shares
                entry_price = p
                entry_idx = idx
                cash -= shares * p
                trades += 1
        # Exit on sell and have position
        elif sig == -1 and position > 0:
            proceeds = position * p
            profit = proceeds - (position * entry_price)
            if profit > 0:
                wins += 1
            # record trade
            try:
                entries.append({
                    "entry_date": entry_idx,
                    "exit_date": idx,
                    "entry_price": float(entry_price),
                    "exit_price": float(p),
                    "shares": int(position),
                    "profit": float(profit),
                })
            except Exception:
                pass
            cash += proceeds
            position = 0
            entry_idx = None
        total = cash + position * p
        equity.append(total)

    final_value = cash + position * p
    returns_pct = ((final_value - initial_cash) / initial_cash) * 100.0
    out["final_value"] = float(final_value)
    out["returns_pct"] = float(returns_pct)
    out["trades"] = int(trades)
    out["wins"] = int(wins)
    out["equity_curve"] = pd.Series(equity, index=price.index)
    out["trades_list"] = entries
    out["summary"] = {"final_value": out["final_value"], "returns_pct": out["returns_pct"], "trades": out["trades"], "wins": out["wins"]}
    return out
