import vectorbt as vbt

def run_backtest(price_series, entry_signal, exit_signal, init_cash=10000):
    pf = vbt.Portfolio.from_signals(
        price_series,
        entries=entry_signal,
        exits=exit_signal,
        init_cash=init_cash
    )
    stats = {
        "win_rate": pf.win_rate(),
        "sharpe_ratio": pf.sharpe_ratio(),
        "max_drawdown": pf.max_drawdown(),
        "total_return": pf.total_return()
    }
    return pf, stats
