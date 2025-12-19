from typing import Dict, List
from utils import sqlite_store


class TradeSimulator:
    """Lightweight in-memory trade simulator for UI experiments.

    Stores positions and cash locally in memory. Not for production use.
    """
    def __init__(self, initial_cash: float = 10000.0):
        self.cash = float(initial_cash)
        self.positions: Dict[str, int] = {}
        try:
            sqlite_store.init()
        except Exception:
            pass

    def buy(self, symbol: str, price: float, qty: int) -> bool:
        cost = price * qty
        if cost > self.cash or qty <= 0:
            return False
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        try:
            sqlite_store.save_trade(int(__import__('time').time()), symbol, qty, price, 'buy')
        except Exception:
            pass
        return True

    def sell(self, symbol: str, price: float, qty: int) -> bool:
        if qty <= 0 or self.positions.get(symbol, 0) < qty:
            return False
        proceeds = price * qty
        self.cash += proceeds
        self.positions[symbol] -= qty
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        try:
            sqlite_store.save_trade(int(__import__('time').time()), symbol, qty, price, 'sell')
        except Exception:
            pass
        return True

    def history(self, limit: int = 100) -> List[Dict]:
        try:
            return sqlite_store.load_trades(limit=limit)
        except Exception:
            return []

    def net_worth(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for s, q in self.positions.items():
            p = prices.get(s, 0.0)
            total += p * q
        return float(total)
