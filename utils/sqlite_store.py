"""Tiny SQLite-backed store for simulator history.
"""
from __future__ import annotations
import sqlite3
import os
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'simulator.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _conn():
    return sqlite3.connect(DB_PATH)

def init():
    with _conn() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            symbol TEXT,
            qty REAL,
            price REAL,
            side TEXT
        )''')

def save_trade(ts: int, symbol: str, qty: float, price: float, side: str) -> None:
    with _conn() as c:
        c.execute('INSERT INTO trades (ts, symbol, qty, price, side) VALUES (?,?,?,?,?)', (ts, symbol, qty, price, side))

def load_trades(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as c:
        cur = c.execute('SELECT ts, symbol, qty, price, side FROM trades ORDER BY id DESC LIMIT ?', (limit,))
        rows = cur.fetchall()
    return [{'ts': r[0], 'symbol': r[1], 'qty': r[2], 'price': r[3], 'side': r[4]} for r in rows]
