"""End-to-end smoke run on the committed sample dataset.

    PYTHONPATH=. python examples/run_sample.py

Proves the repository works from a fresh clone: tiny data, classical strategy, both modes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bte import DataStore, build_grid, run_fast, run_rigorous
from strategies.classic import EqualWeight, MACross


def load_sample() -> DataStore:
    path = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices_sample.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return DataStore({c: df[c].dropna() for c in df.columns})


def main() -> int:
    store = load_sample()
    grid = build_grid(store.calendar, "month_end")
    print(f"sample: {len(store.symbols)} symbols, {len(store.calendar)} days, "
          f"{len(grid)} monthly decisions")

    for strat in (EqualWeight(), MACross()):
        a = run_rigorous(store, strat, grid, initial_cash=10_000, cost_bps=10)
        b = run_fast(store, strat, grid, initial_cash=10_000, cost_bps=10)
        s = a.stats()
        rel = abs(a.equity.iloc[-1] - b.equity.iloc[-1]) / a.equity.iloc[-1]
        print(f"{strat.name:12s} CAGR {s['cagr']:+6.2f}%  Sharpe {s['sharpe']:5.2f}  "
              f"MaxDD {s['maxdd']:+6.2f}%  final {a.equity.iloc[-1]:,.0f}  "
              f"fast-vs-rigorous {rel:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
