"""Throughput check: rigorous (1 core) vs fast (N processes)."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from bte import DataStore, build_grid, run_fast, run_rigorous
from examples.demo import Alphabet


def make(assets: int, days: int, seed: int = 1) -> DataStore:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=days)
    return DataStore({f"S{k:03d}": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, days))), index=idx)
                      for k in range(assets)})


def bench(assets: int, days: int, freq: str, k: int, workers: int, cost_bps: float = 10.0):
    store = make(assets, days)
    grid = build_grid(store.calendar, freq)
    strat = Alphabet(k)

    t0 = time.perf_counter()
    a = run_rigorous(store, strat, grid, initial_cash=1e6, cost_bps=cost_bps)
    t1 = time.perf_counter()
    b = run_fast(store, strat, grid, initial_cash=1e6, cost_bps=cost_bps, workers=workers)
    t2 = time.perf_counter()

    common = a.equity.index.intersection(b.equity.index)
    rel = float(np.max(np.abs(a.equity[common].values - b.equity[common].values))
                / max(abs(a.equity[common].values).max(), 1e-12))
    print(f"{assets:4d} assets {len(grid):5d} cycles {freq:9s} "
          f"rigorous {t1-t0:6.2f}s  fast({workers}) {t2-t1:6.2f}s  "
          f"speedup {(t1-t0)/max(t2-t1,1e-9):5.1f}x  |diff| {rel:.1e}")


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    bench(200, 3000, "month_end", 20, w)
    bench(500, 6000, "month_end", 20, w)
    bench(500, 3000, "week_end", 20, w)
    bench(300, 1500, "daily", 20, w)
