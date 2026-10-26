"""How far does the fast loop scale on this box?"""
from __future__ import annotations

import os
import time

from bte import build_grid, run_fast, run_rigorous
from examples.bench import make
from examples.demo import Alphabet


def run(assets=500, days=4000, freq="week_end", k=20):
    store = make(assets, days, seed=5)
    grid = build_grid(store.calendar, freq)
    strat = Alphabet(k)

    t0 = time.perf_counter()
    run_rigorous(store, strat, grid, initial_cash=1e6, cost_bps=10)
    base = time.perf_counter() - t0
    print(f"cpu_count={os.cpu_count()}  assets={assets}  cycles={len(grid)}  rigorous={base:.2f}s")
    print(f"{'workers':>8} {'time':>8} {'speedup':>8}")
    for w in (1, 2, 4, 8, 16, 24, 32, 64, 128):
        t0 = time.perf_counter()
        run_fast(store, strat, grid, initial_cash=1e6, cost_bps=10, workers=w)
        dt = time.perf_counter() - t0
        print(f"{w:8d} {dt:7.2f}s {base/dt:7.1f}x")


if __name__ == "__main__":
    run()
