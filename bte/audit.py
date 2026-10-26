"""Leakage gates. If any of these fails, no number from the engine may be quoted."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .contract import Strategy
from .data import DataStore
from .runner import run_fast, run_rigorous, stats


def gate_access(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp]) -> dict:
    """Every decision goes through a strict view; a future read would have raised."""
    res = run_rigorous(store, strategy, grid)
    return {"passed": True, "cycles": len(res.equity)}


def gate_truncation(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
                    cut: pd.Timestamp, cost_bps: float = 0.0) -> dict:
    """Decisions up to `cut` must be identical when the later data does not exist."""
    cut = pd.Timestamp(cut)
    g = [t for t in grid if t <= cut]
    a = run_rigorous(store, strategy, g, cost_bps=cost_bps).equity
    b = run_rigorous(store.sliced(cut), strategy, g, cost_bps=cost_bps).equity
    same = a.index.equals(b.index) and np.allclose(a.values, b.values, rtol=0, atol=1e-12)
    diff = float(np.max(np.abs(a.values - b.values))) if len(a) == len(b) else float("inf")
    return {"passed": bool(same), "max_diff": diff, "points": len(a)}


def gate_poison(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
                cut: pd.Timestamp, cost_bps: float = 0.0, seed: int = 7) -> dict:
    """Replace everything after `cut` with garbage; results up to `cut` must not move."""
    cut = pd.Timestamp(cut)
    g = [t for t in grid if t <= cut]
    a = run_rigorous(store, strategy, g, cost_bps=cost_bps).equity
    b = run_rigorous(store.poisoned(cut, seed=seed), strategy, g, cost_bps=cost_bps).equity
    same = a.index.equals(b.index) and np.allclose(a.values, b.values, rtol=0, atol=1e-12)
    diff = float(np.max(np.abs(a.values - b.values))) if len(a) == len(b) else float("inf")
    return {"passed": bool(same), "max_diff": diff}


def gate_fast_equals_rigorous(store: DataStore, strategy: Strategy,
                              grid: Sequence[pd.Timestamp],
                              cost_bps: float = 0.0, initial_cash: float = 1.0,
                              workers: int = 16) -> dict:
    a = run_rigorous(store, strategy, grid, initial_cash=initial_cash, cost_bps=cost_bps)
    b = run_fast(store, strategy, grid, initial_cash=initial_cash, cost_bps=cost_bps,
                 workers=workers)
    common = a.equity.index.intersection(b.equity.index)
    diff = float(np.max(np.abs(a.equity[common].values - b.equity[common].values))) if len(common) else float("inf")
    scale = float(np.max(np.abs(a.equity[common].values))) if len(common) else 1.0
    rel = diff / (scale or 1.0)
    return {"passed": rel < 1e-9, "max_diff": diff, "rel_diff": rel, "points": len(common)}


def gate_lag(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
             cost_bps: float = 0.0) -> dict:
    """Delaying execution by one bar must degrade smoothly, not collapse."""
    base = stats(run_rigorous(store, strategy, grid, cost_bps=cost_bps).returns)
    lagged = stats(run_rigorous(store, strategy, list(grid)[:-1], cost_bps=cost_bps).returns)
    return {"base_cagr": base["cagr"], "lagged_cagr": lagged["cagr"]}
