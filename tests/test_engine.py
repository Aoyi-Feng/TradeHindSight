"""The four gates. Run with `python -m tests.test_engine` or pytest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hindsight import DataStore, Strategy, TargetPortfolio, build_grid
from hindsight.contract import DecisionContext
from hindsight.data import LookAheadError
from hindsight import audit, run_fast, run_rigorous


def synthetic(n_days: int = 1200, n_assets: int = 5, seed: int = 0) -> DataStore:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    close = {}
    for k in range(n_assets):
        r = rng.normal(0.0004, 0.012, n_days)
        close[f"A{k}"] = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    return DataStore(close)


class BuyFirst(Strategy):
    name = "buy_first"
    warmup = 0

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        return TargetPortfolio({ctx.universe[0]: 1.0})


class EqualWeight(Strategy):
    name = "equal_weight"
    warmup = 0

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        u = ctx.universe
        return TargetPortfolio({s: 1.0 / len(u) for s in u})


class Peeker(Strategy):
    name = "peeker"
    warmup = 0

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        future = ctx.as_of + pd.Timedelta(days=400)
        ctx.view.price("A0", future)          # must raise
        return TargetPortfolio({ctx.universe[0]: 1.0})


def test_no_future_access():
    store = synthetic()
    grid = build_grid(store.calendar, "month_end")
    try:
        run_rigorous(store, Peeker(), grid)
    except LookAheadError:
        return
    raise AssertionError("strategy read the future without an error")


def test_truncation():
    store = synthetic()
    grid = build_grid(store.calendar, "month_end")
    cut = grid[len(grid) // 2]
    out = audit.gate_truncation(store, EqualWeight(), grid, cut, cost_bps=5)
    assert out["passed"], out


def test_poison():
    store = synthetic()
    grid = build_grid(store.calendar, "month_end")
    cut = grid[len(grid) // 2]
    out = audit.gate_poison(store, EqualWeight(), grid, cut, cost_bps=5)
    assert out["passed"], out


def test_fast_equals_rigorous():
    store = synthetic()
    grid = build_grid(store.calendar, "month_end")
    out = audit.gate_fast_equals_rigorous(store, EqualWeight(), grid, cost_bps=5)
    assert out["passed"], out


def test_cash_and_costs_reconcile():
    store = synthetic()
    grid = build_grid(store.calendar, "month_end")
    res = run_rigorous(store, EqualWeight(), grid, initial_cash=1000.0, cost_bps=10)
    # equity must fall by exactly the sum of costs when prices are irrelevant to it
    drift = res.equity.iloc[0] - 1000.0
    assert drift <= 1e-9, f"first cycle gained {drift} before any price move"
    assert res.costs.sum() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
