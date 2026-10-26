"""Rigorous and fast backtest loops.

rigorous — single thread, cycle by cycle, tracks cash / shares / nav exactly.
fast     — evaluates every cycle independently (process pool) and chains the returns.

Both decide at t using data <= t and execute at the close of t.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import RunConfig
from .contract import DecisionContext, Strategy, TargetPortfolio
from .data import DataStore


@dataclass
class Trade:
    ts: pd.Timestamp
    symbol: str
    shares: float
    price: float
    notional: float


@dataclass
class Result:
    mode: str
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: List[Trade] = field(default_factory=list)
    weights: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def stats(self) -> dict:
        return stats(self.returns)


def stats(rets: pd.Series) -> dict:
    r = pd.Series(rets).dropna()
    if not len(r):
        return {"n": 0, "cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    years = len(r) / 12.0
    growth = float((1 + r).prod())
    cagr = growth ** (1 / years) - 1
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(12)) if r.std(ddof=1) > 0 else float("nan")
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    return {"n": len(r), "cagr": cagr * 100, "sharpe": sharpe, "maxdd": maxdd * 100}


def _decide(strategy: Strategy, store: DataStore, t: pd.Timestamp,
            shares: Dict[str, float], cash: float) -> tuple[TargetPortfolio, float]:
    prices = {s: store.close_of(s, t) for s in shares}
    nav = cash + sum(q * prices[s] for s, q in shares.items()
                     if q and np.isfinite(prices[s]))
    view = store.view(t, strict=True)              # any future read raises
    weights = {s: q * prices[s] / nav for s, q in shares.items()
               if q and nav and np.isfinite(prices[s])}
    ctx = DecisionContext(as_of=t, view=view, holdings=weights, cash=cash, nav=nav)
    target = strategy.decide(ctx)
    if not isinstance(target, TargetPortfolio):
        target = TargetPortfolio(dict(target))
    return target.validate().normalised(), nav


def _cfg(cfg, cost_bps) -> RunConfig:
    if cfg is not None:
        return cfg
    return RunConfig(initial_cash=1.0, cost_mode="bps_per_side", cost_bps=cost_bps)


def run_rigorous(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
                 initial_cash: float = 1.0, cost_bps: float = 0.0,
                 cfg: Optional[RunConfig] = None) -> Result:
    conf = _cfg(cfg, cost_bps)
    cash, shares = float(initial_cash), {}
    equity, cost_s, trades, wrows = {}, {}, [], []
    grid = [pd.Timestamp(x) for x in grid]

    for k, t in enumerate(grid):
        if k == len(grid) - 1:            # last point: mark only, nothing follows
            nav = cash + sum(q * store.close_of(s, t) for s, q in shares.items() if q)
            equity[t] = nav
            cost_s[t] = 0.0
            wrows.append({"ts": t, **{s: q * store.close_of(s, t) / nav
                                      for s, q in shares.items() if q and nav}})
            break

        target, nav_pre = _decide(strategy, store, t, shares, cash)

        # cost fraction from weights (identical convention to the fast loop), then size
        # on the NAV that remains so post-trade weights equal the targets exactly
        w_now = {}
        for s, q in shares.items():
            if q:
                px = store.close_of(s, t)
                if np.isfinite(px) and px > 0:
                    w_now[s] = q * px / nav_pre
        names = set(target) | set(w_now)
        deltas = {s: abs(target.get(s, 0.0) - w_now.get(s, 0.0)) for s in names}
        turnover = sum(deltas.values())
        n_trades = sum(1 for v in deltas.values() if v > 1e-12)
        cost = conf.cost_for(turnover * nav_pre, n_trades)
        nav_eff = nav_pre - cost
        traded = 0.0

        for s, q in list(shares.items()):
            if q and s not in target:
                px = store.close_of(s, t)
                if np.isfinite(px) and px > 0:
                    cash += q * px
                    traded += q * px
                    trades.append(Trade(t, s, -q, px, -q * px))
                shares[s] = 0.0

        for s, w in target.items():
            px = store.close_of(s, t)
            if not np.isfinite(px) or px <= 0:
                continue
            delta = w * nav_eff - shares.get(s, 0.0) * px
            if abs(delta) < 1e-12:
                continue
            dq = delta / px
            shares[s] = shares.get(s, 0.0) + dq
            cash -= delta
            traded += abs(delta)
            trades.append(Trade(t, s, dq, px, delta))

        cash -= cost

        nav = cash + sum(q * store.close_of(s, t) for s, q in shares.items() if q)
        equity[t] = nav
        cost_s[t] = cost
        wrows.append({"ts": t, **{s: q * store.close_of(s, t) / nav
                                  for s, q in shares.items() if q and nav}})

    eq = pd.Series(equity).sort_index()
    return Result(mode="rigorous", equity=eq, returns=eq.pct_change().dropna(),
                  trades=trades,
                  weights=pd.DataFrame(wrows).set_index("ts") if wrows else pd.DataFrame(),
                  costs=pd.Series(cost_s).sort_index())


# ---------------------------------------------------------------- fast (process pool)
_CTX: dict = {}


def _worker_init(store, strategy, initial_cash):
    _CTX["store"] = store
    _CTX["strategy"] = strategy
    _CTX["cash"] = initial_cash


def _worker_cycles(pairs):
    store, strategy, cash = _CTX["store"], _CTX["strategy"], _CTX["cash"]
    out = []
    for t, t_next in pairs:
        target, _ = _decide(strategy, store, t, {}, cash)
        gross = {}
        for s, w in target.items():
            p0, p1 = store.close_of(s, t), store.close_of(s, t_next)
            gross[s] = (p1 / p0) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else 1.0
        out.append((t, t_next, dict(target), gross))
    return out


def run_fast(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
             initial_cash: float = 1.0, cost_bps: float = 0.0,
             workers: int = 0, cfg: Optional[RunConfig] = None) -> Result:
    """Cycle returns are independent -> parallel; the chain is a cheap sequential pass.

    Uses fork so the price store is inherited copy-on-write instead of pickled.
    """
    if strategy.stateful:
        raise ValueError("fast mode cannot run a stateful strategy — use rigorous")
    conf = _cfg(cfg, cost_bps)

    grid = [pd.Timestamp(x) for x in grid]
    pairs = list(zip(grid[:-1], grid[1:]))
    ncpu = os.cpu_count() or 1
    if not workers:
        # auto: 16 — measured optimum on a 32-thread box; beyond that it only adds overhead
        workers = min(16, ncpu)
    if workers > ncpu:                           # never oversubscribe the box
        workers = ncpu
    workers = max(1, min(workers, len(pairs)))
    chunks = [pairs[i::workers] for i in range(workers)]

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(store, strategy, initial_cash)) as pool:
            parts = pool.map(_worker_cycles, chunks)
    except (ValueError, OSError):
        _worker_init(store, strategy, initial_cash)
        parts = [_worker_cycles(pairs)]

    merged = {rec[0]: rec for part in parts for rec in part}
    cycles = [merged[t] for t, _ in pairs]

    value, prev, equity = float(initial_cash), {}, {}
    for t, t_next, w, gross in cycles:
        names = set(w) | set(prev)
        # one-sided traded notional, same convention as the rigorous loop
        deltas = {s: abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in names}
        turnover = sum(deltas.values())
        n_trades = sum(1 for v in deltas.values() if v > 1e-12)
        cost = conf.cost_for(turnover * value, n_trades)   # paid at t, before the interval
        value_at_t = value - cost
        equity[t] = value_at_t
        port = sum(w.get(s, 0.0) * gross[s] for s in w) + (1.0 - sum(w.values()))
        value = value_at_t * port
        equity[t_next] = value
        prev = {s: w.get(s, 0.0) * gross[s] / port for s in w} if port else {}

    eq = pd.Series(equity).sort_index()
    return Result(mode="fast", equity=eq, returns=eq.pct_change().dropna())
