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
    src = pd.Series(rets)
    r = src.dropna()
    # capital cannot fall below zero: clamp any period worse than -100%, otherwise a
    # negative growth factor flips sign on the next multiplication and CAGR explodes
    # into nonsense (a 200bp cost run once printed +6e9% a year).
    r = r.clip(lower=-0.999999)
    if not len(r):
        return {"n": 0, "cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    # periods per year: a monthly series has 12, a weekly one 52. Assuming 12 for weekly
    # data understates CAGR and Sharpe by the ratio of the periods (a 22.8% weekly run
    # once printed 4.7%). Snap to the common calendars so monthly results stay bit-equal.
    ppy = 12.0
    if isinstance(src.index, pd.DatetimeIndex) and len(src.index) >= 3:
        gaps = np.diff(src.index.values).astype("timedelta64[s]").astype(float) / 86400.0
        gaps = gaps[gaps > 0]
        if len(gaps):
            ppy = float(np.clip(365.25 / np.median(gaps), 1.0, 366.0))
            for cand in (12.0, 52.0, 252.0):
                if abs(ppy - cand) <= cand * 0.25:
                    ppy = cand
                    break
    years = len(r) / ppy
    growth = max(float((1 + r).prod()), 0.0)
    cagr = growth ** (1 / years) - 1 if growth > 0 else -1.0
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)) if r.std(ddof=1) > 0 else float("nan")
    eq = (1 + r).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())
    return {"n": len(r), "cagr": cagr * 100, "sharpe": sharpe, "maxdd": maxdd * 100}


def _decide(strategy: Strategy, store: DataStore, t: pd.Timestamp,
            shares: Dict[str, float], cash: float,
            decision_dates: Sequence[pd.Timestamp] = ()) -> tuple[TargetPortfolio, float]:
    prices = {s: store.close_of(s, t) for s in shares}
    nav = cash + sum(q * prices[s] for s, q in shares.items()
                     if q and np.isfinite(prices[s]))
    view = store.view(t, strict=True)              # any future read raises
    weights = {s: q * prices[s] / nav for s, q in shares.items()
               if q and nav and np.isfinite(prices[s])}
    ctx = DecisionContext(as_of=t, view=view, holdings=weights, cash=cash, nav=nav,
                          decision_dates=tuple(decision_dates))
    target = strategy.decide(ctx)
    if not isinstance(target, TargetPortfolio):
        target = TargetPortfolio(dict(target))
    return target.validate().normalised(), nav


def _cfg(cfg, cost_bps, initial_cash: float = 1.0) -> RunConfig:
    if cfg is not None:
        return cfg
    return RunConfig(initial_cash=initial_cash, cost_mode="bps_per_side",
                     cost_bps=cost_bps)


def run_rigorous(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
                 initial_cash: float = 1.0, cost_bps: float = 0.0,
                 cfg: Optional[RunConfig] = None,
                 strict: bool = False) -> Result:
    """strict=True re-emits mode-independent cycles with mode-independent cost.

    Every signal is evaluated at (t-1) on data <= t-1, and executed at the close of t-1;
    the interval (t-1, t] is then pure price movement. Each cycle therefore produces
    exactly one return row, with no latency and no shared cash, which is the only way the
    result can equal a per-cycle chain to the last bit. Costs come from `cfg.cost_mode`:
    legacy_fraction charges the research engine's formula, the notional models charge a
    proportional cost applied to the growth factor.
    """
    conf = _cfg(cfg, cost_bps, initial_cash)
    if strict:
        return _run_strict(store, strategy, grid, conf)
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

        target, nav_pre = _decide(strategy, store, t, shares, cash, grid[:k + 1])

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


def execution_dates(store: DataStore, grid: Sequence[pd.Timestamp],
                    conf: RunConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(decision date, execution date) for every mark.

    same_close     fill at the decision close
    next_close     fill at the next trading day's close
    delay_bars     fill after conf.execute_delay trading days
    next_decision  the target only becomes effective at the following mark
    """
    cal = store.calendar
    out = []
    for k, d in enumerate(grid):
        if conf.execute == "same_close":
            e = d
        elif conf.execute == "next_close":
            pos = min(cal.searchsorted(d, side="right"), len(cal) - 1)
            e = cal[pos]
        elif conf.execute == "delay_bars":
            pos = min(cal.searchsorted(d, side="right") + max(conf.execute_delay - 1, 0),
                      len(cal) - 1)
            e = cal[pos]
        elif conf.execute == "next_decision":
            e = grid[k + 1] if k + 1 < len(grid) else d
        else:
            raise ValueError(f"unknown execute mode {conf.execute!r}")
        out.append((d, pd.Timestamp(e)))
    return out


def _proportional(conf: RunConfig) -> bool:
    """True for cost modes that scale with traded notional (fractional, bit-comparable)."""
    return conf.cost_mode in ("bps_per_side", "bps_per_turnover", "spread")


def _strict_cycle(store: DataStore, strategy: Strategy, t_prev, t,
                  held: Sequence[str], prev_held: Sequence[str],
                  exp: float, exp_prev: float, conf: RunConfig,
                  prev_exp: float = 1.0) -> tuple[float, list]:
    top = conf.top or len(held) or 1
    gross = 0.0
    for s in held:
        p0, p1 = store.close_of(s, t_prev), store.close_of(s, t)
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            gross += p1 / p0 - 1.0
    gross /= max(top, 1)                       # unfilled slots stay in cash
    if conf.cost_mode == "legacy_fraction":
        c = (len(set(held) - set(prev_held)) + len(set(prev_held) - set(held))) / max(top, 1)
        c += abs(exp - prev_exp)                   # reference quirk: its own prev_exp
        c *= conf.cost_bps / 1e4
        return gross * exp - c, []
    # notional modes: charge on the turnover fraction, applied to the gross growth factor
    names = set(held) | set(prev_held)
    swapped = sum(1 for s in names if s not in set(prev_held) or s not in set(held))
    turnover = swapped / max(top, 1)
    if abs(exp - exp_prev) > 0:
        turnover += abs(exp - exp_prev)
    cost_frac = conf.cost_for(turnover, swapped) if conf.cost_mode != "flat_per_trade" else 0.0
    return gross * exp - cost_frac, []



def _active_index(grid, schedule, k):
    """Which decision is actually in the account at grid[k-1]."""
    best = None
    for j in range(0, k):
        if schedule[grid[j]] <= grid[k - 1]:
            best = j
    return best


def _cycle_return(store, plan, grid, schedule, k, conf):
    """Return of cycle k, splitting the window where an order fills inside it.

    plan[j] = (held, exp) decided at grid[j]; schedule maps a decision date to its fill.
    """
    t_prev, t = grid[k - 1], grid[k]
    active = _active_index(grid, schedule, k)
    new = k - 1

    # no fill inside the window: whatever is in the account earns the whole period
    fill = schedule[grid[new]]
    if fill <= t_prev or active == new:
        held, exp = plan[active] if active is not None else ([], 1.0)
        return _strict_cycle(store, None, t_prev, t, held, [], exp, 1.0, conf, exp)[0]

    held_old, exp_old = plan[active] if active is not None else ([], 1.0)
    held_new, exp_new = plan[new]
    seg1 = _strict_cycle(store, None, t_prev, fill, held_old, [], exp_old, 1.0, conf, exp_old)[0]
    seg2 = _strict_cycle(store, None, fill, t, held_new, held_old, exp_new, exp_old, conf, exp_new)[0]
    return (1 + seg1) * (1 + seg2) - 1


def _run_strict(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
                conf: RunConfig) -> Result:
    """One decision per mark, one return per mark, no shared cash between cycles."""
    grid = [pd.Timestamp(x) for x in grid]
    if len(grid) < 2:
        return Result(mode="rigorous", equity=pd.Series(dtype=float))
    plan = []
    for k, t in enumerate(grid):
        view = store.view(t, strict=True)
        ctx = DecisionContext(as_of=t, view=view, decision_dates=tuple(grid[:k + 1]))
        target = strategy.decide(ctx).validate().normalised()
        held = list(target)
        plan.append((held, float(sum(target.values())) if held else 0.0))
    return _chain(store, plan, grid, execution_dates(store, grid, conf), conf,
                  float(conf.initial_cash), mode="rigorous")


def _chain(store, plan, grid, sched, conf, cash, mode="rigorous") -> Result:
    schedule = dict(sched)
    value = float(cash)
    equity, rets = {grid[0]: value}, {}
    for k in range(1, len(grid)):
        r = _cycle_return(store, plan, grid, schedule, k, conf)
        rets[grid[k]] = r
        value *= 1.0 + r
        equity[grid[k]] = value
    return Result(mode=mode, equity=pd.Series(equity).sort_index(),
                  returns=pd.Series(rets).sort_index(),
                  costs=pd.Series({d: 0.0 for d in rets}).sort_index())


# ---------------------------------------------------------------- fast (process pool)
_CTX: dict = {}


def _worker_init(store, strategy, initial_cash, grid=()):
    _CTX["store"] = store
    _CTX["strategy"] = strategy
    _CTX["cash"] = initial_cash
    _CTX["grid"] = tuple(grid)


def _worker_cycles(pairs):
    store, strategy, cash = _CTX["store"], _CTX["strategy"], _CTX["cash"]
    out = []
    for t, t_next in pairs:
        target, _ = _decide(strategy, store, t, {}, cash, _CTX.get("grid", ()))
        gross = {}
        for s, w in target.items():
            p0, p1 = store.close_of(s, t), store.close_of(s, t_next)
            gross[s] = (p1 / p0) if (np.isfinite(p0) and np.isfinite(p1) and p0 > 0) else 1.0
        out.append((t, t_next, dict(target), gross))
    return out


def run_fast(store: DataStore, strategy: Strategy, grid: Sequence[pd.Timestamp],
             initial_cash: float = 1.0, cost_bps: float = 0.0,
             workers: int = 0, cfg: Optional[RunConfig] = None,
             strict: bool = True) -> Result:
    """Cycle returns are independent -> parallel; the chain is a cheap sequential pass.

    strict=True (default) uses the same per-cycle arithmetic as the strict rigorous loop:
    each decision depends only on data up to its own date, so the cycles can be evaluated
    in any order and chained afterwards. The result equals the rigorous loop bit for bit.

    Uses fork so the price store is inherited copy-on-write instead of pickled.
    """
    if strategy.stateful:
        raise ValueError("fast mode cannot run a stateful strategy — use rigorous")
    conf = _cfg(cfg, cost_bps, initial_cash)
    grid = [pd.Timestamp(x) for x in grid]
    if not strict:
        return _run_fast_legacy(store, strategy, grid, conf, workers)

    pairs = list(range(len(grid)))
    ncpu = os.cpu_count() or 1
    workers = workers or min(16, ncpu)
    workers = max(1, min(workers, ncpu, len(pairs) or 1))
    chunks = [pairs[i::workers] for i in range(workers)]

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_strict_init, initargs=(store, strategy, grid)) as pool:
            parts = pool.map(_strict_cycles, chunks)
    except (ValueError, OSError):
        _strict_init(store, strategy, grid)
        parts = [_strict_cycles(pairs)]

    plan_map = {rec[0]: rec[1:] for part in parts for rec in part}
    plan = [plan_map.get(k, ([], 0.0)) for k in range(len(grid))]
    return _chain(store, plan, grid, execution_dates(store, grid, conf), conf,
                  float(initial_cash), mode="fast")
    value = float(initial_cash)
    equity = {grid[0]: value}
    rets = {}
    for k in range(1, len(grid)):
        t_prev, t = grid[k - 1], grid[k]
        # the first window is empty in the strict loop too: nothing was held before the
        # first decision, so that cycle earns no return and only pays for entering
        held, exp = plan[k - 1] if k >= 2 else ([], 1.0)
        prev_held, _ = plan[k - 2] if k >= 3 else ([], 1.0)
        prev_exp = plan[k - 1][1]
        r, _ = _strict_cycle(store, strategy, t_prev, t, held, prev_held, exp, 1.0, conf, prev_exp)
        rets[t] = r
        value *= 1.0 + r
        equity[t] = value
    return Result(mode="fast", equity=pd.Series(equity).sort_index(),
                  returns=pd.Series(rets).sort_index())


_SCTX: dict = {}


def _strict_init(store, strategy, grid):
    _SCTX["store"] = store
    _SCTX["strategy"] = strategy
    _SCTX["grid"] = tuple(grid)


def _strict_cycles(items):
    """Evaluate the decision at each mark. Nothing here depends on cash."""
    store, strategy, grid = _SCTX["store"], _SCTX["strategy"], _SCTX["grid"]
    out = []
    for k in items:
        t = grid[k]
        view = store.view(t, strict=True)
        ctx = DecisionContext(as_of=t, view=view, decision_dates=tuple(grid[:k + 1]))
        target = strategy.decide(ctx).validate().normalised()
        held = list(target)
        exp = float(sum(target.values())) if held else 0.0
        out.append((k, held, exp))
    return out


def _run_fast_legacy(store, strategy, grid, conf, workers):
    pairs = list(zip(grid[:-1], grid[1:]))
    ncpu = os.cpu_count() or 1
    workers = workers or min(16, ncpu)
    workers = max(1, min(workers, ncpu, len(pairs) or 1))
    chunks = [pairs[i::workers] for i in range(workers)]
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(store, strategy, conf.initial_cash, grid)) as pool:
            parts = pool.map(_worker_cycles, chunks)
    except (ValueError, OSError):
        _worker_init(store, strategy, conf.initial_cash, grid)
        parts = [_worker_cycles(pairs)]

    merged = {rec[0]: rec for part in parts for rec in part}
    cycles = [merged[t] for t, _ in pairs]
    value, prev, equity = float(conf.initial_cash), {}, {}
    for t, t_next, w, gross in cycles:
        names = set(w) | set(prev)
        deltas = {s: abs(w.get(s, 0.0) - prev.get(s, 0.0)) for s in names}
        turnover = sum(deltas.values())
        n_trades = sum(1 for v in deltas.values() if v > 1e-12)
        cost = conf.cost_for(turnover * value, n_trades)
        value_at_t = value - cost
        equity[t] = value_at_t
        port = sum(w.get(s, 0.0) * gross[s] for s in w) + (1.0 - sum(w.values()))
        value = value_at_t * port
        equity[t_next] = value
        prev = {s: w.get(s, 0.0) * gross[s] / port for s in w} if port else {}
    eq = pd.Series(equity).sort_index()
    return Result(mode="fast", equity=eq, returns=eq.pct_change().dropna())
