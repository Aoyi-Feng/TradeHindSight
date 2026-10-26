"""Ask a strategy what it would do at the latest date in the data.

No backtest, no simulation: one decision at the most recent bar the data actually contains,
through the same PITView and the same contract as the engine. A strategy cannot behave
differently here than it does in a run.

If the strategy implements ``explain(ctx)`` — returning ``{symbol: score}`` or a list of
``(symbol, score)`` — the ranking behind the decision is reported too, so the output shows
*why* those names and not the others.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .clock import build_grid
from .contract import DecisionContext, Strategy


def resolve_bar(store, at) -> pd.Timestamp:
    """Map a requested date onto the last bar at or before the end of that day.

    A date-only input means "that session's close". Taking midnight literally drops the
    session itself (a bar stamped 09:30 is greater than midnight), which silently shifts
    every window one day back — the difference between a $10B filter admitting 437 names
    and 443 of them.
    """
    if at is None:
        return pd.Timestamp(store.calendar[-1])
    ts = pd.Timestamp(at)
    if ts in store.calendar:
        return ts
    eod = ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    pos = store.calendar.searchsorted(eod, side="right") - 1
    if pos < 0:
        raise ValueError(f"no bar at or before {ts.date()} in the data")
    return pd.Timestamp(store.calendar[pos])


def latest_date(store) -> pd.Timestamp:
    """The newest date for which the store has a price."""
    return pd.Timestamp(store.calendar[-1])


def _as_ranking(raw) -> list[tuple[str, float]] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        items = raw
    out = []
    for sym, score in items:
        try:
            out.append((str(sym), float(score)))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda kv: -kv[1])


def latest_decision(
    store,
    strategy: Strategy,
    at: pd.Timestamp | None = None,
    holdings: Mapping[str, float] | None = None,
    cash: float = 1.0,
    nav: float = 1.0,
    params: Mapping[str, object] | None = None,
    frequency: str = "month_end",
    marks: int = 24,
) -> dict:
    """Run one decision at ``at`` (default: the newest date in the data)."""
    t = resolve_bar(store, at)

    grid = build_grid(store.calendar, frequency)
    recent = [d for d in grid if d <= t][-marks:]
    if not recent or recent[-1] != t:
        recent = recent + [t]

    held = dict(holdings or {})
    view = store.view(t, strict=True)
    ctx = DecisionContext(
        as_of=t, view=view, holdings=held, cash=cash, nav=nav,
        params=dict(params or {}), decision_dates=tuple(recent),
    )
    target = strategy.decide(ctx).validate().normalised()
    weights = {str(k): float(v) for k, v in dict(target).items()}

    ranking = None
    if hasattr(strategy, "explain"):
        try:
            ranking = _as_ranking(strategy.explain(ctx))
        except Exception as exc:              # a broken explainer must not hide the decision
            ranking = None
            ranking_error = f"{type(exc).__name__}: {exc}"
        else:
            ranking_error = None
    else:
        ranking_error = None

    orders = []
    for sym in sorted(set(held) | set(weights)):
        before = float(held.get(sym, 0.0))
        after = float(weights.get(sym, 0.0))
        delta = after - before
        if abs(delta) > 1e-9:
            orders.append({
                "symbol": sym, "from": before, "to": after, "delta": delta,
                "action": "SELL" if delta < 0 else "BUY",
                "notional": delta * nav,
                "price": float(view.price(sym)) if sym in view.symbols() else None,
            })

    return {
        "as_of": t,
        "data_latest": latest_date(store),
        "stale_days": int((latest_date(store) - t).days),
        "strategy": getattr(strategy, "name", type(strategy).__name__),
        "cash_weight": 1.0 - float(sum(weights.values())),
        "weights": weights,
        "orders": orders,
        "n_names": len(weights),
        "ranking": ranking,
        "ranking_error": ranking_error,
        "prices": {s: float(view.price(s)) for s in weights if s in view.symbols()},
    }


def format_decision(dec: dict, top: int = 10) -> str:
    """Human/agent readable summary of a decision, with the ranking when available."""
    t = dec["as_of"]
    out = [
        f"strategy     {dec['strategy']}",
        f"as of        {t.date()} ({t.strftime('%A')})",
    ]
    if dec["stale_days"]:
        out.append(f"WARNING      asked for {t.date()}, newest data is "
                   f"{dec['data_latest'].date()} ({dec['stale_days']}d stale)")
    else:
        out.append("data         newest bar is the decision bar")
    out.append(f"positions    {dec['n_names']}   cash {dec['cash_weight']:.1%}")
    for sym, w in sorted(dec["weights"].items(), key=lambda kv: -kv[1]):
        px = dec["prices"].get(sym)
        out.append(f"  {sym:<8} {w:6.2%}" + (f"   last {px:,.2f}" if px else ""))

    if dec.get("ranking_error"):
        out.append(f"ranking      unavailable ({dec['ranking_error']})")

    ranking = dec.get("ranking")
    if ranking:
        held = set(dec["weights"])
        out.append(f"ranking      {len(ranking)} scored")
        for i, (sym, score) in enumerate(ranking[:top], 1):
            mark = "*" if sym in held else " "
            out.append(f"  {i:>3}.{mark}{sym:<8} {score:8.4f}")

    if dec["orders"]:
        out.append("orders")
        for o in dec["orders"]:
            out.append(f"  {o['action']:<4} {o['symbol']:<8} {o['from']:6.1%} -> {o['to']:6.1%}"
                       f"   {o['notional']:+,.0f}")
    else:
        out.append("orders       none (already at target)")
    return "\n".join(out)
