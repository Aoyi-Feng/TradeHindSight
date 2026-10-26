"""Classical, textbook strategies. No tuned parameters, nothing proprietary.

These exist so the repository can be cloned and exercised end to end.
"""
from __future__ import annotations

from tradehindsight.contract import DecisionContext, Strategy, TargetPortfolio


class EqualWeight(Strategy):
    """Hold every eligible name, equally weighted, rebalanced on the decision clock."""
    name = "equal_weight"
    stateful = False
    warmup = 1

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        names = [s for s in ctx.universe if ctx.view.price(s) > 0]
        if not names:
            return TargetPortfolio({}, cash_weight=1.0)
        w = 1.0 / len(names)
        return TargetPortfolio({s: w for s in names})


class MACross(Strategy):
    """Textbook moving-average crossover: long when the fast MA is above the slow MA.

    Windows are deliberately the classic 20/60 — not any parameter used in production.
    """
    name = "ma_cross"
    stateful = False

    def __init__(self, fast: int = 20, slow: int = 60, k: int = 3):
        self.fast, self.slow, self.k = fast, slow, k
        self.warmup = slow

    def explain(self, ctx: DecisionContext):
        """Score every name by fast/slow deviation — the ranking behind the picks."""
        out = {}
        for s in ctx.universe:
            h = ctx.view.history(s, n=self.slow)
            if len(h) < self.slow:
                continue
            out[s] = float(h.iloc[-self.fast:].mean() / h.mean() - 1.0)
        return out

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        picks = []
        for s in ctx.universe:
            h = ctx.view.history(s, n=self.slow)
            if len(h) < self.slow:
                continue
            fast = float(h.iloc[-self.fast:].mean())
            slow = float(h.mean())
            if fast > slow and h.iloc[-1] > 0:
                picks.append((s, fast / slow - 1))
        picks.sort(key=lambda x: -x[1])
        picks = picks[: self.k]
        if not picks:
            return TargetPortfolio({}, cash_weight=1.0)
        w = 1.0 / len(picks)
        return TargetPortfolio({s: w for s, _ in picks})


class RandomPicker(Strategy):
    """Deterministic pseudo-random pick — a baseline with no signal at all."""
    name = "random"
    stateful = False
    warmup = 1

    def __init__(self, k: int = 3, seed: int = 0):
        self.k, self.seed = k, seed

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        names = sorted(s for s in ctx.universe if ctx.view.price(s) > 0)
        if not names:
            return TargetPortfolio({}, cash_weight=1.0)
        rng = __import__("random").Random(f"{self.seed}:{ctx.as_of.date()}")
        picked = rng.sample(names, min(self.k, len(names)))
        w = 1.0 / len(picked)
        return TargetPortfolio({s: w for s in picked})
