"""A deliberately boring demo strategy — it exists to exercise the plumbing, nothing else.

Picks the first k symbols in alphabetical order. No signal, no parameters worth copying.
"""

from __future__ import annotations

from tradehindsight.contract import DecisionContext, Strategy, TargetPortfolio


class Alphabet(Strategy):
    name = "alphabet"
    stateful = False
    warmup = 1

    def __init__(self, k: int = 3):
        self.k = k

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        names = [s for s in ctx.universe if ctx.view.price(s) > 0][: self.k]
        if not names:
            return TargetPortfolio({}, cash_weight=1.0)
        w = 1.0 / len(names)
        return TargetPortfolio({s: w for s in names})
