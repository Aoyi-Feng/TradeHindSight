"""Minimal example strategy: hold one symbol, cash otherwise."""
from __future__ import annotations

from bte.contract import DecisionContext, Strategy, TargetPortfolio


class BuyHold(Strategy):
    name = "buy_and_hold"
    stateful = False

    def __init__(self, symbol: str | None = None):
        self.symbol = symbol

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        sym = self.symbol or ctx.universe[0]
        return TargetPortfolio({sym: 1.0})
