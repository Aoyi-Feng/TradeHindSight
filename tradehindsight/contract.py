"""The engine's contract with a strategy. The engine knows nothing about signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import pandas as pd

from .data import PITView


@dataclass
class DecisionContext:
    """Everything a strategy is allowed to see at one decision time."""
    as_of: pd.Timestamp
    view: PITView
    holdings: Dict[str, float] = field(default_factory=dict)   # symbol -> weight, marked at as_of
    cash: float = 0.0
    nav: float = 0.0
    params: Mapping = field(default_factory=dict)
    decision_dates: tuple = ()          # past decision marks, oldest first, ending at as_of

    @property
    def universe(self):
        return self.view.eligible()


class TargetPortfolio(dict):
    """Target weights, decided at ``as_of``, executed at the same close by default.

    Long-only, weights must sum to <= 1; the remainder is held in cash. Anything else
    raises here rather than producing a silently different portfolio downstream.
    """

    def __init__(self, weights: Optional[Mapping[str, float]] = None, cash_weight: float = 0.0):
        super().__init__(weights or {})
        self.cash_weight = float(cash_weight)

    def validate(self) -> "TargetPortfolio":
        bad = {k: v for k, v in self.items() if v < -1e-12}
        if bad:
            raise ValueError(f"negative weights not supported: {bad}")
        total = sum(self.values()) + self.cash_weight
        if total > 1.0 + 1e-9:
            raise ValueError(f"weights sum to {total:.6f} > 1")
        if self.cash_weight < 0:
            raise ValueError("cash_weight must be >= 0")
        return self

    def normalised(self) -> "TargetPortfolio":
        total = sum(self.values()) + self.cash_weight
        if total <= 0:
            return TargetPortfolio({}, cash_weight=1.0)
        if total <= 1.0 + 1e-9:
            return self
        return TargetPortfolio({k: v / total for k, v in self.items()},
                               cash_weight=self.cash_weight / total)


class Strategy:
    """Subclass this. The engine only ever calls ``decide``.

    The one thing the engine takes from a strategy is the portfolio it wants at a moment.
    How that portfolio was produced is invisible to it: a ranking, a fixed list, one name,
    a rule that says "hold what I already hold", or an empty dict meaning all cash are all
    the same answer. The engine never sees scores, ranks or reasons — it sees weights.
    """

    name = "strategy"
    warmup: int = 0            # bars of history required before the first decision
    stateful: bool = False     # True -> fast mode refuses to run

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:  # pragma: no cover
        raise NotImplementedError

    # optional hook, called once before the first decision
    def on_start(self, ctx: DecisionContext) -> None:
        return None
