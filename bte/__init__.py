"""A leakage-free backtest engine. The strategy is injected, never imported."""

from .contract import DecisionContext, Strategy, TargetPortfolio
from .data import DataStore, LookAheadError, PITView
from .clock import build_grid, month_ends, week_ends, every_k_bars
from .runner import Result, run_fast, run_rigorous, stats
from . import audit

__all__ = [
    "DecisionContext", "Strategy", "TargetPortfolio",
    "DataStore", "LookAheadError", "PITView",
    "build_grid", "month_ends", "week_ends", "every_k_bars",
    "Result", "run_fast", "run_rigorous", "stats", "audit",
]
__version__ = "0.1.0"
