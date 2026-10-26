"""Decision/execution grid. One timestamp per cycle: the last bar with data."""
from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd


def month_ends(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Last trading day of each month — the decision date for that month."""
    s = pd.Series(calendar, index=calendar)
    out = s.groupby([s.index.year, s.index.month]).max()
    return [pd.Timestamp(d) for d in out.values]


def week_ends(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    out = s.groupby([s.index.isocalendar().year, s.index.isocalendar().week]).max()
    return [pd.Timestamp(d) for d in out.values]


def every_k_bars(calendar: pd.DatetimeIndex, k: int) -> List[pd.Timestamp]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return [pd.Timestamp(d) for d in calendar[::k]]


def build_grid(calendar: pd.DatetimeIndex, freq: str | Callable | int,
               start: Optional[str] = None, end: Optional[str] = None) -> List[pd.Timestamp]:
    """freq: 'month_end' | 'week_end' | 'daily' | int k | callable(calendar) -> list."""
    if callable(freq):
        grid = list(freq(calendar))
    elif isinstance(freq, int):
        grid = every_k_bars(calendar, freq)
    elif freq == "month_end":
        grid = month_ends(calendar)
    elif freq == "week_end":
        grid = week_ends(calendar)
    elif freq == "daily":
        grid = [pd.Timestamp(d) for d in calendar]
    else:
        raise ValueError(f"unknown frequency {freq!r}")

    if start:
        grid = [d for d in grid if d >= pd.Timestamp(start)]
    if end:
        grid = [d for d in grid if d <= pd.Timestamp(end)]
    return sorted(set(grid))
