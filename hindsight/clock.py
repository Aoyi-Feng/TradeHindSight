"""Decision/execution grid. One timestamp per cycle: the last bar with data."""
from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd


def month_ends(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Last trading day of each month — the decision date for that month."""
    s = pd.Series(calendar, index=calendar)
    out = s.groupby([s.index.year, s.index.month]).max()
    return [pd.Timestamp(d) for d in out.values]


def month_starts(calendar: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """The last trading day on or before the 1st of each month.

    This is the "decide at the turn of the month" convention: for August 1999 the mark is
    1999-07-30 (the 1st was a Sunday), for September 1999 it is 1999-09-01.
    """
    if not len(calendar):
        return []
    start = pd.Timestamp(calendar[0].year, calendar[0].month, 1)
    end = pd.Timestamp(calendar[-1].year, calendar[-1].month, 1)
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        pos = calendar.searchsorted(m, side="right") - 1
        if pos >= 0:
            out.append(pd.Timestamp(calendar[pos]))
    # de-duplicate while keeping order (a month can resolve to the previous mark)
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


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
    """freq: 'month_end' | 'month_start' | 'week_end' | 'daily' | int k | callable."""
    if callable(freq):
        grid = list(freq(calendar))
    elif isinstance(freq, int):
        grid = every_k_bars(calendar, freq)
    elif freq == "month_end":
        grid = month_ends(calendar)
    elif freq == "month_start":
        grid = month_starts(calendar)
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
