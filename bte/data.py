"""Point-in-time data access, backed by one contiguous matrix.

The only way a strategy can touch prices. Every access is bounded by ``as_of``:
asking for anything later raises ``LookAheadError`` instead of silently returning it.

Storage is one float64 matrix (dates x symbols) plus a positional index, so a lookup is
O(1) instead of a boolean mask over a Series — this is what makes the loop fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LookAheadError(RuntimeError):
    """Raised when a strategy asks for data that did not exist at the decision time."""


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    close: float
    volume: float = float("nan")


class PITView:
    """Read-only, truncated view of the price matrix at one decision time."""

    __slots__ = ("_store", "_row", "_as_of", "_accesses", "_strict")

    def __init__(self, store: "DataStore", row: int, as_of: pd.Timestamp, strict: bool = True):
        self._store = store
        self._row = row
        self._as_of = as_of
        self._accesses: List[str] = []
        self._strict = strict

    @property
    def as_of(self) -> pd.Timestamp:
        return self._as_of

    @property
    def accesses(self) -> List[str]:
        return list(self._accesses)

    def _row_of(self, ts) -> int:
        ts = pd.Timestamp(ts)
        if ts > self._as_of:
            msg = f"look-ahead: requested {ts} while as_of={self._as_of}"
            self._accesses.append(msg)
            if self._strict:
                raise LookAheadError(msg)
        return self._store.row_of(ts)

    # ------------------------------------------------------------------ reads
    def history(self, symbol: str, n: Optional[int] = None) -> pd.Series:
        col = self._store.col_of(symbol)
        if col is None:
            return pd.Series(dtype=float)
        start = 0 if n is None else max(0, self._row + 1 - n)
        vals = self._store.close_arr[start:self._row + 1, col]
        return pd.Series(vals, index=self._store.calendar[start:self._row + 1])

    def history_volume(self, symbol: str, n: Optional[int] = None) -> pd.Series:
        col = self._store.col_of(symbol)
        if col is None:
            return pd.Series(dtype=float)
        start = 0 if n is None else max(0, self._row + 1 - n)
        vals = self._store.vol_arr[start:self._row + 1, col]
        return pd.Series(vals, index=self._store.calendar[start:self._row + 1])

    def price(self, symbol: str, ts=None) -> float:
        row = self._row if ts is None else self._row_of(ts)
        col = self._store.col_of(symbol)
        if col is None:
            return float("nan")
        return float(self._store.close_arr[row, col])

    def volume(self, symbol: str, ts=None) -> float:
        row = self._row if ts is None else self._row_of(ts)
        col = self._store.col_of(symbol)
        if col is None:
            return float("nan")
        return float(self._store.vol_arr[row, col])

    def bar(self, symbol: str) -> Bar:
        return Bar(self._as_of, self.price(symbol), self.volume(symbol))

    def symbols(self) -> List[str]:
        return list(self._store.symbols)

    def eligible(self, min_history: int = 1) -> List[str]:
        return self._store.eligible(self._row, min_history)


class DataStore:
    """Holds the whole history as arrays; hands out truncated views."""

    def __init__(self, close: Dict[str, pd.Series],
                 volume: Optional[Dict[str, pd.Series]] = None):
        close = {k: v.sort_index() for k, v in close.items() if len(v)}
        if not close:
            raise ValueError("empty price store")
        symbols = sorted(close)
        idx = sorted({d for s in close.values() for d in s.index})

        self.calendar = pd.DatetimeIndex(idx)
        self.symbols = symbols
        self._col = {s: i for i, s in enumerate(symbols)}
        self._row = {d: i for i, d in enumerate(self.calendar)}

        arr = np.full((len(self.calendar), len(symbols)), np.nan, dtype=np.float64)
        for j, s in enumerate(symbols):
            arr[:, j] = close[s].reindex(self.calendar).to_numpy(dtype=np.float64)
        self.close_arr = arr

        vol = np.full_like(arr, np.nan)
        for j, s in enumerate(symbols):
            v = (volume or {}).get(s)
            if v is not None:
                vol[:, j] = v.sort_index().reindex(self.calendar).to_numpy(dtype=np.float64)
        self.vol_arr = vol

    # ------------------------------------------------------------------ lookups
    def row_of(self, ts) -> int:
        ts = pd.Timestamp(ts)
        i = self._row.get(ts)
        if i is not None:
            return i
        pos = self.calendar.searchsorted(ts, side="right") - 1
        return int(max(pos, 0))

    def col_of(self, symbol: str) -> Optional[int]:
        return self._col.get(symbol)

    def close_of(self, symbol: str, ts) -> float:
        col = self._col.get(symbol)
        if col is None:
            return float("nan")
        return float(self.close_arr[self.row_of(ts), col])

    def view(self, as_of, strict: bool = True) -> PITView:
        row = self.row_of(pd.Timestamp(as_of))
        return PITView(self, row, self.calendar[row], strict=strict)

    def eligible(self, row: int, min_history: int = 1) -> List[str]:
        lo = max(0, row + 1 - min_history)
        ok = np.isfinite(self.close_arr[lo:row + 1, :]).sum(axis=0)
        return [self.symbols[j] for j in np.nonzero(ok >= min_history)[0]]

    # ------------------------------------------------------------------ copies
    @classmethod
    def from_arrays(cls, calendar: pd.DatetimeIndex, symbols: List[str],
                    close_arr: np.ndarray, vol_arr: Optional[np.ndarray] = None) -> "DataStore":
        obj = cls.__new__(cls)
        obj.calendar = pd.DatetimeIndex(calendar)
        obj.symbols = list(symbols)
        obj._col = {s: i for i, s in enumerate(obj.symbols)}
        obj._row = {d: i for i, d in enumerate(obj.calendar)}
        obj.close_arr = np.asarray(close_arr, dtype=np.float64)
        obj.vol_arr = (np.asarray(vol_arr, dtype=np.float64) if vol_arr is not None
                       else np.full_like(obj.close_arr, np.nan))
        return obj

    def sliced(self, cut) -> "DataStore":
        """Copy with everything after ``cut`` removed (audit gate 2)."""
        cut = pd.Timestamp(cut)
        keep = self.calendar <= cut
        return DataStore.from_arrays(self.calendar[keep], self.symbols,
                                     self.close_arr[keep], self.vol_arr[keep])

    def poisoned(self, cut, seed: int = 7) -> "DataStore":
        """Copy with everything after ``cut`` replaced by garbage (audit gate 3)."""
        cut = pd.Timestamp(cut)
        rng = np.random.default_rng(seed)
        close = self.close_arr.copy()
        vol = self.vol_arr.copy()
        mask = np.asarray(self.calendar > cut)
        n = int(mask.sum())
        if n:
            close[mask] = rng.uniform(0.01, 1e6, size=(n, close.shape[1]))
            vol[mask] = rng.uniform(1.0, 1e9, size=(n, vol.shape[1]))
        return DataStore.from_arrays(self.calendar, self.symbols, close, vol)
