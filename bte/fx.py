"""Currency conversion. Prices are converted once, at load time, into the base currency.

The engine and the strategy only ever see base-currency prices, so an FX move is part of
the P&L exactly as it would be for a holder of that account.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from .config import RunConfig


def _rate_frame(cfg: RunConfig, calendar: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """ccy -> rate series aligned to the calendar (constant or from CSV)."""
    out: Dict[str, pd.Series] = {}
    for ccy, r in (cfg.fx_rates or {}).items():
        if ccy == cfg.base_currency:
            out[ccy] = pd.Series(1.0, index=calendar)
        else:
            out[ccy] = pd.Series(float(r), index=calendar)

    if cfg.fx_series:
        df = pd.read_csv(cfg.fx_series, parse_dates=["date"])
        for ccy, grp in df.groupby("ccy"):
            s = grp.set_index("date")["rate"].sort_index().reindex(calendar).ffill().bfill()
            out[ccy] = s.astype(float)
    return out


def convert_to_base(close: Dict[str, pd.Series], cfg: RunConfig) -> Dict[str, pd.Series]:
    """Multiply each symbol's local-currency prices by its rate into the base currency."""
    if not cfg.currency:
        return close

    calendar = pd.DatetimeIndex(sorted({d for s in close.values() for d in s.index}))
    rates = _rate_frame(cfg, calendar)

    out: Dict[str, pd.Series] = {}
    for sym, s in close.items():
        ccy = cfg.currency.get(sym, cfg.base_currency)
        if ccy == cfg.base_currency:
            out[sym] = s
            continue
        r = rates.get(ccy)
        if r is None:
            raise KeyError(f"no FX rate for {ccy} (symbol {sym}); add it to fx.rates")
        out[sym] = (s * r.reindex(s.index).ffill().bfill()).astype(float)
    return out
