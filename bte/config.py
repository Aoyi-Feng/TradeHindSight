"""Run configuration: costs, FX, clocks. Every number the engine uses comes from here.

Precedence: CLI flag > config file > default. Nothing is read from the environment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


COST_MODES = ("bps_per_side", "bps_per_turnover", "flat_per_trade", "spread",
              "per_name_change")


@dataclass
class RunConfig:
    initial_cash: float = 1.0
    base_currency: str = "CHF"

    # decision / execution
    decision_freq: Any = "month_end"        # month_end | week_end | daily | int k
    execute: str = "same_close"             # same_close | next_close

    # costs
    cost_mode: str = "bps_per_side"
    cost_bps: float = 10.0
    spread_bps: float = 4.0
    min_per_trade: float = 1.0

    # fx: symbol -> currency, currency -> rate into base (constant or "series.csv")
    currency: Dict[str, str] = field(default_factory=dict)
    fx_rates: Dict[str, float] = field(default_factory=dict)
    fx_series: Optional[str] = None          # CSV: date,ccy,rate

    # execution
    workers: int = 0                         # 0 = min(16, cpu_count)
    mode: str = "rigorous"                   # rigorous | fast

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}
        flat: Dict[str, Any] = {}
        for key, val in (d or {}).items():
            if key in ("decision", "cost", "fx") and isinstance(val, dict):
                if key == "decision":
                    flat["decision_freq"] = val.get("frequency", "month_end")
                    flat["execute"] = val.get("execute", "same_close")
                elif key == "cost":
                    for a, b in (("mode", "cost_mode"), ("bps", "cost_bps"),
                                 ("spread_bps", "spread_bps"),
                                 ("min_per_trade", "min_per_trade")):
                        if a in val:
                            flat[b] = val[a]
                elif key == "fx":
                    flat["base_currency"] = val.get("base", "CHF")
                    flat["fx_rates"] = dict(val.get("rates", {}) or {})
                    flat["fx_series"] = val.get("series")
            elif key == "base_currency" and val:
                flat["base_currency"] = val
            elif key == "fx":
                continue
            elif key in known:
                flat[key] = val
        cfg = cls(**{k: v for k, v in flat.items() if k in known})
        cfg.validate()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:      # keep PyYAML optional
                raise RuntimeError("reading .yaml configs needs PyYAML (pip install pyyaml)") from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data or {})

    def overlay(self, **cli) -> "RunConfig":
        """CLI values win, but only where they were actually provided (not None)."""
        d = asdict(self)
        d.update({k: v for k, v in cli.items() if v is not None})
        cfg = RunConfig(**d)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ checks
    def validate(self) -> "RunConfig":
        if self.cost_mode not in COST_MODES:
            raise ValueError(f"cost_mode must be one of {COST_MODES}")
        if self.mode not in ("rigorous", "fast"):
            raise ValueError("mode must be rigorous or fast")
        if self.execute not in ("same_close", "next_close"):
            raise ValueError("execute must be same_close or next_close")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.cost_bps < 0 or self.spread_bps < 0 or self.min_per_trade < 0:
            raise ValueError("costs cannot be negative")
        if self.mode == "fast" and self.execute != "same_close":
            raise ValueError("fast mode only supports same_close execution")
        return self

    # ------------------------------------------------------------------ costs
    def cost_for(self, traded_notional: float, n_trades: int) -> float:
        """Cost charged on one rebalance, in base currency."""
        m = self.cost_mode
        if m == "bps_per_side":
            return traded_notional * self.cost_bps / 1e4
        if m == "bps_per_turnover":
            return 0.5 * traded_notional * self.cost_bps / 1e4
        if m == "spread":
            return traded_notional * self.spread_bps / 2 / 1e4
        if m == "flat_per_trade":
            return n_trades * self.min_per_trade
        if m == "per_name_change":
            # legacy convention: charge one bps block per name swapped, ignoring drift
            return n_trades * self.cost_bps / 1e4
        raise ValueError(m)

    def describe(self) -> str:
        return (f"{self.mode} | {self.decision_freq} | {self.execute} | "
                f"cost={self.cost_mode} bps={self.cost_bps:g} spread={self.spread_bps:g} "
                f"min={self.min_per_trade:g} | base={self.base_currency} | workers={self.workers or 'auto'}")
