"""Command line entry point.

    python -m tradehindsight.cli --config config.yaml --strategy strategies.buyhold:BuyHold
    python -m tradehindsight.cli --strategy strategies.buyhold:BuyHold --synthetic 200,3000 --mode fast
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_grid, run_fast, run_rigorous
from .config import RunConfig
from .data import DataStore
from .fx import convert_to_base


def load_strategy(spec: str):
    """'package.module:Class' or 'path/to/file.py:Class'."""
    if ":" not in spec:
        raise SystemExit("--strategy must look like module:Class")
    mod_path, cls_name = spec.split(":", 1)
    if mod_path.endswith(".py"):
        p = Path(mod_path).resolve()
        module_spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mod)          # type: ignore[union-attr]
    else:
        mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def load_data(args) -> dict[str, pd.Series]:
    if args.synthetic:
        n_assets, n_days = (int(x) for x in args.synthetic.split(","))
        rng = np.random.default_rng(args.seed)
        idx = pd.bdate_range("2005-01-03", periods=n_days)
        return {f"S{k:03d}": pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days))),
                                       index=idx) for k in range(n_assets)}
    if args.data:
        df = pd.read_pickle(args.data) if args.data.endswith(".pkl") else pd.read_csv(
            args.data, index_col=0, parse_dates=True)
        return {c: df[c].dropna() for c in df.columns}
    raise SystemExit("give --data file.pkl|.csv or --synthetic N,D")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hindsight")
    ap.add_argument("--config", help="yaml/json run configuration")
    ap.add_argument("--strategy", required=True, help="module:Class")
    ap.add_argument("--data", help="pkl/csv of close prices, one column per symbol")
    ap.add_argument("--synthetic", help="N,D generate N random assets over D days")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["rigorous", "fast"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--cost-mode", dest="cost_mode",
                    choices=["bps_per_side", "bps_per_turnover", "flat_per_trade", "spread"])
    ap.add_argument("--cost-bps", dest="cost_bps", type=float)
    ap.add_argument("--spread-bps", dest="spread_bps", type=float)
    ap.add_argument("--min-per-trade", dest="min_per_trade", type=float)
    ap.add_argument("--initial-cash", dest="initial_cash", type=float)
    ap.add_argument("--freq", dest="freq")
    ap.add_argument("--save", metavar="DIR", nargs="?", const="runs", default=None,
                    help="write this run (manifest, equity, returns, weights, trades) to DIR")
    ap.add_argument("--label", default="", help="free-text label stored in the manifest")
    ap.add_argument("--json", action="store_true", help="print the config and results as json")
    ap.add_argument("--now", action="store_true",
                    help="do not backtest: run one decision at the newest date in the data")
    ap.add_argument("--at", help="with --now: the date to decide at (YYYY-MM-DD)")
    ap.add_argument("--holdings", help='with --now: current weights, e.g. {"AAA":0.3,"BBB":0.3}')
    ap.add_argument("--nav", type=float, default=1.0, help="with --now: NAV for order sizes")
    ap.add_argument("--top", type=int, default=10, help="with --now: ranking rows to print")
    args = ap.parse_args(argv)

    cfg = RunConfig.from_file(args.config) if args.config else RunConfig()
    cfg = cfg.overlay(
        mode=args.mode, workers=args.workers, cost_mode=args.cost_mode,
        cost_bps=args.cost_bps, spread_bps=args.spread_bps,
        min_per_trade=args.min_per_trade, initial_cash=args.initial_cash,
        decision_freq=args.freq,
    )

    close = load_data(args)
    close = convert_to_base(close, cfg)
    store = DataStore(close)
    grid = build_grid(store.calendar, cfg.decision_freq if not str(cfg.decision_freq).isdigit()
                      else int(cfg.decision_freq))
    strat = load_strategy(args.strategy)

    if args.now:
        from .live import format_decision, latest_decision
        held = json.loads(args.holdings) if args.holdings else {}
        dec = latest_decision(store, strat, at=args.at, holdings=held, nav=args.nav,
                              frequency=cfg.decision_freq)
        if args.json:
            print(json.dumps({k: (str(v) if isinstance(v, pd.Timestamp) else v)
                              for k, v in dec.items()}, default=str, indent=2))
        else:
            print(format_decision(dec, top=args.top))
        return 0

    runner = run_fast if cfg.mode == "fast" else run_rigorous
    kw = {"cfg": cfg, "initial_cash": cfg.initial_cash}
    if cfg.mode == "fast":
        kw["workers"] = cfg.workers
    res = runner(store, strat, grid, **kw)
    s = res.stats()

    if args.json:
        print(json.dumps({"config": cfg.describe(), "stats": s,
                          "cycles": len(res.equity),
                          "final": float(res.equity.iloc[-1])}, indent=2))
    else:
        print(f"config      {cfg.describe()}")
        print(f"cycles      {len(grid)}  ({store.calendar[0].date()} .. {store.calendar[-1].date()})")
        print(f"CAGR        {s['cagr']:+.2f}%")
        print(f"Sharpe      {s['sharpe']:.3f}")
        print(f"MaxDD       {s['maxdd']:+.2f}%")
        print(f"final NAV   {res.equity.iloc[-1]:,.2f}")
    if args.save:
        from .archive import save_run
        where = save_run(res, cfg, store=store, strategy=strat, out_dir=args.save,
                         label=args.label)
        print(f"saved       {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
