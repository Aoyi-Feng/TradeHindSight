"""Persist a run so it can be inspected later without re-running it.

A run directory looks like this:

    runs/20261026-132501_a1b2c3/
        manifest.json    run id, UTC time, git commit, config, strategy, data fingerprint
        equity.csv       NAV after every decision, one row per cycle
        returns.csv      per-cycle return
        weights.csv      post-trade weights at each decision
        trades.csv       every fill: time, symbol, shares, price, notional
        costs.csv        cost charged at each decision

Nothing is written unless you ask for it: call save_run() or pass --save on the CLI.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .runner import Result


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _data_fingerprint(store) -> dict:
    arr = np.asarray(store.close_arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(np.nan_to_num(arr, nan=-9.87654e30).tobytes())
    return {
        "symbols": len(store.symbols),
        "days": len(store.calendar),
        "first": str(store.calendar[0].date()),
        "last": str(store.calendar[-1].date()),
        "matrix_sha256": h.hexdigest()[:16],
    }


def save_run(result: Result, cfg, store=None, strategy=None, out_dir="runs",
             label: str = "", extra: Optional[dict] = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = stamp
    root = Path(out_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)

    stats = result.stats()
    manifest = {
        "run_id": run_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "engine_git_commit": _git_commit(),
        "mode": result.mode,
        "strategy": type(strategy).__name__ if strategy is not None else None,
        "strategy_params": {k: v for k, v in vars(strategy).items()
                            if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))
                            } if strategy is not None else None,
        "config": cfg.describe() if hasattr(cfg, "describe") else repr(cfg),
        "stats": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                  for k, v in stats.items()},
        "cycles": int(len(result.returns)),
        "final_nav": float(result.equity.iloc[-1]) if len(result.equity) else None,
        "trades": len(result.trades),
        "total_cost": float(result.costs.sum()) if len(result.costs) else 0.0,
        "data": _data_fingerprint(store) if store is not None else None,
    }
    if extra:
        manifest["extra"] = extra
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    result.equity.rename("nav").to_csv(path / "equity.csv", index_label="date")
    result.returns.rename("return").to_csv(path / "returns.csv", index_label="date")
    if len(result.costs):
        result.costs.rename("cost").to_csv(path / "costs.csv", index_label="date")
    if len(result.weights):
        result.weights.to_csv(path / "weights.csv", index_label="date")
    if result.trades:
        pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(path / "trades.csv", index=False)

    return path
