# Using Hindsight as an agent

You are reading a repository you may be asked to run, extend, or debug. This document is the
short path: what the code guarantees, how to drive it, and where the traps are. Read it before
touching `hindsight/`.

---

## 1. What this is

A backtest engine with **no strategy inside it**. You pass a `Strategy` object; the engine owns
the clock, the data guard, the ledger and the reporting. There are two execution paths:

| path | what it does | when to use it |
|---|---|---|
| `run_rigorous(..., strict=True)` | one decision per mark, one return per mark, single thread | the number you publish |
| `run_fast(...)` | the same arithmetic, decisions evaluated in a process pool | parameter/model sweeps |

Both produce **bit-identical** output (measured `0.00e+00`). If they ever disagree, that is a
bug in the engine, not a tolerance to widen.

`run_rigorous(..., strict=False)` is the cash-and-share ledger: it tracks actual cash, actual
shares and charges costs in currency. It is slower and *not* comparable to `fast`; use it when
you want to inspect fills (`trades.csv`), not when you want a headline number.

---

## 2. Install and verify in one minute

```bash
git clone https://github.com/Aoyi-Feng/backtest-engine.git
cd backtest-engine
python -m venv .venv && . .venv/bin/activate
pip install -e .            # pandas, numpy; PyYAML optional (config files)
python -m tests.test_engine # expect 5 PASS lines
python examples/run_sample.py
```

`run_sample.py` must print `fast-vs-rigorous 0.0e+00` for both strategies. If it does not, stop
and inspect before trusting any number you produce.

Requirements: Python 3.10+, pandas, numpy.

---

## 3. The contract you implement

```python
from hindsight.contract import DecisionContext, Strategy, TargetPortfolio

class MyStrategy(Strategy):
    name = "my_strategy"
    stateful = False        # True => fast mode refuses to run it
    warmup = 252            # bars of history you need before the first real decision

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        ...
        return TargetPortfolio({"AAPL": 0.5, "MSFT": 0.3})   # remainder is cash
```

`DecisionContext` gives you:

| field | meaning |
|---|---|
| `as_of` | the decision timestamp — **the only "now" you may use** |
| `view` | a point-in-time view; any read past `as_of` raises `LookAheadError` |
| `holdings` | current weights, symbol → weight |
| `cash` | current cash in base currency |
| `nav` | current net asset value |
| `decision_dates` | past decision marks, oldest first, ending at `as_of` |
| `params` | free-form dict from the caller |

`PITView` methods: `price(symbol)`, `history(symbol, n)`, `history_volume(symbol, n)`,
`volume(symbol)`, `eligible(min_history=1)`, `symbols`, `calendar`, `as_of`.

Rules the engine enforces for you:

- long only, weights must sum to ≤ 1, negative weights raise;
- the target is normalised before use;
- a decision may only touch data ≤ `as_of` — this is a hard error, not a warning.

---

## 4. Driving it

```python
from hindsight import DataStore, build_grid, run_fast, run_rigorous
from hindsight.config import RunConfig

store = DataStore({"AAPL": close_series, "MSFT": close_series})   # DatetimeIndex, float
grid = build_grid(store.calendar, "month_end")                    # or month_start / week_end / daily / int k

cfg = RunConfig(cost_mode="bps_per_side", cost_bps=10, execute="same_close")
res = run_rigorous(store, MyStrategy(), grid, initial_cash=1_000_000, cfg=cfg, strict=True)

res.stats()          # {'n', 'cagr', 'sharpe', 'maxdd'}
res.equity           # NAV at each mark
res.returns          # per-cycle return
res.trades           # fills (ledger mode only)
```

CLI:

```bash
python -m hindsight.cli --config config.yaml --strategy path/to/file.py:MyStrategy \
       --data prices.csv --mode fast --freq month_end --save runs --label "sweep-01"
```

`--strategy` accepts `module:Class`, `path.py:Class`, or an importable dotted path.

---

## 5. Clock and execution semantics (the part people get wrong)

**`t` is the decision mark.** Everything the strategy may see is `≤ t`; the fill happens at `t`
(or later, per `execute`). There is no "decide today, earn from last month".

| `execute` | fill at |
|---|---|
| `same_close` | the decision close |
| `next_close` | the next trading day's close |
| `delay_bars` | after `execute_delay` trading days |
| `next_decision` | the following mark |

A fill landing inside a cycle splits it: the old position earns until the fill, the new one
after it. Both modes implement the same split.

`month_end` = last trading day of the month. `month_start` = last trading day on or before the
1st. **They differ by up to a month in effect** — pick the one your reference uses, and pin it,
because changing it changes every number.

---

## 6. Costs

| `cost_mode` | formula |
|---|---|
| `bps_per_side` | traded notional × bps / 1e4 |
| `bps_per_turnover` | 0.5 × Σ\|Δw\| × NAV × bps / 1e4 |
| `flat_per_trade` | n_trades × `min_per_trade` |
| `spread` | traded notional × `spread_bps` / 2 / 1e4 |
| `per_name_change` | one bps block per name swapped (drift ignored) |
| `legacy_fraction` | (names swapped / `top` + \|Δexposure\|) × bps / 100 — reference-compatible |

---

## 7. Verification you can run yourself

| check | command | expected |
|---|---|---|
| five gates | `python -m tests.test_engine` | 5 × PASS |
| fast ≡ rigorous | `python examples/run_sample.py` | `0.0e+00` |
| scaling | `python examples/scaling.py` | 16 workers ≈ 10× |
| save/load a run | `--save runs` | `manifest.json` + 4 CSVs |

`hindsight/audit.py` exposes the gates individually: `gate_access`, `gate_truncation`, `gate_poison`,
`gate_fast_equals_rigorous`, `gate_latency`.

---

## 8. Traps

1. **Deriving month ends inside a strategy.** Use `ctx.decision_dates`. Rebuilding the calendar
   yourself silently changes which months count — it flipped nine defensive months here.
2. **NaNs that look like data.** A stock listed three months ago has 252 rows in the matrix but
   60 real prints. `score()` must return `None` unless the window is fully finite; otherwise
   `min/max` swallow the NaN and invent a score.
3. **`fast` + `stateful=True`** raises by design. A strategy that carries state across cycles
   cannot be evaluated out of order.
4. **Comparing modes with different arithmetic.** `strict=True` and the ledger are different
   books. Compare like with like.
5. **Widening a tolerance to make a gate pass.** A disagreement here means look-ahead or an
   accounting bug. Find it.
6. **Survivorship.** The engine does not know which names existed on a date; that is the data's
   job. A universe taken from today's index will flatter the past.

---

## 9. File map

```
hindsight/data.py       DataStore (dates × symbols matrix), PITView, LookAheadError
hindsight/clock.py      month_end / month_start / week_end / daily / every-k grids
hindsight/contract.py   Strategy protocol, DecisionContext, TargetPortfolio
hindsight/runner.py     strict loop, ledger loop, process-pool fast path, execution schedule
hindsight/config.py     RunConfig, cost modes
hindsight/fx.py         currency conversion into the base currency at load time
hindsight/audit.py      the five gates
hindsight/archive.py    run directories (manifest + CSVs)
hindsight/cli.py        command line
strategies/       textbook examples only (buy & hold, equal weight, 20/60, random)
```

---

## 10. What to do when asked to "add a strategy"

1. Create `strategies/<name>.py` implementing `Strategy`.
2. Only read through `ctx.view`; never touch `store` internals.
3. Run `python -m tests.test_engine` first — the engine must still be green.
4. Run the new strategy through `run_rigorous(strict=True)` to get the number, then
   `run_fast` and confirm `0.0e+00`.
5. Save the run (`--save runs`) so the result carries its config, git commit and data fingerprint.
