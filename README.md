# backtest-engine

A backtest engine that owns data access, the clock, the ledger and the cost model — and
nothing else. Strategies are injected; the engine never contains a signal, a factor or a
defence rule.

The design target is narrow: **a number you can defend**. Every place where a future price
could leak into a decision is blocked at the data layer, and the engine ships with the tests
that prove it.

---

## Guarantees

1. **No look-ahead.** A strategy only ever sees `PITView`, a view truncated at the decision
   time. Any read past it raises `LookAheadError` — it does not silently return the future.
2. **One clock.** Decision and execution both happen at the close of `t`, the last bar that
   has data. There is no second timestamp to get wrong.
3. **Strategy external.** No signal, factor or overlay lives in the engine. An exposure overlay is a strategy
   output, not engine magic — so it can never be priced into the period it was measured on.
4. **Fast equals rigorous.** The parallel loop reproduces the ledger loop to floating-point
   tolerance (measured: 2e-14 to 2e-13 relative).

---

## Two modes

| | `rigorous` | `fast` |
|---|---|---|
| Purpose | **the final, correct simulation** | **multi-model / multi-parameter comparison** |
| Execution | one process, cycle by cycle | process pool, all cycles in parallel |
| Ledger | cash, shares, NAV after every trade | per-cycle return, chained at the end |
| Output | equity curve, trade list, weights, costs | equity curve, returns |
| Stateful strategies | supported | **refused** |
| Speed (500 assets, 800 weekly cycles) | 5.05 s | **0.46 s (11.0×)** |

Use `fast` to sweep hundreds of configurations; re-run the winner under `rigorous` before you
believe it. The gates below exist precisely so the two can never disagree.

---

## Layout

```
bte/
  contract.py  Strategy protocol, TargetPortfolio (long-only, weights <= 1)
  data.py      DataStore (dates x symbols matrix), PITView (the look-ahead guard)
  clock.py     decision grids: month_end / week_end / daily / every_k_bars
  runner.py    run_rigorous (ledger) + run_fast (process pool)
  config.py    RunConfig: clocks, cost modes, FX, workers
  fx.py        convert prices into the base currency at load time
  audit.py     the five gates
  cli.py       command line entry point
strategies/    buyhold.py (the only example kept in the repo)
tests/         test_engine.py — the gates
examples/      demo.py (trivial), bench.py, scaling.py
```

---

## Quick start

```bash
PYTHONPATH=. python -m bte.cli \
    --strategy strategies.buyhold:BuyHold \
    --synthetic 120,2000 --mode fast
```

```
config      fast | month_end | same_close | cost=bps_per_side bps=10 | base=CHF | workers=auto
cycles      92  (2005-01-03 .. 2012-08-31)
CAGR        +7.85%
Sharpe      0.604
MaxDD       -19.58%
```

With real data (a pickle or CSV of closes, one column per symbol):

```bash
PYTHONPATH=. python -m bte.cli --config config.yaml \
    --strategy strategies.buyhold:BuyHold --data prices.pkl --mode rigorous
```

From Python:

```python
from bte import DataStore, build_grid, run_rigorous

store = DataStore(close)                       # {symbol: Series}
grid = build_grid(store.calendar, "month_end")
res = run_rigorous(store, MyStrategy(), grid, initial_cash=1e6, cost_bps=10)
print(res.stats(), res.trades[:3])
```

---

## Writing a strategy

```python
from bte.contract import DecisionContext, Strategy, TargetPortfolio

class EqualWeight(Strategy):
    stateful = False          # True -> fast mode refuses to run it
    warmup = 1

    def decide(self, ctx: DecisionContext) -> TargetPortfolio:
        names = [s for s in ctx.universe if ctx.view.price(s) > 0]
        if not names:
            return TargetPortfolio({}, cash_weight=1.0)
        w = 1.0 / len(names)
        return TargetPortfolio({s: w for s in names})
```

`DecisionContext` gives `as_of`, the truncated `view`, the current `holdings`, `cash` and
`nav`. `TargetPortfolio` is long-only, weights must sum to <= 1, and the remainder is cash.
Returning an over-weight or a negative raises immediately rather than silently changing the
portfolio downstream.

Strategies are private: they live in `strategies/` on your machine and are excluded from this
repository by `.gitignore`. The engine ships one trivial example, `BuyHold`.

## Configuration

Precedence: **CLI flag > config file > default**. Nothing is read from the environment.

```yaml
initial_cash: 6000
base_currency: CHF

decision:
  frequency: month_end        # month_end | week_end | daily | <int k bars>
  execute: same_close         # same_close | next_close (fast mode: same_close only)

cost:
  mode: bps_per_side          # bps_per_side | bps_per_turnover | flat_per_trade | spread
  bps: 10
  spread_bps: 4
  min_per_trade: 1.0

fx:
  base: CHF
  rates: {USD: 0.88, EUR: 0.95}
  # series: fx.csv            # date,ccy,rate — overrides the constants

mode: rigorous                # rigorous | fast
workers: 0                    # 0 = min(16, cpu_count)
```

### Cost modes

| mode | charged |
|---|---|
| `bps_per_side` | `traded_notional × bps / 1e4` — buys and sells each counted |
| `bps_per_turnover` | `0.5 × traded_notional × bps / 1e4` — the industry turnover convention |
| `flat_per_trade` | `n_trades × min_per_trade` — minimum commission |
| `spread` | `traded_notional × spread_bps / 2 / 1e4` — crossing half the spread |

Costs are paid **at `t`, before the interval**, and positions are sized on the NAV that
remains after the cost. Both modes share this convention; getting it wrong is what makes a
"10 bps" backtest drift by 0.1% over a decade.

### FX

Prices are converted into the base currency **once, at load time**: `price_base =
price_local × rate(t)`. The strategy never sees currencies, and an FX move lands in the P&L
exactly as it would for a real account holding that currency.

---

## The gates

Run them with `PYTHONPATH=. python -m tests.test_engine`. If any fails, no number from the
engine may be quoted.

| gate | what it proves |
|---|---|
| `test_no_future_access` | a strategy that asks for tomorrow raises `LookAheadError` |
| `test_truncation` | delete all data after T → every decision up to T is byte-identical |
| `test_poison` | replace everything after T with random garbage → results up to T do not move |
| `test_fast_equals_rigorous` | the parallel loop reproduces the ledger loop (≤ 1e-9 relative) |
| `test_cash_and_costs_reconcile` | the first cycle loses exactly the cost, nothing else |

The truncation and poison gates are the real proof: a leak anywhere in the signal, the
universe filter or the overlay changes the decisions, and both tests catch it.

---

## Performance

500 assets, 800 weekly decisions, 20-name portfolio (this box: 32 logical / 16 physical cores):

| workers | time | speedup |
|---|---|---|
| rigorous (1 core) | 5.05 s | 1.0× |
| 4 | 1.34 s | 3.8× |
| 8 | 0.74 s | 6.9× |
| **16** | **0.46 s** | **11.0×** |
| 32 | 0.50 s | 10.2× |
| 128 | 0.59 s | 8.6× |

Scaling peaks at the physical core count and then decays: past 16 the only thing growing is
pool overhead. `workers` is clamped to `cpu_count`, so asking for 128 on a 32-thread box
gives 32 — and 32 is already slower than 16.

Two things make it fast, neither of them the pool:

- **Data is one `dates × symbols` float64 matrix**, so a lookup is a positional index
  (O(1)) instead of a boolean mask over a Series. This alone made the sequential loop ~5×
  faster.
- **Fork, not pickle.** Workers inherit the price matrix copy-on-write.

---

## Known limits

Deliberate omissions, not bugs. Each changes the numbers materially, so it must be stated
before any conclusion is quoted:

- **No corporate actions yet.** Splits and dividends are not applied; prices are assumed
  total-return adjusted upstream.
- **No delisting handling.** A symbol that stops trading keeps its last price forever — which
  is also a survivorship bias if the universe is today's index membership.
- **Long-only.** Negative weights raise; no leverage, shorting or margin.
- **Universe is whatever you pass.** Point-in-time index membership is not in the engine.
- **Fast mode is state-free only.** It refuses `stateful=True` rather than return a number
  that quietly ignores cash constraints, minimum lots or taxes.

---

## License

