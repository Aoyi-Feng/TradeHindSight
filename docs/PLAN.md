# Plan and status

## Why this exists

The previous research engine produced three different CAGR figures for the same strategy
(39.6 / 42.8 / 42.5) because four implementations of the same portfolio maths coexisted. It
also priced the defence overlay into the month it was measured on, which inflated every
defence number by up to 12 percentage points.

This repository is the answer to both: one implementation of the maths, and a data layer that
makes a future read impossible rather than merely discouraged.

## Roadmap

| phase | scope | acceptance | state |
|---|---|---|---|
| **P0** | data guard, clock, rigorous ledger, four gates | a strategy runs 1999–2026; truncation and poison gates pass | **done** |
| **P1** | fast mode, process pool, equivalence gate | fast reproduces rigorous to ≤1e-9 relative | **done** |
| **P2** | config layer: cost modes, FX, CLI | four cost modes and FX conversion verified | **done (costs, FX)** |
| **P2b** | corporate actions: splits, dividends, delisting | three split cases reconcile by hand | **open** |
| **P3** | reconcile against the audited research numbers using the private strategy | matches the audited CAGR / Sharpe / MaxDD | **in progress — 0.95 monthly correlation after fixing the clock convention; residual gap open** |
| **P4** | gates in CI on every commit | a failing gate blocks the commit | **open** |

## What is done

### Engine core
- `DataStore`: prices as one `dates × symbols` float64 matrix, O(1) positional lookup
  (this alone made the sequential loop ~5× faster).
- `PITView`: every read bounded by `as_of`; a later timestamp raises `LookAheadError`.
- `run_rigorous`: cash/share ledger, cost charged at `t` before the interval, positions sized
  on post-cost NAV.
- `run_fast`: cycle returns in a fork-based process pool (default `min(16, cpu_count)`),
  chained sequentially; refuses stateful strategies.

### Where it paid off immediately
- Old fast/rigorous mismatch: 4,163 currency units on a 3.9M NAV (0.106%) at 2016-01-29.
  Root cause: rigorous sized positions on pre-cost NAV, fast on post-cost NAV. Aligned, the
  residual is now 2e-14 relative.
- Throughput: 500 assets × 800 weekly decisions, 5.05 s single-threaded → 0.46 s with 16
  workers (11.0×). Beyond 16 workers everything gets slower; `workers` is clamped to
  `cpu_count`.

### Configuration
- `RunConfig` from yaml/json, CLI overrides win, validation refuses fast+next_close.
- Cost modes: `bps_per_side`, `bps_per_turnover`, `flat_per_trade`, `spread`.
- FX converted to the base currency at load time (constant or CSV series).

### Gates (all passing)
`test_no_future_access`, `test_truncation`, `test_poison`, `test_fast_equals_rigorous`,
`test_cash_and_costs_reconcile`.

## What is not done

- **Corporate actions** — splits and dividends are assumed handled upstream. Until they are,
  any total-return comparison against another engine will differ.
- **Delisting** — a dead symbol keeps its last price forever.
- **Point-in-time universe** — the engine takes whatever universe the strategy filters; index
  membership history is not supplied.
- **Reconciliation** — the private strategy has not yet been re-run through this engine.
  Until it reproduces the audited figures, one of the two engines is still wrong.

## Next step

**P3 — reconcile.** Run the private strategy against real data through this engine and
compare with the audited figures. Everything else (P2b, P4) is polish by comparison: this is the step that
proves the engine.
