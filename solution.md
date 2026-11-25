# SPL Mint Anomaly & Rug-Pull Radar Scorer

Standalone Python 3.12 heuristic scorer (`scorer.py`, 6,533 bytes, stdlib only,
no network, no keys). It ingests a newly-initialized SPL mint + DEX pool
creation event and outputs a calibrated rug-pull probability in [0, 1] for the
first 15 blocks of the launch (1 = very likely a rug).

## Signals

Four weighted risk components (weights sum to 1.0), combined and passed
through a sigmoid calibration (k=6.0, midpoint=0.45):

| Component | Weight | What it measures |
|---|---|---|
| authority | 0.25 | Mint authority still live (+0.65) and/or freeze authority live (+0.35) — the creator can mint or freeze |
| concentration | 0.25 | Top-10 holder Gini coefficient; +0.15 if top-10 owns >60% of supply |
| liquidity | 0.30 | LP not burned and not locked = 1.0; locked LP scales down with lock length (365d → 0.10, 30d → 0.55, 7d → 0.75); burned LP = 0.0 |
| bundler | 0.20 | Slot-bundler pattern: many txs in the creation slot, volume with ≤3 distinct first-block buyers, creator pre-funding ≥5 wallets |

Labels: p ≥ 0.70 → `LIKELY_RUG`, 0.40–0.70 → `SUSPICIOUS`, < 0.40 → `LIKELY_LEGIT`.

## How to run

```bash
python3 scorer.py --selftest        # runs the 5-case backtest below
python3 scorer.py event.json        # scores one JSON launch event, prints JSON
```

Event JSON fields: `mint_authority`, `freeze_authority` (bool),
`top10` (list of holder shares 0..1), `lp_burned`, `lp_locked` (bool),
`lp_lock_days` (float), `slot_tx_count`, `first_block_buyers`,
`creator_wallets` (ints). See the docstring in `scorer.py`.

## Backtest — SIMULATED fixtures (synthetic, not real launches)

`--selftest` scores 5 hand-built synthetic cases; all 5 match their
expected label (see `test-log.txt` for the full run):

| Case | p | Label | Why |
|---|---|---|---|
| legit-fair-launch | 0.092 | LIKELY_LEGIT | authorities revoked, LP burned, wide holders |
| legit-locked-lp | 0.112 | LIKELY_LEGIT | authorities revoked, LP locked 365d |
| rug-mint-live-no-lp | 0.944 | LIKELY_RUG | mint+freeze live, LP unlocked, 31 slot txs, 2 buyers, 9 creator wallets |
| rug-bundled-snipe | 0.729 | LIKELY_RUG | 47 slot txs, 1 buyer, 12 creator wallets, 7-day lock |
| grey-short-lock-whale | 0.452 | SUSPICIOUS | freeze live, 30-day lock, concentrated top-10 |

## Limitations

Heuristic only — it cannot prove a rug, only rank early risk. Gini is
computed on the top-10 snapshot available at launch; LP-lock checks trust the
locker state as reported in the event. Thresholds are tuned on the synthetic
fixtures above and should be re-validated on real labelled launches.
