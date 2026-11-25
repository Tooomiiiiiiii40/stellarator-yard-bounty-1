#!/usr/bin/env python3
"""Standalone heuristic scorer: early-momentum vs rug-pull probability
for newly-initialized SPL mints + DEX pool creation events.

Scores a launch within its first 15 blocks and returns a calibrated
probability in [0, 1] that the launch is a rug-pull (1 = very likely rug).
Pure stdlib, no network, no keys. Python 3.12.

Event schema (JSON dict):
  mint_authority: bool   True if mint authority is still active
  freeze_authority: bool True if freeze authority is still active
  top10: list[float]     top-10 holder shares of total supply (0..1 each)
  lp_burned: bool        LP tokens sent to a burn address
  lp_locked: bool        LP tokens locked in a locker contract
  lp_lock_days: float    lock duration in days (0 if not locked)
  slot_tx_count: int     txs touching the mint in the creation slot
  first_block_buyers: int distinct buyers in the first block
  creator_wallets: int   distinct wallets funded by the creator pre-launch
"""
import json
import math
import sys

# Component weights (sum to 1.0 before calibration).
W = {"authority": 0.25, "gini": 0.25, "liquidity": 0.30, "bundler": 0.20}

# Sigmoid calibration: maps raw weighted risk to a probability.
CAL_K, CAL_MID = 6.0, 0.45


def gini(shares):
    """Gini coefficient of a list of non-negative shares (0 = equal)."""
    xs = sorted(float(x) for x in shares if x is not None)
    n = len(xs)
    if n == 0 or sum(xs) <= 0:
        return 0.0
    total = sum(xs)
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return max(0.0, min(1.0, (2.0 * cum) / (n * total) - (n + 1.0) / n))


def authority_risk(ev):
    """Mint/freeze authority still live lets the creator mint or freeze."""
    r = 0.0
    if ev.get("mint_authority"):
        r += 0.65  # can mint infinite supply
    if ev.get("freeze_authority"):
        r += 0.35  # can freeze holders' tokens
    return min(1.0, r)


def concentration_risk(ev):
    """Top-10 holder Gini; >0.80 means a few wallets own the float."""
    top10 = ev.get("top10") or []
    g = gini(top10)
    share = sum(top10)
    r = g
    if share > 0.60:  # top-10 owns most of supply on top of high Gini
        r = min(1.0, r + 0.15)
    return r


def liquidity_risk(ev):
    """LP neither burned nor locked = creator can pull the pool."""
    if ev.get("lp_burned"):
        return 0.0
    if ev.get("lp_locked"):
        days = float(ev.get("lp_lock_days") or 0)
        if days >= 365:
            return 0.10
        if days >= 90:
            return 0.30
        if days >= 30:
            return 0.55
        return 0.75  # short lock is barely better than no lock
    return 1.0


def bundler_risk(ev):
    """Slot-bundler pattern: many txs in the creation slot, few real buyers,
    creator pre-funding a cluster of wallets."""
    n_tx = int(ev.get("slot_tx_count") or 0)
    buyers = int(ev.get("first_block_buyers") or 0)
    wallets = int(ev.get("creator_wallets") or 0)
    r = 0.0
    if n_tx >= 20:
        r += 0.45
    elif n_tx >= 8:
        r += 0.25
    if 0 < buyers <= 3 and n_tx >= 5:
        r += 0.30  # volume with almost no distinct buyers
    if wallets >= 5:
        r += 0.25  # sybil cluster funded by the creator
    return min(1.0, r)


def calibrate(raw):
    """Sigmoid calibration to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-CAL_K * (raw - CAL_MID)))


def score(ev):
    """Score one launch event. Returns (probability, components dict)."""
    comp = {
        "authority": authority_risk(ev),
        "concentration": concentration_risk(ev),
        "liquidity": liquidity_risk(ev),
        "bundler": bundler_risk(ev),
    }
    raw = (W["authority"] * comp["authority"]
           + W["gini"] * comp["concentration"]
           + W["liquidity"] * comp["liquidity"]
           + W["bundler"] * comp["bundler"])
    return calibrate(raw), comp


def label(p):
    if p >= 0.70:
        return "LIKELY_RUG"
    if p >= 0.40:
        return "SUSPICIOUS"
    return "LIKELY_LEGIT"


# SIMULATED fixtures for --selftest / backtest (synthetic, not real launches).
CASES = [
    ("legit-fair-launch",
     {"mint_authority": False, "freeze_authority": False,
      "top10": [0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01],
      "lp_burned": True, "lp_locked": False, "lp_lock_days": 0,
      "slot_tx_count": 3, "first_block_buyers": 42, "creator_wallets": 1},
     "LIKELY_LEGIT"),
    ("legit-locked-lp",
     {"mint_authority": False, "freeze_authority": False,
      "top10": [0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01],
      "lp_burned": False, "lp_locked": True, "lp_lock_days": 365,
      "slot_tx_count": 5, "first_block_buyers": 18, "creator_wallets": 2},
     "LIKELY_LEGIT"),
    ("rug-mint-live-no-lp",
     {"mint_authority": True, "freeze_authority": True,
      "top10": [0.28, 0.12, 0.09, 0.06, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01],
      "lp_burned": False, "lp_locked": False, "lp_lock_days": 0,
      "slot_tx_count": 31, "first_block_buyers": 2, "creator_wallets": 9},
     "LIKELY_RUG"),
    ("rug-bundled-snipe",
     {"mint_authority": False, "freeze_authority": False,
      "top10": [0.35, 0.18, 0.11, 0.07, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01],
      "lp_burned": False, "lp_locked": True, "lp_lock_days": 7,
      "slot_tx_count": 47, "first_block_buyers": 1, "creator_wallets": 12},
     "LIKELY_RUG"),
    ("grey-short-lock-whale",
     {"mint_authority": False, "freeze_authority": True,
      "top10": [0.16, 0.11, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01],
      "lp_burned": False, "lp_locked": True, "lp_lock_days": 30,
      "slot_tx_count": 9, "first_block_buyers": 7, "creator_wallets": 3},
     "SUSPICIOUS"),
]


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        for name, ev, expect in CASES:
            p, comp = score(ev)
            got = label(p)
            ok = "OK " if got == expect else "MISS"
            print(f"{ok} {name:28s} p={p:.3f} {got:13s} "
                  f"(want {expect}) comp={ {k: round(v, 2) for k, v in comp.items()} }")
        return 0
    if len(argv) == 2:
        with open(argv[1]) as fh:
            ev = json.load(fh)
        p, comp = score(ev)
        print(json.dumps({"rug_probability": round(p, 4), "label": label(p),
                          "components": {k: round(v, 3) for k, v in comp.items()}},
                         indent=2))
        return 0
    print("usage: scorer.py [--selftest | event.json]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
