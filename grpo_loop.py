#!/usr/bin/env python3
"""
grpo_loop.py — Group Relative Policy Optimization data shaping over the ledger.

What this is
------------
GRPO (Group Relative Policy Optimization, DeepSeek-style) needs, per prompt,
a GROUP of candidate completions each carrying a scalar reward, from which a
RELATIVE advantage is computed:

    advantage_i = (reward_i - mean(group)) / (std(group) + eps)

No critic/value network is required — the group baseline replaces it. This
module turns the Sentinel ledger (records: verdict/latency/tokens, keyed by
prompt_hash) into GRPO-ready groups and writes a JSONL the trainer consumes.

It is data-shaping only (technique, not weights): it does NOT run a GPU
training step. Pair the emitted JSONL with a TRL/verl GRPO trainer when GPU
is available. This keeps the loop runnable on any machine — the heavy step
is deferred, exactly like the rest of the stack.

Reward model (transparent, tunable)
-----------------------------------
    base   : CLEAN=1.0  MINOR=0.6  MAJOR=0.1  BLOCK=0.0
    latency: + up to +0.15 for fast answers (normalized within group)
    length : - small penalty for runaway tokens_out vs group median
Reward is clamped to [0, 1.15]. Groups are formed by prompt_hash; singleton
groups (no peers) get advantage 0 (nothing relative to compare) but are kept
with their raw reward so SFT-style use still works.

Schema bound to sentinel-ledger/ledger.py:
    records(id, ts, session_id, model, prompt_hash,
            verdict[CLEAN|MINOR|MAJOR|BLOCK], latency_ms,
            tokens_in, tokens_out, notes)

Usage
-----
    python grpo_loop.py --db ledger.db --output grpo_dataset.jsonl
    python grpo_loop.py --db ledger.db --min-group 2   # drop singletons
    python grpo_loop.py --stats                         # reward distribution only
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

VERDICT_BASE = {"CLEAN": 1.0, "MINOR": 0.6, "MAJOR": 0.1, "BLOCK": 0.0}
EPS = 1e-6
REWARD_MIN, REWARD_MAX = 0.0, 1.15


# ---------------------------------------------------------------------------
# Ledger read
# ---------------------------------------------------------------------------
def _iter_records(db_path: str, limit: int | None = None) -> Iterable[sqlite3.Row]:
    if not Path(db_path).exists():
        print(f"[grpo] {db_path} not found", file=sys.stderr)
        return
    sql = """
        SELECT id, ts, session_id, model, prompt_hash, verdict,
               latency_ms, tokens_in, tokens_out, notes
        FROM records
        ORDER BY ts DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(sql):
            yield row
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def _base_reward(verdict: str) -> float:
    return VERDICT_BASE.get((verdict or "").upper(), 0.0)


def _shape_group(rows: List[sqlite3.Row]) -> List[dict]:
    """Compute reward + group-relative advantage for one prompt_hash group."""
    # latency normalization within group (fast → bonus)
    lats = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    lo, hi = (min(lats), max(lats)) if lats else (0.0, 0.0)
    span = (hi - lo) or 1.0
    # length reference within group
    outs = [r["tokens_out"] for r in rows if r["tokens_out"] is not None]
    med_out = statistics.median(outs) if outs else 0

    shaped = []
    rewards = []
    for r in rows:
        rew = _base_reward(r["verdict"])
        # latency bonus: fastest in group gets +0.15, slowest +0.0
        if r["latency_ms"] is not None and lats:
            frac_fast = 1.0 - ((r["latency_ms"] - lo) / span)
            rew += 0.15 * frac_fast
        # length penalty: punish runaway output beyond 1.5x group median
        if r["tokens_out"] and med_out and r["tokens_out"] > 1.5 * med_out:
            rew -= 0.10
        rew = max(REWARD_MIN, min(REWARD_MAX, rew))
        rewards.append(rew)
        shaped.append({"row": r, "reward": rew})

    # group-relative advantage (GRPO core)
    mean = statistics.fmean(rewards) if rewards else 0.0
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    for s in shaped:
        s["advantage"] = (s["reward"] - mean) / (std + EPS) if std > 0 else 0.0
    return shaped


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def _to_example(s: dict, group_id: str, group_size: int) -> dict:
    r = s["row"]
    return {
        "messages": [
            {"role": "user",
             "content": f"[session:{r['session_id'] or 'unknown'}] "
                        f"[hash:{r['prompt_hash'] or 'none'}]"},
            {"role": "assistant",
             "content": r["notes"] or f"Council verdict: {r['verdict']}"},
        ],
        "group_id": group_id,
        "group_size": group_size,
        "reward": round(s["reward"], 4),
        "advantage": round(s["advantage"], 4),
        "verdict": r["verdict"],
        "model": r["model"],
        "latency_ms": r["latency_ms"],
        "tokens_in": r["tokens_in"],
        "tokens_out": r["tokens_out"],
        "ts": r["ts"],
        "source": "ledger-grpo",
    }


def build(db_path: str, min_group: int = 1, limit: int | None = None) -> List[dict]:
    groups: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in _iter_records(db_path, limit=limit):
        key = row["prompt_hash"] or f"_solo_{row['id']}"
        groups[key].append(row)

    examples: List[dict] = []
    for key, rows in groups.items():
        if len(rows) < min_group:
            continue
        shaped = _shape_group(rows)
        for s in shaped:
            examples.append(_to_example(s, group_id=key, group_size=len(rows)))
    return examples


def _print_stats(examples: List[dict]) -> None:
    if not examples:
        print("no examples")
        return
    rewards = [e["reward"] for e in examples]
    advs = [e["advantage"] for e in examples]
    by_v: Dict[str, int] = defaultdict(int)
    for e in examples:
        by_v[e["verdict"]] += 1
    gsizes = {e["group_id"]: e["group_size"] for e in examples}
    multi = sum(1 for v in gsizes.values() if v > 1)
    print(f"examples      : {len(examples)}")
    print(f"groups        : {len(gsizes)}  (multi-completion: {multi})")
    print(f"reward  mean  : {statistics.fmean(rewards):.4f}  "
          f"min {min(rewards):.3f}  max {max(rewards):.3f}")
    print(f"advantage rng : [{min(advs):.3f}, {max(advs):.3f}]")
    print(f"by verdict    : {dict(by_v)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO data shaping over the Sentinel ledger")
    ap.add_argument("--db", default="ledger.db", help="path to ledger.db")
    ap.add_argument("--output", default="grpo_dataset.jsonl", help="output JSONL")
    ap.add_argument("--min-group", type=int, default=1,
                    help="drop prompt groups smaller than this (2 = require peers)")
    ap.add_argument("--limit", type=int, default=None, help="cap ledger rows read")
    ap.add_argument("--stats", action="store_true", help="print stats, do not write")
    args = ap.parse_args()

    examples = build(args.db, min_group=args.min_group, limit=args.limit)
    _print_stats(examples)
    if args.stats:
        return
    with open(args.output, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples → {args.output}")


if __name__ == "__main__":
    main()
