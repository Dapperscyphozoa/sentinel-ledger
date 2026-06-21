#!/usr/bin/env python3
"""
distill_pipeline.py — council → student distillation data builder (§6).

The loop this closes
--------------------
The 93-model council produces verified outputs (ledger verdict CLEAN/MINOR).
Those verified outputs are the best teacher signal Sentinel has. This pipeline
collects them — plus high-quality agent trajectories from the OpenHands corpus —
into a single distillation dataset that trains a SMALL student model. That
student is then served by the §3 local inference layer, so the council's
expensive, verified reasoning becomes a cheap, local, rate-limit-free model.

  teacher (93-model council, ledger-verified)  ─┐
                                                ├─→ distill_dataset.jsonl ─→ student LoRA ─→ §3 local serve
  OpenHands agent trajectories (SWE-quality)   ─┘

Sources
-------
1. **Ledger** (`ledger.db`): CLEAN/MINOR records → (prompt_ref, council_answer).
   The ledger stores prompt HASHES, not raw text; pair with a prompt log via
   --prompt-log to recover inputs, else emit hash-stub references (still useful
   for response-style distillation).
2. **GRPO dataset** (`grpo_dataset.jsonl`, from grpo_loop.py): reuse the
   advantage-weighted, group-shaped examples — high-advantage completions are
   the strongest distillation targets.
3. **OpenHands corpus** (`--openhands DIR`): mine resolved-task trajectories
   (instruction → solution) as agentic/coding distillation examples. Best-effort
   parser over common trajectory shapes; skips anything it can't parse.

Output
------
JSONL, OpenAI-messages format with a `weight` field (distillation importance):
    {"messages":[{user},{assistant}], "weight": 1.0, "source": "...", ...}

Heavy step (the actual LoRA fine-tune) is deferred to gpu_train.py — this is
data-shaping only, runnable anywhere.

Usage
-----
    python distill_pipeline.py --ledger ledger.db --output distill_dataset.jsonl
    python distill_pipeline.py --grpo grpo_dataset.jsonl --openhands ../OpenHands-main
    python distill_pipeline.py --stats   # distribution only
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Verdict → distillation weight. CLEAN answers are the gold teacher signal.
VERDICT_WEIGHT = {"CLEAN": 1.0, "MINOR": 0.6}


# ---------------------------------------------------------------------------
# Source 1: ledger
# ---------------------------------------------------------------------------
def _load_prompt_log(path: Optional[str]) -> Dict[str, str]:
    """Optional hash→prompt map to recover raw inputs the ledger doesn't store."""
    if not path or not Path(path).exists():
        return {}
    m: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    h = rec.get("prompt_hash") or rec.get("hash")
                    p = rec.get("prompt") or rec.get("text")
                    if h and p:
                        m[h] = p
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return m


def from_ledger(db_path: str, prompt_log: Optional[str] = None,
                limit: int | None = None) -> Iterable[dict]:
    if not Path(db_path).exists():
        print(f"[distill] {db_path} not found — skipping ledger", file=sys.stderr)
        return
    hmap = _load_prompt_log(prompt_log)
    sql = """
        SELECT session_id, model, prompt_hash, verdict, notes, ts
        FROM records WHERE verdict IN ('CLEAN','MINOR') ORDER BY ts DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        for r in con.execute(sql):
            h = r["prompt_hash"] or "none"
            user = hmap.get(h) or f"[session:{r['session_id'] or 'unknown'}] [hash:{h}]"
            answer = r["notes"]
            if not answer:
                continue  # no teacher output to distill
            yield {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "weight": VERDICT_WEIGHT.get(r["verdict"], 0.5),
                "source": "council-ledger",
                "teacher_model": r["model"],
                "verdict": r["verdict"],
            }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Source 2: GRPO dataset (advantage-weighted)
# ---------------------------------------------------------------------------
def from_grpo(path: str, limit: int | None = None) -> Iterable[dict]:
    if not Path(path).exists():
        print(f"[distill] {path} not found — skipping grpo", file=sys.stderr)
        return
    n = 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except json.JSONDecodeError:
                continue
            msgs = ex.get("messages")
            if not msgs:
                continue
            # map advantage (can be negative) → non-negative distill weight.
            adv = float(ex.get("advantage", 0.0))
            reward = float(ex.get("reward", 0.5))
            # only distill from positive-advantage (better-than-group) outputs
            if adv <= 0:
                continue
            yield {
                "messages": msgs,
                "weight": round(min(1.0, 0.5 + adv * 0.25 + reward * 0.25), 4),
                "source": "council-grpo",
                "advantage": adv,
            }
            n += 1
            if limit and n >= limit:
                break


# ---------------------------------------------------------------------------
# Source 3: OpenHands trajectories
# ---------------------------------------------------------------------------
def from_openhands(root: str, limit: int | None = None) -> Iterable[dict]:
    """
    Mine instruction→solution pairs from the OpenHands corpus. Best-effort over
    common shapes (eval outputs, trajectory json). Skips unparseable files —
    never crashes the build.
    """
    base = Path(root)
    if not base.exists():
        print(f"[distill] {root} not found — skipping openhands", file=sys.stderr)
        return
    n = 0
    # look for jsonl/json trajectory or eval-output files
    candidates = []
    for pat in ("**/output.jsonl", "**/*trajector*.json*", "**/*eval*output*.json*"):
        candidates.extend(base.glob(pat))
    for fp in candidates:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines() if fp.suffix == ".jsonl" else [text]
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            instr = (rec.get("instruction") or rec.get("task")
                     or rec.get("problem_statement") or rec.get("query"))
            soln = (rec.get("solution") or rec.get("answer")
                    or rec.get("model_patch") or rec.get("result")
                    or rec.get("final_message"))
            # resolved/passed gate where available
            resolved = rec.get("resolved", rec.get("success", True))
            if instr and soln and resolved:
                yield {
                    "messages": [
                        {"role": "user", "content": str(instr)[:8000]},
                        {"role": "assistant", "content": str(soln)[:12000]},
                    ],
                    "weight": 0.8,
                    "source": "openhands",
                }
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def build(args) -> List[dict]:
    out: List[dict] = []
    if args.ledger:
        out.extend(from_ledger(args.ledger, args.prompt_log, args.limit))
    if args.grpo:
        out.extend(from_grpo(args.grpo, args.limit))
    if args.openhands:
        out.extend(from_openhands(args.openhands, args.limit))
    # de-dupe by (user, assistant) text
    seen = set()
    deduped = []
    for ex in out:
        m = ex["messages"]
        key = (m[0]["content"], m[1]["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def _stats(examples: List[dict]) -> None:
    from collections import defaultdict
    if not examples:
        print("no examples")
        return
    by_src: Dict[str, int] = defaultdict(int)
    wsum = 0.0
    for e in examples:
        by_src[e["source"]] += 1
        wsum += e.get("weight", 0)
    print(f"examples   : {len(examples)}")
    print(f"by source  : {dict(by_src)}")
    print(f"avg weight : {wsum/len(examples):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Council→student distillation builder (§6)")
    ap.add_argument("--ledger", help="ledger.db (CLEAN/MINOR teacher answers)")
    ap.add_argument("--grpo", help="grpo_dataset.jsonl (advantage-weighted)")
    ap.add_argument("--openhands", help="OpenHands corpus dir (agent trajectories)")
    ap.add_argument("--prompt-log", help="hash→prompt jsonl to recover raw inputs")
    ap.add_argument("--output", default="distill_dataset.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap per-source rows")
    ap.add_argument("--stats", action="store_true", help="print stats, do not write")
    args = ap.parse_args()

    if not (args.ledger or args.grpo or args.openhands):
        ap.error("provide at least one source: --ledger / --grpo / --openhands")

    examples = build(args)
    _stats(examples)
    if args.stats:
        return
    with open(args.output, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(examples)} examples → {args.output}")
    print("# next: python gpu_train.py --data", args.output, "--mode sft")


if __name__ == "__main__":
    main()
