#!/usr/bin/env python3
"""lora_dataset_builder.py — Export Sentinel council records to LoRA fine-tuning JSONL.

Reads two sources:
  1. sentinel-ledger SQLite (ledger.db) — real council CLEAN/MINOR verdict records
     with prompt_hash, model, latency, tokens
  2. ASI-Evolve experiment database (nodes.json) — scored evolution nodes
     with motivation, code, analysis, score

Outputs:
  lora_dataset.jsonl   — each line is a training example:
    {"messages": [{"role":"user","content":"<prompt>"},
                  {"role":"assistant","content":"<response>"}],
     "source": "ledger|evolve",
     "verdict": "CLEAN|MINOR",
     "score": <float>}

Only CLEAN and MINOR records are exported (MAJOR and BLOCK are excluded —
they represent failures the model should NOT learn from).

Usage:
    python lora_dataset_builder.py [options]

    --ledger-db    PATH   path to ledger.db         (default: ./ledger.db)
    --evolve-dir   PATH   path to ASI-Evolve experiment database_data/ dir
    --output       PATH   output JSONL path          (default: ./lora_dataset.jsonl)
    --min-score    FLOAT  minimum node score to include from evolve (default: 0.0)
    --limit        INT    max records per source     (default: unlimited)
    --dry-run             print stats only, do not write file
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator


# ---------------------------------------------------------------------------
# Source 1: sentinel-ledger SQLite
# ---------------------------------------------------------------------------

def _iter_ledger(
    db_path: str,
    limit: int | None = None,
) -> Generator[dict, None, None]:
    """Yield CLEAN/MINOR records from ledger.db.

    Each record has prompt_hash (not the raw prompt — ledger stores hashes only).
    We emit a synthetic training example with the hash as a stub identifier.
    In production, pair with a prompt log to recover the original text.
    """
    if not Path(db_path).exists():
        print(f"[ledger] {db_path} not found — skipping", file=sys.stderr)
        return

    sql = """
        SELECT id, ts, session_id, model, prompt_hash, verdict,
               latency_ms, tokens_in, tokens_out, notes
        FROM records
        WHERE verdict IN ('CLEAN', 'MINOR')
        ORDER BY ts DESC
    """
    if limit:
        sql += f" LIMIT {limit}"

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(sql):
            yield {
                "messages": [
                    {
                        "role": "user",
                        "content": f"[session:{row['session_id'] or 'unknown'}] "
                                   f"[hash:{row['prompt_hash'] or 'none'}]",
                    },
                    {
                        "role": "assistant",
                        "content": row["notes"] or f"Council verdict: {row['verdict']}",
                    },
                ],
                "source": "ledger",
                "verdict": row["verdict"],
                "model": row["model"],
                "latency_ms": row["latency_ms"],
                "tokens_in": row["tokens_in"],
                "tokens_out": row["tokens_out"],
                "ts": row["ts"],
                "score": 1.0 if row["verdict"] == "CLEAN" else 0.7,
            }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Source 2: ASI-Evolve nodes.json
# ---------------------------------------------------------------------------

def _iter_evolve(
    database_dir: str,
    min_score: float = 0.0,
    limit: int | None = None,
) -> Generator[dict, None, None]:
    """Yield evolution nodes from ASI-Evolve database_data/nodes.json.

    Each node becomes a training example:
      user    = motivation (what the researcher was trying to achieve)
      assistant = code + analysis (what was produced and how it was evaluated)

    Only nodes with score >= min_score are included.
    """
    nodes_file = Path(database_dir) / "nodes.json"
    if not nodes_file.exists():
        print(f"[evolve] {nodes_file} not found — skipping", file=sys.stderr)
        return

    try:
        with open(nodes_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[evolve] failed to load {nodes_file}: {e}", file=sys.stderr)
        return

    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        # Some versions store as dict keyed by id
        nodes = list(nodes.values()) if isinstance(nodes, dict) else []

    # Sort by score descending
    nodes.sort(key=lambda n: float(n.get("score", 0.0)), reverse=True)

    count = 0
    for node in nodes:
        score = float(node.get("score", 0.0))
        if score < min_score:
            continue

        motivation = (node.get("motivation") or "").strip()
        code = (node.get("code") or "").strip()
        analysis = (node.get("analysis") or "").strip()

        if not motivation or not code:
            continue  # skip incomplete nodes

        user_content = motivation
        assistant_parts = []
        if code:
            assistant_parts.append(f"```python\n{code}\n```")
        if analysis:
            assistant_parts.append(f"\n**Analysis:** {analysis}")
        assistant_content = "\n".join(assistant_parts)

        yield {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "evolve",
            "verdict": "CLEAN",
            "score": score,
            "node_id": node.get("id"),
            "node_name": node.get("name"),
            "visit_count": node.get("visit_count", 0),
        }

        count += 1
        if limit and count >= limit:
            return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger-db",  default="./ledger.db",
                        help="Path to sentinel-ledger SQLite DB")
    parser.add_argument("--evolve-dir", default="",
                        help="Path to ASI-Evolve experiment database_data/ directory")
    parser.add_argument("--output",     default="./lora_dataset.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--min-score",  type=float, default=0.0,
                        help="Minimum node score for evolve records (default: 0.0)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Max records per source")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print stats only, do not write file")
    args = parser.parse_args()

    records: list[dict] = []

    # Ledger records
    ledger_count = 0
    for rec in _iter_ledger(args.ledger_db, limit=args.limit):
        records.append(rec)
        ledger_count += 1

    # Evolve records
    evolve_count = 0
    if args.evolve_dir:
        for rec in _iter_evolve(args.evolve_dir, min_score=args.min_score, limit=args.limit):
            records.append(rec)
            evolve_count += 1

    # Stats
    verdict_counts: dict[str, int] = {}
    for r in records:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    print(f"[builder] total={len(records)}  ledger={ledger_count}  evolve={evolve_count}")
    print(f"[builder] verdicts: {verdict_counts}")
    print(f"[builder] output: {args.output}")

    if args.dry_run:
        print("[builder] dry-run — nothing written")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_kb = out_path.stat().st_size // 1024
    print(f"[builder] wrote {len(records)} records → {out_path} ({size_kb} KB)")
    print(f"[builder] generated: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
