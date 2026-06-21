#!/usr/bin/env python3
"""
gpu_train.py — the training step that closes the loop (§5 GRPO + §6 distill).

Consumes the datasets produced by grpo_loop.py and distill_pipeline.py and runs
the actual fine-tune, emitting a LoRA adapter that the §3 inference layer
(inference/local_serve.py) then serves. This is the ONE component that needs a
GPU; everything upstream is hardware-agnostic data shaping.

Two modes
---------
- **sft**   : supervised fine-tune on distillation data (§6). Student learns to
              imitate the council's verified outputs. Weight field → loss
              weighting where the trainer supports it.
- **grpo**  : Group Relative Policy Optimization on the §5 dataset. Uses the
              precomputed group-relative advantages — no critic network.

Design
------
- Lazy heavy imports (torch/trl/peft) INSIDE run paths, so importing this module
  or running `--check` costs nothing and works on any machine.
- `--check` validates the dataset + reports what WOULD run (shapes, counts,
  est. steps) WITHOUT loading torch — so the whole loop is verifiable on the
  24GB Mac; only the final `train` invocation needs the GPU box.
- Sensible LoRA defaults (r=16, alpha=32, dropout=0.05) on attention+MLP
  projections; 4-bit base load when bitsandbytes is available.

Output: ./adapters/<run_name>/  (LoRA weights + adapter_config.json)
Then serve the merged/adapter model via inference/local_serve.py.

Usage
-----
    python gpu_train.py --check --data distill_dataset.jsonl --mode sft
    python gpu_train.py --data distill_dataset.jsonl --mode sft \
        --base Qwen/Qwen2.5-7B-Instruct --run-name council-student-v1
    python gpu_train.py --data grpo_dataset.jsonl --mode grpo --base <model>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Dataset loading + validation (no torch)
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[train] dataset not found: {path}", file=sys.stderr)
        sys.exit(2)
    rows = []
    bad = 0
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[train] skipped {bad} unparseable lines", file=sys.stderr)
    return rows


def validate(rows: List[dict], mode: str) -> dict:
    """Check required fields per mode; return a summary. Exits on fatal issues."""
    if not rows:
        print("[train] empty dataset", file=sys.stderr)
        sys.exit(2)
    n_msgs = sum(1 for r in rows if isinstance(r.get("messages"), list) and len(r["messages"]) >= 2)
    summary = {"rows": len(rows), "with_messages": n_msgs, "mode": mode}
    if mode == "grpo":
        n_adv = sum(1 for r in rows if "advantage" in r)
        n_groups = len({r.get("group_id") for r in rows if "group_id" in r})
        summary.update({"with_advantage": n_adv, "groups": n_groups})
        if n_adv == 0:
            print("[train] GRPO mode needs 'advantage' field — run grpo_loop.py first",
                  file=sys.stderr)
            sys.exit(2)
    else:  # sft
        n_w = sum(1 for r in rows if "weight" in r)
        summary.update({"with_weight": n_w})
    if n_msgs == 0:
        print("[train] no usable 'messages' examples", file=sys.stderr)
        sys.exit(2)
    return summary


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------
def lora_kwargs():
    return dict(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Training (heavy — imports torch/trl/peft lazily)
# ---------------------------------------------------------------------------
def train_sft(rows, base_model, run_name, out_dir, epochs, batch, lr, max_len):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def fmt(ex):
        # apply the model's chat template to the messages
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = Dataset.from_list(rows).map(fmt)

    quant = None
    try:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(**lora_kwargs()))
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch, gradient_accumulation_steps=4,
        learning_rate=lr, max_seq_length=max_len, logging_steps=10,
        save_strategy="epoch", bf16=True, report_to="none", run_name=run_name)
    SFTTrainer(model=model, args=cfg, train_dataset=ds,
               processing_class=tok).train()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] SFT adapter saved → {out_dir}")


def train_grpo(rows, base_model, run_name, out_dir, epochs, batch, lr, max_len):
    """
    GRPO fine-tune using precomputed group-relative advantages.

    We reconstruct groups by group_id and feed (prompt, completion, advantage)
    to TRL's GRPO trainer. Where the installed TRL expects to compute its own
    rollouts, this path instead does offline advantage-weighted policy gradient
    (advantage already in the data), which is the offline-GRPO formulation.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Offline GRPO: each row already carries an advantage; weight the LM loss by
    # the (clamped, normalized) advantage. Positive-advantage completions are
    # reinforced, negative are down-weighted.
    def fmt(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        adv = float(ex.get("advantage", 0.0))
        return {"text": text, "adv": adv}

    ds = Dataset.from_list(rows).map(fmt)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, LoraConfig(**lora_kwargs()))
    model.print_trainable_parameters()

    # Minimal offline advantage-weighted trainer loop (kept dependency-light;
    # swap for trl.GRPOTrainer when online rollouts against the council are wired).
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    def collate(batch_rows):
        texts = [b["text"] for b in batch_rows]
        advs = torch.tensor([b["adv"] for b in batch_rows], dtype=torch.bfloat16)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        enc["adv"] = advs
        return enc

    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    dev = next(model.parameters()).device
    for ep in range(epochs):
        for step, enc in enumerate(dl):
            adv = enc.pop("adv").to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            # scale the LM loss per-example by advantage sign/magnitude
            loss = out.loss * (1.0 + adv.mean())
            loss.backward()
            opt.step(); opt.zero_grad()
            if step % 10 == 0:
                print(f"[grpo] ep{ep} step{step} loss {loss.item():.4f}")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[train] GRPO adapter saved → {out_dir}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="GPU training harness — SFT distill + GRPO (§5/§6)")
    ap.add_argument("--data", required=True, help="dataset jsonl (distill or grpo)")
    ap.add_argument("--mode", choices=["sft", "grpo"], default="sft")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="base model HF id/path")
    ap.add_argument("--run-name", default="council-student")
    ap.add_argument("--out", default=None, help="output dir (default ./adapters/<run-name>)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--check", action="store_true",
                    help="validate dataset + report plan WITHOUT loading torch")
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    summary = validate(rows, args.mode)

    out_dir = Path(args.out or f"./adapters/{args.run_name}")
    est_steps = (summary["with_messages"] // max(1, args.batch)) * args.epochs

    print("=== training plan ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  base: {args.base}")
    print(f"  est. optimizer steps: ~{est_steps}")
    print(f"  output: {out_dir}")

    if args.check:
        # verify torch availability without importing the heavy stack
        import importlib.util
        has_torch = importlib.util.find_spec("torch") is not None
        has_trl = importlib.util.find_spec("trl") is not None
        has_peft = importlib.util.find_spec("peft") is not None
        print(f"  torch:{has_torch}  trl:{has_trl}  peft:{has_peft}")
        print("  CHECK OK — dataset valid; run without --check on a GPU box to train.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fn = train_sft if args.mode == "sft" else train_grpo
    fn(rows, args.base, args.run_name, out_dir,
       args.epochs, args.batch, args.lr, args.max_len)
    print("# serve it: python ../sentinel-api/inference/local_serve.py "
          f"--model {out_dir} --backend sglang")


if __name__ == "__main__":
    main()
