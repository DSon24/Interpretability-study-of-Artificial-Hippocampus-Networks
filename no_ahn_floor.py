#!/usr/bin/env python
"""Table 1 Primary no-AHN floor (r54).

Stock Qwen2.5-3B-Instruct with sliding-window attention forced on at 8064 for every
layer, no AHN merge, no attention sinks. The backbone is shared across the GDN /
DeltaNet / Mamba2 cells, so this one run is the floor for all three:

    "Any retention we measure has to exceed what this configuration explains."
    -- proposal, Table 1

Two cohorts, both behavioural (generation + task metric, not a lens readout):

  * RULER NIAH  n=60  (simonjegou/ruler, config 16384) -- the headline floor number
    is accuracy on the EVICTED subset: needles past the 8064 window are simply not in
    context for this model, so evicted accuracy near zero is what makes AHN's evicted
    numbers attributable to AHN.
  * LongBench-E HotpotQA  n=60  -- length-stratified, same cohort construction as
    notebook 03. Floor number is mean first-line F1 with no memory pathway at all.

Unattended:  CUDA_VISIBLE_DEVICES=MIG-xxxx python no_ahn_floor.py
Output:      results/run_3b_floor/06_no_ahn_floor.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import ahn_interp as ai


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
def load_floor_model(cfg: dict, force_window: bool = True):
    """Stock Qwen2.5-3B-Instruct, SWA forced on for EVERY layer, no AHN, no sinks.

    Stock config ships use_sliding_window=false and max_window_layers=70 (> 36 layers),
    so all three fields have to be overridden or the window never actually applies.

    force_window=False is the control: stock config untouched (full ~32K context). Used
    to tell "the 8064 clip legitimately kills retrieval" apart from "the load path is
    broken" -- if in-window RULER accuracy is low in BOTH, the prompt/format is wrong.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    mp = cfg["model_path"]
    hf_cfg = AutoConfig.from_pretrained(mp)
    if force_window:
        hf_cfg.use_sliding_window = True
        hf_cfg.sliding_window = int(cfg["sliding_window"])
        hf_cfg.max_window_layers = int(cfg.get("max_window_layers", 0))

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    attn = cfg.get("attn_impl", "flash_attention_2")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            mp, config=hf_cfg, torch_dtype=dtype,
            attn_implementation=attn, device_map="auto",
        )
    except (ImportError, ValueError) as e:
        print(f"! {attn} unavailable ({e}); falling back to sdpa")
        attn = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            mp, config=hf_cfg, torch_dtype=dtype,
            attn_implementation=attn, device_map="auto",
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(mp)

    n_layers = model.config.num_hidden_layers
    print(f"loaded {mp}")
    print(f"  attn_impl           : {attn}")
    print(f"  n_layers            : {n_layers}")
    print(f"  use_sliding_window  : {getattr(model.config, 'use_sliding_window', None)}")
    print(f"  sliding_window      : {getattr(model.config, 'sliding_window', None)}")
    print(f"  max_window_layers   : {getattr(model.config, 'max_window_layers', None)}")
    if force_window:
        win_ok = (
            getattr(model.config, "use_sliding_window", False)
            and int(getattr(model.config, "sliding_window", 0)) == int(cfg["sliding_window"])
            and int(getattr(model.config, "max_window_layers", n_layers)) == 0
        )
        if not win_ok:
            raise RuntimeError(
                "sliding window is NOT active on every layer -- the floor would not evict. "
                "Check the transformers version honours use_sliding_window + max_window_layers=0."
            )
        print("  -> SWA active on all layers, no AHN, no attention sinks")
    else:
        print("  -> CONTROL: stock config, no window override (full context)")
    return model, tok, attn


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int):
    ins = tok(prompt, return_tensors="pt").to(model.device)
    n_in = int(ins["input_ids"].shape[1])
    out = model.generate(
        **ins, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    return text, n_in


def first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


# --------------------------------------------------------------------------------------
# RULER NIAH
# --------------------------------------------------------------------------------------
def run_ruler(model, tok, sliding_window: int, n: int, ruler_config: str, seed: int):
    ex = ai.load_ruler(config=ruler_config, split="test", n=n, seed=seed)
    print(f"\nRULER NIAH: {len(ex)} examples, config={ruler_config}")
    rows, t0 = [], time.time()
    for i, e in enumerate(ex):
        prompt, gold = e["prompt"], e["answer"]
        hit = prompt.find(gold)
        if hit < 0:
            print(f"  ! example {i}: gold not found in prompt, skipping")
            continue
        needle_pos = len(tok(prompt[:hit], add_special_tokens=True)["input_ids"])
        try:
            pred, n_in = generate(model, tok, prompt, e.get("max_new_tokens", 32))
        except torch.cuda.OutOfMemoryError as err:
            ai.free_cuda()
            print(f"  ! OOM at example {i}, skipping: {err}")
            continue
        window_start = n_in - sliding_window
        evicted = needle_pos < window_start  # no attention sinks in the floor
        gold_n, pred_n = ai.normalize_answer(gold), ai.normalize_answer(pred)
        rows.append({
            "idx": i,
            "task": e.get("task", "niah"),
            "n_tokens": n_in,
            "needle_pos": needle_pos,
            "compression_boundary": window_start,
            "eviction_distance": window_start - needle_pos,
            "needle_is_evicted": bool(evicted),
            "gold": gold,
            "prediction": pred,
            "substring_match": float(gold_n in pred_n) if gold_n else 0.0,
            "exact_match": ai.exact_match(pred, gold),
            "f1": ai.qa_f1_score(pred, gold),
            "f1_first_line": ai.qa_f1_score(first_line(pred), gold),
        })
        r = rows[-1]
        tag = "EVICT" if r["needle_is_evicted"] else "inwin"
        print(f"  [{i+1:2d}/{len(ex)}] {tag} nt={r['n_tokens']:6d} "
              f"evd={r['eviction_distance']:6d} hit={r['substring_match']:.0f} "
              f"gold={r['gold']!r:16.16} pred={r['prediction']!r:.40}")
        if (i + 1) % 10 == 0:
            acc = np.mean([rr["substring_match"] for rr in rows])
            print(f"  -- [{i+1}/{len(ex)}] running substr-acc={acc:.3f}  ({(time.time()-t0)/60:.1f} min)")
        ai.free_cuda()

    ev = [r for r in rows if r["needle_is_evicted"]]
    iw = [r for r in rows if not r["needle_is_evicted"]]

    def _agg(subset):
        if not subset:
            return None
        return {
            "n": len(subset),
            "substring_match": float(np.mean([r["substring_match"] for r in subset])),
            "exact_match": float(np.mean([r["exact_match"] for r in subset])),
            "f1": float(np.mean([r["f1"] for r in subset])),
            "f1_first_line": float(np.mean([r["f1_first_line"] for r in subset])),
        }

    by_task = {}
    for t in sorted({r["task"] for r in rows}):
        by_task[t] = _agg([r for r in rows if r["task"] == t])

    summary = {
        "n_scored": len(rows),
        "n_evicted": len(ev),
        "n_in_window": len(iw),
        "all": _agg(rows),
        "evicted": _agg(ev),
        "in_window": _agg(iw),
        "by_task": by_task,
        "ruler_config": ruler_config,
        "wall_min": (time.time() - t0) / 60,
    }
    print(f"RULER done: {len(rows)} scored ({len(ev)} evicted) in {summary['wall_min']:.1f} min")
    print("  task mix:", {t: v["n"] for t, v in by_task.items()})
    if summary["evicted"]:
        print(f"  EVICTED substring-acc = {summary['evicted']['substring_match']:.3f}  "
              f"(this is the floor number for RQ2)")
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------------------
# LongBench-E HotpotQA  (cohort construction mirrors notebook 03 cell 4)
# --------------------------------------------------------------------------------------
def run_hotpot(model, tok, sliding_window: int, n: int, max_input: int, seed: int,
               cohort_gate_extra: int = 128):
    """cohort_gate_extra mirrors notebook 03's num_attn_sinks (128) in the eligibility
    threshold ONLY, so the 60 HotpotQA examples line up 1:1 with the AHN runs. The floor
    model itself has no attention sinks."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", "hotpotqa_e", split="test")
    tmpl = (
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    )
    eligible = []
    for e in ds:
        p = tmpl.format(context=e["context"], input=e["input"])
        nt = len(tok.encode(p))
        # keep only prompts long enough that the answer *could* be evicted past the
        # window -- the same eligibility gate notebook 03 uses so the cohorts line up
        if nt > max_input or nt <= sliding_window + cohort_gate_extra:
            continue
        eligible.append({"prompt": p, "answers": e["answers"], "n_tokens": nt,
                         "length_bucket": e.get("length"), "_id": e.get("_id", len(eligible))})

    eligible.sort(key=lambda r: r["n_tokens"])
    third = max(1, len(eligible) // 3)
    buckets = {"short": eligible[:third], "mid": eligible[third:2 * third], "long": eligible[2 * third:]}
    rng = np.random.default_rng(seed)
    cohort, per = [], max(1, n // 3)
    for name, b in buckets.items():
        idx = rng.choice(len(b), size=min(per, len(b)), replace=False)
        for j in idx:
            r = dict(b[int(j)]); r["stratum"] = name; cohort.append(r)
    print(f"\nHotpotQA: eligible {len(eligible)} -> cohort {len(cohort)} "
          f"{ {k: sum(1 for c in cohort if c['stratum'] == k) for k in buckets} }")

    rows, t0 = [], time.time()
    for i, c in enumerate(cohort):
        try:
            pred, n_in = generate(model, tok, c["prompt"], 32)
        except torch.cuda.OutOfMemoryError as err:
            ai.free_cuda()
            print(f"  ! OOM at example {i}, skipping: {err}")
            continue
        fl = first_line(pred)
        rows.append({
            "id": c["_id"], "stratum": c["stratum"], "n_tokens": n_in,
            "prediction": pred, "prediction_first_line": fl, "gold": c["answers"],
            "f1": max(ai.qa_f1_score(pred, g) for g in c["answers"]),
            "f1_first_line": max(ai.qa_f1_score(fl, g) for g in c["answers"]),
            "exact_match_first_line": max(ai.exact_match(fl, g) for g in c["answers"]),
        })
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(cohort)}] F1_fl={np.mean([r['f1_first_line'] for r in rows]):.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)")
        ai.free_cuda()

    def _by(key):
        return {s: float(np.mean([r[key] for r in rows if r["stratum"] == s]))
                for s in ("short", "mid", "long") if any(r["stratum"] == s for r in rows)}

    summary = {
        "n_scored": len(rows),
        "f1": float(np.mean([r["f1"] for r in rows])) if rows else None,
        "f1_first_line": float(np.mean([r["f1_first_line"] for r in rows])) if rows else None,
        "exact_match_first_line": float(np.mean([r["exact_match_first_line"] for r in rows])) if rows else None,
        "f1_first_line_by_stratum": _by("f1_first_line"),
        "wall_min": (time.time() - t0) / 60,
    }
    f1fl = summary["f1_first_line"]
    print(f"HotpotQA done: {len(rows)} scored in {summary['wall_min']:.1f} min  "
          f"mean first-line F1 = {f1fl:.3f}" if f1fl is not None else
          f"HotpotQA done: {len(rows)} scored -- no rows to score")
    return {"rows": rows, "summary": summary}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_3b_floor.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--ruler-config", default="16384")
    ap.add_argument("--max-input", type=int, default=32000)
    ap.add_argument("--skip-ruler", action="store_true")
    ap.add_argument("--skip-hotpot", action="store_true")
    ap.add_argument("--no-window", action="store_true",
                    help="control: do not force SWA, use stock full context. Writes "
                         "06_no_ahn_floor_nowindow.json. Compare its in-window RULER "
                         "accuracy to the windowed run to check the load path.")
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    cfg = ai.load_run_config(args.config)
    seed = int(cfg.get("seed", ai.SEED))
    ai.set_seed(seed)
    results_dir = os.path.join("results", cfg["run_name"])
    os.makedirs(results_dir, exist_ok=True)
    ai.set_results_dir(results_dir)
    out_name = "06_no_ahn_floor_nowindow.json" if args.no_window else "06_no_ahn_floor.json"
    out_path = os.path.join(results_dir, out_name)

    if os.path.exists(out_path) and not args.allow_overwrite:
        raise SystemExit(
            f"{out_path} already exists. Pass --allow-overwrite for a deliberate rerun."
        )

    print(json.dumps(cfg, indent=2))
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- set CUDA_VISIBLE_DEVICES to a MIG slice")

    model, tok, attn = load_floor_model(cfg, force_window=not args.no_window)
    sw = int(cfg["sliding_window"])

    payload = {
        "config": cfg,
        "attn_impl_used": attn,
        "seed": seed,
        "window_forced": not args.no_window,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Table 1 Primary no-AHN floor (r54). Stock Qwen2.5-3B-Instruct, SWA at "
                "8064 on every layer, no AHN, no attention sinks. RULER evicted "
                "substring accuracy is the number RQ2 must exceed.",
    }
    if not args.skip_ruler:
        payload["ruler_niah"] = run_ruler(model, tok, sw, args.n, args.ruler_config, seed)
        ai.save_json(payload, out_name)  # checkpoint after the long cohort
    if not args.skip_hotpot:
        payload["hotpot_qa"] = run_hotpot(model, tok, sw, args.n, args.max_input, seed)

    ai.save_json(payload, out_name)
    print(f"\nsaved -> {out_path}")

    r = payload.get("ruler_niah", {}).get("summary", {})
    h = payload.get("hotpot_qa", {}).get("summary", {})
    print("\n=== FLOOR SUMMARY ===")
    if r.get("evicted"):
        print(f"RULER NIAH evicted   : substr-acc {r['evicted']['substring_match']:.3f}  "
              f"F1 {r['evicted']['f1']:.3f}  (n={r['evicted']['n']})")
    if r.get("in_window"):
        print(f"RULER NIAH in-window : substr-acc {r['in_window']['substring_match']:.3f}  "
              f"F1 {r['in_window']['f1']:.3f}  (n={r['in_window']['n']})")
    if h and h.get("f1_first_line") is not None:
        print(f"HotpotQA first-line  : F1 {h['f1_first_line']:.3f}  "
              f"EM {h['exact_match_first_line']:.3f}  (n={h['n_scored']})")


if __name__ == "__main__":
    main()
