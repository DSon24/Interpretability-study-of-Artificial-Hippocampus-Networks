#!/usr/bin/env python
"""Table 3 check 1 -- final-layer identity -- for the J-lens (row 73).

Both 02_table3_jlens_validation*.json record `final_layer_identity` as skipped, because
layer 35 (= n_layers - 1) is not in the fitted `source_layers` [9, 18, 27]. The check is
the cheapest decisive test of the instrument: at the final layer the Jacobian should be
~identity, so transporting the last residual through J35 and decoding (final norm ->
lm_head) must reproduce the model's own next-token distribution.

  PASS (KL < 0.01)  the transport + final-norm + unembed composition is sound, and
                    checks 2/3 failing is a depth-specific property of the lens.
  FAIL              the composition is wrong and every downstream readout -- the whole
                    RULER control battery -- inherits the error.

Rather than refit the 3-layer lens with a 4th layer (~0.6 GPU-h/layer, and a mid-fit
box shutdown loses everything), this fits layer 35 ALONE on a small wikitext corpus,
runs the check, and appends J35 into the existing .pt maps so it is not lost.

    CUDA_VISIBLE_DEVICES=MIG-xxxx python fit_layer35.py                # n=300, ~15 min
    CUDA_VISIBLE_DEVICES=MIG-xxxx python fit_layer35.py --n-prompts 500 --merge-into \\
        results/run_3b_gdn/jlens_qwen25_3b.pt results/run_3b_gdn/jlens_qwen25_3b_1000ctx.pt

Output: results/run_3b_gdn/02_final_layer_identity.json  (+ updates the
final_layer_identity block of any --table3-json files, + merges J35 into --merge-into .pt).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

import ahn_interp as ai

BACKBONE = "Qwen/Qwen2.5-3B-Instruct"
# skip=0 matches corpus A (the map of record); wikitext-103-raw is disjoint from
# RULER / LongBench / LV-Eval, same as notebook 02 cell 6.
CORPUS_SKIP = 0


@torch.no_grad()
def resid_at(model, tok, prompt: str, layer: int, pos: int = -1) -> torch.Tensor:
    store = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    try:
        model(**tok(prompt, return_tensors="pt").to(model.device))
    finally:
        h.remove()
    return store["h"][0, pos].float()


@torch.no_grad()
def true_logits(model, tok, prompt: str) -> torch.Tensor:
    return model(**tok(prompt, return_tensors="pt").to(model.device)).logits[0, -1].float()


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    P = torch.softmax(p_logits, -1)
    Q = torch.softmax(q_logits, -1)
    return float((P * (P.clamp_min(1e-12).log() - Q.clamp_min(1e-12).log())).sum())


def build_corpus(n: int, skip: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        t = ex["text"].strip()
        if len(t) > 400:
            out.append(t[:1500])
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=300,
                    help="fit corpus size for the single-layer fit. 300 ~15 min; bump to "
                         "500 if the KL comes back borderline (0.005-0.02).")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--skip-first", type=int, default=4)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=8, help="held-out prompts for the KL check")
    ap.add_argument("--results-dir", default="results/run_3b_gdn")
    ap.add_argument("--merge-into", nargs="*", default=[
        "results/run_3b_gdn/jlens_qwen25_3b.pt",
        "results/run_3b_gdn/jlens_qwen25_3b_1000ctx.pt",
    ], help="existing lens .pt files to append J35 into (backed up to <path>.bak first). "
            "Missing paths are skipped with a warning.")
    ap.add_argument("--table3-json", nargs="*", default=[
        "results/run_3b_gdn/02_table3_jlens_validation.json",
        "results/run_3b_gdn/02_table3_jlens_validation_1000ctx.json",
    ], help="Table 3 JSONs whose final_layer_identity block to fill in.")
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    ai.set_seed()
    os.makedirs(args.results_dir, exist_ok=True)
    ai.set_results_dir(args.results_dir)
    out_name = "02_final_layer_identity.json"
    out_path = os.path.join(args.results_dir, out_name)
    if os.path.exists(out_path) and not args.allow_overwrite:
        raise SystemExit(f"{out_path} exists. Pass --allow-overwrite for a rerun.")

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- set CUDA_VISIBLE_DEVICES to a MIG slice")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    n_layers = model.config.num_hidden_layers
    final_layer = n_layers - 1
    print(f"backbone loaded ({time.time()-t0:.0f}s), n_layers={n_layers}, "
          f"final_layer={final_layer}")

    base = ai.ModelBundle(
        model=model, tokenizer=tok, n_layers=n_layers, hidden=model.config.hidden_size,
        vocab=model.lm_head.weight.shape[0],
        num_heads=model.config.num_attention_heads,
        head_dim=getattr(model.config, "head_dim",
                         model.config.hidden_size // model.config.num_attention_heads),
        sliding_window=None, num_attn_sinks=0, use_ahn_router=False, use_q_proj=False,
        use_normalized_l2=False, ahn_layers=[], ahn_impl="none", model_path=BACKBONE,
    )

    corpus = build_corpus(args.n_prompts + args.n_eval, skip=CORPUS_SKIP)
    fit_prompts, eval_prompts = corpus[:args.n_prompts], corpus[args.n_prompts:]
    print(f"corpus: {len(fit_prompts)} fit / {len(eval_prompts)} eval "
          f"(wikitext-103-raw, skip={CORPUS_SKIP})")

    # --- reference: the no-lens decode path (lens=None) should already reproduce the
    # model logits (KL ~ 0). If THIS is not ~0 the problem is norm/unembed, not the fit.
    ref = [kl(true_logits(model, tok, p),
             ai.readout_logits(resid_at(model, tok, p, final_layer), base, lens=None))
           for p in eval_prompts]
    print(f"\nreference (lens=None) KL: mean {np.mean(ref):.5f}  max {np.max(ref):.5f}  "
          f"(should be ~0; this is the norm+unembed path with no transport)")

    # --- fit layer 35 alone -------------------------------------------------------
    t0 = time.time()
    lens35 = ai.fit_jacobian_lens(
        base, fit_prompts, [final_layer],
        max_seq_len=args.max_seq_len, skip_first=args.skip_first, dim_batch=args.dim_batch,
    )
    fit_h = (time.time() - t0) / 3600
    lens35.meta["fit_gpu_hours"] = fit_h
    lens35.meta["single_layer_fit"] = final_layer
    lens35.meta["corpus"] = f"wikitext-103-raw skip={CORPUS_SKIP} n={args.n_prompts}"
    print(f"fit layer {final_layer} in {fit_h*60:.1f} min")
    lens35.save(os.path.join(args.results_dir, "jlens_qwen25_3b_layer35.pt"))

    # --- the check: J35 transport should be ~identity ---------------------------------
    # also test resid from layer 34 -> J35, to catch the jlens vs forward-hook off-by-one
    # convention flagged in notebook 02 cell 25 note 4.
    per_prompt = []
    for p in eval_prompts:
        tl = true_logits(model, tok, p)
        row = {"kl_L35_resid": kl(tl, ai.readout_logits(
            resid_at(model, tok, p, final_layer), base, lens=lens35, layer=final_layer))}
        if final_layer - 1 >= 0:
            row["kl_L34_resid"] = kl(tl, ai.readout_logits(
                resid_at(model, tok, p, final_layer - 1), base, lens=lens35, layer=final_layer))
        # top-5 agreement on the L35-resid path
        j = ai.readout_logits(resid_at(model, tok, p, final_layer), base,
                              lens=lens35, layer=final_layer)
        t5 = set(int(i) for i in tl.topk(5).indices)
        j5 = set(int(i) for i in j.topk(5).indices)
        row["top5_overlap"] = len(t5 & j5) / 5.0
        per_prompt.append(row)

    mean_kl = float(np.mean([r["kl_L35_resid"] for r in per_prompt]))
    max_kl = float(np.max([r["kl_L35_resid"] for r in per_prompt]))
    mean_kl_34 = (float(np.mean([r["kl_L34_resid"] for r in per_prompt]))
                  if "kl_L34_resid" in per_prompt[0] else None)
    mean_top5 = float(np.mean([r["top5_overlap"] for r in per_prompt]))
    passed = mean_kl < 0.01

    print("\n=== FINAL-LAYER IDENTITY ===")
    print(f"  J35 @ L35-resid : mean KL {mean_kl:.5f}   max {max_kl:.5f}   "
          f"top-5 overlap {mean_top5:.2f}")
    if mean_kl_34 is not None:
        print(f"  J35 @ L34-resid : mean KL {mean_kl_34:.5f}   "
              f"(if this is much lower, jlens indexes one block earlier than the hook)")
    print(f"  reference       : mean KL {np.mean(ref):.5f}")
    verdict = ("PASS (KL < 0.01)" if passed else
               "FAIL -- composition suspect; refit at n=500 before concluding, then tell Gautam")
    print(f"  -> {verdict}")

    identity_block = {
        "kl": mean_kl,
        "kl_max": max_kl,
        "kl_L34_resid": mean_kl_34,
        "kl_reference_no_lens": float(np.mean(ref)),
        "top5_overlap": mean_top5,
        "passed": bool(passed),
        "layer": final_layer,
        "n_eval": len(eval_prompts),
        "per_prompt": per_prompt,
        "fit": {"n_prompts": args.n_prompts, "max_seq_len": args.max_seq_len,
                "skip_first": args.skip_first, "corpus": lens35.meta["corpus"],
                "fit_gpu_hours": fit_h},
        "note": "Layer 35 fitted alone (not part of the 3-layer map of record). The "
                "identity check is a property of the transport+norm+unembed composition, "
                "not of the averaging corpus, so this result applies to both the 500- and "
                "1000-context maps.",
    }

    payload = {
        "backbone": BACKBONE,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_layer_identity": identity_block,
    }
    ai.save_json(payload, out_name)
    print(f"\nsaved -> {out_path}")

    # --- append J35 into the existing maps ------------------------------------------
    for pt_path in args.merge_into:
        real = pt_path
        if not os.path.exists(real):
            try:
                real = ai.resolve_lens_path(pt_path)
            except FileNotFoundError:
                print(f"  ! {pt_path}: not found, skipping merge")
                continue
        lens = ai.JacobianLens.load(real, map_location="cpu")
        if final_layer in lens.jacobians and not args.allow_overwrite:
            print(f"  {real}: already has layer {final_layer}, leaving it "
                  f"(pass --allow-overwrite to replace)")
            continue
        shutil.copy2(real, real + ".bak")
        lens.jacobians[final_layer] = lens35.jacobians[final_layer].cpu()
        lens.meta.setdefault("appended_layers", {})[str(final_layer)] = lens35.meta["corpus"]
        lens.save(real)
        print(f"  {real}: appended J{final_layer}  (backup at {real}.bak)  "
              f"layers now {sorted(lens.jacobians)}")

    # --- fill in the Table 3 JSON blocks ------------------------------------------
    for j_path in args.table3_json:
        if not os.path.exists(j_path):
            print(f"  ! {j_path}: not found, skipping")
            continue
        with open(j_path) as f:
            doc = json.load(f)
        doc["final_layer_identity"] = identity_block
        with open(j_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  {j_path}: final_layer_identity updated (passed={passed}). "
              f"TABLE_3_PASSED still gated on checks 2/3 + map_stability -- unchanged here.")

    print("\ndone.")


if __name__ == "__main__":
    main()
