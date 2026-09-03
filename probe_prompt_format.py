#!/usr/bin/env python3
"""
The answer-prefix half of the construction ladder — runs on a laptop, no AHN, no GPU.

WHY THIS SPLITS OFF FROM probe_construction.py
----------------------------------------------
probe_construction.py needs the merged Qwen2.5-3B + AHN-GDN checkpoint and CUDA kernels
(flash-linear-attention, FlashAttention-2). But its single sharpest question does not
involve AHN at all:

    At the position where the readout is taken, is the needle even a plausible
    next token?

`build_niah_prompt` ends the prompt at "What was the special word?" and every readout is
taken at pos=-1. The natural continuation after "?" is " The" or a newline -- the needle
sits several tokens away. The known-fact probe that works ("The capital of France is")
puts the target in exactly the next-token slot. And `ahn_interp.load_ruler` appends
RULER's own `answer_prefix`, so the two code paths in that module already disagree.

That is a property of the PROMPT and the BACKBONE, not of compressive memory. It is
testable on the base model, on a laptop, with no lens and no AHN. If the needle is
unreadable here, no J-Lens on an intermediate layer of an AHN-merged model can rescue it,
and the C1/C2/C3 failures are explained without reference to AHN at all.

Runs on MPS or CPU. Prompts are kept short so no sliding window is involved.

    python probe_prompt_format.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT = "results/run_3b_gdn/06_prompt_format_probe.json"
NEEDLE = "Paris"
QUESTION = "What was the special word?"
PREFIX = " The special word is"
FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")


def variants(needle):
    """(name, prompt, has_answer_prefix) -- one changed thing at a time."""
    stem = f"The special word is {needle}. "
    pad = FILLER * 12
    return [
        ("A_known_fact",              "The capital of France is", True),
        ("B_niah_short_no_prefix",    stem + "The meeting continued. " + QUESTION, False),
        ("C_niah_short_with_prefix",  stem + "The meeting continued. " + QUESTION + PREFIX, True),
        ("D_niah_padded_no_prefix",   stem + pad + QUESTION, False),
        ("E_niah_padded_with_prefix", stem + pad + QUESTION + PREFIX, True),
        ("F_ruler_style_prefix",      stem + pad + QUESTION + " Answer:", True),
        ("G_no_question_bare_stem",   stem + pad + "The special word is", True),
    ]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        dtype = torch.bfloat16 if dev == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(dev)
    except Exception:
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(dev)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    ids = tok.encode(f" {NEEDLE}", add_special_tokens=False)
    assert len(ids) == 1, f"' {NEEDLE}' is {len(ids)} tokens; pick a single-token needle"
    tid = ids[0]
    vocab = model.lm_head.weight.shape[0]
    print(f"device={dev} dtype={dtype} vocab={vocab}  ' {NEEDLE}'=[{tid}]  chance rank={vocab//2}\n")

    rows = []
    for name, prompt, has_prefix in variants(NEEDLE):
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1].float().cpu()
        order = logits.argsort(descending=True)
        rank = int((order == tid).nonzero()[0].item())
        prob = float(torch.softmax(logits, -1)[tid])
        top5 = [tok.decode([i]) for i in order[:5].tolist()]
        rows.append({
            "variant": name, "has_answer_prefix": has_prefix,
            "n_tokens": int(enc["input_ids"].shape[1]),
            "last_tokens": tok.decode(enc["input_ids"][0, -6:].tolist()),
            "rank_needle": rank, "prob_needle": prob, "top5": top5,
        })
        print(f"{name:26s} prefix={str(has_prefix):5s} n={rows[-1]['n_tokens']:5d}  "
              f"rank={rank:7d}  p={prob:.3e}  top5={top5}")

    def r(n):
        return next(x["rank_needle"] for x in rows if x["variant"] == n)

    verdict = {
        "short_prefix_gain_B_to_C": r("B_niah_short_no_prefix") - r("C_niah_short_with_prefix"),
        "padded_prefix_gain_D_to_E": r("D_niah_padded_no_prefix") - r("E_niah_padded_with_prefix"),
    }
    print("\n=== effect of adding the answer prefix (positive = prefix helps) ===")
    print(f"  short  B -> C : {verdict['short_prefix_gain_B_to_C']:+d} ranks")
    print(f"  padded D -> E : {verdict['padded_prefix_gain_D_to_E']:+d} ranks")

    json.dump({
        "note": "Base Qwen2.5-3B-Instruct, no AHN, no J-lens. Tests whether the needle is a "
                "plausible next token at the position where 04's readout is taken (pos=-1).",
        "model": MODEL, "device": dev, "dtype": str(dtype), "vocab": vocab,
        "chance_rank": vocab / 2, "needle": NEEDLE, "needle_token_id": tid,
        "rows": rows, "verdict": verdict,
    }, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
