#!/usr/bin/env python3
"""
The construction ladder — the experiment that decides between candidate (b) and (c).

BACKGROUND
----------
Two results disagree and the project is blocked on which one is right:

  * On eight isolated known-fact prompts ("The capital of France is") the J-lens
    beats the plain logit lens by 8-204x and gets ` Paris` to rank 8 at layer 27.
  * On the full NIAH sweep the same lens is at chance (C1 mean rank 87,675 against a
    chance rank of 75,968), with no needle/distractor separation (C2) and shuffled
    context scoring BETTER than ordered (C3).

Candidate (a), an unconverged lens, is ruled out -- map stability passes at
0.91/0.87/0.89. What is left is:

  (b) the NIAH sweep's prompt construction differs from the known-fact test's in a
      way that makes the target unreadable, or
  (c) C3's result is real and AHN is closer to a recency mechanism than a content
      store -- which the pre-registration flags as the most publishable outcome here.

THE SPECIFIC SUSPICION
----------------------
`build_niah_prompt` ends the prompt at the question mark:

    <filler> "The special word is Paris. " <filler> "What was the special word?"

and `AHNProbe.Capture.o_t()` defaults to `pos=-1`. So the readout asks "what token
follows '?'" and then measures the rank of ` Paris`. The natural continuation there
is ` The` or a newline -- the needle is several tokens away, not next. The known-fact
prompt puts the target in exactly the next-token slot, which is why it works.

Two pieces of corroborating evidence already in the repo:

  * `ahn_interp.load_ruler` builds `context + question + answer_prefix`. RULER itself
    supplies an answer prefix; the hand-built NIAH prompt has none. The two code paths
    in our own module disagree.
  * Gate C in `01_instrumentation_gates.json` has the model answering
    'The special word in this text is "Here we go."' -- that is filler text from
    _FILLERS[0], not the needle. The model fails the task behaviourally, so asking
    whether the memory READOUT holds the needle is asking about information the model
    demonstrably is not using.

WHAT THIS SCRIPT DOES
---------------------
Walks a ladder of conditions from the one that works to the one that fails, changing
exactly one thing per rung, and reports where the signal dies.

  L0  bare known fact, no padding                      -- the condition that works
  L1  short NIAH, needle in-window, no answer prefix   -- swaps in the NIAH phrasing
  L2  short NIAH, needle in-window, WITH answer prefix -- adds the answer prefix
  L3  full-length, needle in-window, no answer prefix  -- adds 8k of filler (= C4)
  L4  full-length, needle in-window, WITH answer prefix
  L5  full-length, needle EVICTED, no answer prefix    -- the current C1 condition
  L6  full-length, needle EVICTED, WITH answer prefix
  L7  L6 with natural filler instead of the degenerate repeated filler

READING THE RESULT
------------------
  * Rank recovers at L2/L4/L6 -> the answer prefix was the defect. Candidate (b).
    Fix `build_niah_prompt`, re-run notebook 04, and every downstream null is void.
  * L4 is good but L6 is at chance -> construction is fine and the needle genuinely
    does not survive eviction. Candidate (c), the recency story, and it is now clean
    enough to be the paper's headline.
  * Nothing recovers anywhere, including L0 reproduced here -> the readout path itself
    is broken and neither (b) nor (c) is established.

Step 0 below is the cheapest check and runs first: the model's OWN final-layer logits
at the readout position. If the full model cannot rank the needle there, no lens on an
intermediate layer will, and the question is settled before any lens is involved.

PARTIALLY ANSWERED ALREADY -- run probe_prompt_format.py first, it needs no GPU.
On the base model the needle sits at rank 31-35 of 151,936 at the readout position even
WITHOUT an answer prefix, and at rank 0 with one. So the prompt format is not what puts
the C1 readout at rank 87,688: it costs about 30 ranks, not 87,000. This script's job is
now the remaining half -- what happens once the needle is actually evicted into AHN's
compressive memory and read through o_t rather than the full residual stream.

UNTESTED as written -- it has not been run on a GPU. Treat the first execution as a
debugging pass, exactly like every other notebook in this repo.

    python probe_construction.py --model-path ./merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN
"""

import argparse
import json
import os

import torch

import ahn_interp as ai

LAYERS = [9, 18, 27]
NEEDLE = "Paris"
DISTRACTOR = "London"
# " Answer:" beat " The special word is" on the base model (rank 0, p=0.959 vs rank 1,
# p=0.046) in probe_prompt_format.py, and it is the format RULER itself uses via
# load_ruler's answer_prefix. See results/run_3b_gdn/06_prompt_format_probe.json.
ANSWER_PREFIX = " Answer:"
NATURAL_FILLER = (
    "The committee reconvened on Thursday to review the quarterly figures, and "
    "several members raised concerns about the revised timetable for the coastal "
    "survey. Rainfall through the spring had been unusually heavy, which delayed "
    "the fieldwork by nearly a month. "
)


def build_ladder(bundle, tok):
    """Return [(name, prompt, expects_ahn)] -- one entry per rung."""
    sinks = bundle.num_attn_sinks
    window = bundle.sliding_window or 0
    rungs = []

    # L0 -- the condition that demonstrably works
    rungs.append(("L0_known_fact", "The capital of France is", False))

    # L1/L2 -- NIAH phrasing, short, needle trivially in-window
    short = f"The special word is {NEEDLE}. The meeting continued for some time. What was the special word?"
    rungs.append(("L1_short_no_prefix", short, False))
    rungs.append(("L2_short_with_prefix", short + ANSWER_PREFIX, False))

    # L3/L4 -- full length, needle in-window (never compressed, so AHN cannot be blamed)
    p_in = ai.build_niah_prompt(tok, NEEDLE, bundle, eviction_distance=0, in_window=True)
    rungs.append(("L3_full_inwindow_no_prefix", p_in["prompt"], True))
    rungs.append(("L4_full_inwindow_with_prefix", p_in["prompt"] + ANSWER_PREFIX, True))

    # L5/L6 -- full length, needle evicted: the actual C1 condition
    p_ev = ai.build_niah_prompt(tok, NEEDLE, bundle, eviction_distance=1024, in_window=False)
    rungs.append(("L5_full_evicted_no_prefix", p_ev["prompt"], True))
    rungs.append(("L6_full_evicted_with_prefix", p_ev["prompt"] + ANSWER_PREFIX, True))

    # L7 -- as L6 but with non-degenerate filler. _FILLERS repeats one 5-sentence
    # string thousands of times; shuffling that changes almost nothing, which may be
    # why C3 is uninformative rather than negative.
    n_rep = max(1, (window + 1024) // len(tok.encode(NATURAL_FILLER)))
    body = NATURAL_FILLER * (sinks // 20 + 2)
    tail = NATURAL_FILLER * n_rep
    rungs.append((
        "L7_full_evicted_natural_filler",
        body + f"The special word is {NEEDLE}. " + tail + "What was the special word?" + ANSWER_PREFIX,
        True,
    ))
    return rungs, p_in, p_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--lens-path", default="results/run_3b_gdn/jlens_qwen25_3b.pt")
    ap.add_argument("--results-dir", default="results/run_3b_gdn")
    ap.add_argument("--sliding-window", type=int, default=8064)
    ap.add_argument("--num-attn-sinks", type=int, default=128)
    # ahn_interp.load_ahn_model defaults to "flash_attention_2", which needs a source
    # build matching the system CUDA toolkit exactly. On at least one EC2 AMI
    # (Deep Learning OSS Nvidia Driver AMI, Ubuntu 26.04, driver 595.91.07) the system
    # nvcc reports CUDA 13.0 while the torch wheel needed for transformers==4.51.0
    # compatibility is cu124 -- torch.utils.cpp_extension refuses to build against the
    # mismatch. "sdpa" is torch's built-in fused attention, needs no compilation, and
    # is plenty for these prompt lengths (at most a few thousand tokens on a 3B model).
    ap.add_argument("--attn-implementation", default="sdpa",
                     choices=["sdpa", "eager", "flash_attention_2"])
    args = ap.parse_args()

    ai.set_seed()
    ai.set_results_dir(args.results_dir)

    bundle = ai.load_ahn_model(
        args.model_path,
        sliding_window=args.sliding_window,
        num_attn_sinks=args.num_attn_sinks,
        attn_implementation=args.attn_implementation,
    )
    tok = bundle.tokenizer

    ids = ai.single_token_needles(tok, [NEEDLE, DISTRACTOR])
    needle_id, distractor_id = ids[NEEDLE], ids[DISTRACTOR]
    chance = bundle.vocab / 2.0

    lens = None
    if os.path.exists(args.lens_path):
        lens = ai.JacobianLens.load(args.lens_path)
        print(f"J-lens loaded from {args.lens_path}: layers {sorted(lens.jacobians)}")
    else:
        print(f"!! no lens at {args.lens_path} -- falling back to the plain logit lens.")
        print("   Pull it with: huggingface-cli download gautam-dphs/ahn-interp-jlens-qwen25-3b")

    rungs, p_in, p_ev = build_ladder(bundle, tok)
    probe = ai.AHNProbe(bundle)
    out = []

    for name, prompt, expects_ahn in rungs:
        inputs = tok(prompt, return_tensors="pt").to(bundle.model.device)
        n_tok = int(inputs["input_ids"].shape[1])
        row = {
            "rung": name,
            "n_tokens": n_tok,
            "expects_ahn": expects_ahn,
            "ahn_active": n_tok > (bundle.sliding_window or 0) + bundle.num_attn_sinks,
        }

        cap = probe.run(inputs, layers=LAYERS, keep_logits=True)

        # -- step 0: the model's own final logits at the readout position. If the
        # needle is not findable here, nothing downstream can find it either.
        final = cap.logits[0, -1].float()
        row["model_rank_needle"] = ai.token_rank(final, needle_id)
        row["model_prob_needle"] = ai.token_prob(final, needle_id)
        row["model_rank_distractor"] = ai.token_rank(final, distractor_id)
        row["model_top5"] = [tok.decode([i]) for i in final.topk(5).indices.tolist()]

        # -- the lens readouts, per layer, on both views
        for L in LAYERS:
            try:
                resid = cap.residual(L, pos=-1)
                lg = ai.readout_logits(resid, bundle, lens=lens, layer=L)
                row[f"resid_rank_L{L}"] = ai.token_rank(lg, needle_id)
            except KeyError:
                row[f"resid_rank_L{L}"] = None
            if row["ahn_active"]:
                try:
                    o = cap.o_t(L, pos=-1)
                    lg = ai.readout_logits(o, bundle, lens=lens, layer=L)
                    row[f"o_t_rank_L{L}"] = ai.token_rank(lg, needle_id)
                    row[f"o_t_rank_distractor_L{L}"] = ai.token_rank(lg, distractor_id)
                except KeyError:
                    row[f"o_t_rank_L{L}"] = None
            else:
                row[f"o_t_rank_L{L}"] = None

        out.append(row)
        print(
            f"{name:34s} n={n_tok:6d}  ahn={row['ahn_active']!s:5s}  "
            f"model_rank={row['model_rank_needle']:7d}  "
            f"o_t@18={row.get('o_t_rank_L18')}  top5={row['model_top5']}"
        )
        ai.free_cuda()

    result = {
        "chance_rank": chance,
        "vocab": bundle.vocab,
        "needle": NEEDLE,
        "distractor": DISTRACTOR,
        "answer_prefix": ANSWER_PREFIX,
        "lens_used": lens is not None,
        "lens_validated": False,  # Table 3 checks 2 and 3 still fail
        "in_window_spec": {k: v for k, v in p_in.items() if k != "prompt"},
        "evicted_spec": {k: v for k, v in p_ev.items() if k != "prompt"},
        "rungs": out,
        "cfg": bundle.summary(),
    }
    path = ai.save_json(result, "06_construction_ladder.json")
    print(f"\nsaved -> {path}")
    print(f"chance rank = {chance:.0f}. A rung is 'alive' if the rank is far below it.")


if __name__ == "__main__":
    main()
