"""CPU-only: is the RULER needle actually evicted, or is it sitting in the local window?

Run 024 put layer 27 at median rank 31,656 on the RULER cohort -- better than chance, and
better than the homemade in-window C4 ceiling (90,902). A supposedly-evicted read beating
the pre-eviction ceiling has one obvious alternative explanation: the needle was never
evicted. RULER varies needle depth, and at ~15.7K tokens against an 8064 window roughly
half the context is inside the window and directly visible to attention.

`measure_ruler` records n_tokens but not needle position, so the rows cannot answer this.
This script locates the gold answer inside each prompt, converts that to a token index,
and applies the same eviction test `build_niah_prompt` uses:

    window_start = n_tokens - sliding_window
    evicted      = num_attn_sinks <= needle_pos < window_start

It then splits the saved layer-27 ranks by that verdict. If the below-chance result is
carried by in-window examples, it is an attention artefact and not evidence about AHN. If
it survives on the evicted subset, candidate (b) stands.

No GPU: tokenizer only.

    python ruler_needle_position.py
"""

from __future__ import annotations

import json
import os
import statistics as st
from typing import Dict, List

import ahn_interp as ai

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
ROWS_PATH = os.path.join(RESULTS_DIR, "04g_ruler_niah_rows.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04h_ruler_needle_position.json")

CHANCE = 75968
# Config of record. Deliberately not read from the model -- this script never loads one.
SLIDING_WINDOW = 8064
NUM_ATTN_SINKS = 128


def main() -> None:
    from transformers import AutoTokenizer

    with open(ROWS_PATH) as f:
        blob = json.load(f)
    rows = blob["rows"]
    ruler_cfg = blob["ruler_cfg"]
    layers = sorted({int(r["layer"]) for r in rows})

    # measure_ruler appends one row per layer per example, in example order, and run 024
    # skipped nothing. Assert that before relying on positional joining.
    n_examples = len(rows) // len(layers)
    assert len(rows) == n_examples * len(layers), (
        f"{len(rows)} rows does not divide by {len(layers)} layers -- some examples were "
        "skipped, so rows cannot be joined to examples by position. Re-run with an "
        "explicit example index in the row schema."
    )
    for i in range(n_examples):
        got = [int(r["layer"]) for r in rows[i * len(layers):(i + 1) * len(layers)]]
        assert got == layers, f"example {i} has layer order {got}, expected {layers}"

    tok = AutoTokenizer.from_pretrained(
        ai.resolve_ckpt("Qwen-2.5-Instruct-3B-AHN-GDN")
    )
    examples = ai.load_ruler(**ruler_cfg)
    assert len(examples) == n_examples, (
        f"loaded {len(examples)} examples but rows imply {n_examples}; "
        "ruler_cfg does not reproduce the run"
    )

    per_example: List[dict] = []
    for idx, ex in enumerate(examples):
        prompt, answer = ex["prompt"], ex["answer"]
        n_tokens = len(tok(prompt)["input_ids"])
        window_start = n_tokens - SLIDING_WINDOW

        hit = prompt.find(answer)
        occurrences = prompt.count(answer)
        if hit < 0:
            per_example.append({
                "example": idx, "n_tokens": n_tokens, "window_start": window_start,
                "answer_found": False, "occurrences": 0, "verdict": "answer_not_in_prompt",
            })
            continue

        needle_pos = len(tok(prompt[:hit], add_special_tokens=True)["input_ids"])
        evicted = NUM_ATTN_SINKS <= needle_pos < window_start
        in_sinks = needle_pos < NUM_ATTN_SINKS
        per_example.append({
            "example": idx,
            "n_tokens": n_tokens,
            "window_start": window_start,
            "needle_pos": needle_pos,
            "eviction_distance": window_start - needle_pos,
            "answer_found": True,
            "occurrences": occurrences,
            "verdict": "in_sink_region" if in_sinks else ("evicted" if evicted else "in_window"),
        })

    counts: Dict[str, int] = {}
    for rec in per_example:
        counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1

    print(f"RULER cohort needle placement — config={ruler_cfg}, n={len(per_example)}")
    print(f"window={SLIDING_WINDOW}, sinks={NUM_ATTN_SINKS}\n")
    for verdict, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {verdict:20s} {n:3d}  ({n / len(per_example):5.1%})")

    multi = sum(r.get("occurrences", 0) > 1 for r in per_example)
    if multi:
        print(f"\n  ! {multi} prompt(s) contain the answer string more than once -- the "
              "first occurrence was used, which may not be the needle")

    # split the saved ranks by placement verdict
    print(f"\nlayer-27 rank by placement (D-resid basis, chance {CHANCE}):")
    split: Dict[str, List[int]] = {}
    for rec in per_example:
        row = rows[rec["example"] * len(layers) + layers.index(27)]
        split.setdefault(rec["verdict"], []).append(int(row["rank_c1_residual"]))
    for verdict, xs in sorted(split.items(), key=lambda kv: -len(kv[1])):
        beat = sum(x < CHANCE for x in xs) / len(xs)
        print(f"  {verdict:20s} n={len(xs):3d}  median={st.median(xs):8.0f}  "
              f"beating chance {beat:5.1%}")

    out = {
        "ruler_cfg": ruler_cfg,
        "sliding_window": SLIDING_WINDOW,
        "num_attn_sinks": NUM_ATTN_SINKS,
        "chance_rank": CHANCE,
        "counts": counts,
        "layer27_rank_by_verdict": {
            k: {"n": len(v), "median_rank": st.median(v),
                "frac_beating_chance": sum(x < CHANCE for x in v) / len(v)}
            for k, v in split.items()
        },
        "per_example": per_example,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_PATH}")

    if counts.get("in_window", 0) > len(per_example) * 0.2:
        print(
            "\n! A material share of needles were never evicted. The layer-27 result "
            "cannot be read as evidence about AHN memory until it is shown to hold on "
            "the evicted subset alone."
        )


if __name__ == "__main__":
    main()
