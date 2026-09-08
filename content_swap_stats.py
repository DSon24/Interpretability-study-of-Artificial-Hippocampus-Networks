"""CPU-only: does layer 27's C2 sign depend on needle CONTENT, or on something else?

Gautam, 8 Sep, asked two things that turned out to be one experiment: run the
length-matched control before rescoping RQ2, and explain why layer 27's sign appeared to
flip between digit needles (RULER, positive) and common-word needles (the homemade set,
negative).

RULER and the homemade cohort differ in three tangled ways -- construction, length, and
needle content -- and only the third had never been varied on its own. The sweeps behind
this script hold construction (`build_niah_prompt`) fixed and cross needle content
(7 single-token words x 8 seven-digit strings) with eviction distance
(64 / 512 / 2048 / 4096 / 8192). Distance 8192 gives 8064 + 128 + 8192 = 16,384 tokens
against RULER's 15,679, so the long end is length-matched by construction.

WHAT THIS FOUND, and it withdraws a claim made the day before:

1. No content effect. Word and digit needles track each other at essentially every
   layer x distance cell. The 7 Sep "layer 27's sign is needle-content-dependent" entry
   is NOT supported and is withdrawn.

2. Layer 27 is null in this construction -- every distance, both content types, and
   pooled (word 0.973x p=0.59, digit 0.995x p=0.76).

3. Son's homemade multi-control result (layer 27, pooled, 0.863x, p=0.016) does not
   replicate. His used 4 hand-picked needles; with 7 the pooled effect vanishes. This is
   the same lesson as the 4-5 Sep needle-category finding -- the original 4-needle sample
   was not representative of the needle population.

4. Layer 18 carries real but incoherent structure: 5 of 30 cells survive Holm, but the
   sign oscillates with distance (+1.43 at 64, -0.80 at 512, +1.19 at 4096, -0.80 at
   8192). Not a monotone decay, not obviously a mechanism. Flagged, unexplained.

Multiple comparisons are corrected here rather than left to the reader: 30 cells, Holm at
0.05. Nine cells clear an uncorrected 0.05 against ~1.5 expected by chance, so there is
real structure -- but only five survive correction, and reporting the uncorrected nine
would overstate it.

    python content_swap_stats.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
LONG_PATH = os.path.join(RESULTS_DIR, "04l_content_swap_rows.json")
SHORT_PATH = os.path.join(RESULTS_DIR, "04m_content_swap_short_rows.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04n_content_swap_stats.json")

SEED = 20260820
N_BOOT = 10000
EPS = 1e-30
DISTANCES = [64, 512, 2048, 4096, 8192]
LAYERS = [9, 18, 27]
CONTENTS = ["word", "digit"]

# Son's homemade multi-control result, for the replication check (04-C2-debug cells 105-110)
SON_LAYER27_POOLED = {"fold": 0.863, "p": 0.016, "n_needles": 4}


def per_target_deltas(df: pd.DataFrame, content: str, layer: int,
                      distance: Optional[int] = None) -> np.ndarray:
    """Multi-control corrected log effect, one value per target.

    Targets are the sampling unit: the claim is about needle content, so the bootstrap
    resamples needles, not (needle, filler) observations, which would treat three fillers
    of the same needle as independent evidence about that needle's content.
    """
    d = df[(df["content"] == content) & (df["layer"] == layer)]
    if distance is not None:
        d = d[d["requested_distance"] == distance]

    own = d[(d["is_target"]) & (d["stored"] == d["tested_target"])][
        ["tested_target", "requested_distance", "filler_idx", "p_target"]
    ].rename(columns={"p_target": "p_own"})

    ctrl = d[~d["is_target"]].copy()
    ctrl["log_p"] = np.log(ctrl["p_target"] + EPS)
    ctrl_mean = (ctrl.groupby(["tested_target", "requested_distance", "filler_idx"])["log_p"]
                     .mean().reset_index(name="log_p_ctrl"))

    merged = own.merge(ctrl_mean, on=["tested_target", "requested_distance", "filler_idx"])
    merged["delta_log"] = np.log(merged["p_own"] + EPS) - merged["log_p_ctrl"]
    return merged.groupby("tested_target")["delta_log"].mean().to_numpy()


def boot(xs: Sequence[float], seed: int = SEED, n_boot: int = N_BOOT):
    """Bootstrap over targets. Returns (fold, ci_low, ci_high, two-sided p)."""
    xs = np.asarray(xs)
    rng = np.random.default_rng(seed)
    reps = np.array([rng.choice(xs, len(xs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
    return (float(np.exp(xs.mean())), float(np.exp(lo)), float(np.exp(hi)),
            float(max(p, 1.0 / n_boot)))


def holm(cells: List[dict]) -> List[dict]:
    """Holm-Bonferroni over every cell tested, monotonicity enforced."""
    cells = sorted(cells, key=lambda c: c["p"])
    m = len(cells)
    for i, c in enumerate(cells):
        c["p_holm"] = min(1.0, c["p"] * (m - i))
    for i in range(1, m):
        cells[i]["p_holm"] = max(cells[i]["p_holm"], cells[i - 1]["p_holm"])
    return cells


def main() -> None:
    rows = (json.load(open(LONG_PATH))["rows"] + json.load(open(SHORT_PATH))["rows"])
    df = pd.DataFrame(rows)
    print(f"content swap — {len(df)} rows, distances {sorted(df['requested_distance'].unique())}")
    print(f"bootstrap over targets, {N_BOOT} resamples, seed {SEED}\n")

    cells: List[dict] = []
    for layer in LAYERS:
        for content in CONTENTS:
            for dist in DISTANCES:
                xs = per_target_deltas(df, content, layer, dist)
                if len(xs) < 3:
                    continue
                fold, lo, hi, p = boot(xs)
                cells.append({"layer": layer, "content": content, "distance": dist,
                              "n_targets": len(xs), "fold": fold,
                              "ci95": [lo, hi], "p": p})
    cells = holm(cells)

    for layer in LAYERS:
        print(f"=== layer {layer} ===")
        print(f"{'distance':>9} {'word':>30} {'digit':>30}")
        for dist in DISTANCES:
            cols = []
            for content in CONTENTS:
                c = next((x for x in cells if x["layer"] == layer
                          and x["content"] == content and x["distance"] == dist), None)
                if c is None:
                    cols.append(" " * 30); continue
                mark = "*" if c["p_holm"] < 0.05 else " "
                cols.append(f"{c['fold']:>7.3f}x [{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]{mark}")
            print(f"{dist:>9} {cols[0]:>30} {cols[1]:>30}")
        print()

    survivors = [c for c in cells if c["p_holm"] < 0.05]
    uncorrected = [c for c in cells if c["p"] < 0.05]
    print(f"* = survives Holm at 0.05 across all {len(cells)} cells")
    print(f"survivors: {len(survivors)}   uncorrected p<0.05: {len(uncorrected)} "
          f"(≈{0.05*len(cells):.1f} expected by chance)\n")

    pooled = {}
    print("--- pooled across distances, the aggregation Son used ---")
    for layer in LAYERS:
        for content in CONTENTS:
            xs = per_target_deltas(df, content, layer, None)
            fold, lo, hi, p = boot(xs)
            pooled[f"{layer}_{content}"] = {"layer": layer, "content": content,
                                            "n_targets": len(xs), "fold": fold,
                                            "ci95": [lo, hi], "p": p}
            print(f"  layer {layer:>2} {content:>5}  {fold:.3f}x [{lo:.3f}, {hi:.3f}]  "
                  f"p={p:.4f}  (n={len(xs)} targets)")

    l27w = pooled["27_word"]
    print(f"\nreplication check — Son's layer-27 pooled result on 4 needles: "
          f"{SON_LAYER27_POOLED['fold']}x, p={SON_LAYER27_POOLED['p']}")
    print(f"  same construction, {l27w['n_targets']} word needles: "
          f"{l27w['fold']:.3f}x, p={l27w['p']:.4f} -> "
          f"{'REPLICATES' if l27w['p'] < 0.05 and l27w['fold'] < 1 else 'DOES NOT REPLICATE'}")

    json.dump({"seed": SEED, "n_boot": N_BOOT, "n_rows": len(df),
               "cells": cells, "pooled": pooled,
               "son_layer27_pooled": SON_LAYER27_POOLED},
              open(OUT_PATH, "w"), indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
