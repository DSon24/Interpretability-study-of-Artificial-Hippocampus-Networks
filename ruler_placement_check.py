"""CPU-only: is the below-chance layer-27 RULER read carried by needles that were
never evicted? -- redone on the corrected digit-sequence rows (Run 026).

`ruler_needle_position.py` answered this on `04g_ruler_niah_rows.json`, whose target
was token 220 (a leading space, identical for all 60 examples -- the target-scoring bug
fixed by Run 026, 81cf877). `04i_ruler_controls_rows.json` already carries, per row,
`needle_pos` / `placement` / `eviction_distance` / `needle_is_evicted` AND the corrected
`mean_digit_rank`, so this splits the corrected metric by placement with no GPU and no
tokenizer.

Two views:
  * by `placement` -- the coarse in_sink_region / evicted / in_window verdict, matching
    the original script.
  * by fractional needle depth (needle_pos / n_tokens) in terciles -- the literal
    "does front/middle/back matter" question.

`occurrences` (answer string appearing more than once, so the first hit may not be the
needle) is joined in from the old 04h per-example records, which already computed it.

    python ruler_placement_check.py

Output: results/run_3b_gdn/04q_ruler_placement_check.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import ahn_interp as ai

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
ROWS_PATH = os.path.join(RESULTS_DIR, "04i_ruler_controls_rows.json")
OLD_04H_PATH = os.path.join(RESULTS_DIR, "04h_ruler_needle_position.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04q_ruler_placement_check.json")

SEED = 20260820
N_BOOT = 10000
CHANCE = 75968


def _agg(ranks: list) -> dict:
    a = np.asarray(ranks, dtype=float)
    pt, lo, hi = ai.bootstrap_ci(a, stat=np.median, n_boot=N_BOOT, seed=SEED)
    return {
        "n": int(a.size),
        "median_rank": float(pt),
        "ci95": [float(lo), float(hi)],
        "mean_rank": float(a.mean()),
        "frac_beating_chance": float((a < CHANCE).mean()),
    }


def main() -> None:
    blob = json.load(open(ROWS_PATH))
    rows = [r for r in blob["rows"]
            if r["condition"] == "ordered" and r["readout"] == "jlens"]
    layers = sorted({r["layer"] for r in rows})

    # occurrences, keyed by example index, from the old placement script
    occ = {}
    if os.path.exists(OLD_04H_PATH):
        for rec in json.load(open(OLD_04H_PATH)).get("per_example", []):
            occ[rec["example"]] = rec.get("occurrences")
    multi_occ = sorted(i for i, c in occ.items() if c and c > 1)

    out = {
        "source_rows": ROWS_PATH,
        "note": "Layer-wise RULER digit rank split by needle placement, corrected "
                "digit-sequence scoring (Run 026). Redo of 04h, which ran on the "
                "space-token-buggy 04g rows. ordered context, jlens readout.",
        "chance_rank": CHANCE,
        "multi_occurrence_examples": multi_occ,
        "old_04h_layer27_by_verdict": (
            json.load(open(OLD_04H_PATH)).get("layer27_rank_by_verdict")
            if os.path.exists(OLD_04H_PATH) else None
        ),
        "by_placement": {},
        "by_depth_tercile": {},
    }

    print(f"RULER placement check (corrected scoring) -- {len(rows)} ordered rows, "
          f"layers {layers}, chance {CHANCE}\n")

    for L in layers:
        lr = [r for r in rows if r["layer"] == L]

        by_place = {}
        for pl in ("in_sink_region", "evicted", "in_window"):
            xs = [r["mean_digit_rank"] for r in lr if r["placement"] == pl]
            if xs:
                by_place[pl] = _agg(xs)
        out["by_placement"][str(L)] = by_place

        depth = np.array([r["needle_pos"] / r["n_tokens"] for r in lr])
        rank = np.array([r["mean_digit_rank"] for r in lr], dtype=float)
        edges = np.quantile(depth, [0, 1 / 3, 2 / 3, 1.0]); edges[-1] += 1e-9
        by_ter = {}
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), 1):
            m = (depth >= lo) & (depth < hi)
            if m.sum():
                d = _agg(rank[m].tolist())
                d["depth_frac_range"] = [float(lo), float(min(hi, 1.0))]
                by_ter[f"t{k}"] = d
        out["by_depth_tercile"][str(L)] = by_ter

        print(f"layer {L}")
        for pl, d in by_place.items():
            print(f"  {pl:16s} n={d['n']:3d}  median {d['median_rank']:8.0f}  "
                  f"CI [{d['ci95'][0]:.0f}, {d['ci95'][1]:.0f}]  "
                  f"beat-chance {d['frac_beating_chance']:.2f}")
        for tk, d in by_ter.items():
            lo, hi = d["depth_frac_range"]
            print(f"  depth {tk} ({lo:.2f}-{hi:.2f})  n={d['n']:3d}  "
                  f"median {d['median_rank']:8.0f}  beat-chance {d['frac_beating_chance']:.2f}")
        print()

    if multi_occ:
        print(f"! {len(multi_occ)} example(s) contain the answer string more than once "
              f"(examples {multi_occ}); the first occurrence was used for needle_pos.\n")

    ai.set_results_dir(RESULTS_DIR)
    ai.save_json(out, os.path.basename(OUT_PATH))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
