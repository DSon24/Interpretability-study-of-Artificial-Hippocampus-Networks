#!/usr/bin/env python3
"""
Re-run the C1-C4 control battery PER LAYER instead of pooled. CPU only, no model.

WHY
---
`04_table4_controls.json` reports one number per control, pooled across layers 9, 18
and 27. Every control except C4 fails, and the headline reading has been "the readout
is at chance." But the three layers are not interchangeable instruments, and pooling
them averages a degenerate readout together with a working one.

Reads `results/run_3b_gdn/04_retention_rows.json` (624 rows already on disk) and
recomputes each control within layer, with bootstrap CIs.

READOUT BASIS -- read this before using the numbers.
Expected_Tables_and_Figures section 1 pre-registers the Delta-readout on the RESIDUAL
STREAM ("rank_c1_residual"), not the raw module output ("rank"). The first version of
this script reported only `rank` (the o_t basis) and concluded that layer 27 was above
chance. In the pre-registered basis it is not: 109,793 against a chance rank of 75,968.
Both bases are reported below; `d_resid` is primary.

    python per_layer_controls.py
"""

import json
import math
import random
import statistics as st

ROWS = "results/run_3b_gdn/04_retention_rows.json"
OUT = "results/run_3b_gdn/04c_per_layer_controls.json"
LAYERS = (9, 18, 27)
N_BOOT = 10000
SEED = 20260820


def boot_median(vals, n_boot=N_BOOT, seed=SEED):
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    n = len(vals)
    for _ in range(n_boot):
        meds.append(st.median([vals[rng.randrange(n)] for _ in range(n)]))
    meds.sort()
    return (meds[int(0.025 * n_boot)], meds[int(0.975 * n_boot)])


def main():
    blob = json.load(open(ROWS))
    rows = blob["rows"]
    vocab = 151936
    chance = vocab / 2.0
    uniform_entropy = math.log(vocab)

    out = {
        "note": "per-layer recomputation of Table 4 from 04_retention_rows.json. CPU only.",
        "vocab": vocab,
        "chance_rank": chance,
        "uniform_entropy_nats": uniform_entropy,
        "n_rows": len(rows),
        "lens_validated": False,
        "layers": {},
    }

    for L in LAYERS:
      for basis, key in (("d_resid_preregistered", "rank_c1_residual"), ("o_t", "rank")):
        at = [r for r in rows if r["layer"] == L]
        ev = [r for r in at if not r["in_window"] and not r["shuffled"]]
        iw = [r for r in at if r["in_window"]]
        sh = [r for r in at if r["shuffled"]]
        dis = [r for r in ev if "p_distractor" in r]

        ent = [r["entropy"] for r in at]
        ev_rank = [r[key] for r in ev]
        iw_rank = [r[key] for r in iw]
        sh_rank = [r[key] for r in sh]

        ratios = [r["p_mem"] / r["p_distractor"] for r in dis if r["p_distractor"] > 0]
        wins = sum(1 for r in dis if r["rank"] < r["rank_distractor"])

        # readout health: an entropy far below uniform means the readout has collapsed
        # onto a handful of tokens, which is a different failure from "at chance".
        ent_med = st.median(ent)
        degenerate = ent_med < 1.0

        rec = {
            "n_rows": len(at),
            "readout_entropy_median_nats": ent_med,
            "readout_degenerate": degenerate,
            "C1_evicted_rank_median": st.median(ev_rank),
            "C1_evicted_rank_ci": boot_median(ev_rank),
            "C1_better_than_chance": st.median(ev_rank) < chance,
                # NOTE: this is the RAW ratio. Son's cross-pair diagnostic shows it is
            # dominated by pair-identity baseline preference; see
            # results/run_3b_gdn/04d_c2_baseline_corrected.json for the corrected
            # version, in which every layer's effect disappears.
        "C2_median_prob_ratio": st.median(ratios) if ratios else None,
            "C2_needle_win_rate": wins / len(dis) if dis else None,
            "C2_n": len(dis),
            "C2_passed_10x": bool(ratios and st.median(ratios) >= 10),
            "C3_ordered_rank_median": st.median(ev_rank),
            "C3_shuffled_rank_median": st.median(sh_rank) if sh_rank else None,
            "C4_in_window_rank_median": st.median(iw_rank) if iw_rank else None,
            "C4_in_window_ci": boot_median(iw_rank),
            "C4_passed": (st.median(iw_rank) < st.median(ev_rank)) if iw_rank else None,
            "frac_rows_better_than_chance": sum(1 for r in at if r[key] < chance) / len(at),
            "best_in_window_rank": min(iw_rank) if iw_rank else None,
        }
        out["layers"].setdefault(str(L), {})[basis] = rec
        if basis != "d_resid_preregistered":
            continue

        print(f"\n--- layer {L}  (pre-registered D-resid basis) " + "-" * 18)
        print(f"  readout entropy median : {ent_med:8.3f} nats "
              f"(uniform {uniform_entropy:.2f}){'   <-- DEGENERATE' if degenerate else ''}")
        print(f"  C1 evicted rank        : {rec['C1_evicted_rank_median']:8.0f} "
              f"95% CI [{rec['C1_evicted_rank_ci'][0]:.0f}, {rec['C1_evicted_rank_ci'][1]:.0f}]  "
              f"chance {chance:.0f}  -> {'below chance (good)' if rec['C1_better_than_chance'] else 'at/above chance'}")
        if rec["C2_median_prob_ratio"] is not None:
            print(f"  C2 needle/distractor   : {rec['C2_median_prob_ratio']:8.3f}x   "
                  f"needle wins {wins}/{len(dis)} ({rec['C2_needle_win_rate']:.0%})")
        print(f"  C4 in-window ceiling   : {rec['C4_in_window_rank_median']:8.0f} "
              f"vs evicted {rec['C1_evicted_rank_median']:.0f}  -> {'PASS' if rec['C4_passed'] else 'FAIL'}")
        print(f"  rows better than chance: {rec['frac_rows_better_than_chance']:.1%}")

    print("\n=== basis comparison: median evicted rank (chance = 75,968) ===")
    print(f"{'layer':>6} {'D-resid (pre-reg)':>19} {'o_t':>10}")
    for L in LAYERS:
        d = out["layers"][str(L)]
        print(f"{L:>6} {d['d_resid_preregistered']['C1_evicted_rank_median']:>19.0f} "
              f"{d['o_t']['C1_evicted_rank_median']:>10.0f}")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
