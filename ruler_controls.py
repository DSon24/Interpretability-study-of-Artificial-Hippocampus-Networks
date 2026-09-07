"""CPU-only: C2, C3-context and C3-lens for the RULER cohort.

Run 025 put layer 27 at median rank 17,250 on evicted RULER needles, but that is a
C1-shaped result. C2 is the control that withdrew the 2 Sep layer-27 claim -- its raw
5.03x ratio turned out to be pure pair-identity baseline -- so a C1 pass on a new cohort
does not licence an RQ2 claim on its own.

WHAT EACH CONTROL IS HERE

TARGET. RULER NIAH answers here are 7-digit numbers, and Qwen tokenizes " 7700828" as
[' ', '7', '7', '0', '0', '8', '2', '8'] -- the leading space is its own token. Scoring
encode(" " + answer)[0] therefore scored token 220, a bare space, identically for all 60
examples; that is what run 025 did. The measurement now scores the answer as a sequence:
the readout at each of the 7 digit positions, from which the log-probability of any
candidate 7-digit answer can be assembled.

C2, baseline-corrected, cross-example. RULER has no distractor, so the other examples'
answers are the distractor pool. For an ordered pair (i, j):

    fold_ij = [ P_i(a_i) / P_i(a_j) ] / [ P_j(a_i) / P_j(a_j) ]

P_i(a_j) is the readout probability of example j's ANSWER SEQUENCE evaluated at example
i's digit positions. The denominator measures the same pair with the other example stored, so a
token that is simply more probable in general cancels out. Under the null that the readout
is indifferent to what was stored, fold = 1. This is the ratio-of-ratios shape Son's 28 Aug
correction arrived at, applied across examples instead of across hand-picked needle pairs.

Note fold_ji = 1 / fold_ij exactly, so only unordered pairs i < j are counted; counting
both directions would force the geometric mean to 1 by construction.

The fold is the SQUARE of the per-example effect, and this is the easiest way to overstate
the result by a factor of ten. If storing example i multiplies p(a_i) by k, then the
numerator gains k and the denominator loses k, so fold = k^2. A synthetic cohort with a
planted 50x effect reports fold = 2500x. The pre-registered C2 bar (10x, Expected Tables
Table 4) is stated in per-example terms, so it is sqrt(fold) that must clear it. Both are
reported; `effect_per_example` is the one to quote and the one tested against the bar.

C3-context. Own-answer rank with the context word order destroyed, paired against the
ordered condition, question and answer_prefix left intact.

C3-lens (DIVERGENCE 3a). Own-answer rank decoded through a row-permuted J-lens map.
Structure that survives this is an artefact of the decoding procedure. This is the control
the proposal's Table 4 actually specifies; the repo had only ever implemented shuffled
context.

Every statistic is restricted to placement == "evicted" by default: the claim is about
what survives compression, and 28 of 60 RULER needles sit inside the local window.

    python ruler_controls.py
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics as st
from typing import Dict, List, Sequence, Tuple

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
ROWS_PATH = os.path.join(RESULTS_DIR, "04i_ruler_controls_rows.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04i_ruler_controls_stats.json")

CHANCE = 75968
N_BOOT = 10000
SEED = 20260820
EPS = 1e-12
# Pre-registered C2 bar, from Expected Tables Table 4.
C2_BAR = 10.0


def geo_mean(xs: Sequence[float]) -> float:
    return math.exp(sum(math.log(max(x, EPS)) for x in xs) / len(xs))


def boot_ci(xs: Sequence[float], stat, n_boot: int = N_BOOT, seed: int = SEED):
    rng = random.Random(seed)
    reps = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n_boot))
    return reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot)]


def answer_logprob(table: Sequence[Sequence[float]], digits: Sequence[int]) -> float:
    """Log-probability of a candidate digit sequence under one example's readout table."""
    n = min(len(table), len(digits))
    return sum(table[t][digits[t]] for t in range(n))


def c2_folds(rows_by_example: Dict[int, dict], keep: Sequence[int],
             answer_seqs: Sequence[Sequence[int]]) -> Tuple[List[float], int]:
    """Baseline-corrected C2 fold for every unordered pair of kept examples.

    Worked in log space: log fold = [lp_i(a_i) - lp_i(a_j)] - [lp_j(a_i) - lp_j(a_j)].
    """
    folds, dropped = [], 0
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            i, j = keep[a], keep[b]
            di, dj = answer_seqs[i], answer_seqs[j]
            if di is None or dj is None or di == dj:
                dropped += 1
                continue
            ti = rows_by_example[i]["digit_logprobs"]
            tj = rows_by_example[j]["digit_logprobs"]
            log_fold = ((answer_logprob(ti, di) - answer_logprob(ti, dj))
                        - (answer_logprob(tj, di) - answer_logprob(tj, dj)))
            folds.append(math.exp(max(min(log_fold, 700.0), -700.0)))
    return folds, dropped


def permutation_p(folds: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED) -> float:
    """Sign-flip test on log folds: H0 is that storing i rather than j does nothing,
    under which each pair's log fold is equally likely to carry either sign."""
    logs = [math.log(max(f, EPS)) for f in folds]
    obs = sum(logs) / len(logs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        flipped = sum(x if rng.random() < 0.5 else -x for x in logs) / len(logs)
        if abs(flipped) >= abs(obs):
            hits += 1
    return hits / n_boot


def main() -> None:
    with open(ROWS_PATH) as f:
        blob = json.load(f)
    rows = blob["rows"]
    answer_seqs = blob["answer_digit_seqs"]
    layers = sorted({int(r["layer"]) for r in rows})

    print(f"RULER control battery — {len(rows)} rows, config={blob['ruler_cfg']}")
    print(f"readout={rows[0]['readout']}  lens_validated={rows[0]['lens_validated']}")
    print(f"chance rank = {CHANCE}; C2 pre-registered bar = {C2_BAR}x\n")

    out: dict = {"chance_rank": CHANCE, "c2_bar": C2_BAR, "n_boot": N_BOOT,
                 "seed": SEED, "ruler_cfg": blob["ruler_cfg"],
                 "readout": rows[0]["readout"],
                 "lens_validated": rows[0]["lens_validated"], "layers": {}}

    for L in layers:
        ordered = {int(r["example"]): r for r in rows
                   if int(r["layer"]) == L and r["condition"] == "ordered"}
        shuffled = {int(r["example"]): r for r in rows
                    if int(r["layer"]) == L and r["condition"] == "shuffled_context"}
        evicted = sorted(i for i, r in ordered.items() if r["placement"] == "evicted")

        rec: dict = {"n_evicted": len(evicted)}
        print(f"--- layer {L}  (n evicted = {len(evicted)}) " + "-" * 26)

        # ---- C1 on the sequence target --------------------------------------------
        if evicted:
            mranks = [ordered[i]["mean_digit_rank"] for i in evicted]
            lo, hi = boot_ci(mranks, st.median)
            rec["C1_sequence"] = {
                "n": len(mranks), "median_mean_digit_rank": st.median(mranks),
                "ci95": [lo, hi],
                "below_chance": bool(hi < CHANCE),
                "median_answer_logprob": st.median([ordered[i]["answer_logprob"] for i in evicted]),
            }
            v = "BELOW chance" if hi < CHANCE else ("above chance" if lo > CHANCE else "spans chance")
            print(f"  C1 mean digit rank {st.median(mranks):9.0f} [{lo:.0f}, {hi:.0f}]  -> {v}")

        # ---- C2 -------------------------------------------------------------------
        folds, dropped = c2_folds(ordered, evicted, answer_seqs)
        if folds:
            g = geo_mean(folds)
            lo, hi = boot_ci(folds, geo_mean)
            p = permutation_p(folds)
            # fold = (per-example effect)^2; the bar is stated per example
            eff, eff_lo, eff_hi = math.sqrt(g), math.sqrt(lo), math.sqrt(hi)
            # The per-example effect compounds over the answer's digits, so a modest
            # per-digit preference looks large end to end: 1.4x per digit over 7 digits
            # is already 10.5x. The pre-registered 10x bar was written for a
            # single-token needle, so the per-digit figure is the honest comparison and
            # both are reported.
            ndig = len(answer_seqs[evicted[0]]) if evicted and answer_seqs[evicted[0]] else 1
            per_digit = eff ** (1.0 / ndig) if ndig else eff
            rec["C2_cross_example"] = {
                "n_pairs": len(folds), "pairs_dropped": dropped,
                "corrected_fold_geomean": g, "ci95_fold": [lo, hi],
                "effect_per_example": eff, "ci95_effect_per_example": [eff_lo, eff_hi],
                "answer_digits_scored": ndig,
                "effect_per_digit": per_digit,
                "ci95_effect_per_digit": [eff_lo ** (1.0 / ndig), eff_hi ** (1.0 / ndig)],
                "permutation_p": p,
                "passes_preregistered_bar": bool(eff_lo > C2_BAR),
                "excludes_null": bool(eff_lo > 1.0),
            }
            verdict = (f"PASSES the {C2_BAR:.0f}x bar" if eff_lo > C2_BAR else
                       f"above 1.0 but below the {C2_BAR:.0f}x bar" if eff_lo > 1.0 else
                       "does not exclude 1.0")
            print(f"  C2 per-example effect {eff:8.3f}x  CI [{eff_lo:.3f}, {eff_hi:.3f}]  "
                  f"perm p={p:.4f}  n_pairs={len(folds)}")
            print(f"     (ratio-of-ratios fold {g:.3g}x = effect squared; "
                  f"{per_digit:.3f}x per digit over {ndig} digits)")
            print(f"     -> {verdict}")

        # ---- C3-context -----------------------------------------------------------
        pairs = [(ordered[i]["mean_digit_rank"], shuffled[i]["mean_digit_rank"])
                 for i in evicted if i in shuffled]
        if pairs:
            deltas = [s - o for o, s in pairs]          # positive = shuffling hurt
            lo, hi = boot_ci(deltas, st.median)
            rec["C3_context"] = {
                "n": len(pairs),
                "median_rank_ordered": st.median([o for o, _ in pairs]),
                "median_rank_shuffled": st.median([s for _, s in pairs]),
                "median_delta_shuffled_minus_ordered": st.median(deltas),
                "ci95_delta": [lo, hi],
                "order_sensitive": bool(lo > 0 or hi < 0),
            }
            direction = ("shuffling HURTS" if lo > 0 else
                         "shuffling HELPS" if hi < 0 else "no order sensitivity")
            print(f"  C3-context ordered {st.median([o for o,_ in pairs]):8.0f} -> "
                  f"shuffled {st.median([s for _,s in pairs]):8.0f}   "
                  f"delta {st.median(deltas):+8.0f} [{lo:+.0f}, {hi:+.0f}]  -> {direction}")

        # ---- C3-lens --------------------------------------------------------------
        lens_pairs = [(ordered[i]["answer_logprob"], ordered[i]["answer_logprob_shuffled_lens"])
                      for i in evicted if "answer_logprob_shuffled_lens" in ordered[i]]
        if lens_pairs:
            # answer log-probability: higher is better, so a real lens should beat the
            # permuted one and the delta should be negative
            deltas = [s - o for o, s in lens_pairs]
            lo, hi = boot_ci(deltas, st.median)
            rec["C3_lens"] = {
                "n": len(lens_pairs),
                "median_answer_logprob_real": st.median([o for o, _ in lens_pairs]),
                "median_answer_logprob_shuffled": st.median([s for _, s in lens_pairs]),
                "median_delta": st.median(deltas),
                "ci95_delta": [lo, hi],
                "signal_collapses": bool(hi < 0),
            }
            verdict = ("signal COLLAPSES under the permuted map, as it should" if hi < 0
                       else "signal SURVIVES the permuted map — DECODING ARTEFACT")
            print(f"  C3-lens    real logP {st.median([o for o,_ in lens_pairs]):+8.2f} -> "
                  f"permuted {st.median([s for _,s in lens_pairs]):+8.2f}   "
                  f"delta {st.median(deltas):+7.2f} [{lo:+.2f}, {hi:+.2f}]")
            print(f"     -> {verdict}")

        out["layers"][str(L)] = rec
        print()

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
