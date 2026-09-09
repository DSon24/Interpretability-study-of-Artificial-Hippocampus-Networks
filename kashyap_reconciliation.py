"""CPU-only: reconcile our RQ1 numbers against Kashyap (2026)'s reported bands.

Kashyap reports, for removing all recurrent writes on LongBench:
    * mean F1 moves 0.4-2.3 points
    * 38-42% of answers change

Our nb03 rows carry two scorings -- full generation and first line (`*_fl`) -- and
neither matches those bands: first-line gives dF1 +3.34 pts / 33.3% changed (GDN),
full generation gives -1.44 pts / 91.7%. His change rate sits *between* ours, and his
F1 band brackets our full-generation magnitude, so the discrepancy plausibly is the
scoring convention rather than the science.

This script re-scores the saved generations under every convention that could
reasonably be his -- including the *official* LongBench scorers vendored at
`eval/longbench/` -- and reports which land inside his bands. No GPU, no model: it
reads `03_nowrite_reproduction.json` only.

It also jackknifes the DeltaNet dF1, whose CI [+0.2, +13.7] barely excludes zero at
n=60, to test whether the one result that genuinely breaks his band survives dropping
a handful of examples.

    python kashyap_reconciliation.py

Output:
    results/05_kashyap_reconciliation.json
"""
from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import Callable, Sequence

import numpy as np

import ahn_interp as ai

RUNS = [("GatedDeltaNet", "results/run_3b_gdn"),
        ("DeltaNet", "results/run_3b_dn"),
        ("Mamba2", "results/run_3b_m2")]
OUT = "results/05_kashyap_reconciliation.json"

# Kashyap (2026), as quoted in the proposal's related-work section.
KASHYAP_F1_POINTS = (0.4, 2.3)      # |mean F1 shift|, percentage points
KASHYAP_CHANGE_RATE = (0.38, 0.42)


# --------------------------------------------------------------------------------------
# official LongBench scoring, transcribed from eval/longbench/{metrics,eval}.py
#
# Note the normalisation order differs subtly from ahn_interp.normalize_answer: the
# official one strips punctuation BEFORE articles, ours strips articles first. Kept
# faithful here so "official" means official.
# --------------------------------------------------------------------------------------
def lb_normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def lb_qa_f1(prediction: str, ground_truth: str) -> float:
    p = lb_normalize_answer(prediction).split()
    g = lb_normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def lb_scorer_e_prep(pred: str) -> str:
    """LongBench-E (`scorer_e`) prediction prep for hotpotqa.

    The `.lstrip("\\n").split("\\n")[0]` first-line truncation in scorer_e applies only
    to trec / triviaqa / samsum / lsht. hotpotqa is NOT in that list, so LongBench-E
    scores the FULL generation. This function is therefore the identity -- kept
    explicit because that fact is the whole point of the comparison.
    """
    return pred


def lb_scorer_prep(pred: str) -> str:
    """Non-E `scorer` prediction prep: a chain of prefix splits, then strip."""
    for sep in (".assistant", "\n\nQuestion", "</s>", "(Document", "\n\nAnswer", "(Passage"):
        pred = pred.split(sep)[0]
    return pred.strip()


def score_max_over_golds(pred: str, golds: Sequence[str],
                         metric: Callable[[str, str], float]) -> float:
    """Official scorers take max over the reference answers; ours took gold[0]."""
    return max((metric(pred, g) for g in golds), default=0.0)


# --------------------------------------------------------------------------------------
# metric variants
# --------------------------------------------------------------------------------------
def f1_variants(row: dict) -> dict:
    """dF1 (AHN - NOWRITE) for this example under each scoring convention."""
    golds = row["gold"] if isinstance(row["gold"], list) else [row["gold"]]
    full_a, full_n = row["answer_ahn"], row["answer_nowrite"]
    fl_a, fl_n = row["answer_ahn_fl"], row["answer_nowrite_fl"]

    out = {
        # what the repo already reports, straight off the saved fields
        "first_line": row["delta_f1_fl"],
        "full_generation": row["delta_f1"],
    }

    # official LongBench-E: no truncation for hotpotqa, official normalisation,
    # max over reference answers
    a = score_max_over_golds(lb_scorer_e_prep(full_a), golds, lb_qa_f1)
    n = score_max_over_golds(lb_scorer_e_prep(full_n), golds, lb_qa_f1)
    out["longbench_e_official"] = a - n

    # official non-E scorer: prefix-split chain, then the same metric
    a = score_max_over_golds(lb_scorer_prep(full_a), golds, lb_qa_f1)
    n = score_max_over_golds(lb_scorer_prep(full_n), golds, lb_qa_f1)
    out["longbench_nonE_official"] = a - n

    # official metric applied to the first line -- isolates "which text" from
    # "which normalisation"
    a = score_max_over_golds(fl_a, golds, lb_qa_f1)
    n = score_max_over_golds(fl_n, golds, lb_qa_f1)
    out["longbench_official_first_line"] = a - n
    return out


def change_variants(row: dict) -> dict:
    """Did the answer change? Under each plausible definition."""
    full_a, full_n = row["answer_ahn"], row["answer_nowrite"]
    fl_a, fl_n = row["answer_ahn_fl"], row["answer_nowrite_fl"]
    return {
        "raw_exact_full_generation": full_a != full_n,
        "raw_exact_first_line": fl_a != fl_n,
        "normalized_em_full_generation":
            lb_normalize_answer(full_a) != lb_normalize_answer(full_n),
        "normalized_em_first_line":
            lb_normalize_answer(fl_a) != lb_normalize_answer(fl_n),
        "nonE_prepped_normalized":
            lb_normalize_answer(lb_scorer_prep(full_a))
            != lb_normalize_answer(lb_scorer_prep(full_n)),
        # "the score moved", rather than "the string moved"
        "f1_changed_full_generation": row["f1_ahn"] != row["f1_nowrite"],
        "f1_changed_first_line": row["f1_ahn_fl"] != row["f1_nowrite_fl"],
    }


def in_band(value: float, band: tuple) -> bool:
    return band[0] <= value <= band[1]


# --------------------------------------------------------------------------------------
# item 4: how deep is the DeltaNet result?
# --------------------------------------------------------------------------------------
def jackknife(deltas: Sequence[float], n_boot: int = 4000) -> dict:
    """Leave-one-out, then greedy most-supportive-first deletion until the CI
    stops excluding zero. `k_to_kill` is the fragility depth."""
    d = np.asarray(deltas, dtype=float)
    full = ai.bootstrap_ci(d, n_boot=n_boot)

    loo = []
    for i in range(d.size):
        kept = np.delete(d, i)
        loo.append(ai.bootstrap_ci(kept, n_boot=n_boot))
    loo_points = [p for p, _, _ in loo]
    loo_excludes_zero = [lo > 0 or hi < 0 for _, lo, hi in loo]

    # greedy: repeatedly drop the example most supportive of the effect
    order = np.argsort(-d) if full[0] > 0 else np.argsort(d)
    kept = d.copy()
    k_to_kill = None
    for k in range(1, min(11, d.size)):
        kept = np.delete(d, order[:k])
        _, lo, hi = ai.bootstrap_ci(kept, n_boot=n_boot)
        if not (lo > 0 or hi < 0):
            k_to_kill = k
            break

    sig = bool(full[1] > 0 or full[2] < 0)
    return {
        "n": int(d.size),
        "full_sample": {"mean_points": full[0] * 100,
                        "ci_points": [full[1] * 100, full[2] * 100],
                        "excludes_zero": sig},
        "leave_one_out": {
            "min_mean_points": float(np.min(loo_points) * 100),
            "max_mean_points": float(np.max(loo_points) * 100),
            # only meaningful when the full sample was significant; otherwise there is
            # no significance to lose and the count is trivially n.
            "n_deletions_that_lose_significance":
                int(sum(not x for x in loo_excludes_zero)) if sig else None,
        },
        "greedy_deletion": {
            "k_to_lose_significance": k_to_kill if sig else None,
            "note": "examples dropped, most-supportive-first, before the 95% CI stops "
                    "excluding zero. null => the full sample was not significant, so "
                    "there is nothing to break.",
        },
    }


def main() -> None:
    report = {
        "source": "results/run_3b_*/03_nowrite_reproduction.json (saved generations, no GPU)",
        "comparison_target": {
            "citation": "Kashyap (2026), write-attrition study, under review ACL",
            "f1_shift_points": list(KASHYAP_F1_POINTS),
            "change_rate": list(KASHYAP_CHANGE_RATE),
        },
        "scoring_note":
            "LongBench-E `scorer_e` truncates the prediction to its first line ONLY for "
            "trec / triviaqa / samsum / lsht. hotpotqa is not in that list, so official "
            "LongBench-E scoring reads the FULL generation. Our primary metric of record "
            "is first-line, which is therefore NOT the official convention.",
        "cells": {},
    }

    for name, run in RUNS:
        path = os.path.join(run, "03_nowrite_reproduction.json")
        if not os.path.exists(path):
            print(f"! {name}: {path} missing, skipped")
            continue
        per = json.load(open(path))["per_example"]

        f1s = [f1_variants(r) for r in per]
        chg = [change_variants(r) for r in per]

        f1_rows = {}
        for key in f1s[0]:
            vals = [x[key] for x in f1s]
            pt, lo, hi = ai.bootstrap_ci(vals)
            f1_rows[key] = {
                "delta_f1_points": pt * 100,
                "ci_points": [lo * 100, hi * 100],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "magnitude_in_kashyap_band": in_band(abs(pt * 100), KASHYAP_F1_POINTS),
            }

        chg_rows = {}
        for key in chg[0]:
            rate = float(np.mean([x[key] for x in chg]))
            chg_rows[key] = {
                "change_rate": rate,
                "in_kashyap_band": in_band(rate, KASHYAP_CHANGE_RATE),
            }

        report["cells"][name] = {
            "run_dir": run, "n": len(per),
            "delta_f1_by_scoring": f1_rows,
            "change_rate_by_definition": chg_rows,
        }
        print(f"\n=== {name} (n={len(per)}) ===")
        for k, v in f1_rows.items():
            flag = "  <-- in Kashyap band" if v["magnitude_in_kashyap_band"] else ""
            print(f"  dF1 {k:32s} {v['delta_f1_points']:+7.2f} pts "
                  f"[{v['ci_points'][0]:+7.2f}, {v['ci_points'][1]:+7.2f}]{flag}")
        for k, v in chg_rows.items():
            flag = "  <-- in Kashyap band" if v["in_kashyap_band"] else ""
            print(f"  chg {k:32s} {v['change_rate']:6.1%}{flag}")

    # item 4 -- jackknife every cell on the first-line metric (the one that produced
    # the DeltaNet claim), so the fragility numbers are comparable across cells.
    report["jackknife_first_line"] = {}
    for name, run in RUNS:
        path = os.path.join(run, "03_nowrite_reproduction.json")
        if not os.path.exists(path):
            continue
        per = json.load(open(path))["per_example"]
        jk = jackknife([r["delta_f1_fl"] for r in per])
        report["jackknife_first_line"][name] = jk
        print(f"\n=== jackknife (first-line dF1) · {name} ===")
        print(f"  full sample {jk['full_sample']['mean_points']:+.2f} pts "
              f"[{jk['full_sample']['ci_points'][0]:+.2f}, "
              f"{jk['full_sample']['ci_points'][1]:+.2f}] "
              f"excludes_zero={jk['full_sample']['excludes_zero']}")
        print(f"  leave-one-out range {jk['leave_one_out']['min_mean_points']:+.2f} .. "
              f"{jk['leave_one_out']['max_mean_points']:+.2f} pts")
        if jk["full_sample"]["excludes_zero"]:
            print(f"  {jk['leave_one_out']['n_deletions_that_lose_significance']}/"
                  f"{jk['n']} single deletions lose significance; greedy k = "
                  f"{jk['greedy_deletion']['k_to_lose_significance']}")
        else:
            print("  full sample not significant — no significance to break")

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
