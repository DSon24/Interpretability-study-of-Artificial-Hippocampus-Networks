"""CPU-only: summarise the RULER NIAH cohort rows into Table-4-comparable statistics.

Exists so the numbers in README and docs/FINDINGS.md regenerate from the saved rows
instead of being transcribed out of a notebook cell. That transcription is what produced
the RQ1 long-stratum artefact bug (Findings, 3 Sep), where README prose and the saved JSON
disagreed and neither was obviously wrong.

READOUT BASIS -- read this before using the numbers.

`d_resid_preregistered` (row key `rank_c1_residual`) is the basis of record, per
Expected_Tables section 1 and the 3 Sep correction. `o_t` (row key `rank`) is reported
alongside it because the withdrawn "layer 27 is a working instrument" claim came from
reading `o_t` alone. Both are printed; d_resid is primary.

    python ruler_cohort_stats.py
"""

from __future__ import annotations

import json
import os
import random
import statistics as st
from typing import Dict, List, Sequence, Tuple

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
ROWS_PATH = os.path.join(RESULTS_DIR, "04g_ruler_niah_rows.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04g_ruler_niah_stats.json")

SEED = 20260820
N_BOOT = 10000

# Expected rank of a uniformly random token over Qwen2.5's 151,936-token vocabulary,
# (V - 1) / 2. The same constant every control in this project is scored against.
CHANCE = 75968

# Homemade build_niah_prompt sweep, pre-registered D-resid basis, evicted median rank
# per layer (04c_per_layer_controls.json; Findings, 2-3 Sep). The comparison point.
HOMEMADE_DRESID_EVICTED = {9: 94478, 18: 119805, 27: 109793}

BASES: Sequence[Tuple[str, str]] = (
    ("d_resid_preregistered", "rank_c1_residual"),
    ("o_t", "rank"),
)


def bootstrap_median_ci(
    xs: Sequence[int], n_boot: int = N_BOOT, seed: int = SEED
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the median. Seeded, so the CI is reproducible."""
    rng = random.Random(seed)
    reps = sorted(st.median([rng.choice(xs) for _ in xs]) for _ in range(n_boot))
    return reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot)]


def verdict(lo: float, hi: float) -> str:
    """Where the CI sits relative to chance. Lower rank is better, so 'below chance'
    is the readout succeeding."""
    if hi < CHANCE:
        return "below_chance"
    if lo > CHANCE:
        return "above_chance"
    return "spans_chance"


def summarise(rows: List[dict]) -> dict:
    by_layer: Dict[int, List[dict]] = {}
    for r in rows:
        by_layer.setdefault(int(r["layer"]), []).append(r)

    out: dict = {
        "chance_rank": CHANCE,
        "n_rows": len(rows),
        "n_boot": N_BOOT,
        "seed": SEED,
        "homemade_dresid_evicted_median": HOMEMADE_DRESID_EVICTED,
        "layers": {},
        "pooled": {},
    }

    for basis, key in BASES:
        xs = [int(r[key]) for r in rows]
        lo, hi = bootstrap_median_ci(xs)
        out["pooled"][basis] = {
            "n": len(xs),
            "mean_rank": st.mean(xs),
            "median_rank": st.median(xs),
            "ci95_median": [lo, hi],
            "verdict_vs_chance": verdict(lo, hi),
            "frac_beating_chance": sum(x < CHANCE for x in xs) / len(xs),
        }

    for layer in sorted(by_layer):
        layer_rows = by_layer[layer]
        rec: dict = {}
        for basis, key in BASES:
            xs = [int(r[key]) for r in layer_rows]
            lo, hi = bootstrap_median_ci(xs)
            rec[basis] = {
                "n": len(xs),
                "mean_rank": st.mean(xs),
                "median_rank": st.median(xs),
                "ci95_median": [lo, hi],
                "verdict_vs_chance": verdict(lo, hi),
                "frac_beating_chance": sum(x < CHANCE for x in xs) / len(xs),
            }
        hm = HOMEMADE_DRESID_EVICTED.get(layer)
        if hm is not None:
            rec["vs_homemade_dresid"] = {
                "homemade_median": hm,
                "ruler_median": rec["d_resid_preregistered"]["median_rank"],
                "delta": rec["d_resid_preregistered"]["median_rank"] - hm,
            }
        out["layers"][str(layer)] = rec

    return out


def main() -> None:
    with open(ROWS_PATH) as f:
        blob = json.load(f)
    rows = blob["rows"]
    stats = summarise(rows)
    stats["ruler_cfg"] = blob.get("ruler_cfg")
    stats["lens_validated"] = bool(rows and rows[0].get("lens_validated"))
    stats["readout"] = rows[0].get("readout") if rows else None

    print(f"RULER NIAH cohort — {len(rows)} rows, config={stats['ruler_cfg']}")
    print(f"readout={stats['readout']}  lens_validated={stats['lens_validated']}")
    print(f"chance rank = {CHANCE}; lower rank is better, so 'below_chance' = success\n")

    hdr = f"{'layer':>6} {'basis':>23} {'median':>9} {'95% CI':>20} {'verdict':>14} {'homemade':>10}"
    print(hdr)
    print("-" * len(hdr))
    for layer in sorted(stats["layers"], key=int):
        rec = stats["layers"][layer]
        for basis, _ in BASES:
            d = rec[basis]
            lo, hi = d["ci95_median"]
            hm = HOMEMADE_DRESID_EVICTED[int(layer)] if basis == "d_resid_preregistered" else ""
            print(
                f"{layer:>6} {basis:>23} {d['median_rank']:>9.0f} "
                f"[{lo:>7.0f},{hi:>7.0f}] {d['verdict_vs_chance']:>14} {str(hm):>10}"
            )

    print()
    for basis, _ in BASES:
        d = stats["pooled"][basis]
        print(
            f"pooled {basis:>23}: median {d['median_rank']:>8.0f}  "
            f"mean {d['mean_rank']:>8.0f}  beating chance {d['frac_beating_chance']:>6.1%}"
        )

    with open(OUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
