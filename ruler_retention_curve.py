"""CPU-only: RULER NIAH retention as a function of eviction distance (Run 025, redone).

The original retention sweep (Findings, 7 Sep, "Retention does not decay across the range
measured") was a two-bin near/far split on the space-token-buggy RULER scores -- the
target was token 220 (a leading space), identical for all 60 examples, so it measured
nothing about the digits. Run 026 (81cf877) rebuilt RULER scoring on the digit sequence
and `results/run_3b_gdn/04i_ruler_controls_rows.json` already carries per-example
`eviction_distance` and `mean_digit_rank`. This re-does the retention curve on those
corrected rows: bin the evicted examples by how far past the compression boundary the
needle sat, and ask whether the digit rank degrades with distance (decay) or is flat.

No GPU: everything needed is in 04i. Regenerate with

    python ruler_retention_curve.py

Output: results/run_3b_gdn/04p_ruler_retention_curve.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import ahn_interp as ai

RESULTS_DIR = os.path.join("results", "run_3b_gdn")
ROWS_PATH = os.path.join(RESULTS_DIR, "04i_ruler_controls_rows.json")
OUT_PATH = os.path.join(RESULTS_DIR, "04p_ruler_retention_curve.json")

SEED = 20260820
N_BOOT = 10000
N_BINS = 4
# Expected rank of a uniformly random token over Qwen2.5's 151,936-token vocabulary.
CHANCE = 75968
# Findings 7 Sep, two-bin split on the BUGGY rows -- the number this replaces.
PRIOR_TWOBIN_BUGGY = {"near_median_rank": 15844, "far_median_rank": 17250, "verdict": "flat"}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """rho = Pearson on the rank-transformed data. numpy-only, no scipy."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def perm_p(x: np.ndarray, y: np.ndarray, rho_obs: float, n_perm: int = 10000) -> float:
    """Two-sided permutation p-value for Spearman rho."""
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(n_perm):
        if abs(spearman(x, rng.permutation(y))) >= abs(rho_obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def curve_for_layer(rows: list) -> dict:
    dist = np.array([r["eviction_distance"] for r in rows], dtype=float)
    rank = np.array([r["mean_digit_rank"] for r in rows], dtype=float)

    order = np.argsort(dist)
    dist, rank = dist[order], rank[order]

    # equal-count (quantile) bins so no bin is empty at n=32
    edges = np.quantile(dist, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() == 0:
            continue
        pt, clo, chi = ai.bootstrap_ci(rank[m], stat=np.median, n_boot=N_BOOT, seed=SEED)
        bins.append({
            "dist_lo": float(lo), "dist_hi": float(hi),
            "n": int(m.sum()),
            "mean_eviction_distance": float(dist[m].mean()),
            "median_rank": float(pt), "ci95": [float(clo), float(chi)],
            "frac_beating_chance": float((rank[m] < CHANCE).mean()),
        })

    rho = spearman(dist, rank)
    slope = float(np.polyfit(dist, rank, 1)[0])  # rank per token of eviction distance
    p = perm_p(dist, rank, rho)
    # decay = rank gets WORSE (higher) as eviction distance grows -> rho > 0, significant
    if p >= 0.05:
        verdict = "flat -- no distance dependence over the range measured"
    elif rho > 0:
        verdict = "decays -- digit rank degrades with eviction distance"
    else:
        verdict = "improves with distance -- opposite of a decay curve, treat as artefact"

    return {
        "n": len(rows),
        "eviction_distance_range": [float(dist.min()), float(dist.max())],
        "overall_median_rank": float(np.median(rank)),
        "spearman_rho": rho,
        "perm_p": p,
        "ols_slope_rank_per_token": slope,
        "verdict": verdict,
        "bins": bins,
    }


def main() -> None:
    blob = json.load(open(ROWS_PATH))
    rows = blob["rows"]

    # the real measurement: ordered context, jlens readout, needle actually evicted
    ordered = [r for r in rows
               if r["condition"] == "ordered"
               and r["readout"] == "jlens"
               and r["needle_is_evicted"]]
    layers = sorted({r["layer"] for r in ordered})

    out = {
        "source_rows": ROWS_PATH,
        "note": "RULER NIAH retention vs eviction distance, corrected digit-sequence "
                "scoring (Run 026). Evicted examples only, ordered context, jlens "
                "readout. Replaces the 7 Sep two-bin split on the buggy space-token "
                "rows.",
        "prior_twobin_on_buggy_rows": PRIOR_TWOBIN_BUGGY,
        "chance_rank": CHANCE,
        "n_bins": N_BINS,
        "lens_validated": bool(ordered[0].get("lens_validated", False)) if ordered else None,
        "ruler_config": ordered[0]["ruler_config"] if ordered else None,
        "layers": {},
    }

    print(f"RULER retention curve -- {len(ordered)} evicted rows over layers {layers}\n")
    for L in layers:
        L_rows = [r for r in ordered if r["layer"] == L]
        res = curve_for_layer(L_rows)
        out["layers"][str(L)] = res
        print(f"layer {L}:  n={res['n']}  dist {res['eviction_distance_range'][0]:.0f}"
              f"..{res['eviction_distance_range'][1]:.0f}  "
              f"rho={res['spearman_rho']:+.3f}  p={res['perm_p']:.3f}")
        print(f"  {res['verdict']}")
        for b in res["bins"]:
            print(f"    d~{b['mean_eviction_distance']:6.0f}  n={b['n']:2d}  "
                  f"median rank {b['median_rank']:8.0f}  "
                  f"CI [{b['ci95'][0]:.0f}, {b['ci95'][1]:.0f}]  "
                  f"beat-chance {b['frac_beating_chance']:.2f}")
        print()

    ai.set_results_dir(RESULTS_DIR)
    ai.save_json(out, os.path.basename(OUT_PATH))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
