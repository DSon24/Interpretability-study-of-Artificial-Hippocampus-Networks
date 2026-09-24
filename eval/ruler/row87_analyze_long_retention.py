from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


SEED = 20260820
N_PERM = 10000
N_BOOT = 10000

SOURCES = {
    "16k": Path("results/run_3b_gdn/04i_ruler_controls_rows.json"),
    "32k": Path("results/validation/row87_ruler_32768_rows.json"),
    "64k": Path("results/validation/row87_ruler_65536_rows.json"),
}

OUT = Path(
    "results/validation/"
    "row87_ruler_long_context_retention.json"
)


def load_rows():
    out = []

    for cohort, path in SOURCES.items():
        with open(path) as f:
            blob = json.load(f)

        for r in blob["rows"]:
            if r["condition"] != "ordered":
                continue
            if not r["needle_is_evicted"]:
                continue

            out.append({
                **r,
                "length_cohort": cohort,
            })

    return out


def fixed_effect_slope(dist, rank, cohort):
    """
    rank ~ intercept + distance + I(32k) + I(64k)

    This estimates the distance slope while allowing each context-length
    cohort to have its own intercept.
    """
    dist = np.asarray(dist, dtype=float)
    rank = np.asarray(rank, dtype=float)
    cohort = np.asarray(cohort)

    X = np.column_stack([
        np.ones(len(dist)),
        dist,
        (cohort == "32k").astype(float),
        (cohort == "64k").astype(float),
    ])

    beta = np.linalg.lstsq(X, rank, rcond=None)[0]

    return float(beta[1])


def stratified_bootstrap(dist, rank, cohort, n=N_BOOT):
    rng = np.random.default_rng(SEED)

    dist = np.asarray(dist)
    rank = np.asarray(rank)
    cohort = np.asarray(cohort)

    groups = {
        g: np.where(cohort == g)[0]
        for g in sorted(set(cohort))
    }

    slopes = []

    for _ in range(n):
        idx = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True)
            for ix in groups.values()
        ])

        slopes.append(
            fixed_effect_slope(
                dist[idx],
                rank[idx],
                cohort[idx],
            )
        )

    return [
        float(np.quantile(slopes, 0.025)),
        float(np.quantile(slopes, 0.975)),
    ]


def stratified_permutation_p(
    dist,
    rank,
    cohort,
    observed,
    n=N_PERM,
):
    """
    Permute rank only WITHIN each context-length cohort.

    This tests distance dependence without allowing differences between
    16k/32k/64k cohorts themselves to create a false trend.
    """
    rng = np.random.default_rng(SEED)

    dist = np.asarray(dist)
    rank = np.asarray(rank)
    cohort = np.asarray(cohort)

    groups = {
        g: np.where(cohort == g)[0]
        for g in sorted(set(cohort))
    }

    hits = 0

    for _ in range(n):
        yp = rank.copy()

        for ix in groups.values():
            yp[ix] = rng.permutation(yp[ix])

        slope = fixed_effect_slope(
            dist,
            yp,
            cohort,
        )

        if abs(slope) >= abs(observed):
            hits += 1

    return (hits + 1) / (n + 1)


rows = load_rows()

layers = sorted(set(int(r["layer"]) for r in rows))

result = {
    "task": "Tracker row 87 — extended RULER retention range",
    "sources": {k: str(v) for k, v in SOURCES.items()},
    "analysis": (
        "Evicted examples only; ordered context; corrected Run-026 "
        "digit-sequence J-Lens readout. Linear distance effect estimated "
        "with context-length fixed effects; permutation shuffles ranks "
        "within 16k/32k/64k cohorts."
    ),
    "n_perm": N_PERM,
    "n_boot": N_BOOT,
    "layers": {},
}

print("=== ROW 87 LONG-CONTEXT RETENTION ===")

for L in layers:
    rs = [r for r in rows if int(r["layer"]) == L]

    dist = np.array(
        [r["eviction_distance"] for r in rs],
        dtype=float,
    )
    rank = np.array(
        [r["mean_digit_rank"] for r in rs],
        dtype=float,
    )
    cohort = np.array(
        [r["length_cohort"] for r in rs]
    )

    rho, rho_p = spearmanr(dist, rank)

    slope = fixed_effect_slope(
        dist,
        rank,
        cohort,
    )

    ci = stratified_bootstrap(
        dist,
        rank,
        cohort,
    )

    perm_p = stratified_permutation_p(
        dist,
        rank,
        cohort,
        slope,
    )

    by_cohort = {}

    for g in ["16k", "32k", "64k"]:
        gr = [
            r for r in rs
            if r["length_cohort"] == g
        ]

        ds = np.array(
            [r["eviction_distance"] for r in gr]
        )
        ys = np.array(
            [r["mean_digit_rank"] for r in gr]
        )

        by_cohort[g] = {
            "n": len(gr),
            "distance_range": [
                int(ds.min()),
                int(ds.max()),
            ],
            "median_distance":
                float(np.median(ds)),
            "median_rank":
                float(np.median(ys)),
            "mean_rank":
                float(np.mean(ys)),
        }

    if perm_p >= 0.05:
        verdict = (
            "no significant distance dependence "
            "over the extended range"
        )
    elif slope > 0:
        verdict = (
            "retention degrades with increasing "
            "eviction distance"
        )
    else:
        verdict = (
            "rank improves with increasing distance; "
            "opposite of decay"
        )

    result["layers"][str(L)] = {
        "n": len(rs),
        "full_distance_range": [
            int(dist.min()),
            int(dist.max()),
        ],
        "pooled_spearman_rho": float(rho),
        "pooled_spearman_p": float(rho_p),
        "fixed_effect_slope_rank_per_token":
            slope,
        "fixed_effect_slope_rank_per_10k_tokens":
            slope * 10000,
        "bootstrap_ci95_slope_per_token": ci,
        "stratified_permutation_p": float(perm_p),
        "verdict": verdict,
        "by_cohort": by_cohort,
    }

    print(f"\nL{L}")
    print(
        f"  n={len(rs)}  distance="
        f"{int(dist.min())}..{int(dist.max())}"
    )
    print(
        f"  pooled Spearman rho={rho:+.3f} "
        f"p={rho_p:.4g}"
    )
    print(
        f"  adjusted slope={slope*10000:+.1f} "
        f"rank / 10k tokens"
    )
    print(
        f"  bootstrap CI="
        f"[{ci[0]*10000:+.1f}, "
        f"{ci[1]*10000:+.1f}] / 10k"
    )
    print(
        f"  stratified permutation p="
        f"{perm_p:.4f}"
    )
    print("  =>", verdict)

    for g in ["16k", "32k", "64k"]:
        x = by_cohort[g]

        print(
            f"     {g}: n={x['n']:2d} "
            f"median_d={x['median_distance']:7.0f} "
            f"median_rank={x['median_rank']:8.1f}"
        )

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

print("\nsaved ->", OUT)
