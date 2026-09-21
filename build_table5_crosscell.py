"""Table 5 cross-cell RQ1 comparison: paired bootstrap / sign-flip permutation / Holm.

CPU-only. Reads results/run_3b_{gdn,dn,m2}/<input>. Method matches the 18 Sep artefact
(results/05_table5_rq1_crosscell.json): mean paired difference in per-example
AHN-minus-NOWRITE delta F1, 10000-example paired bootstrap CI, 10000-draw paired
sign-flip permutation test, Holm across the 3 pairwise contrasts, seed 20260820.

    python build_table5_crosscell.py --score delta_f1_fl --out results/05_table5_rq1_crosscell.json
    python build_table5_crosscell.py --score delta_f1 --out results/05_table5_rq1_crosscell_official.json
"""
import argparse, json
import numpy as np

CELLS = {"GatedDeltaNet": "gdn", "DeltaNet": "dn", "Mamba2": "m2"}
SEED, N_BOOT, N_PERM = 20260820, 10000, 10000


def load(cell, name, score):
    rows = json.load(open(f"results/run_3b_{cell}/{name}"))["per_example"]
    return {r["id"]: float(r[score]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="delta_f1_fl")
    ap.add_argument("--input", default="03_nowrite_reproduction.json")
    ap.add_argument("--input-dn", default=None)
    ap.add_argument("--input-m2", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = {"gdn": a.input, "dn": a.input_dn or a.input, "m2": a.input_m2 or a.input}
    data = {c: load(k, names[k], a.score) for c, k in CELLS.items()}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    assert len(ids) == 60, f"paired ids = {len(ids)}"
    v = {c: np.array([d[i] for i in ids]) for c, d in data.items()}
    rng = np.random.default_rng(SEED)
    within = {c: {"delta_f1_points": round(100 * float(x.mean()), 2)} for c, x in v.items()}
    pw, raw_p = {}, {}
    for a_, b_ in [("DeltaNet", "GatedDeltaNet"), ("DeltaNet", "Mamba2"), ("GatedDeltaNet", "Mamba2")]:
        diff = v[a_] - v[b_]
        obs = diff.mean()
        boots = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(1)
        ci = np.percentile(boots, [2.5, 97.5])
        signs = rng.choice([-1, 1], (N_PERM, len(diff)))
        p = float((np.sum(np.abs((signs * diff).mean(1)) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1))
        key = f"{a_}_vs_{b_}"
        pw[key] = {"difference_f1_points": round(100 * float(obs), 2),
                   "ci95": [round(100 * float(ci[0]), 2), round(100 * float(ci[1]), 2)],
                   "permutation_p": round(p, 4)}
        raw_p[key] = p
    order = sorted(raw_p, key=raw_p.get)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * raw_p[k]))
        pw[k]["holm_p"] = round(run, 4)
    out = {"analysis": "RQ1 paired cross-cell comparison", "n_examples": len(ids),
           "paired_examples": True, "score_field": a.score, "inputs": names, "seed": SEED,
           "method": {"effect": "mean paired difference in AHN-minus-NOWRITE delta F1",
                      "confidence_interval": f"{N_BOOT}-example paired bootstrap",
                      "hypothesis_test": f"{N_PERM}-draw paired sign-flip permutation test",
                      "multiple_testing": "Holm correction across 3 pairwise comparisons"},
           "within_cell": within, "pairwise": pw}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({"within": within, "pairwise": pw}, indent=1))


if __name__ == "__main__":
    main()
