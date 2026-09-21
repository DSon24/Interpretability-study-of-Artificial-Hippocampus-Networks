"""CPU-only: build Figure 3 (RQ1 forest plot) for all three cells from the nb03 artefacts.

Rebuilds each cell's `05_table5_rq1.json` (same schema notebook 05 writes: first-line
primary, full-generation robustness, percentile-bootstrap CI per length stratum) from
`results/run_3b_<cell>/03_nowrite_reproduction.json`, then plots the three cells together.
Pooled paired cross-cell contrasts live in `build_table5_crosscell.py`.

    python build_fig3.py

Output:
    results/run_3b_{gdn,dn,m2}/05_table5_rq1.json
    results/figures/fig3_rq1_forest.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = [("GatedDeltaNet", "gdn", "#c0392b", "o"),
         ("DeltaNet", "dn", "#2471a3", "s"),
         ("Mamba2", "m2", "#239b56", "^")]
STRATA = ["short", "mid", "long", "all"]
METRICS = [("first_line", "_fl", True), ("full_generation", "", False)]
N_BOOT, SEED = 4000, 20260820
OUT = "results/figures/fig3_rq1_forest.png"


def boot_ci(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def build_table5(per, cell_name, suffix, name, primary):
    rows = []
    for s in ("short", "mid", "long"):
        rs = [r for r in per if r["stratum"] == s]
        rows.append({
            "scale": "3B", "cell": cell_name, "stratum": s, "n": len(rs),
            "metric": name, "primary": primary,
            "mean_f1": float(np.mean([r[f"f1_ahn{suffix}"] for r in rs])),
            "delta_f1_vs_nowrite": boot_ci([r[f"delta_f1{suffix}"] for r in rs]),
            "answer_change_rate": float(np.mean([r[f"answer_changed{suffix}"] for r in rs])),
        })
    return rows


def main():
    tables = {}
    for cell_name, key, *_ in CELLS:
        per = json.load(open(f"results/run_3b_{key}/03_nowrite_reproduction.json"))["per_example"]
        assert len(per) == 60
        tabs = {n: build_table5(per, cell_name, sfx, n, p) for n, sfx, p in METRICS}
        json.dump({"primary_metric": "first_line", "tables": tabs},
                  open(f"results/run_3b_{key}/05_table5_rq1.json", "w"), indent=2)
        pooled = {n: boot_ci([r[f"delta_f1{sfx}"] for r in per]) for n, sfx, _ in METRICS}
        tables[cell_name] = (tabs, pooled, len(per))

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    y_of = {s: i for i, s in enumerate(reversed(STRATA))}
    nudge = 0.2
    for ci_, (cell_name, key, colour, marker) in enumerate(CELLS):
        tabs, pooled, n_all = tables[cell_name]
        off = (ci_ - 1) * nudge
        pts = {r["stratum"]: r["delta_f1_vs_nowrite"] for r in tabs["first_line"]}
        pts["all"] = pooled["first_line"]
        ys = [y_of[s] + off for s in STRATA]
        p = [pts[s][0] for s in STRATA]
        ax.errorbar(p, ys, xerr=[[pts[s][0] - pts[s][1] for s in STRATA],
                                 [pts[s][2] - pts[s][0] for s in STRATA]],
                    fmt=marker, capsize=3, color=colour, lw=1.4, label=cell_name, zorder=3)
    ax.axvline(0, c="k", lw=1, zorder=1)
    ax.set_yticks([y_of[s] for s in STRATA])
    ax.set_yticklabels([f"{s}\n(n={60 if s == 'all' else 20})" for s in STRATA])
    ax.set_ylim(-0.6, len(STRATA) - 0.4)
    ax.set_xlabel(r"$\Delta$F1  (AHN $-$ NOWRITE),  95% percentile-bootstrap CI")
    ax.set_title("Figure 3 — RQ1 effect sizes by recurrent cell · 3B, Qwen chat template\n"
                 "first-line scoring (full-generation differs by a constant 0.5 pt, one NOWRITE example)")
    ax.grid(alpha=.3, axis="x")
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for c, (tabs, pooled, _) in tables.items():
        print(c, {k: [round(100 * v, 2) for v in val] for k, val in pooled.items()})
    print("saved", OUT)


if __name__ == "__main__":
    main()
