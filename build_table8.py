"""Table 8 (RQ3): Spearman correlation of J-lens retention readouts with per-example dF1.

CPU-only. Reads results/run_3b_<cell>/04b_joined_retention_task.json (chat-template 04b,
modal_04b.py) and the cell's 03_nowrite_reproduction.json (boundary JS). Same statistics
as `ahn_interp.spearman` (Spearman rho, paired percentile bootstrap CI, n_boot=10000,
seed 20260820) plus a Holm correction over the three pre-registered predictors.

    python build_table8.py

Writes results/run_3b_<cell>/05_table8_rq3.json (same row schema as before, plus holm_p)
and results/05_table8_rq3_crosscell.json (all cells, plus layer-wise and full-generation
sensitivity rows).
"""
import json

import numpy as np
from scipy import stats

SEED, N_BOOT = 20260820, 10000
CELLS = {"GatedDeltaNet": "gdn", "DeltaNet": "dn", "Mamba2": "m2"}
PRIMARY = [("target rank @ prompt end", "rank_at_prompt_end"),
           ("target mass @ prompt end", "p_mem_at_prompt_end"),
           ("readout entropy @ prompt end", "entropy_at_prompt_end")]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    idx = np.random.default_rng(SEED).integers(0, x.size, size=(N_BOOT, x.size))
    b = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    b = b[np.isfinite(b)]
    lo, hi = float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
    return {"rho": float(rho), "p": float(p), "n": int(x.size), "ci": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    allout = {}
    for cell, key in CELLS.items():
        d = f"results/run_3b_{key}"
        j = json.load(open(f"{d}/04b_joined_retention_task.json"))
        assert j.get("prompt_format") == "qwen_chat_template", f"{key}: 04b is not the chat-template run"
        rows = j["rows"]
        assert len(rows) == 60 and j["cell"] == cell
        js = {str(r["id"]): r["boundary_js"] for r in
              json.load(open(f"{d}/03_nowrite_reproduction.json"))["per_example"]}
        y = [r["delta_f1"] for r in rows]
        table = [{"predictor": "retention half-life", "outcome": "per-example dF1",
                  "rho": None, "p": None, "ci": None, "n": 0,
                  "note": "no per-example half-life from a single task prompt"}]
        prim = []
        for name, f in PRIMARY:
            t = {"predictor": name, "outcome": "per-example dF1", **spearman([r[f] for r in rows], y)}
            prim.append(t)
        for t, a in zip(prim, holm([t["p"] for t in prim])):
            t["holm_p"] = float(a)
        table += prim
        table.append({"predictor": "boundary JS divergence", "outcome": "per-example dF1",
                      **spearman([js[str(r["id"])] for r in rows], y),
                      "prior_work_range": [-0.09, 0.0]})
        json.dump(table, open(f"{d}/05_table8_rq3.json", "w"), indent=2)

        layer = []
        for L in ("9", "18", "27"):
            for oname, outcome in (("dF1 first-line", "delta_f1"),
                                   ("dF1 full-generation", "delta_f1_full_generation")):
                layer.append({"layer": int(L), "outcome": oname,
                              **spearman([r["layers"][L]["rank"] for r in rows],
                                         [r[outcome] for r in rows])})
        for t, a in zip([l for l in layer if l["outcome"] == "dF1 first-line"],
                        holm([l["p"] for l in layer if l["outcome"] == "dF1 first-line"])):
            t["holm_p_over_layers"] = float(a)
        allout[cell] = {"primary_layer18": table, "rank_by_layer": layer}
        print(f"\n{cell} (n=60)")
        for t in table[1:]:
            print(f"  {t['predictor']:32s} rho={t['rho']:+.3f} p={t['p']:.4f} "
                  f"CI=[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}]" + (f" holm={t['holm_p']:.3f}" if 'holm_p' in t else ""))
        for l in layer:
            if l["outcome"] == "dF1 first-line":
                print(f"  rank L{l['layer']:<2d} rho={l['rho']:+.3f} p={l['p']:.4f} holm={l['holm_p_over_layers']:.3f}")
    json.dump({"table": 8, "prompt_format": "qwen_chat_template", "n_boot": N_BOOT, "seed": SEED,
               "cells": allout}, open("results/05_table8_rq3_crosscell.json", "w"), indent=2)


if __name__ == "__main__":
    main()
