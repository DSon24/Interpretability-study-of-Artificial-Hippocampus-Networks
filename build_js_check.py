"""Row 80: Gautam's headline correlations on the chat-format data, per cell.

CPU-only. rho(boundary JS, answer changed) and rho(boundary JS, F1), from
results/run_3b_<cell>/03_nowrite_reproduction.json (Qwen chat template, n=60). Same
statistics as build_table8.py (Spearman, paired percentile bootstrap, seed 20260820).

Which "F1" Gautam means is unconfirmed (protocol-alignment row 12), so all three are
reported: per-example dF1 (AHN - NOWRITE), F1 under NOWRITE, and F1 under AHN. "Answer
changed" is reported under the headline first-line normalised-EM definition and under
raw full-generation string inequality. Prior work (Kashyap 2026, via the proposal):
rho(JS, changed) 0.34-0.41, rho(JS, F1) -0.09 to 0.00. No tuning toward those numbers.

    python build_js_check.py    ->  results/05_rq3_js_gautam_check.json
"""
import json

import numpy as np
from build_table8 import CELLS, spearman

TARGETS = [
    ("rho(JS, answer changed) - first-line normalised EM (headline)", "answer_changed_fl", [0.34, 0.41]),
    ("rho(JS, answer changed) - full-generation", "answer_changed", [0.34, 0.41]),
    ("rho(JS, dF1)  [AHN - NOWRITE, first-line]", "delta_f1_fl", [-0.09, 0.0]),
    ("rho(JS, F1 under NOWRITE, first-line)", "f1_nowrite_fl", [-0.09, 0.0]),
    ("rho(JS, F1 under AHN, first-line)", "f1_ahn_fl", [-0.09, 0.0]),
]


def main():
    out = {"method": "Spearman, paired percentile bootstrap n_boot=10000, seed 20260820",
           "prompt_format": "qwen_chat_template", "prior_work": {"changed": [0.34, 0.41], "F1": [-0.09, 0.0]},
           "cells": {}}
    for cell, key in CELLS.items():
        per = json.load(open(f"results/run_3b_{key}/03_nowrite_reproduction.json"))["per_example"]
        assert len(per) == 60
        js = [r["boundary_js"] for r in per]
        rows = []
        for label, field, band in TARGETS:
            y = [float(r[field]) for r in per]
            s = spearman(js, y)
            rows.append({"quantity": label, "field": field, "prior_work_band": band, **s})
        out["cells"][cell] = rows
        print(f"\n{cell}")
        for r in rows:
            print(f"  {r['quantity'][:62]:62s} rho={r['rho']:+.3f} p={r['p']:.4f} "
                  f"CI=[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  prior {r['prior_work_band']}")
    json.dump(out, open("results/05_rq3_js_gautam_check.json", "w"), indent=2)
    print("\nsaved results/05_rq3_js_gautam_check.json")


if __name__ == "__main__":
    main()
