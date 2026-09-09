"""CPU-only: build Figure 3 (RQ1 forest plot) straight from the Table 5 artefact.

PROPOSAL.md lists Figure 3 as "RQ1 forest plot", owned by `05_analysis_and_figures.ipynb`.
That notebook rebuilds Table 5 from `03_nowrite_reproduction.json` and then plots it; this
script skips the rebuild and reads the frozen Table 5 JSON, whose `delta_f1_vs_nowrite`
rows are already `[point, lo, hi]` percentile-bootstrap intervals.

    python build_fig3.py [results/run_3b_gdn]

Input:
    <run>/05_table5_rq1.json          (from notebook 05)
Output:
    results/figures/fig3_rq1_forest.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/run_3b_gdn"
SRC = os.path.join(RUN, "05_table5_rq1.json")
OUTDIR = "results/figures"
OUT = os.path.join(OUTDIR, "fig3_rq1_forest.png")

STRATA = ["short", "mid", "long"]
# (metric key in the artefact, label, colour, marker, filled?)
SERIES = [
    ("first_line", "first-line  (primary)", "crimson", "o", True),
    ("full_generation", "full generation  (robustness)", "#888", "s", False),
]


def main() -> None:
    with open(SRC) as fh:
        doc = json.load(fh)

    tables = doc["tables"]
    primary_metric = doc.get("primary_metric", "first_line")
    any_row = next(r for t in tables.values() for r in t)
    cell, scale = any_row["cell"], any_row["scale"]

    # y position per stratum; series are nudged off the shared row so CIs don't overlap.
    y_of = {s: i for i, s in enumerate(reversed(STRATA))}  # long at bottom
    nudge = 0.16

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    seen_any = False
    txt_lines = {s: [] for s in STRATA}  # right-margin numeric column

    for si, (mkey, label, colour, marker, filled) in enumerate(SERIES):
        rows = tables.get(mkey)
        if not rows:
            continue
        seen_any = True
        off = (si - (len(SERIES) - 1) / 2) * nudge
        by_stratum = {r["stratum"]: r for r in rows}
        ys, pts, los, his = [], [], [], []
        for s in STRATA:
            r = by_stratum.get(s)
            if r is None:
                continue
            p, lo, hi = r["delta_f1_vs_nowrite"]
            ys.append(y_of[s] + off)
            pts.append(p)
            los.append(p - lo)
            his.append(hi - p)
            crosses = "" if lo <= 0 <= hi else "  *"
            txt_lines[s].append((colour, f"{label.split('  ')[0]:>16}: "
                                         f"{p:+.3f}  [{lo:+.3f}, {hi:+.3f}]{crosses}"))
        ax.errorbar(
            pts, ys, xerr=[los, his], fmt=marker, capsize=4,
            color=colour, markerfacecolor=colour if filled else "white",
            markeredgecolor=colour, label=label, lw=1.5, zorder=3,
        )

    if not seen_any:
        raise SystemExit(f"no usable rows in {SRC}")

    ax.axvline(0, c="k", lw=1, zorder=1)
    ax.set_yticks([y_of[s] for s in STRATA])
    ax.set_yticklabels([f"{s}\n(n={tables[primary_metric][STRATA.index(s)]['n']})"
                        for s in STRATA])
    ax.set_ylim(-0.6, len(STRATA) - 0.4)
    ax.set_xlim(-0.28, 0.28)
    ax.set_xlabel(r"$\Delta$F1  (AHN $-$ NOWRITE)     ( * = 95% CI excludes 0 )")
    ax.set_title(f"Figure 3 — RQ1 effect sizes  ·  AHN-{cell} {scale}\n"
                 f"length-stratified, {primary_metric.replace('_', ' ')} scoring is primary")
    ax.grid(alpha=.3, axis="x")
    ax.legend(fontsize=8, loc="lower left", framealpha=.9)

    # numeric column to the right of the axes
    for s in STRATA:
        base = y_of[s]
        for j, (colour, line) in enumerate(txt_lines[s]):
            ax.text(1.02, base + 0.18 - j * 0.30, line, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.5, family="monospace", color=colour)

    os.makedirs(OUTDIR, exist_ok=True)
    fig.subplots_adjust(right=0.62)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
