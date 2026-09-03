#!/usr/bin/env python3
"""
Extract the baseline-corrected C2 results from notebooks/04-C2-debug.ipynb into JSON.

WHY THIS EXISTS
---------------
Sơn's C2 investigation ran the pair-baseline correction across the full design and
per layer, but the results live only as printed cell output inside the notebook.
Nothing under results/ holds them, so Tables 4 and 8 would have to be transcribed by
hand into the paper. This parses the saved outputs instead.

It does NOT recompute anything. Regenerating the underlying numbers requires re-running
the GPU sweep in cells 40-52 (2 x 84 forward passes on the merged 3B checkpoint), which
is why the source-of-record stays the notebook and this is an extraction step.

    python extract_c2_corrected.py
"""

import json
import re
import subprocess

NB = "notebooks/04-C2-debug.ipynb"
OUT = "results/run_3b_gdn/04d_c2_baseline_corrected.json"


def cell_text(nb, i):
    t = ""
    for o in nb["cells"][i].get("outputs", []):
        t += "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
    return t


def all_output(nb):
    return "\n".join(cell_text(nb, i) for i in range(len(nb["cells"])))


def parse_robustness(txt):
    """The FINAL ROBUSTNESS SUMMARY block, one stanza per layer."""
    out = {}
    for m in re.finditer(
        r"LAYER (\d+)\s*\n"
        r"\s*Main corrected fold:\s*([\d.]+)x\s*\n"
        r"\s*Main 95% CI:\s*\[([\d.]+), ([\d.]+)\]\s*\n"
        r"\s*Main t-test p:\s*([\d.eE+-]+)\s*\n"
        r"\s*Leave-distance folds:\s*([\d.]+)x to ([\d.]+)x\s*\n"
        r"\s*Leave-filler folds:\s*([\d.]+)x to ([\d.]+)x\s*\n"
        r"\s*Per-needle folds:\s*([\d.]+)x to ([\d.]+)x\s*\n"
        r"\s*Block permutation p:\s*([\d.eE+-]+)", txt):
        L, fold, lo, hi, p, dlo, dhi, flo, fhi, nlo, nhi, pp = m.groups()
        out[L] = {
            "corrected_fold": float(fold),
            "ci_95": [float(lo), float(hi)],
            "t_test_p": float(p),
            "leave_one_distance_out_fold_range": [float(dlo), float(dhi)],
            "leave_one_filler_out_fold_range": [float(flo), float(fhi)],
            "per_needle_fold_range": [float(nlo), float(nhi)],
            "blocked_permutation_p": float(pp),
        }
    return out


def parse_scrambled(txt):
    """The C2 #6 scrambled-needle content control table."""
    blk = txt.split("C2 #6 SCRAMBLED-NEEDLE CONTENT CONTROL")[-1]
    out = {}
    for m in re.finditer(
        r"^\s*(9|18|27)\s+(\d+)\s+(-?[\d.eE+-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.eE+-]+)\s*$",
        blk, re.M):
        L, n, mlog, fold, lo, hi, t, p = m.groups()
        if L in out:
            continue
        out[L] = {"n_conditions": int(n), "geom_fold": float(fold),
                  "ci_95": [float(lo), float(hi)], "p": float(p)}
    return out


def parse_raw(txt):
    """Raw (uncorrected) per-layer C2, from the earlier diagnostic cell."""
    blk = txt.split("=== C2 per layer, corrected epsilon")[-1]
    out = {}
    for m in re.finditer(
        r"layer (\d+): n=(\d+)\s+median_ratio=([\d.]+)\s+"
        r"frac_needle>distractor=([\d.]+)\s+frac_pass_10x=([\d.]+)", blk):
        L, n, r, w, p10 = m.groups()
        out[L] = {"n": int(n), "raw_median_ratio": float(r),
                  "needle_win_rate": float(w), "frac_pass_10x": float(p10)}
    return out


def main():
    nb = json.load(open(NB))
    txt = all_output(nb)
    rob, scr, raw = parse_robustness(txt), parse_scrambled(txt), parse_raw(txt)
    assert set(rob) == {"9", "18", "27"}, f"robustness parse got {sorted(rob)}"
    assert set(scr) == {"9", "18", "27"}, f"scrambled parse got {sorted(scr)}"
    assert set(raw) == {"9", "18", "27"}, f"raw parse got {sorted(raw)}"

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    layers = {}
    for L in ("9", "18", "27"):
        c = rob[L]
        survives = (c["ci_95"][0] > 1.0 or c["ci_95"][1] < 1.0) and c["blocked_permutation_p"] < 0.05
        layers[L] = {
            "raw": raw[L],
            "baseline_corrected": c,
            "scrambled_needle_control": scr[L],
            "corrected_effect_survives": bool(survives),
            "survives_scrambled_control": bool(
                scr[L]["ci_95"][0] > 1.0 or scr[L]["ci_95"][1] < 1.0),
        }

    json.dump({
        "note": "Baseline-corrected C2, per layer. EXTRACTED from notebook cell outputs, "
                "not recomputed. Regenerating needs the GPU sweep in 04-C2-debug.ipynb "
                "cells 40-52 (2 x 84 forward passes on the merged 3B checkpoint).",
        "source_notebook": NB,
        "extracted_at_commit": sha,
        "design": "3 layers x 7 eviction distances x 3 fillers x 4 needle/distractor pairs "
                  "= 252 matched observations, aggregated to 21 conditions per layer",
        "correction": "each pair's needle/distractor log-ratio when that needle is stored, "
                      "minus the same pair's mean log-ratio when the other three are stored",
        "lens_validated": False,
        "layers": layers,
    }, open(OUT, "w"), indent=2)

    print(f"{'layer':>6} {'raw':>9} {'corrected':>11} {'95% CI':>18} {'perm p':>9} "
          f"{'scrambled':>11} {'verdict':>26}")
    for L in ("9", "18", "27"):
        d = layers[L]
        c, s = d["baseline_corrected"], d["scrambled_needle_control"]
        verdict = ("effect survives control" if d["corrected_effect_survives"] and d["survives_scrambled_control"]
                   else "fails scrambled control" if d["corrected_effect_survives"]
                   else "no corrected effect")
        print(f"{L:>6} {d['raw']['raw_median_ratio']:>8.2f}x {c['corrected_fold']:>10.4f}x "
              f"[{c['ci_95'][0]:.4f}, {c['ci_95'][1]:.4f}] {c['blocked_permutation_p']:>9.4f} "
              f"{s['geom_fold']:>10.4f}x {verdict:>26}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
