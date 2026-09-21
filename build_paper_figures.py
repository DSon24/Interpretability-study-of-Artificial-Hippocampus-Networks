"""CPU-only: build Figures 1, 2, 4, 5, 6, 7, 8 of the rev4 artefact set.

Figure 3 (RQ1 forest) is built by build_fig3.py. Every figure here is drawn from
committed result JSONs; none uses synthetic data. Sources are printed per figure.

    python build_paper_figures.py

Output: results/figures/fig{1,2,4,5,6,7,8}_*.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUTDIR = "results/figures"
CELLS = [("GatedDeltaNet", "gdn", "#c0392b"), ("DeltaNet", "dn", "#2471a3"), ("Mamba2", "m2", "#239b56")]
LAYERS = [9, 18, 27]
CHANCE = 75968.0
UNIFORM_ENTROPY = 11.931214658529285
ANALYSIS_LAYER = 27


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def stats(key):
    return json.load(open(f"results/run_3b_{key}/04i_ruler_controls_stats.json"))["layers"]


def ruler_rows(key, layer=ANALYSIS_LAYER, condition="ordered", evicted=True):
    rows = json.load(open(f"results/run_3b_{key}/04i_ruler_controls_rows.json"))["rows"]
    return [r for r in rows if r["layer"] == layer and r["condition"] == condition
            and bool(r["needle_is_evicted"]) == evicted]


# ---------------------------------------------------------------- Figure 1
def fig1():
    """Schematic: what one row of raw data is. No data; this is the unit-of-analysis diagram."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    # context bar
    ax.text(2, 46.5, "One long-context example", fontsize=10, weight="bold")
    ax.add_patch(Rectangle((2, 36), 4, 6, fc="#f0c419", ec="k", lw=.8))
    ax.text(4, 43.2, "sinks", ha="center", fontsize=7.5)
    ax.add_patch(Rectangle((6, 36), 26, 6, fc="#dfe6e9", ec="k", lw=.8))
    ax.text(20.5, 39, "evicted region — past the window", ha="center", va="center", fontsize=8.5)
    ax.add_patch(Rectangle((32, 36), 12, 6, fc="#b2eabf", ec="k", lw=.8))
    ax.text(38, 39, "sliding window\n8064", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((11.5, 36), 1.6, 6, fc="crimson", ec="k", lw=.8))
    ax.text(12.3, 43.2, "needle $y^*$", ha="center", fontsize=8, color="crimson")
    ax.annotate("", xy=(12.3, 35.2), xytext=(32, 35.2),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.1))
    ax.text(22.2, 32.8, "eviction distance", ha="center", fontsize=8, color="crimson")

    # model stack
    ax.add_patch(Rectangle((52, 22), 11, 22, fc="#f5f6fa", ec="k", lw=1))
    ax.text(57.5, 46.5, "AHN model", ha="center", fontsize=9.5, weight="bold")
    for i, (L, y) in enumerate(zip(LAYERS, [26, 32, 38])):
        ax.add_patch(Rectangle((52.6, y), 9.8, 3.4, fc="#dff1fb", ec="#2471a3", lw=.9))
        ax.text(57.5, y + 1.7, f"layer {L}   $o_t$", ha="center", va="center", fontsize=8.5)
        ax.add_patch(FancyArrowPatch((62.6, y + 1.7), (71, y + 1.7),
                                     arrowstyle="->", mutation_scale=11, color="#2471a3", lw=1))
    ax.add_patch(FancyArrowPatch((44.4, 39), (51.6, 39), arrowstyle="->", mutation_scale=13, color="k", lw=1.2))

    # readout
    ax.add_patch(Rectangle((71, 24), 26, 18, fc="#fff8e7", ec="#b8860b", lw=1))
    ax.text(84, 46.5, "J-lens readout (vocabulary space)", ha="center", fontsize=9.5, weight="bold")
    for txt, y in [("rank($y^*$)  — where the answer sits", 38.2),
                   ("$P_{mem}(y^*)$  — mass on the answer", 34.6),
                   ("readout entropy — blank vs confident", 31.0),
                   ("task F1, and F1 under NOWRITE", 27.4)]:
        ax.text(72.4, y, "•  " + txt, fontsize=8.6, va="center")

    ax.add_patch(Rectangle((2, 4), 95, 13, fc="#eef4ee", ec="#2d6a4f", lw=1))
    ax.text(49.5, 14.6, "One row of raw data", ha="center", fontsize=10, weight="bold", color="#2d6a4f")
    ax.text(49.5, 10.8, "(example id, cell, scale, layer, eviction distance, "
                        "$P_{mem}(y^*)$, rank($y^*$), readout entropy, task F1, F1 under NOWRITE)",
            ha="center", fontsize=9, family="monospace")
    ax.text(49.5, 6.6, "Every table and figure in the paper is an aggregation over these rows. "
                       "Because task score and readout come from the same forward pass,\n"
                       "RQ3 is a within-example paired comparison — which is what makes n=60 adequate.",
            ha="center", fontsize=8.4, style="italic")
    ax.set_title("Figure 1 — The unit of analysis", fontsize=12, weight="bold", loc="left")
    save(fig, "fig1_unit_of_analysis.png")


# ---------------------------------------------------------------- Figure 2
def fig2():
    """C1 / C2 / C3 at the analysis layer, all three cells. Never-cut validation figure."""
    S = {n: stats(k)[str(ANALYSIS_LAYER)] for n, k, _ in CELLS}
    names = [n for n, _, _ in CELLS]
    cols = [c for _, _, c in CELLS]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    v = [S[n]["C1_sequence"]["median_mean_digit_rank"] for n in names]
    lo = [S[n]["C1_sequence"]["ci95"][0] for n in names]
    hi = [S[n]["C1_sequence"]["ci95"][1] for n in names]
    ax.bar(x, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], capsize=4, color=cols)
    ax.axhline(CHANCE, ls="--", c="k", alpha=.8)
    ax.text(len(names) - .45, CHANCE, " chance", va="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("median mean-digit-rank of the answer")
    ax.set_title("C1 — evicted-needle readout\nbelow chance = memory carries the answer", fontsize=10)
    ax.invert_yaxis(); ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    v = [S[n]["C2_cross_example"]["effect_per_example"] for n in names]
    lo = [S[n]["C2_cross_example"]["ci95_effect_per_example"][0] for n in names]
    hi = [S[n]["C2_cross_example"]["ci95_effect_per_example"][1] for n in names]
    ax.errorbar(x, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o", capsize=4, color="k", zorder=3)
    ax.axhline(1.0, ls="--", c="grey"); ax.text(len(names) - .45, 1.0, " null", va="center", fontsize=8.5)
    ax.axhline(10.0, ls="-.", c="crimson")
    ax.text(len(names) - .45, 10.0, " pre-registered 10x bar", va="center", fontsize=8.5, color="crimson")
    for xi, n in zip(x, names):
        p = S[n]["C2_cross_example"]["permutation_p"]
        ax.annotate("p<1e-4" if p < 1e-4 else f"p={p:.3g}", (xi, S[n]["C2_cross_example"]["effect_per_example"]),
                    textcoords="offset points", xytext=(13, 2), ha="left", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(0.4, 20)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("p(stored answer) / p(distractor)")
    ax.set_title("C2 — distractor\n>1 = selective; the 10x bar is never met", fontsize=10)
    ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    v = [S[n]["C3_context"]["median_delta_shuffled_minus_ordered"] for n in names]
    lo = [S[n]["C3_context"]["ci95_delta"][0] for n in names]
    hi = [S[n]["C3_context"]["ci95_delta"][1] for n in names]
    ax.errorbar(x, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="s", capsize=4, color="k", zorder=3)
    ax.axhline(0, ls="--", c="grey")
    ax.text(len(names) - .45, 0, " no order effect", va="center", fontsize=8.5)
    for xi, n in zip(x, names):
        ax.annotate("order-sensitive" if S[n]["C3_context"]["order_sensitive"] else "not sensitive",
                    (xi, v[names.index(n)]), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=7.5)
    ax.set_ylim(min(lo) * 1.25, max(hi) * 1.45)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Δ median rank (shuffled − ordered)")
    ax.set_title("C3 — shuffled context\nrank should worsen if order matters", fontsize=10)
    ax.grid(alpha=.3, axis="y")

    fig.suptitle(f"Figure 2 — The three primary controls, RULER NIAH n=60 (32 evicted), "
                 f"layer {ANALYSIS_LAYER}, 3B   ·   lens_validated = False",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_primary_controls.png")


# ---------------------------------------------------------------- Figure 4
def fig4():
    """Retention vs eviction distance. Rev4 expects exponential decay (straight line on log y)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    ax = axes[0]
    for name, key, col in CELLS:
        r = ruler_rows(key)
        d = np.array([x["eviction_distance"] for x in r], float)
        y = np.array([x["answer_logprob"] for x in r], float)
        o = np.argsort(d)
        ax.scatter(d, y, s=22, alpha=.55, color=col, label=name, edgecolors="none")
        b, a = np.polyfit(d, y, 1)
        ax.plot(d[o], a + b * d[o], color=col, lw=1.6)
    ax.set_xlabel("eviction distance (tokens past the compression boundary)")
    ax.set_ylabel("log P(answer digit sequence)   [nats]")
    ax.set_title(f"A · RULER NIAH, layer {ANALYSIS_LAYER}, evicted needles (n=32/cell)\n"
                 "y is already a log probability: exponential decay = a downward straight line", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    rows = json.load(open("results/run_3b_gdn/04_retention_rows.json"))["rows"]
    for L, m in zip(LAYERS, ["o", "s", "^"]):
        sel = [r for r in rows if r["layer"] == L and not r["shuffled"] and not r["in_window"]]
        by = {}
        for r in sel:
            by.setdefault(r["requested_distance"], []).append(r["p_mem"])
        ds = sorted(by)
        med = [float(np.median(by[d])) for d in ds]
        ax.plot(ds, med, marker=m, lw=1.5, label=f"layer {L}")
    ax.set_yscale("log")
    ax.set_xlabel("requested eviction distance (tokens)")
    ax.set_ylabel("median $P_{mem}(y^*)$")
    ax.set_title("B · Homemade single-token needle sweep, GatedDeltaNet only\n"
                 "the only controlled distance sweep in the project (supporting analysis)", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")

    fig.suptitle("Figure 4 — Does retention decay with eviction distance?   No decay is visible in either cohort.",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig4_retention_decay.png")


# ---------------------------------------------------------------- Figure 5
def fig5():
    """Rank vs distance — the interpretable version of Figure 4."""
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for name, key, col in CELLS:
        r = ruler_rows(key)
        d = np.array([x["eviction_distance"] for x in r], float)
        y = np.array([x["mean_digit_rank"] for x in r], float)
        ax.scatter(d, y, s=26, alpha=.6, color=col, label=name, edgecolors="none")
        b, a = np.polyfit(d, y, 1)
        o = np.argsort(d)
        ax.plot(d[o], a + b * d[o], color=col, lw=1.6)
    ax.axhline(CHANCE, ls="--", c="k", alpha=.8)
    ax.text(ax.get_xlim()[1], CHANCE, " chance (75,968)", va="bottom", ha="right", fontsize=9)
    ax.axhline(1000, ls="-.", c="crimson", alpha=.9)
    ax.text(ax.get_xlim()[1], 1000, " top-1000 — the readable-answer threshold", va="bottom", ha="right",
            fontsize=9, color="crimson")
    ax.set_yscale("log")
    ax.set_xlabel("eviction distance (tokens past the compression boundary)")
    ax.set_ylabel("mean rank of the answer digits  (lower = stronger)")
    ax.set_title(f"Figure 5 — Where does the answer actually sit?   RULER NIAH, layer {ANALYSIS_LAYER}, "
                 f"evicted needles\nNo example is close to the top-1000 line at any distance.",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")
    save(fig, "fig5_rank_vs_distance.png")


# ---------------------------------------------------------------- Figure 6
def fig6():
    """Layer × cell heatmap, with the control verdicts overlaid."""
    M = np.zeros((len(CELLS), len(LAYERS)))
    note = [["" for _ in LAYERS] for _ in CELLS]
    for i, (name, key, _) in enumerate(CELLS):
        S = stats(key)
        for j, L in enumerate(LAYERS):
            s = S[str(L)]
            M[i, j] = s["C1_sequence"]["median_mean_digit_rank"] / CHANCE
            ok = s["C3_lens"]["signal_collapses"] and s["C4_layer_permutation"]["map_is_layer_specific"]
            note[i][j] = ("real readout" if ok else "decoding artefact")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels([f"layer {L}" for L in LAYERS])
    ax.set_yticks(range(len(CELLS))); ax.set_yticklabels([n for n, _, _ in CELLS])
    for i in range(len(CELLS)):
        for j in range(len(LAYERS)):
            dark = M[i, j] < 0.3 or M[i, j] > 1.3
            fg = "white" if dark else "#111"
            ax.text(j, i - .13, f"{M[i, j]:.2f}x chance", ha="center", va="center",
                    fontsize=10, weight="bold", color=fg)
            ax.text(j, i + .17, note[i][j], ha="center", va="center", fontsize=8.5,
                    style="italic", color=fg)
    fig.colorbar(im, ax=ax, label="median answer rank ÷ chance   (<1 = better than chance)")
    ax.set_title("Figure 6 — Is retention concentrated in particular layers?\n"
                 "Colour is the C1 readout; the label says whether C3-lens and C4 accept it as a real readout",
                 fontsize=10.5, weight="bold")
    fig.tight_layout()
    save(fig, "fig6_layer_heatmap.png")


# ---------------------------------------------------------------- Figure 7
def fig7():
    """Entropy: blank memory (entropy -> uniform) vs confidently wrong (entropy low, rank collapsed)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    c = json.load(open("results/run_3b_gdn/04c_per_layer_controls.json"))["layers"]
    v = [c[str(L)]["d_resid_preregistered"]["readout_entropy_median_nats"] for L in LAYERS]
    ax.bar([f"layer {L}" for L in LAYERS], v, color="#2471a3", width=.55)
    ax.axhline(UNIFORM_ENTROPY, ls="--", c="crimson")
    ax.text(2.4, UNIFORM_ENTROPY, " uniform (11.93)", va="bottom", ha="right", fontsize=9, color="crimson")
    for i, y in enumerate(v):
        ax.annotate(f"{y:.2f}", (i, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylim(0, 13.2); ax.set_ylabel("median readout entropy (nats)")
    ax.set_title("A · GatedDeltaNet, homemade needle cohort\n"
                 "layer 9 is degenerate — confident, and wrong", fontsize=9.5)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    data, labels, cols = [], [], []
    for name, key, col in CELLS:
        rows = json.load(open(f"results/run_3b_{key}/04b_joined_retention_task.json"))["rows"]
        data.append([r["entropy_at_prompt_end"] for r in rows]); labels.append(name); cols.append(col)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(.55)
    ax.axhline(UNIFORM_ENTROPY, ls="--", c="crimson")
    ax.text(3.45, UNIFORM_ENTROPY, " uniform", va="bottom", ha="right", fontsize=9, color="crimson")
    ax.set_ylabel("readout entropy at prompt end (nats)")
    ax.set_title("B · LongBench-E HotpotQA, layer 18, all three cells (n=60 each)\n"
                 "chat-template 04b run", fontsize=9.5)
    ax.grid(alpha=.3, axis="y")

    fig.suptitle("Figure 7 — Blank memory or confidently wrong?", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig7_readout_entropy.png")


# ---------------------------------------------------------------- Figure 8
def fig8():
    """RQ3: retention content vs dF1, side by side with boundary JS vs dF1."""
    from scipy import stats as sps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, (pred, lab, title) in zip(axes, [
            ("rank", "target rank at prompt end (layer 18)",
             "A · Retention content  —  this paper's predictor"),
            ("js", "boundary Jensen–Shannon divergence",
             "B · Output divergence  —  the predictor prior work tested")]):
        for name, key, col in CELLS:
            rows = json.load(open(f"results/run_3b_{key}/04b_joined_retention_task.json"))["rows"]
            js = {str(r["id"]): r["boundary_js"] for r in
                  json.load(open(f"results/run_3b_{key}/03_nowrite_reproduction.json"))["per_example"]}
            y = np.array([r["delta_f1"] for r in rows], float)
            x = np.array([r["rank_at_prompt_end"] if pred == "rank" else js[str(r["id"])] for r in rows], float)
            rho, p = sps.spearmanr(x, y)
            ax.scatter(x, y, s=26, alpha=.6, color=col, edgecolors="none",
                       label=f"{name}   ρ={rho:+.2f} (p={p:.2f})")
        ax.axhline(0, ls="--", c="grey", lw=1)
        ax.set_xlabel(lab); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=.3)
        if pred == "rank":
            ax.set_ylabel("per-example ΔF1  (AHN − NOWRITE)")
    fig.suptitle("Figure 8 — Does what the memory retains predict which examples AHN helps?\n"
                 "Neither predictor separates the examples; ΔF1 is zero on 52–55 of 60 examples in every cell.",
                 fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig8_rq3_scatter.png")


if __name__ == "__main__":
    fig1(); fig2(); fig4(); fig5(); fig6(); fig7(); fig8()
