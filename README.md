# What Does the Artificial Hippocampus Store?

**A cell-family ablation of compressive memory content in long-context language models**

Hannah Q. Kim\*, Son Nguyen\*, Solomon Muwanguzi, Gautam Kashyap

\*Equal contribution

---

Artificial Hippocampus Networks buy a 74% KV-cache reduction by compressing evicted
key–value pairs into a fixed-size recurrent state. We ask **what survives that
compression**, whether it differs across the three released recurrent cell families, and
whether what survives predicts task performance where existing distributional measures
do not.

This repository is a research fork of [ByteDance-Seed/AHN](https://github.com/ByteDance-Seed/AHN)
(Fang et al., 2025). The upstream code under `src/ahn/` and `examples/` is unmodified;
under `eval/`, we made one case-insensitivity fix to Qwen model-name matching in
`eval/lveval/utils.py` and added the long-context RULER runner `eval/ruler/row87_*.py`.
`ahn_interp.py`, `scripts/`, `configs/`, `notebooks/`, `docs/` and `results/` are ours.
The original project README is preserved at [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).

## Research questions

| | Question | Layer |
|---|---|---|
| **RQ1** | Once analysed with adequate statistical power, does the choice of recurrent cell (DeltaNet, GatedDeltaNet, Mamba2) change long-context task behaviour — and where? | Behavioural |
| **RQ2** | What does each cell's compressed memory actually retain, read out in vocabulary space, and how fast does that content decay with eviction distance? | Content |
| **RQ3** | Does content retention measured in RQ2 predict the per-example task differences in RQ1, where output-distribution divergence demonstrably does not? | The join |

RQ3 is the contribution. RQ1 alone is a benchmark paper; RQ2 alone is a lens
demonstration. Together they test a falsifiable claim: that a content-specific measure of
what compressed memory holds succeeds at predicting task outcomes where a
distribution-general measure fails.

The gap is quantified. The concurrent write-attrition study (Kashyap, 2026, under review
at ACL) found that boundary Jensen–Shannon divergence tracks changed answers
(ρ = .34–.41) but **not** F1 (ρ = −.09 to .00), and states in its Limitations that its
interventions *"do not identify what recurrent states store."* That sentence defines RQ2;
the ρ ≈ 0 result defines RQ3.

## Main results

All numbers are for Qwen2.5-3B-Instruct with the released AHN modules under the settings
of record below, unless marked 7B. The complete dated analysis log, including superseded
and withdrawn results, is [docs/FINDINGS.md](docs/FINDINGS.md).

**No-AHN floor.** With stock Qwen and the same 8,064-token sliding window, RULER NIAH
evicted-needle substring accuracy is **0.000** (n = 32; in-window 0.393, n = 28) and
LongBench-E HotpotQA first-line F1 is 0.076. At 7B, evicted accuracy is 0.000 for the floor
*and* for all three AHN cells (in-window 1.000). The released modules do not restore
behavioural retrieval of an evicted needle.

**RQ1 — behaviour (Table 5, Figure 3).** First-line ΔF1 (AHN − NOWRITE) on LongBench-E
HotpotQA, n = 60 paired, Qwen chat template:

| Cell | ΔF1 (pts) [95% CI] | Answers changed |
|---|---:|---:|
| GatedDeltaNet | −4.10 [−8.75, −0.58] | 30.0% |
| DeltaNet | −3.96 [−9.49, +0.94] | 31.7% |
| Mamba2 | −4.63 [−11.44, +0.90] | 30.0% |

No pairwise cell contrast survives Holm correction (all adjusted p = 1.0). The cells shift
behaviour by the same amount and in the same direction.

**RQ2 — content (Tables 4, 7; Figures 2, 4–6).** On the RULER NIAH cohort, read through the
J-lens, layer 27 carries a memory-specific signal that is statistically real but small,
and its sign differs by cell: C2 = 1.076×/digit for GatedDeltaNet (p < 1e-4), 0.969× for
DeltaNet (p = 6e-4) and 1.095× for Mamba2 (p = 0.40, null). None approaches the
pre-registered 10× bar. Layers 9 and 18 fail the C3-lens control and are treated as
decoding artefacts. For GatedDeltaNet, the layer-27 signal shows no detectable decay with
eviction distance across 46–56,640 tokens (slope p = 0.92). This does not show that
retention is flat; it shows that the cohort and readout cannot resolve any decay.

**RQ3 — the join (Table 8, Figure 8).** Retention readouts (target rank, target mass,
readout entropy) do not predict per-example ΔF1 in any cell (|ρ| ≤ 0.18; Holm p ≥ 0.48).
The test has low power because ΔF1 is nonzero on only 5–8 of the 60 examples.

**Caveats.** The J-lens passes map stability but fails two of the five Table 3 validation
checks, so every RQ2/RQ3 readout is stamped `lens_validated = False`. The layer-27 C1/C2
signal did not reproduce on a deeply evicted BABILong `qa1` cohort.


## Repository layout

```
ahn_interp.py            shared instrumentation: model loading, hooks, J-lens readout, stats
configs/                 run configurations of record (3B and 7B: GDN, DN, Mamba2, no-AHN floor)
scripts/
  build_paper_figures.py   Figures 1, 2, 4–8 from committed results (CPU)
  build_fig3.py            Figure 3, the RQ1 forest plot (CPU)
  build_tables_1_2.py      Tables 1–2: ablation grid and evaluation cohorts
  build_table5_crosscell.py  Table 5 cross-cell contrasts (paired bootstrap + Holm)
  build_table8.py          Table 8, the RQ3 join across cells; build_js_check.py adds boundary JS
  build_table10.py         Table 10 compute accounting
  ruler_controls.py        RULER C1–C4 control battery per cell (CPU)
  ruler_retention_curve.py, ruler_placement_check.py, ruler_needle_position.py, ruler_cohort_stats.py
  no_ahn_floor.py          GPU: stock Qwen + forced sliding window, no AHN (Table 1 floor)
  fit_layer35.py           GPU: J-lens fit at the final layer (Table 3 check 1)
  per_layer_controls.py, content_swap_stats.py, extract_c2_corrected.py,
  kashyap_reconciliation.py, probe_construction.py, probe_prompt_format.py
                           supporting analyses referenced in docs/FINDINGS.md
notebooks/
  00_setup_and_config_audit.ipynb      load a checkpoint; record window / sinks / router
  01_instrumentation_gate.ipynb        prove the hook captures the memory's contribution
  02_jlens_fit_and_validate.ipynb      fit the J-lens; run Table 3's five checks
  02b_jlens_fit_1000ctx.ipynb          1,000-context refit for the map-stability check
  03_nowrite_reproduction.ipynb        RQ1: AHN vs NOWRITE on LongBench-E HotpotQA
  04_niah_retention.ipynb              RQ2: retention readout + controls C1–C4
  04_c2_investigation.ipynb            C2 readout-bias investigation
  04_niah_C3_analyze.ipynb             C3 shuffled-context analysis
  04b_rq3_join.ipynb                   RQ3: join retention readouts to per-example ΔF1
  05_analysis_and_figures.ipynb        CPU only — Tables 5–9, figures
  05b_table7_crosscell.ipynb           Table 7 cross-cell retention comparison
  06_factrecall_reproduction.ipynb     LV-Eval FactRecall
  06b_ruler_answer_prefix_ablation.ipynb  RULER answer-prefix ablation
  07_babilong_pilot.ipynb, 07c_babilong_c1c2_full.ipynb  BABILong robustness check
  archive/                             superseded notebooks (pilot, raw-prompt runs), kept for provenance
results/
  run_3b_{gdn,dn,m2}/      per-cell outputs; run_3b_gdn is the most complete run
  run_3b_floor/            no-AHN floor
  run_7b_{gdn,dn,m2,floor}/  7B behavioural RULER NIAH
  05_table*.json           cross-cell paper tables
  figures/                 paper figures
  validation/              follow-up validation runs, incl. long-context RULER (32k, 64k)
  babilong/                BABILong construction-robustness runs
  personal_interest_experiments/  exploratory, not pre-registered
  pilot_2026-08-18/        superseded pilot, kept for provenance
docs/                    findings log, pre-registration amendments, decision records
src/ahn/, examples/      upstream AHN implementation and scripts (unmodified)
eval/                    upstream evaluation harnesses (see note above)
```

Large artefacts are not in git: merged checkpoints (`merged_ckpt/`) and fitted J-lens
weights (`*.pt`).


## Reproducing the results

### Rebuilding the figures (no GPU, no model)

Every figure can be rebuilt on a laptop from the result JSONs committed under `results/`.
Run scripts from the repository root:

```bash
pip install numpy scipy matplotlib
python scripts/build_paper_figures.py   # Figures 1, 2, 4, 5, 6, 7, 8  (Figure 1 is a schematic)
python scripts/build_fig3.py            # Figure 3, the RQ1 forest plot, all three cells
```

Output lands in `results/figures/`. All plotted values come from committed results; none
is synthetic. `scripts/build_fig3.py` also rewrites each cell's
`results/run_3b_<cell>/05_table5_rq1.json` from that cell's `03_nowrite_reproduction.json`. The RQ2/RQ3 readouts behind Figures 2 and
4–8 are stamped `lens_validated = False` — see [FINDINGS](docs/FINDINGS.md).

### Re-running the experiments (GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval]"
python ./examples/scripts/utils/merge_weights.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --ahn-path  ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B \
  --output-path ./merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN
```

- **FlashAttention-2 is required for correctness.** Sliding-window attention is not
  implemented for `sdpa`; without FA2 the window is silently ignored and AHN never activates.
- **Other cells:** swap `AHN-GDN` for `AHN-DN` or `AHN-Mamba2`. Mamba2 also needs the forked
  mamba (`pip install "git+https://github.com/yuweihao/mamba.git"`).
- **Checkpoint location:** configs carry `ckpt_name`, resolved by `ai.resolve_ckpt()` against
  `$AHN_CKPT_ROOT` (default `<repo>/merged_ckpt`).
- **J-lens:** notebook 02 needs a separate environment with `transformers>=5` and
  [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens). The fitted `.pt`
  is a plain tensor dict and loads under the main `transformers==4.51.0` environment.

Run the notebooks in order `00 → 01 → 02 → 03 → 04 → 05`. Each ends in an explicit gate;
`05_analysis_and_figures.ipynb` is CPU-only.

## Experimental settings of record

| Setting | Value | Note |
|---|---|---|
| Backbone | Qwen2.5-3B-Instruct, then 7B | 14B dropped — ~28 GB BF16 exceeds the cards we get (24 GB when the call was made; 20 GB MIG slices on the current H100 box) |
| Cells | DeltaNet, GatedDeltaNet, Mamba2 | released checkpoints, used as-is; no training |
| Sliding window | **8064** | shrunk from the 32K default so AHN activates at affordable lengths; held fixed across all cells so it cannot confound the comparison |
| Attention sinks | **128** | upstream eval default; needles must be placed past it |
| Cohorts | RULER NIAH n=60, LongBench-E HotpotQA n=60, LV-Eval FactRecall n=30 | matches the concurrent study's sizes so numbers are directly comparable |
| Statistics | paired bootstrap, length-stratified resampling, Holm for the three pairwise half-life tests | |
| Seed | 20260820 | `ai.set_seed()` |

## License

Upstream code is Apache-2.0; see [`LICENSE`](LICENSE). Our additions are released under
the same terms.
