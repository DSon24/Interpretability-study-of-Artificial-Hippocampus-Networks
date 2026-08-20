# What Does the Artificial Hippocampus Store?

**A cell-family ablation of compressive memory content in long-context language models**

Hannah Kim · Sơn Nguyễn — Algoverse LLM track
Mentor: Gautam Siddharth Kashyap, School of Computing, Macquarie University
Target: ARR / NAACL 2027 — 12 October 2026

---

Artificial Hippocampus Networks buy a 74% KV-cache reduction by compressing evicted
key–value pairs into a fixed-size recurrent state. We ask **what survives that
compression**, whether it differs across the three released recurrent cell families, and
whether what survives predicts task performance where existing distributional measures
do not.

This repository is a research fork of [ByteDance-Seed/AHN](https://github.com/ByteDance-Seed/AHN)
(Fang et al., 2025). The upstream code under `src/ahn/`, `eval/` and `examples/` is
unmodified; everything under `notebooks/`, `ahn_interp.py`, `docs/` and `results/` is ours.
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

## Where the project stands

Sequencing follows Gautam's instruction after the last meeting:

> "Get one 3B checkpoint running, reproduce the existing NOWRITE result, and make sure we
> can hook into the AHN output/state correctly. Then test the J-Lens on that single
> checkpoint with the basic needle-in-a-haystack setup and the pre-eviction/NOWRITE
> controls. If that works, scale to the 3 cells at 3B, plot the retention-vs-eviction
> curves, and only after that move to the 7B checkpoints and RQ3 correlation with task
> performance."

| Step | State | Where |
|---|---|---|
| One 3B checkpoint running | **done** — Qwen2.5-3B + AHN-GDN merged, A100-40GB | Sơn, 18 Aug |
| Hooks fire on the AHN module, NOWRITE zeroes the capture | **done** | `notebooks/pilot/` |
| Hook output verified to be the memory's residual-stream contribution | **not yet** — the pilot decoded a pre-`o_proj` vector; see Findings | `notebooks/01_` |
| Reproduce the published NOWRITE result (38–42% changed answers) | **not started** — no task score has been computed yet | `notebooks/03_` |
| J-lens fitted for Qwen2.5-3B | **partial** — fitted from one 106-char prompt at layer 18; fails its own sanity check | `notebooks/02_` |
| NIAH retention with pre-eviction / NOWRITE controls | **partial** — shape exists, numbers not yet valid | `notebooks/04_` |
| 3 cells at 3B → retention curves | not started (correct — do not start yet) | |
| 7B checkpoints, RQ3 correlation | not started (correct) | |

### Progress log

**18 Aug 2026 — Sơn.** Merged Qwen2.5-3B-Instruct + AHN-GatedDeltaNet and got it running
on an A100-40GB. Built persistent capture hooks across all 36 AHN layers and a NOWRITE
control hook, and verified hook ordering so NOWRITE fires before capture. Ran a layer
profile, a 3-needle Δ-readout sweep and a retention-decay sweep over six eviction
distances; loaded 20 RULER examples at 4K and confirmed AHN activates and NOWRITE zeroes
on all of them. Separately, fitted a Jacobian lens for layer 18 of the base Qwen2.5-3B in
a Colab T4 session (the `jlens` package needs `transformers>=5`, which conflicts with the
repo's `transformers==4.51.0` pin) and saved `J18_qwen25_3b.pt`.

That is real progress on the hardest engineering step — a working merged checkpoint with
live hooks is most of Stage 1. The measurement built on top of it is not yet valid, for
the reasons below.

## Findings from the 18 Aug pilot

Five issues, in the order they need fixing. All five are addressed by
[`ahn_interp.py`](ahn_interp.py) and the numbered notebooks.

**1. The readout was in the wrong basis.** In `src/ahn/transformer/qwen2_ahn/qwen2_ahn.py`
the memory is combined by plain addition **before** the attention output projection:

```python
attn_output[:, -L:, :] = attn_output[:, -L:, :] + ahn_attn_output
hidden_states = self.self_attn.o_proj(attn_output)
hidden_states = residual + hidden_states
```

A forward hook on `layer.ahn` therefore returns a vector in the concatenated-head space,
not the residual stream. On Qwen2.5-3B both happen to be 2048-dimensional (16 heads × 128),
so `o_t @ unembed.T` runs without error and returns noise. The memory's residual-stream
contribution is `o_proj(ahn_attn_output)`. This is also the answer to **Open Question 4**
in the proposal: the combination is a **sum**, so `o_t` isolates cleanly — unless
`use_ahn_router=True` in the checkpoint config, in which case it enters through a learned
sigmoid gate.

**2. Control C1 was vacuous.** Zeroing the AHN output makes the captured vector
identically zero, so `Δ = o_t(AHN) − o_t(NOWRITE)` collapses to `o_t`. The zero-state
control has to be run on the **residual stream** at the same position, not on the module
output. (The pilot notebook's own markdown notices the collapse and proceeds anyway.)

**3. Needles may never have been evicted.** `in_ahn_seq_len = cache_size − sliding_window
− num_attn_sinks`, and memory K/V are taken from index `num_attn_sinks` onward. Upstream
evaluation uses `NUM_ATTENTION_SINK=128`. The pilot placed every needle at roughly token 5 —
inside the sink prefix, where it is never compressed *and* stays losslessly visible to
attention.

**4. The J-lens is not validated.** It was fitted from a single 106-character prompt at
`max_seq_len=64`. Corpus averaging is the entire reason to prefer a J-lens over a logit
lens. The evidence it isn't working is already in the pilot: decoding the **full**
layer-18 residual stream through `J18` returns `<|endoftext|>`, `小镇`, `县公安局`, `ABCDE`.
That is the lens's easiest possible input, and it fails. Table 3 must pass first.

**5. Two diagnostics that should have stopped the pipeline.** The pre-eviction baseline
came out *worse* than the evicted condition (Paris rank 110712 in-window vs ~95000
evicted) — if the ceiling is below the floor, the measurement is not measuring retention.
And every rank in `results/pilot_2026-08-18/` sits between 7k and 149k on a 151,936-token
vocabulary, i.e. indistinguishable from chance.

None of this is wasted work. The instrumentation, the hook-ordering discipline, the
NOWRITE control and the merged checkpoint all carry forward unchanged; what changes is
the vector that gets decoded and where the needle is placed.

## Repository layout

```
ahn_interp.py                 shared instrumentation — upload this next to any notebook
notebooks/
  00_setup_and_config_audit.ipynb    load a checkpoint; record window / sinks / router
  01_instrumentation_gate.ipynb      prove the hook captures the memory's contribution
  02_jlens_fit_and_validate.ipynb    fit the J-lens; run Table 3's five checks
  03_nowrite_reproduction.ipynb      Week-6 milestone: 38–42% changed answers
  04_niah_retention.ipynb            retention curves + controls C1–C4 (Tables 4, 6)
  05_analysis_and_figures.ipynb      CPU only — Tables 5–9, Figures 3–8
  pilot/                             Sơn's 18 Aug notebooks, kept for provenance
docs/
  UPSTREAM_README.md                 ByteDance's original README
  PROPOSAL.md                        pointer to the proposal + expected-artefacts docs
results/
  pilot_2026-08-18/                  superseded — see Findings above
src/ahn/                       upstream AHN implementation (unmodified)
eval/, examples/               upstream harnesses (unmodified)
artifacts/deprecated/          J18 fitted from one prompt; kept, not used
```

## Running an experiment

There is no SSH to the GPU box, so the workflow is: **upload two files, run, download the
JSON.** Every notebook is standalone apart from `ahn_interp.py`.

1. Upload `ahn_interp.py` and the notebook you want into the Jupyter working directory.
2. Edit the `CFG` cell — model path, cell family, `sliding_window`, `num_attn_sinks`.
3. Run top to bottom. Each notebook ends in an explicit **gate**; if the gate fails, fix
   it before moving on rather than proceeding with a caveat.
4. Download the `results/<run>/*.json` files.
5. Run `05_analysis_and_figures.ipynb` **on your laptop**. It needs no GPU and no model.
   GPU time is the scarce resource; analysis time is not.

Order matters: `00 → 01 → 02 → 03 → 04 → 05`. Notebook 02 must be run in a **separate
environment** with `transformers>=5` because `jlens` conflicts with the repo's
`transformers==4.51.0` pin — that is why fitting and use are in different notebooks.

### Merging a checkpoint

```bash
python ./examples/scripts/utils/merge_weights.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --ahn-path  ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B \
  --output-path ./merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN
```

Three cells at 3B: swap `AHN-GDN` for `AHN-DN` and `AHN-Mamba2`. Keep `merged_ckpt/` out
of git (it already is).

## Experimental settings of record

| Setting | Value | Note |
|---|---|---|
| Backbone | Qwen2.5-3B-Instruct, then 7B | 14B dropped — ~28 GB BF16 exceeds our 24 GB GPUs |
| Cells | DeltaNet, GatedDeltaNet, Mamba2 | released checkpoints, used as-is; no training |
| Sliding window | **8064** | shrunk from the 32K default so AHN activates at affordable lengths; held fixed across all cells so it cannot confound the comparison |
| Attention sinks | **128** | upstream eval default; needles must be placed past it |
| Cohorts | RULER NIAH n=60, LongBench-E HotpotQA n=60, LV-Eval FactRecall n=30 | matches the concurrent study's sizes so numbers are directly comparable |
| Statistics | paired bootstrap, length-stratified resampling, Holm for the three pairwise half-life tests | |
| Seed | 20260820 | `ai.set_seed()` |

## Next steps

Ordered so that each one unblocks the next. The first four are the mentor's stated
sequence with the pilot's gaps filled in.

1. **Run `01_instrumentation_gate.ipynb`.** ~10 GPU-minutes. Gate A is the identity
   `resid(AHN) − resid(NOWRITE) = o_proj(ahn_raw)` at the first AHN layer. Until it
   passes, no readout number means anything. Gate B checks that suppressing writes moves
   the output distribution at all.
2. **Run `03_nowrite_reproduction.ipynb`.** 2–4 GPU-hours. This is the Week-6 milestone
   and the first task score the project will have produced. Reproducing 38–42% changed
   answers with |ΔF1| under ~2.5 points is the cleanest evidence the fork behaves like the
   real system — and it is the sentence to put in the next mentor update.
3. **Run `02_jlens_fit_and_validate.ipynb` with a real corpus** (500 contexts, two
   disjoint halves, `max_seq_len=256`, layers 9/18/27). Record the wall-clock: the
   proposal budgets 15–20 GPU-hours and treats **>40 h as the abort signal for RQ2**.
   Check 3 (known-fact recall) is cheap — run it first and stop early if it fails.
4. **Run `04_niah_retention.ipynb` on GDN 3B.** Controls C1 and C4 must pass before
   Table 6 is populated. If C3 (shuffled context) fails, message Gautam that day: it would
   mean AHN is closer to a learned recency mechanism than to content memory, which
   contradicts the framing of the original AHN paper and is arguably the most publishable
   thing in the project.
5. **Build `04b` — the RQ3 join.** *This is the most important missing piece of
   engineering and nothing else substitutes for it.* Right now retention is measured on
   synthetic NIAH prompts and ΔF1 on LongBench-E, so there is no per-example key linking
   them, and Table 8 cannot be computed at all. Extend notebook 04 to read out on the
   **same** LongBench-E examples as notebook 03, using the first token of the gold answer
   as the target. That single change turns RQ3 from aspiration into a computation, and
   `05` already has the cell written and waiting for the file.
6. **Add the boundary-JS column.** Notebook 01's Gate B already computes JS between the
   AHN and NOWRITE next-token distributions. Log it per example in notebook 03 and Table 8
   row 2 — the pre-registered comparison against prior work — comes for free rather than
   needing Gautam's numbers. Worth asking him anyway (his Question 2) whether to use his
   values directly or recompute on our cohort so the two columns are strictly paired.
7. **Only then, the second and third cells at 3B.** Same notebook, change `CFG`. Table 7
   (pairwise half-life ratios with their own CIs) needs two cells minimum; the proposal's
   success criterion is a ratio ≥ 2 with a CI excluding 1 on at least one pair.
8. **7B last**, gated on the J-lens map cost measured in step 3 (Table 10 row 2).

### Two things to raise with Gautam now, not later

- **Open Question 4 is answered**: the combination is a plain sum before `o_proj`, so
  `o_t` isolates cleanly. Worth telling him — it was flagged as the first thing to check
  in Week 1, and it is settled.
- **Open Question 6**: he uses 8,064 on LongBench-E himself, which makes our 8064 window
  easy to defend. Confirm it in writing so it goes in Methods as a stated parameter.

Still open and blocking: the training-only compute-reimbursement rule (Question 1 — Ops,
not Gautam), and whether AHN runs on a free T4 with an FP16 + eager-attention fallback
(Question 3), which notebook 00 now tests directly.

## Citation

The system under study:

```bibtex
@article{fang2025artificial,
  title={Artificial hippocampus networks for efficient long-context modeling},
  author={Fang, Yunhao and Yu, Weihao and Zhong, Shu and Ye, Qinghao and Xiong, Xuehan and Wei, Lai},
  journal={arXiv preprint arXiv:2510.07318},
  year={2025}
}
```

The readout method: *Verbalizable Representations Form a Global Workspace in Language
Models* (Anthropic, 2026), transformer-circuits.pub/2026/workspace ·
github.com/anthropics/jacobian-lens

Evaluation: RULER (Hsieh et al., 2024), LongBench (Bai et al., 2024), LV-Eval (Yuan
et al., 2024). Cells: Gated Delta Networks (Yang et al., 2024), Mamba2 (Dao & Gu, 2024).

Upstream code is Apache-2.0; see [`LICENSE`](LICENSE). Our additions are released under
the same terms.
