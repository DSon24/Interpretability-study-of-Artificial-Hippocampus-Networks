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
| Hook output verified to be the memory's residual-stream contribution | **done** — Gate A passes: `resid(AHN) − resid(NOWRITE) = o_proj(ahn_raw)` | `results/run_3b_gdn/01_instrumentation_gates.json` |
| Reproduce the published NOWRITE result (38–42% changed answers) | **done, with a metric correction** — 33.3% changed (first-line), ΔF1 **+6.1 pts** | `results/run_3b_gdn/03_nowrite_reproduction.json` |
| J-lens fitted for Qwen2.5-3B | **done, map converged** — 500 contexts, layers 9/18/27, 1.92 GPU-h; **Table 3 check 4 (map stability) now passes** (top-10 overlap 0.91/0.87/0.89 across a disjoint second corpus); checks 2 and 3 still fail | `results/run_3b_gdn/02_table3_jlens_validation.json` |
| NIAH retention with pre-eviction / NOWRITE controls | **done — control battery fails.** C4 passes, **C1, C2, C3 all fail**; readout is at chance | `results/run_3b_gdn/04_table4_controls.json` |
| C2 follow-up — is the distractor control measuring anything? | **done, and it reframes C2** — the raw ratio is confounded by pair identity; after baseline correction no layer shows a memory-specific effect | `notebooks/04-C2-debug.ipynb`, Sơn, 28–31 Aug |
| C3 follow-up — corrected shuffled-context rerun | **done** — needle held at a fixed token position; 96 matched rows. C3 does **not** give the expected order-sensitivity in either direction | `notebooks/04_niah_C3_analyze.ipynb`, Sơn, 29 Aug |
| Per-layer recomputation of the control battery | **done, no GPU** — layer 9's readout is degenerate and fails C4. The 2 Sep "layer 27 works" claim is **withdrawn**: wrong readout basis, and its C2 ratio vanishes under baseline correction | `04c_per_layer_controls.json`, `04d_c2_baseline_corrected.json`, 2–3 Sep |
| Retention curves (Table 6) | **run, not usable** — exponential fit inadequate at all three layers (R² < 0); nonparametric half-distance ≈ 11 tokens everywhere | `results/run_3b_gdn/05_table6_retention_summary.json` |
| RQ1 on LongBench-E HotpotQA (Table 5) | **done** — pooled first-line ΔF1 +6.11 pts, but **95% CI [−1.85, +14.42] spans zero** | `results/run_3b_gdn/05_table5_rq1.json` |
| RQ3 join (`04b`) | **built and run by Sơn** — no significant correlation between memory rank and ΔF1 at any of layers 9/18/27, before or after Holm correction; **result is provisional**, inherits the same C1 chance-level readout | `results/run_3b_gdn/04b_joined_retention_task.json` |
| 3 cells at 3B → retention curves | **blocked** — GPU_PLAN says do not proceed past a C1 failure until GDN's instrument is fixed; map-stability passing narrows but does not resolve the diagnosis | `configs/run_3b_dn.json`, `run_3b_m2.json` (unrun) |
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

**19–20 Aug 2026 — Hannah.** Ran notebooks 00 → 03 end to end on the shared A100 box.
Notebook 00 recorded the config of record (window 8064, sinks 128, `use_ahn_router=false`,
`o_proj` has no bias). Notebook 01 passed all three instrumentation gates. Notebook 02 fit
a real 500-context J-lens in 1.92 GPU-hours and ran the Table 3 battery. Notebook 03
produced the project's first task scores on 60 LongBench-E HotpotQA examples. Four bugs
were found and fixed in `ahn_interp.py` along the way (needle-padding precision, the
eviction-distance formula, the Gate A residual-capture hook point, and an OOM from
computing full-sequence logits); one dataset-id fix in notebook 02 (`wikitext` →
`Salesforce/wikitext`, which recent `datasets` versions require). Detail below.

**20 Aug 2026 — Hannah.** Audited the repo against the proposal and Gautam's instruction
(`docs/GPU_PLAN_2026-08-20.md`) — no scope drift, everything is on his stated path in his
stated order. Fixed the blocker that would have crashed `04` on startup (a stray
`AssertionError` when Table 3 fails, now downgraded to a warning that stamps
`lens_validated=False` onto every row), switched `05`'s Table 5 to the first-line metric
fields, and trimmed the eviction-distance sweep from 240 to 210 configs. Ran
`04_niah_retention.ipynb` on GDN 3B: the control battery **fails** (C4 passes; C1, C2, C3
do not), the retention curves don't support an exponential fit at any of the three layers,
and a bootstrap on RQ1's first-line ΔF1 shows the +6.11-point effect does not survive its
own confidence interval at n=60. Detail below.

**20 Aug 2026 (later) — Hannah + Sơn.** Ran Table 3 check 4 (map stability): fit a second
J-lens on a disjoint 500-context corpus, compared top-10 overlap against the first map.
**Passes at all three layers** (0.91 / 0.87 / 0.89, all ≥ the 0.80 bar). This rules out an
undersampled/unconverged fit as the explanation for the C1 control failure — the lens is
stable, so whatever is producing the chance-level readout on the full NIAH sweep is not a
fitting artifact. Separately, Sơn built and ran `04b` (the RQ3 join, `notebooks/0.4b.ipynb`):
joined the 60 LongBench-E examples from notebook 03 to a memory-readout pass, correlated
gold-answer-token rank against ΔF1. No significant correlation at any layer (Spearman
ρ = +0.257 / −0.100 / +0.020 at layers 9/18/27, none surviving Holm correction across the
three layers). A 1000-context refit of the J-lens was also kicked off, in progress as of
this writing, as a further robustness check beyond the 500-context stability pass. Detail
below.

**28–31 Aug 2026 — Sơn.** Took the two failing controls apart on branch
`son-c2-investigation` (now merged to `main`). For C2: established that the failure is
word-dependent rather than distance-dependent, that it is present *before* eviction, that
the plain logit lens shows the same reversal so it is not a J-Lens artifact, and finally
that the raw `p(needle)/p(distractor)` statistic is dominated by a pair-specific baseline
readout preference — `mango` beats `banana` by 31–54× regardless of which needle was
actually stored. Re-ran the whole C2 design baseline-corrected (252 matched observations,
aggregated to 21 conditions per layer). For C3: rebuilt the control so the needle stays at
the same token position in both the ordered and shuffled conditions, and reran it as 96
fully matched pairs. Also fixed the `needle_pos` off-by-four in `ahn_interp.py` and
regenerated `04_table4_controls.json`. Detail below.

## Findings from the 19–20 Aug run

**1. The instrumentation is verified correct.** Gate A — the identity
`resid(AHN) − resid(NOWRITE) = o_proj(ahn_raw)` — passes at the first AHN layer and fails
at a deep layer as expected. Gate B confirms suppressing writes moves the next-token
distribution. This closes the "not yet" row above and retires pilot findings 1 and 2:
the readout is now in the right basis and the C1 control is no longer vacuous.

**2. The J-lens map costs 1.92 GPU-hours, not 15–20.** This is Table 10 row 1, measured at
the proposal's own settings (500 contexts, `max_seq_len=256`, layers 9/18/27, disjoint
corpora verified zero-overlap). The proposal budgeted 15–20 GPU-hours and set **>40 h as
the abort signal for RQ2**. Coming in an order of magnitude under budget materially
de-risks the 7B decision (Table 10 row 2) and means lens re-fits are cheap enough to
iterate on rather than ration.

**3. Table 3 check 2 is mis-specified and should be revised, not chased.** The check
demands ≥60% top-1 agreement with the plain logit lens at mid layers. Measured agreement
is 0% / 0% / 5% at layers 9 / 18 / 27 — but on eight known-fact prompts the logit lens
itself has a **median rank of 12,433** at layer 18. Demanding agreement with a reference
that is itself near-useless at the layers in question is not a validity test. Recommend
replacing it with a direct known-fact criterion.

**4. Table 3 check 3 fails on a strict criterion, but the lens carries real signal.**
Check 3 requires the target token at rank 0. Measured `rank_Paris` is 805 / 59 / **8** at
layers 9 / 18 / 27, and across eight known-fact prompts the J-lens median rank is
1,808 / 61 / **9** against a 151,936-token vocabulary. On the same prompts the plain logit
lens gives 40,154 / 12,433 / 71 — so the J-lens beats it by **22× at layer 9, 204× at
layer 18, and 8× at layer 27**. It reaches top-1 on 0/8 facts; the logit lens manages 1/8
at layer 27 only.

Four candidate explanations were tested and eliminated, all at zero GPU cost:

| hypothesis | test | result |
|---|---|---|
| decode path wrong (hook point, final norm, `lm_head`) | decode layer-35 residual with no lens, compare to the model's own logits | **ruled out** — KL = 1.95×10⁻⁴, exact top-5 match |
| transport orientation transposed (`h @ J` vs `J @ h`) | sweep both orientations on the fitted Jacobians | **ruled out** — `h @ J` gives ranks of 16k–146k |
| layer-index convention off by one between `jlens` and the hook | pair each `J_L` with `resid@L±1, ±2` | **ruled out** — the diagonal is optimal at layers 18 and 27 |
| junk tokens win through large unembedding norm | compare `‖W‖` for `____`, `:**`, ` Paris`, ` Tokyo` | **ruled out** — all ≈ 1.0 |
| an input-independent additive offset in logit space | subtract the mean readout over 32 held-out prompts | **ruled out** — rank got *worse* (8 → 32) |

What remains is that the averaged Jacobian leans toward structural continuations
(`____`, `:**`, `.[`), and those occupy the top slots ahead of the correct answer. Note
that the model itself ranks ` __` second on "The capital of France is", so this is a
plausible-continuation family being over-weighted, not noise. **Whether this disqualifies
the lens for RQ2 is a judgment call for Gautam:** RQ2 needs rank *separation* between
needle and control, not top-1, and notebook 04 tests that property directly.

**5. Notebook 03's pass/fail metric measured the wrong thing.** As written, both
`answer_changed` and the F1 scores are computed on the entire ≤32-token generation. The
model ignores the template's "only give me the answer" instruction and rambles past the
answer until the token cap, so any divergence in that trailing justification counted as a
changed answer:

```
AHN: 'Dallas Cowboys\nThe football maneuver, known as the horse-collar tackle...'
NW:  'Dallas Cowboys\nThe horse-collar tackle is most closely associated...'
```

The answer is identical; the metric scored it as changed. Recomputed on the first line
only — same 60 generations, no re-run:

| metric | full generation | first line | published target |
|---|---|---|---|
| answer-change rate | 91.7% | **33.3%** | 38–42% |
| mean F1 (AHN) | 0.115 | **0.400** | — |
| mean F1 (NOWRITE) | 0.130 | **0.339** | — |
| ΔF1 | −1.44 pts | **+6.11 pts** | +0.4 to +2.3 pts |

The change rate lands inside the notebook's own tolerance band (30–50%) and near the
published range, and ΔF1 **flips sign** to match the published direction. The remaining
gap is magnitude: +6.1 points is larger than the published +0.4 to +2.3, i.e. this
checkpoint's memory appears to help *more* than the concurrent study's did — the opposite
of a null result. Both metrics are kept in the saved JSON (`*_fl` fields alongside the
originals) because the discrepancy is itself a finding: the prompt template is not
eliciting terse answers.

## Findings from the 20 Aug NIAH retention run

Full detail and the reasoning behind every patch is in
[`docs/GPU_PLAN_2026-08-20.md`](docs/GPU_PLAN_2026-08-20.md); this is the summary.

**1. The control battery fails on three of four checks.** Run with `USE_JLENS=True`,
GDN 3B, the same window/sinks as the config of record:

| control | result | passed |
|---|---|---|
| C1 zero-state | mean rank 87,688 vs chance rank 75,968 — indistinguishable from chance | **fail** |
| C2 distractor | median probability ratio 1.00 (need ≥10×) — no needle/distractor separation | **fail** |
| C3 shuffled context | ordered rank 88,702 vs shuffled rank 72,769 — shuffling context makes the rank *better* | **fail** |
| C4 pre-eviction ceiling | in-window rank 77,430 beats evicted rank 87,688, as required | pass |

> These are the **corrected** numbers, regenerated 31 Aug after Sơn fixed an off-by-four in
> `build_niah_prompt` (`needle_pos` was measuring the start of the needle *sentence*, not the
> needle *token*). Every eviction distance in the sweep shifts by ~4 tokens. The battery moves
> by less than 0.2% and **no pass/fail verdict changes**; earlier drafts of this README quoted
> 87,675 / 88,642 / 72,943 / 77,411.

C1 failing is the load-bearing result: it means the readout is not finding the needle
above chance in a 151,936-token vocabulary, full stop. That sits in tension with Finding 4
from the 19–20 Aug run, where the same J-lens beat the logit lens by 8–204× on eight
isolated known-fact prompts. The isolated-fact test and the full NIAH sweep disagree, and
that disagreement — not top-1 accuracy — is now the open question for Gautam.

**2. The retention curves don't support the model the proposal assumes.** Table 6's
exponential fit gives negative R² at all three layers (9, 18, 27), i.e. worse than fitting
a flat line. The nonparametric half-distance fallback lands at ~11 tokens uniformly across
layers — too short relative to the 64–8192-token eviction sweep to read as a real decay
curve rather than noise. Given Finding 1, this is consistent with the readout not tracking
the needle at all, rather than a curve-fitting problem.

**3. RQ1's effect does not survive its own confidence interval.** Bootstrapping the
first-line ΔF1 from notebook 03 (10,000 resamples) gives **+6.11 pts, 95% CI [−1.85,
+14.42]** pooled, and every per-stratum interval also spans zero (short +9.17 [0.00,
+21.67], mid +3.45 [−11.55, +18.45], long +5.72 [−10.56, +22.00]). Two consequences: the
honest RQ1 reading at n=60 is *no significant behavioural effect*, which the proposal
already treats as the expected setup for RQ3 rather than a bad outcome; and Gautam's
published +0.4 to +2.3 pts sits comfortably inside our interval, so the earlier "our ΔF1 is
3× his" gap (Finding 5, 19–20 Aug) is not a real effect-size discrepancy — it's noise at
this cohort size.

> **Read this section with the 2 Sep per-layer re-analysis in hand.** Every number in the
> table above is pooled across layers 9, 18 and 27. Layer 9's readout is degenerate and
> fails C4 on its own; layer 27 is above chance with C2 at 5.03×. The pooled "everything is
> at chance" reading is an artefact of that averaging.

**4. Per Gautam's own instruction, this blocks the next two steps.** His sequencing said
scale to three cells only "if that works." It didn't. `configs/run_3b_dn.json` and
`run_3b_m2.json` exist but are correctly unrun, pending the diagnosis below.

## Findings from the map-stability check and the RQ3 join

**1. Map stability passes — undersampling is ruled out.** A second J-lens fitted on a
disjoint 500-context corpus agrees with the first at top-10 overlap 0.91 / 0.87 / 0.89
across layers 9/18/27, all above the 0.80 bar
(`results/run_3b_gdn/02_table3_jlens_validation.json`). This was the cheapest of the three
candidate explanations for the C1/isolated-fact disagreement, and it's now closed: the lens
converged. A weak or chance-level signal downstream is therefore **not a fitting artifact**
— it's either a genuine property of what AHN retains, or a mismatch between the NIAH
sweep's construction and the known-fact test's, per the two remaining candidates below.
Table 3 overall still reads `TABLE_3_PASSED: false` (checks 2 and 3 still fail on their
original criteria), which is expected and unrelated to this result.

**2. The RQ3 join runs, and finds no correlation — but the result is provisional.** `04b`
(`notebooks/0.4b.ipynb`) joins the 60 LongBench-E HotpotQA examples from notebook 03 to a
fresh memory readout: gold-answer first-token rank at layers 9/18/27, correlated against
`delta_f1` via Spearman.

| layer | ρ | raw p | Holm p | 95% CI |
|---|---|---|---|---|
| 9 | +0.257 | 0.047 | 0.141 | [−0.015, 0.487] |
| 18 | −0.100 | 0.448 | 0.897 | crosses zero |
| 27 | +0.020 | 0.880 | 0.897 | crosses zero |

None survive correction; the layer-9 near-hit doesn't survive its own bootstrap CI either.
Robust to metric choice — repeating against the uncorrected full-generation ΔF1 gives the
same null. Statistically this is careful work (bootstrap CIs, Holm correction across
layers, a sensitivity check), but the result is **downstream of the same open question as
C1**: individual ranks in the join are mostly in the bottom few percent of the 151,936-token
vocabulary (e.g. 150,768; 148,143; 151,375), the same chance-level signature C1 already
flagged. A null correlation here is ambiguous between "content retention genuinely doesn't
predict task benefit" and "the readout isn't measuring retention, so of course it doesn't
correlate with anything" — can't distinguish the two until the C1 diagnosis resolves. Two
added design limitations: `04b` uses only the gold answer's first token as target (reduces
to single letters for most multi-word entities — low-information), and `delta_f1` is
exactly 0 for ~80% of the 60 examples, a heavily tied outcome that limits Spearman's power
regardless of the readout question.

**3. Remaining candidates for the C1 diagnosis, now narrowed to two.** (a) undersampling —
**ruled out** by Finding 1 above. (b) NIAH sweep construction differs from the known-fact
test's conditions (needle placement, padding, prompt format). (c) C3's result is real — AHN
behaves closer to a recency mechanism than a content store. Both remaining candidates need
a comparison the stability check can't provide on its own; this is the open question to
bring to Gautam now that the cheap explanation is closed off.

> **Superseded twice — see the 28–31 Aug and 2 Sep sections below.** Candidate (c) rested on C3 as
> originally run. The corrected C3 does not support a clean recency reading either — the
> effect of shuffling depends on layer, distance and which metric you look at. That leaves
> **(b), construction, as the live candidate**, and it is now the one to test directly.

## Findings from the 18 Aug pilot

Five issues, in the order they need fixing. All five are addressed by
[`ahn_interp.py`](ahn_interp.py) and the numbered notebooks. **Findings 1, 2 and 3 are now
retired** — Gate A's identity holds, C1 runs on the residual stream, and needles are placed
past `num_attn_sinks`. Finding 4 is superseded by the 19–20 Aug run above (the lens is now
fit on a real corpus and diagnosed in detail). Kept here as the record of what changed and
why.

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

## Findings from the 21 Aug C1 diagnosis

Ran Gautam's three requested checks on one evicted needle example (GDN 3B, real project settings: sliding_window=8064, num_attn_sinks=128), needle "Paris" evicted 515 tokens past the compression boundary.

- **Eviction index**: confirmed evicted — not in-window, not in the sink region.
- **WRITE vs NOWRITE**: `raw_off` norm is exactly 0.0000 at every checked layer (9/18/27); the suppression hook fires correctly.
- **Does `o_t` actually change**: yes. `||o_t_on - o_t_off||` = 0.52 / 0.94 / 2.45 at layers 9/18/27, growing with depth rather than flat or near-zero. The memory channel is genuinely active and contributing signal on this example.

Follow-up, evicted vs in-window rank at layer 18, same needle:

- Evicted rank: 108,733
- In-window rank: 95,949 (needle never touched by AHN, fully visible to ordinary attention)
- Chance baseline: ~76,000 — both conditions are near or worse than chance.

The in-window result is the important one: that needle was never compressed by AHN at all, so AHN cannot be responsible for the readout failing there. This points at the NIAH prompt construction itself, not AHN's memory — consistent with the 18 Aug pilot's own note that its in-window baseline scored worse than its evicted condition.

**Caveat**: the rank comparison above used the plain logit lens, not the fitted J-Lens (no lens argument was available on this box at the time). Not a like-for-like comparison with the earlier "J-Lens beats logit lens by up to 204x" result. Re-running this same evicted-vs-in-window comparison through the actual J-Lens is the natural next step, to confirm construction is the issue rather than AHN itself.

## Findings from the 28–31 Aug C2 and C3 investigation

Sơn's work on branch `son-c2-investigation`, merged to `main` on 2 Sep. Everything here is
in `notebooks/04-C2-debug.ipynb` and `notebooks/04_niah_C3_analyze.ipynb`. All of it
remains conditional on the J-Lens not having passed the full Table 3 battery.

**1. A real bug in the eviction-distance measurement, fixed.** `build_niah_prompt` computed
`needle_pos` from the tokenized head *plus the needle sentence's opening words*, so it was
pointing at the start of `"The special word is Paris. "` rather than at ` Paris` itself —
an off-by-four in every reported eviction distance. The fix appends the
`"The special word is"` scaffold before measuring. `04_table4_controls.json` was
regenerated: the battery moves by under 0.2% and **no verdict changes**.

**2. C2 does not fail uniformly — it fails per needle.** Median `p(needle)/p(distractor)`
at layer 27, evicted rows only:

| needle → distractor | median ratio | needle win rate |
|---|---:|---:|
| Paris → London | 9.75× | 100% (23/23) |
| Tokyo → Osaka | 5.75× | 100% (23/23) |
| lantern → torch | 3.93× | 91.3% (21/23) |
| banana → mango | **0.037×** | **0% (0/23)** |

Overall median 4.50× against the 10× bar. Every term was verified to be a single token
with the expected leading space, so tokenization does not explain it.

**3. The `banana` reversal is not caused by eviction, and not caused by the J-Lens.** Two
controls close both explanations:

- **In-window**: `banana` already favours `mango` at 0.074× *before* the needle is ever
  compressed. AHN forgetting the needle cannot be the mechanism.
- **Plain vs J-Lens** at the same layer and distance: `banana → mango` is 0.015 under the
  plain logit lens and 0.024 under the J-Lens; `Paris → London` is 23.33 and 21.55. The
  direction is already present in `o_t` before any lens transform is applied.

**4. The raw C2 statistic is confounded by pair identity — this is the load-bearing
result.** Holding the readout fixed and varying *which needle was actually stored*,
`p(banana)/p(mango)` barely moves:

| stored needle | p(banana)/p(mango) |
|---|---:|
| Paris | 0.0207 |
| Tokyo | 0.0185 |
| banana | 0.0239 |
| lantern | 0.0323 |

`mango` wins regardless of what is in memory. The same holds across all four pairs:
Paris > London by ~15–26×, Tokyo > Osaka by ~6.5–7×, mango > banana by ~31–54×, lantern >
torch by ~9–12.5×, **whichever needle is stored**. So `p(needle)/p(distractor)` is
measuring a baseline property of the vocabulary pair, not memory selectivity. The apparent
success of Paris/Tokyo and failure of banana were both artefacts of that baseline.

**5. Baseline-corrected, no layer shows a memory-specific effect.** Normalising each pair
against its own preference when *other* needles are stored, across the full design
(3 layers × 7 distances × 3 fillers × 4 pairs = 252 matched observations, aggregated to 21
condition-level observations per layer):

| layer | geometric-mean fold | 95% CI | p |
|---:|---:|:---:|---:|
| 9 | 0.995× | [0.983, 1.007] | .412 |
| 18 | **1.047×** | [1.021, 1.074] | **.0012** |
| 27 | 0.991× | [0.962, 1.021] | .546 |

Layer 18's small positive effect then **fails its own scrambled-needle control** — with
scrambled content the same analysis gives 1.009× [0.974, 1.045], p = .594 at layer 18
(and 0.961× [0.937, 0.985], p = .0029 at layer 9, i.e. significant in the *wrong*
direction). A prompt-contamination check came back clean: 0 cases across all 84 prompts.

So the honest reading is that **C2 as specified was never a valid selectivity measure**,
and the corrected version finds no token-level, memory-specific retention signal on this
checkpoint. That is not the same as "AHN retains nothing" — it is "this readout cannot
show that it does."

**6. Corrected C3 does not give order-sensitivity in either direction.** The original C3
let the needle move when the context was shuffled. The rerun holds it at the same token
position: 8 needles × 2 eviction distances (1024, 4096) × ordered/shuffled × 3 layers =
**96 rows, 48 complete matched pairs, none dropped**.

| layer | rank effect | probability effect |
|---|---|---|
| 9 | no consistent direction | shuffled higher at both distances |
| 18 | shuffled better in **16/16** pairs | **reverses with distance** — shuffled far higher at 1024 (ordered/shuffled ≈ 0.106), ordered far higher at 4096 (≈ 104,969) |
| 27 | shuffled better in 13/16, not surviving Holm | shuffled higher in **16/16** |

Read carefully: this is **not** evidence that shuffling helps memory. It is C3 failing to
be the control it was designed to be — the effect depends on layer, on eviction distance,
and on whether you measure rank or probability, and those three disagree. A control whose
sign flips with distance cannot support a content-memory interpretation *or* refute one.

**One statistical caveat worth carrying into the paper.** Each layer × distance cell holds
8 matched pairs, so a two-sided exact sign-flip test has 2⁸ = 256 arrangements and a
minimum achievable p of 2/256 = .0078. After Holm across the six tests, the smallest
possible adjusted p is **.0469**. Any ".047" in this analysis means *all 8 pairs agreed in
direction*, not a finely resolved significance estimate.

**7. What this does to the diagnosis.** Candidate (c) — AHN as a recency mechanism — was
resting on the original C3 result. The corrected C3 does not support it. Combined with
(a) already ruled out by map stability, **candidate (b), prompt construction, is the only
live explanation left**, and §5 of Next steps says how to test it.

> **Reproducibility gap — fix before any of this goes in the paper.** The corrected C2
> statistics and the matched-pair C3 results exist only as markdown inside the two
> notebooks. `04_table4_controls.json` was regenerated but still carries only the original
> four controls. Nothing in `results/` holds the baseline-corrected folds, the CIs, or the
> 96 C3 rows. Export both to JSON under `results/run_3b_gdn/` so Tables 4 and 8 can be
> regenerated by `05_analysis_and_figures.ipynb` rather than transcribed by hand.

## Findings from the 2 Sep per-layer re-analysis — CORRECTED 3 Sep

> **Correction notice.** The first version of this section, pushed 2 Sep in `a9a140c`,
> claimed layer 27 was "a working instrument … above chance, C2 at 5.03×." **That claim was
> wrong on two independent grounds** and is withdrawn below. If you read this section
> before 3 Sep, re-read items 3 and 4. What survives is the layer-*heterogeneity* result,
> not the layer-27 rescue.

No new GPU time. `per_layer_controls.py` recomputes C1–C4 from the 624 rows already in
`04_retention_rows.json`, within layer instead of pooled. `extract_c2_corrected.py` pulls
Sơn's baseline-corrected C2 out of `04-C2-debug.ipynb` into
`results/run_3b_gdn/04d_c2_baseline_corrected.json`.

**1. Readout basis — the error that produced the withdrawn claim.**
`Expected_Tables_and_Figures` §1 pre-registers the Δ-readout **on the residual stream**
(`rank_c1_residual`), not the raw module output (`rank`). The first version of
`per_layer_controls.py` reported only the `o_t` basis. The two disagree sharply:

| layer | D-resid (pre-registered) | o_t (what was reported) |
|---:|---:|---:|
| 9 | 94,478 | 84,406 |
| 18 | 119,805 | 114,104 |
| 27 | **109,793** | **68,132** |

Chance is 75,968. In the pre-registered basis **no layer is above chance** — layer 27 is
33,825 ranks *worse* than chance, not 7,836 better. The script now reports both and
defaults to `d_resid_preregistered`.

**2. Layer 9's readout is degenerate — this survives.** Median entropy 0.088 nats against
a uniform of 11.93 means the readout is putting essentially all mass on one token: not
noisy, *confidently wrong*. It also **fails C4 in both bases** (in-window 123,480 vs
evicted 94,478 in D-resid; 118,723 vs 84,406 in o_t) — the ceiling below the floor, the
exact diagnostic the 18 Aug pilot flagged as proof a measurement is not measuring
retention. Layer 9 should not be pooled with the others.

**3. WITHDRAWN — layer 27 is not a working instrument.** Two independent refutations:

- **Basis.** In the pre-registered D-resid readout it is at 109,793, above chance, with
  only 33.2% of rows beating chance.
- **Baseline correction.** The 5.03× raw C2 ratio is exactly the statistic Sơn showed is
  dominated by pair identity. Corrected, it collapses:

| layer | raw C2 | baseline-corrected | 95% CI | permutation p | scrambled control |
|---:|---:|---:|---|---:|---:|
| 9 | 0.71× | 0.9953× | [0.9816, 1.0092] | .579 | 0.958× |
| 18 | 0.56× | 1.0463× | [1.0201, 1.0732] | **.0054** | 1.010× |
| 27 | **5.03×** | **0.9911×** | [0.9657, 1.0173] | .662 | 0.991× |

Layer 27's apparent 5× advantage is **entirely baseline preference**. Layer 18 is the only
layer with a corrected effect, it is tiny (+4.6%), and it does not survive the
scrambled-needle content control (1.010×, p = .548). Sơn's robustness battery backs this
up: leave-one-distance-out, leave-one-filler-out and a blocked permutation test all agree.

**4. Revised — in-window vs evicted is basis-dependent.** In the `o_t` basis the gap looked
decisive (layer 18: 40,644 in-window vs 114,104 evicted). In the pre-registered basis it
narrows to 102,908 vs 119,805 — C4 still passes at layers 18 and 27, but **the in-window
ceiling is itself above chance**, so this is not "the needle reads well until it is
compressed." Individual in-window rows do reach rank 758–2,581, so the readout is not
incapable; the medians are simply not above chance. The 21 Aug caveat is therefore
*partly* retired: the J-Lens does beat the plain logit lens in-window, but not enough to
put the median below chance.

**5. Candidate list, revised again.**

- **(a) undersampling** — still ruled out.
- **(b) prompt construction** — **back on the table.** Item 4's revision removes the reason
  for downgrading it: in-window needles do *not* read well in the pre-registered basis, so
  "the prompt is fine, compression destroys content" is no longer supported. The
  answer-prefix gap (§Next steps 5) is untested and remains the cheapest thing to check.
- **(c) content does not survive compression** — **unsupported either way.** It requires an
  in-window ceiling meaningfully better than the evicted floor. In the pre-registered basis
  there isn't one.
- **(d) layer selection** — **partly supported.** Layer 9 is demonstrably broken. But
  dropping it does not rescue 18 or 27, so this is a hygiene fix, not a diagnosis.

**6. What this leaves.** In the pre-registered readout basis, with C2 baseline-corrected,
**there is currently no layer and no control on which the instrument demonstrably works.**
That is a stronger and cleaner negative than the pooled battery gave, and it points at the
instrument rather than at AHN. The next test is construction (§Next steps 5), not more
cells and not more scale.

> **Process note, worth carrying into the paper's Limitations.** Both errors corrected here
> — pooling across a degenerate layer, and reading out in a non-pre-registered basis — were
> caught by comparing analysis code against `Expected_Tables_and_Figures`, not by looking at
> the numbers. The pre-registration earned its keep. The 5.03× also survived a first
> reading precisely because it pointed the way the hypothesis wanted.

## Repository layout

```
ahn_interp.py                 shared instrumentation — upload this next to any notebook
per_layer_controls.py         CPU-only: recompute Table 4 within layer, both readout bases
extract_c2_corrected.py       CPU-only: pull Son's baseline-corrected C2 out of the notebook
notebooks/
  00_setup_and_config_audit.ipynb    load a checkpoint; record window / sinks / router
  01_instrumentation_gate.ipynb      prove the hook captures the memory's contribution
  02_jlens_fit_and_validate.ipynb    fit the J-lens; run Table 3's five checks
  03_nowrite_reproduction.ipynb      Week-6 milestone: 38–42% changed answers
  04_niah_retention.ipynb            retention curves + controls C1–C4 (Tables 4, 6)
  04-C2-debug.ipynb                  C2 readout-bias investigation — Sơn, 28–31 Aug
  04_niah_C3_analyze.ipynb           corrected C3 shuffled-context rerun — Sơn, 29 Aug
  05_analysis_and_figures.ipynb      CPU only — Tables 5–9, Figures 3–8
  0.4b.ipynb                         RQ3 join — Sơn's build; see Findings
  pilot/                             Sơn's 18 Aug notebooks, kept for provenance
docs/
  UPSTREAM_README.md                 ByteDance's original README
  PROPOSAL.md                        pointer to the proposal + expected-artefacts docs
  GPU_PLAN_2026-08-20.md             direction audit + 10 GPU-h plan; source for the 20 Aug findings
results/
  run_3b_gdn/                        the run of record — GDN 3B, window 8064, sinks 128
    00_config_audit.json               checkpoint config as loaded
    01_instrumentation_gates.json      Gates A/B/C — all pass
    02_table3_jlens_validation.json    Table 3 checks — 4 (map stability) now passes, 2/3 fail
    jlens_qwen25_3b.pt                 fitted J-lens, layers 9/18/27, 1.92 GPU-h (gitignored)
    jlens_qwen25_3b_corpusB.pt         second fit for the stability check, 1.25 GPU-h (gitignored)
    03_nowrite_reproduction.json       60 examples, per-example rows + both metrics
    04_retention_rows.json             NIAH sweep, 210 configs, per-row ranks
    04_table4_controls.json            C1–C4 battery, pooled — C4 passes, C1/C2/C3 fail
    04c_per_layer_controls.json        the same battery within layer, both bases — Findings, 2–3 Sep
    04d_c2_baseline_corrected.json     baseline-corrected C2 per layer — every effect vanishes
    04b_joined_retention_task.json     RQ3 join — no significant rank/ΔF1 correlation
    05_table5_rq1.json                 RQ1 by stratum, with bootstrap CIs
    05_table6_retention_summary.json   per-layer half-life fit — inadequate, see Findings
  pilot_2026-08-18/                  superseded — see Findings above
src/ahn/                       upstream AHN implementation (unmodified)
eval/, examples/               upstream harnesses (unmodified)
artifacts/deprecated/          J18 fitted from one prompt; kept, not used
```

**Housekeeping — fixed 20 Aug.** `results/run_3b_gdn/jlens_corpusB.ckpt` (~50 MB) had been
committed to git twice — `.pt` was gitignored but `.ckpt` wasn't, so this checkpoint was
bloating repo history the same way the `.pt` rule was meant to prevent. `*.ckpt` is now in
`.gitignore` and the file is untracked (`git rm --cached`, kept on disk). `notebooks/
02-duplicate.ipynb` is **not** a stray duplicate — it's a teammate's in-progress 1000-context
corpus fit through notebook 02, kept intentionally. Neither `.pt` file (corpus A or B) is in
git — they only exist on the GPU box, so they're backed up to Hugging Face Hub instead:
[gautam-dphs/ahn-interp-jlens-qwen25-3b](https://huggingface.co/gautam-dphs/ahn-interp-jlens-qwen25-3b/tree/main)
(private, org-owned). Re-download from there on any new box rather than re-fitting.

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
`transformers==4.51.0` pin — that is why fitting and use are in different notebooks. The
`.pt` the fit produces is a plain tensor dict, so it loads back under the 4.51.0 env
without issue.

```bash
# one-time: the notebook-02 environment
python3 -m venv ~/jlens-venv && source ~/jlens-venv/bin/activate
git clone https://github.com/anthropics/jacobian-lens.git && pip install -e jacobian-lens
pip install datasets accelerate ipykernel
python -m ipykernel install --user --name jlens-venv --display-name "jlens (transformers>=5)"
```

**On a shared box, pin the GPU.** The default `device_map="cuda"` resolves to device 0,
which is usually the most contended. Check `torch.cuda.mem_get_info(i)` across devices and
pass `device_map="cuda:<idle_index>"` to `ai.load_ahn_model`, or set
`CUDA_VISIBLE_DEVICES` before importing torch. Independent notebooks can then be run in
parallel on separate devices — each 3B job needs only 6–16 GB of a 40 GB card, and wall
clock, not GPU-hours, is the binding constraint.

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

Steps 1–4 are done. Gautam's own instruction says not to scale past a failed control
battery, so steps 5–7 are **paused**, not skipped, until the C1/C3 disagreement below is
resolved.

1. ~~Run `01_instrumentation_gate.ipynb`~~ — **done**, all gates pass.
2. ~~Run `03_nowrite_reproduction.ipynb`~~ — **done**; see Findings item 5.
3. ~~Run `02_jlens_fit_and_validate.ipynb` with a real corpus~~ — **done** in 1.92 GPU-h;
   Table 3 checks 2 and 3 fail, diagnosed in Findings items 3 and 4. ~~Check 4 (map
   stability across disjoint corpora)~~ — **done**, passes at all three layers (0.91 / 0.87
   / 0.89). This rules out an unconverged fit as the explanation for the C1 failure below —
   see "Findings from the map-stability check and the RQ3 join."
4. ~~Run `04_niah_retention.ipynb` on GDN 3B~~ — **done**; see "Findings from the 20 Aug
   NIAH retention run" above. **C1, C2 and C3 all fail** — the readout does not find the
   needle above chance in the full NIAH sweep, despite beating the logit lens by 8–204× on
   isolated known-fact prompts (19–20 Aug Finding 4). Bootstrap on RQ1's first-line ΔF1 also
   shows the +6.11-point effect's 95% CI spans zero at n=60.
5. **Diagnose the C1/isolated-fact disagreement — the candidate list has moved twice.**
   (a) undersampling — **ruled out**, map stability passes. (c) the recency reading —
   **weakened** by the corrected C3, which shows no order-sensitivity in either direction,
   but the *related* claim that content simply does not survive compression is
   **strengthened** by the 2 Sep per-layer result (strong in-window signal, weak evicted
   signal at layers 18 and 27). (b) prompt construction — **weakened**: in-window needles
   read at rank ~800, so the prompt is not fatally malformed. Still worth testing directly,
   because `build_niah_prompt` ends at `"What was the special word?"` while `load_ruler`
   appends RULER's own `answer_prefix`, and the readout is taken at `pos=-1` — the two code
   paths disagree about whether the target sits in the next-token slot. (d) **layer
   selection — new and cheapest to act on.** Layer 9's readout is degenerate (0.088 nats)
   and fails C4. Re-run `04` without it, and with deeper layers added, before concluding
   anything about AHN.
6. ~~Build `04b` — the RQ3 join~~ — **done, by Sơn.** No significant correlation between
   memory rank and ΔF1 at any of the three layers, before or after Holm correction. Result
   is provisional, not a green light to proceed — see Finding 2 in the section above for
   why it inherits the same open question as C1.
7. **Add the boundary-JS column.** Notebook 01's Gate B already computes JS between the
   AHN and NOWRITE next-token distributions. Log it per example in notebook 03 and Table 8
   row 2 — the pre-registered comparison against prior work — comes for free rather than
   needing Gautam's numbers. Not blocked by 5; safe to do any time.
8. **The second and third cells at 3B — paused, not skipped.** `configs/run_3b_dn.json`
   and `run_3b_m2.json` exist and are correctly unrun. GPU_PLAN is explicit: a second and
   third cell measured with a broken instrument is nine wasted GPU-hours, so this waits for
   5.
9. **7B last**, gated on the J-lens map cost measured in step 3 (Table 10 row 2) and on the
   C1 diagnosis above.

### To raise with Gautam now, not later

- **Open Question 4 is answered**: the combination is a plain sum before `o_proj`, so
  `o_t` isolates cleanly. Confirmed empirically by Gate A, not just by reading the source.
- **Open Question 6**: he uses 8,064 on LongBench-E himself, which makes our 8064 window
  easy to defend. Confirm it in writing so it goes in Methods as a stated parameter.
- **The J-lens decision has new evidence, and the cheapest explanation is now closed off.**
  The lens beats the logit lens by up to 204× on eight isolated known-fact prompts, but
  notebook 04's full control battery shows the readout at chance (C1 fails, mean rank
  87,688 vs a chance rank of 75,968) with no needle/distractor separation (C2 fails) and
  shuffled context scoring *better* than ordered context (C3 fails). The map-stability
  re-fit (Table 3 check 4) has now run and **passes** at all three layers (0.91/0.87/0.89
  top-10 overlap) — the lens is converged, so this isn't an unconverged fit. `04b` (the RQ3
  join) also ran and finds no significant rank/ΔF1 correlation at any layer, but that
  result inherits the same open question rather than resolving it. What's left: (a) a
  direct comparison of the NIAH sweep's prompt construction against the known-fact test's,
  or (b) accepting C3's result as real — AHN behaving closer to a recency mechanism than a
  content store. Both are his call, not a compute problem anymore.
- **RQ1's effect does not survive its own confidence interval.** Bootstrapped first-line
  ΔF1 is +6.11 pts but the 95% CI is [−1.85, +14.42] at n=60 — it spans zero, and so does
  every per-stratum interval. His published +0.4 to +2.3 pts sits entirely inside that
  interval, so the earlier "our effect is 3× his" framing was a cohort-size artifact, not a
  real effect-size gap — worth telling him in those terms rather than asking about cohort
  length distributions.

~~Question 1 (training-only compute-reimbursement rule)~~ — **resolved**: work has been
running continuously on the GPU box Algoverse provided directly, so the reimbursement
question doesn't block anything in practice.

Still open, not blocking: whether AHN runs on a free T4 with an FP16 + eager-attention
fallback (Question 3), which notebook 00 now tests directly.

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
