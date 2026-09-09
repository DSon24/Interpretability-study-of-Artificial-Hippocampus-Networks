# Findings

Every finding from the project, in the order it was written. Split out of the README on
6 Sep 2026 -- it had grown to 1,085 lines, more than half of them this log, which made the
parts a newcomer actually needs (setup, how to run an experiment, where things stand)
hard to reach.

Nothing here is edited. Later entries correct earlier ones rather than replacing them, so
a claim's history stays visible: read top to bottom and the withdrawals are part of the
record. The 8 Sep content-swap entry is the most recent experimental state of the C1
question, and it withdraws the 7 Sep permutation-test entry's central claim. The mentor
decision immediately after it records how that evidence changes execution.

- [Findings from the 19–20 Aug run](#findings-from-the-1920-aug-run)
- [Findings from the 20 Aug NIAH retention run](#findings-from-the-20-aug-niah-retention-run)
- [Findings from the map-stability check and the RQ3 join](#findings-from-the-map-stability-check-and-the-rq3-join)
- [Findings from the 18 Aug pilot](#findings-from-the-18-aug-pilot)
- [Findings from the 21 Aug C1 diagnosis](#findings-from-the-21-aug-c1-diagnosis)
- [Findings from the 28–31 Aug C2 and C3 investigation](#findings-from-the-2831-aug-c2-and-c3-investigation)
- [Findings from the 2 Sep per-layer re-analysis — CORRECTED 3 Sep](#findings-from-the-2-sep-per-layer-re-analysis-corrected-3-sep)
- [Findings from the 4–5 Sep C1 rank correction, construction ladder, and needle-category test](#findings-from-the-45-sep-c1-rank-correction-construction-ladder-and-needle-category-test)
- [Findings from the 7 Sep RULER cohort run](#findings-from-the-7-sep-ruler-cohort-run)
- [Findings from the 7 Sep J-lens repeat and placement check](#findings-from-the-7-sep-j-lens-repeat-and-placement-check)
- [Findings from the 7 Sep target-scoring bug and the RULER control battery](#findings-from-the-7-sep-target-scoring-bug-and-the-ruler-control-battery)
- [Findings from the 7 Sep permutation test: layer 27's sign is needle-content-dependent](#findings-from-the-7-sep-permutation-test-layer-27s-sign-is-needle-content-dependent) — **WITHDRAWN 8 Sep**
- [Findings from the 8 Sep content swap: content is not the variable](#findings-from-the-8-sep-content-swap-content-is-not-the-variable)
- [Mentor decision after the 8 Sep checks](#mentor-decision-after-the-8-sep-checks)
- [Findings from the DeltaNet RULER control battery (8 Sep)](#findings-from-the-deltanet-ruler-control-battery-8-sep)
- [Findings from the DeltaNet RQ1 rerun (nb03, 9 Sep)](#findings-from-the-deltanet-rq1-rerun-nb03-9-sep)
- [Findings from the Mamba2 RULER control battery (9 Sep)](#findings-from-the-mamba2-ruler-control-battery-9-sep)
- [Findings from the Mamba2 RQ1 run (nb03, 9 Sep)](#findings-from-the-mamba2-rq1-run-nb03-9-sep)
- [Findings from the 1000-context J-lens map-stability refit (9 Sep)](#findings-from-the-1000-context-j-lens-map-stability-refit-9-sep)
- [Findings from the r29 evicted-vs-in-window J-lens re-run (9 Sep)](#findings-from-the-r29-evicted-vs-in-window-j-lens-re-run-9-sep)
- [Findings from the no-AHN floor run (r54, 9 Sep)](#findings-from-the-no-ahn-floor-run-r54-9-sep)
- [Findings from the RULER retention curve on corrected scoring (Run 025 redone, 9 Sep)](#findings-from-the-ruler-retention-curve-on-corrected-scoring-run-025-redone-9-sep)
- [Findings from the RULER needle-placement check on corrected scoring (04q, 9 Sep)](#findings-from-the-ruler-needle-placement-check-on-corrected-scoring-04q-9-sep)
- [Findings from the Kashyap scoring-convention reconciliation (9 Sep)](#findings-from-the-kashyap-scoring-convention-reconciliation-9-sep)

---

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

> **Corrected 3–9 Sep.** The **+6.11 pts** ΔF1 in the table above was recomputed by hand
> from the first-line means, not regenerated from the saved per-example `_fl` rows. The
> 3 Sep pre-registration audit (Experiment Log run 022) and the 9 Sep regeneration
> (`results/run_3b_gdn/03_nowrite_reproduction.json`, commit 363e996 — its `metric_note`
> now states the headline is generated from the first-line fields only) give **+3.34 pts**
> pooled, with the answer-change rate unchanged at 33.3%. Short and mid match to the
> decimal; the **long stratum flips sign** (+5.72 → **−2.61**). The `03`/`05` JSON
> summaries and the README now carry +3.34; this table row is left as first written per
> the append-only convention.

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

> **Superseded 9 Sep.** The +6.11 / [−1.85, +14.42] figures here came from the by-hand
> first-line means. Regenerated from the saved `_fl` rows: pooled first-line ΔF1
> **+3.34 pts, 95% CI [−4.70, +11.10]**; short +9.17 and mid +3.45 unchanged, long
> **−2.61 [−18.00, +11.72]** (sign flip). The n=60 reading is unchanged — no significant
> behavioural effect, every per-stratum interval spans zero.

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

## Findings from the 4–5 Sep C1 rank correction, construction ladder, and needle-category test

> **Revises, does not contradict, the "no layer works" bottom line above.** Item 6 of the 2
> Sep section says there is "no layer and no control on which the instrument demonstrably
> works." That was true of the 4-needle design it was measured on. Extending the needle set
> below shows the instrument does work, for some words, at two of the three layers — the
> earlier negative was an artifact of which four words got tested, not a property of the
> instrument itself.

First GPU access on program-provided A100s (replacing the earlier no-SSH shared box).
Three linked pieces of work, in the order they happened.

**1. C1 baseline-corrected by rank, the same way C2 was.** `04-C2-debug.ipynb` cell 40's
84-forward-pass sweep only ever recorded `p_needle`/`p_distractor`; added `rank_needle`/
`rank_distractor` and persisted the frame (previously it only ever lived in notebook
memory — the same gap C2's `df_bias` had). Matched/clustered design identical to the C2
statistics (21 independent condition-level values per layer, bootstrap CI, sign-flip
permutation test, Holm correction across 3 layers):

| layer | mean rank, target stored | mean rank, other stored | mean delta (95% CI) | Holm p | significant |
|---:|---:|---:|---|---:|:---:|
| 9 | 77,134 | 77,376 | −242.2 [−398.2, −89.4] | .0126 | yes |
| 18 | 90,883 | 91,743 | −860.0 [−1365.0, −396.3] | .0054 | yes |
| 27 | 63,041 | 61,952 | **+1089.6** [+369.2, +1823.7] | .0126 | yes |

Chance rank 75,968. Layers 9 and 18 move in the direction a real memory signal predicts,
but the effect is small — under 1.5% of vocab. Layer 27 is significant in the *wrong*
direction: rank gets worse when the needle is actually stored. None of the individual
layer×needle tests (12 total, n=21 each) survive Holm correction — the effect only
appears once pooled across the 4 needles. This by itself does not resolve C1; item 3
below explains why pooling across needles is exactly the wrong move here.

**2. The construction ladder ran on GPU for the first time.** `probe_construction.py` was
written 3 Sep against a CPU-only base-model check (the answer-prefix costs ~30 ranks, not
the ~87,000 separating C1 from chance) but had never been run through the real merged
model + fitted J-lens + actual eviction — it needed a GPU, and the box available at the
time had no SSH. Two bugs fixed to get a full run: `probe.run(inputs, keep_logits=True)`
was missing `layers=LAYERS`, so it captured all 36 AHN layers instead of 3 (~12× the
needed memory, OOM'd at rung 5 of 8 even after the fix below); and `ModelBundle.summary()`
called `dataclasses.asdict(self)`, which deep-copies every field — including the 3B-param
`model` field — before the dict comprehension filters it out, so building a metadata dict
was cloning the whole model on GPU first. Both fixed in `ahn_interp.py` and
`probe_construction.py`.

With those fixed, all 8 rungs complete. Reading `o_t` (AHN's memory contribution
specifically — the column that actually answers the C1 question, not the model's own
final-layer logits) for the single needle "Paris" at eviction distance ≈1024:

| rung | L9 `o_t` rank | L18 `o_t` rank | L27 `o_t` rank |
|---|---:|---:|---:|
| L3 in-window, no prefix | 116,491 | 109,630 | 2,190 |
| L4 in-window, with prefix | 127,346 | 129,821 | 43,002 |
| L5 evicted, no prefix | 89,792 | 147,618 | **6,703** |
| L6 evicted, with prefix | 109,408 | 146,636 | **33,092** |
| L7 evicted, natural filler | 115,153 | **12,893** | 14,284 |

Layer 27 stays well below the 75,968 chance rank in *every* rung, evicted included — this
single-example number (L5, distance≈1024) matches the population sweep's own Paris/layer
27/distance≈1040 rows exactly (6,703 / 7,984 / 7,008 across the three fillers in item 1's
sweep), cross-validating between the two independently-written scripts. Layer 18 is dead
at chance for repeated filler (L3–L6) but recovers sharply with natural filler instead of
the degenerate repeated 5-sentence filler (L7: 12,893) — one example, but a 10× swing from
changing filler text alone. Layer 9's `o_t` never recovers anywhere, consistent with it
being independently established as degenerate.

**3. The pooled/4-needle C1 failure is a needle-selection artifact, not evidence AHN
retains nothing.** Pulled needle-level means (n=21 conditions each) from item 1's sweep,
split by needle rather than pooled:

| needle | layer 27 mean rank |
|---|---:|
| Paris | 9,767 |
| Tokyo | 17,300 |
| lantern | 77,704 |
| banana | 147,394 |

A 15× gap between the best and worst needle, ranges barely overlapping (Paris tops out at
38,556; banana bottoms out at 143,182) — not noise. The pooled verdict averages a working
signal against a broken one. This is the same pooling mistake the 2 Sep section already
caught with layer 9 (Finding 2 there), just found on a second axis: needle identity, not
layer.

**4. Extended to 24 needles (12 place names, 12 common nouns) to test whether this
generalizes past the original 4.** Tokenizer-verified single-token, same pattern as the
existing `CONTROL_CANDIDATES` cell; one forward pass per (needle, distance, filler), ~500
passes total. Two-sample permutation test on needle-level mean rank, Holm-corrected across
3 layers:

| layer | mean rank, place (n=12) | mean rank, noun (n=12) | diff | Holm p | significant |
|---:|---:|---:|---:|---:|:---:|
| 9 | 82,610 | 92,388 | −9,778 | .488 | no |
| 18 | 71,727 | 89,056 | −17,329 | **.0164** | yes |
| 27 | 24,063 | 92,491 | **−68,428** | **.00045** | yes |

At layer 27, all 12 place names land below chance (worst is Lisbon, 64,543); 10 of 12
nouns land above it. Not a perfectly clean split — `green`/`stone`/`window`/`cloud` also
read below chance alongside the place names — so "place name" is not the whole mechanism,
but the category-level difference is real and Holm-significant, not an artifact of which
24 words got picked this time either.

**5. Candidate list, revised a third time.**

- **(a) undersampling** — still ruled out.
- **(b) prompt construction / answer-prefix** — still weakened; the ~30-rank cost from the
  3 Sep CPU probe is nowhere near the gap to chance. **Narrows to a new, untested version**:
  Finding 2's L7 result suggests the *degenerate repeated filler*, not the missing prefix,
  may be masking signal at layer 18 specifically. One example — needs the same
  population-level test Finding 4 gave the needle-identity question before it can be
  believed.
- **(c) content does not survive compression (recency reading)** — **unsupported**. Layer
  27's `o_t` rank for Paris/Tokyo stays low after eviction, in both the single-example
  ladder and the 21-condition population sweep. Content demonstrably *can* survive
  compression, at least for some words.
- **(e) new — needle-identity / category-dependent retention.** Layers 18 and 27 show
  Holm-significant place-name-vs-common-noun differences at n=12 per category, not just
  the original 4-needle spot check. Whether the operative dimension is really semantic
  category, word frequency, embedding geometry, or something else is untested — 24 needles
  across 2 hand-picked categories establishes the effect is real, not what causes it.

**6. What this leaves.** C1 does not cleanly fail. It fails for some words and works for
others, at layers 18 and 27 (layer 9 remains dead throughout, every finding above is
consistent with its independently-established degeneracy). That is a different thing to
bring to Gautam than either original candidate — not "the instrument is broken," not "AHN
is a recency mechanism," but "AHN's compressed memory retains some word identities far
better than others, and the pooled battery's chance-level verdict is an averaging artifact
over which needles were chosen." The cheapest next test is the same kind of extension
Finding 4 ran: more needles, testing whether the category split holds under a
principled category design (e.g. matched-frequency place names vs. common nouns) rather
than the current hand-picked 24, and whether Finding 2's filler-degeneracy hint replicates
at scale.

> **Process note.** Two OOM bugs (Finding 2) and one needle-selection artifact (Finding 3)
> were all found by reading the actual data closely enough to notice something matched or
> didn't match a prior result, not by the code failing loudly — the OOMs did fail loudly,
> but the needle-selection artifact would have shipped silently as "C1 fails, full stop" if
> the cross-check against the population sweep hadn't been run at all.


## Findings from the 7 Sep RULER cohort run

Run 024. `results/run_3b_gdn/04g_ruler_niah_rows.json` (180 rows),
`04g_ruler_niah_stats.json`; regenerate with `python ruler_cohort_stats.py`. This closes
DIVERGENCE 2 from the 3 Sep pre-registration audit and is the first evidence that moves
candidate (b) since 21 Aug.

**1. The real RULER cohort had never been run, and `load_ruler` had zero call sites.**
Expected Tables row 1 of Table 2 makes RULER NIAH n=60 the *primary* RQ2 cohort — "ground
truth y\* is known by construction" is the proposal's number-one reason-to-believe — and
every NIAH number in this project to date came from the homemade `build_niah_prompt`
instead. `ai.load_ruler()` had existed since notebook 04's first sweep and was never
called by anything.

**2. `config="8192"` skipped 60/60 examples, and the reason matters beyond this run.**
The first attempt used RULER's `"8192"` bucket on the reasoning that
`sliding_window (8064) + num_attn_sinks (128) = 8192`, so it must sit exactly at the
activation threshold. Measured, that bucket tokenizes to a median of **7,881** tokens
under Qwen — *under* the threshold, so AHN never activated and every example was skipped.
`simonjegou/ruler`'s length buckets were built with a different tokenizer, and Qwen's
151,936-token vocabulary compresses English into fewer tokens than whatever built them.
Only `4096`/`8192`/`16384` exist as configs. `16384` measures 15,678–15,684 tokens and
activates for 60/60. **Any future length-matched comparison against a published benchmark
has to re-measure the length in our own tokenizer rather than trusting the bucket name.**

**3. Prompt construction changes the layer-27 readout from worse-than-chance to
better-than-chance.** Same box, same day, same readout (logit lens — see caveat 5), C1
residual-delta basis, bootstrap median CIs over 10,000 resamples:

| layer | homemade evicted (n=168) | RULER (n=60) |
|---|---|---|
| 9 | 110,956 [89,767, 118,623] — above chance | 77,528 [65,652, 91,123] — spans chance |
| 18 | 68,518 [56,571, 77,612] — spans chance | 70,693 [64,498, 80,703] — spans chance |
| 27 | 89,805 [79,992, 99,068] — above chance | **43,139 [27,536, 59,226] — below chance** |

Chance is 75,968; lower rank is better, so "below chance" is the readout succeeding. At
layer 27 the two intervals do not overlap. At layer 9 the RULER cohort is also markedly
better. Layer 18 is unchanged.

**4. This is the strongest evidence yet for candidate (b), and it partly rehabilitates a
withdrawn claim.** The 2 Sep "layer 27 is a working instrument" claim was withdrawn on 3
Sep on two grounds: wrong readout basis, and a C2 ratio that turned out to be pure
pair-identity baseline. Nothing here reinstates it — this is a different cohort in a
different basis and the C2 objection is untouched. What it does say is that the *cohort*
was suppressing the layer-27 signal, not only the basis. Candidate (b) was previously
described as "weakened — in-window needles read at rank ~800, so the prompt is not fatally
malformed". That reasoning is now insufficient: a prompt can be well-formed and still
place the target where the readout cannot reach it. The specific difference remains the
one flagged when this was wired: `build_niah_prompt` ends at `"What was the special
word?"` with nothing after it, `load_ruler` appends RULER's own `answer_prefix`, and the
readout is taken at `pos=-1` in both.

**5. Five caveats, none of them optional when this is quoted.**

- **Logit lens, not the J-lens.** `jlens_qwen25_3b.pt` is gitignored and was not
  re-downloaded after the box reset, so notebook 04 took its documented fallback. Both
  cohorts here are logit-lens, so the *comparison* is sound, but the absolute numbers are
  not the pre-registered instrument — the J-lens beats the logit lens by 8–204× on
  isolated known-fact prompts (Finding 4, 19–20 Aug). **The layer-27 result must be
  re-run with the J-lens before it goes anywhere near the paper.**
- **Not length-matched.** RULER at `16384` is ~15.7K tokens; the homemade evicted rows
  span distances 64–8192. Eviction distance is a known driver of rank, so cohort and
  length are confounded in this comparison.
- **Single-token target.** Only the gold answer's first token is scored, the same
  convention as the RQ3 join. RULER answers are frequently multi-token, so this measures
  something weaker than "the answer was retained".
- **C1-shaped only.** RULER supplies one prompt and one gold answer per example — no
  distractor pair, no clean way to shuffle just the needle region. There is no C2, C3 or
  C4 analog here and none should be claimed.
- **Three layers, no multiplicity correction across them**, and `lens_validated=false`
  travels on every row.

**6. Process note — two errors caught by the writeup, not by the code.** The first read of
this run compared RULER's `o_t` numbers against the homemade *J-lens* D-resid numbers and
concluded the opposite result ("decisive against candidate (b)"). Two separate
mismatches — wrong basis, wrong readout — pointing the same wrong way. Both were caught
only when the numbers were put in a table next to their provenance. `ruler_cohort_stats.py`
exists so these numbers regenerate from the rows with the basis printed next to every
figure, instead of being transcribed.


## Findings from the 7 Sep J-lens repeat and placement check

Run 025. `04g_ruler_niah_rows.json` (J-lens), `04g_ruler_niah_stats.json`,
`04h_ruler_needle_position.json`; regenerate with `python ruler_cohort_stats.py` and
`python ruler_needle_position.py`. This supersedes the logit-lens entry above and is the
first result in this project where the readout finds an evicted needle well above chance
in the pre-registered instrument.

**1. Layer 27 reads the evicted RULER needle at rank 17,250 of 151,936.** J-lens,
pre-registered D-resid basis, bootstrap median CIs over 10,000 resamples, chance 75,968:

| condition | n | median rank | 95% CI | |
|---|---|---|---|---|
| RULER, needle evicted | 32 | **17,250** | [11,318, 30,936] | below chance |
| RULER, needle in-window | 28 | 54,506 | [32,372, 65,194] | below chance |
| homemade `build_niah_prompt`, evicted | 168 | 111,694 | [105,947, 121,525] | above chance |

84.4% of evicted RULER examples beat chance. The same instrument, on the same box, the
same day, reads the homemade prompts at 111,694 — worse than chance. The cohort is the
variable.

**2. The in-window confound was checked and it runs backwards.** RULER varies needle
depth, and 28 of 60 needles (46.7%) landed inside the 8064-token local window, where
attention can read them directly without any compression. That was the obvious
alternative explanation for the layer-27 result, and it is refuted: the **evicted** subset
reads *better* than the in-window subset, 17,250 against 54,506, a median gap of 37,256
with a permutation p of 0.0021.

That direction is mechanistically the right one, which is worth stating because it is easy
to misread as an anomaly. The D-resid basis measures `resid(AHN) − resid(NOWRITE)` — what
the memory pathway specifically contributes. When the needle is in-window, both runs can
read it through attention, so the difference between them is small and noisy. When the
needle is evicted, the memory is the only channel carrying it, so the difference *is* the
signal. The control behaves as designed.

**3. Retention does not decay across the range measured.** Splitting the evicted subset at
its median eviction distance: near half (median 2,753 tokens past the boundary) reads
15,844, far half (median 5,822) reads 17,250, across a full range of 38 to 7,406 tokens.
Flat. Table 6's exponential fit was already reported as inadequate (R² < 0 at all three
layers); this says the reason may be that there is no decay to fit over this range, not
that the fit was mis-specified.

**4. What this does and does not overturn.** It does not reinstate the 2 Sep "layer 27 is
a working instrument" claim, which was withdrawn on 3 Sep for using the `o_t` basis and
for a C2 ratio that was pure pair-identity baseline. This is a different cohort, in the
pre-registered basis, and the C2 objection is untouched — no C2 analog was run here. What
it does overturn is the reading that the C1 failure is a property of AHN. On the primary
pre-registered cohort, in the pre-registered basis, the readout finds evicted content well
above chance at layer 27. The homemade `build_niah_prompt` construction was hiding it.
Candidate (b) is no longer "weakened"; it is the explanation, at layer 27.

**5. Layers 9 and 18 go the other way, and that asymmetry is now the open question.** On
RULER they read 137,550 and 139,168 — worse than chance, near the top of the vocabulary
range, and worse than they read on the homemade prompts. Whatever layer 27 is doing, the
other two probed layers are not doing it. Layer 9's readout was already known to be
degenerate (0.088 nats, fails C4 in both bases). Layer 18 has no such excuse.

**6. Caveats that must travel with any use of this.**

- **`lens_validated=false`.** Table 3 checks 2 and 3 still fail. The instrument has not
  passed its own validation battery, and every row carries the flag.
- **Single-token target.** Only the gold answer's first token is scored. RULER answers are
  frequently multi-token, so this is weaker than "the answer was retained".
- **n=32 evicted examples**, one layer, one cell, one scale.
- **C1-shaped only.** No C2, C3 or C4 analog exists on RULER. C2 is precisely the control
  that killed the previous layer-27 claim, so its absence here is not a small gap.
- **The pre-registration gate still stands.** Expected Tables Table 4 requires C1–C3 to
  pass on the primary condition before RQ2 or RQ3 is populated. This is a C1-shaped pass
  on a cohort that did not exist in the repo four days ago; it does not retroactively pass
  C2 or C3, and the deviation write-up in Methods is still owed.
- **Not length-matched.** RULER at `16384` is ~15.7K tokens; the homemade evicted rows
  span distances 64–8192 at shorter totals.

**7. Process note.** Three readings of this cohort were wrong before this one, each caught
by a check rather than by the code failing. The first compared `o_t` numbers against
J-lens D-resid numbers and concluded the opposite result. The second ran on the logit lens
because the J-lens file exists on the Hub under a different name, and silently overwrote
the J-lens run of record. The third — this one — looked correct until the in-window
question was asked, and would have been a headline claim resting on 28 needles that were
never compressed. The cost of each check was minutes; the cost of publishing any of the
three would not have been.


## Findings from the 7 Sep target-scoring bug and the RULER control battery

Run 026. `04i_ruler_controls_rows.json`, `04i_ruler_controls_stats.json`; regenerate with
`python ruler_controls.py`. **This withdraws the specific numbers in the entry above** (run
025, rank 17,250 at layer 27) and replaces them with a smaller, differently-shaped, but
better-supported result.

**1. Run 025 scored a space character, not the needle.** RULER NIAH answers in this cohort
are 7-digit numbers, and Qwen tokenizes `" 7700828"` as
`[' ', '7', '7', '0', '0', '8', '2', '8']` — the leading space is its own token.
`measure_ruler`'s target was `encode(" " + answer)[0]`, token 220, **identical for all 60
examples**. Run 025's rank-17,250 headline is the rank of a space, not of any digit. This
surfaced immediately on trying to build C2: with one distinct target token, every one of
the 1,770 possible pairs dropped as a same-token collision, and C2 could not run at all.

**2. The measurement now scores the answer as a sequence.** The gold answer is appended to
the prompt; the readout at each of the 7 digit positions gives the log-probability of all
ten digit tokens, a 7x10 table per example. C1 is the example's own answer read from its
own table. C2 is any *other* example's answer read from the same table — the full 60x60
cross matrix costs nothing beyond the forward pass C1 already needed.

**3. C3-lens disqualifies layers 9 and 18.** DIVERGENCE 3a (the proposal's Table 4 defines
C3 as a row-permuted J-lens map, never implemented before this) is decisive here: at
layers 9 and 18 the signal *survives* the permuted map (Δ +98.2 and +225.0 log-nats,
both CIs excluding zero on the wrong side). Whatever those layers show is a property of
the decoding procedure, not of memory — most importantly, layer 18's striking C1 number
(median mean-digit-rank 6,254, deep below chance) **is exactly this artefact** and must
not be reported as evidence of anything.

**4. Layer 27 passes every check, at a real but modest magnitude.**

| control | result |
|---|---|
| C1 | median mean-digit-rank 49,776 [41,803, 54,797] — below chance (75,968) |
| C2, per-digit effect | **1.076x [1.054, 1.098]**, permutation p < 0.0001 |
| C2, per-example effect | 1.672x [1.446, 1.929] — excludes 1.0, does not clear the pre-registered 10x bar |
| C3-context | no order sensitivity (Δ −1,004 [−7,946, +18,689]) |
| C3-lens | **collapses**, Δ −39.2 log-nats [−46.1, −34.4], signal depends on the real map |
| C4 layer-permutation *(added 9 Sep)* | **degrades**, Δ −43.8 log-nats [−58.0, −38.4], readout is layer-specific |

C2's CI excludes the null about as unambiguously as this kind of test produces. It also
does not come close to the magnitude that would count as a clean pass. Both of those are
the finding: a small, statistically robust, memory-specific effect at layer 27, an order
of magnitude short of "AHN clearly retains the content."

**5. The per-example vs per-digit distinction is not a technicality here.** The
ratio-of-ratios C2 statistic is the *square* of the per-example effect (Finding, process
note below), and the per-example effect itself compounds multiplicatively over the
answer's 7 digits. 1.076x per digit is already 1.672x end to end; the pre-registered 10x
bar was written for a single-token needle and is not the number this design should be
measured against. `effect_per_digit` is the figure to quote.

**6. C3-context replicates the 29 Aug null, on a different cohort and a different
construction.** No order sensitivity at layer 27 here; none on the homemade cohort in the
28-31 Aug investigation. Two independent constructions agreeing on "no clean order effect"
is corroboration, not a second failure to explain away.

**7. What this does and does not license.** It does not reinstate the 2 Sep claim (o_t
basis, C2 confound) or the withdrawn run-025 magnitude. It licenses: *a small,
statistically significant, memory-specific signal exists at layer 27 on the primary
pre-registered cohort, that is not an artefact of the decoding procedure, and that falls
well short of the pre-registered threshold for a clear pass.* That is a materially weaker
and more defensible claim than either "the instrument doesn't work" (20 Aug) or "AHN
clearly retains the needle" (run 025, now withdrawn).

**8. Process note — the second self-caught error in two days.** `ruler_controls.py` was
tested against synthetic null and planted-effect cohorts before being run against real
data, the same discipline that caught the fold-squaring bug in the estimator itself. Two
bugs were caught this way before any of this reached a notebook: the space-token target,
found because C2 literally could not execute; and the squared fold, found because a
planted 50x effect reported as 2500x on synthetic data. Neither would have been visible
from the real RULER numbers alone — a 1.076x-per-digit effect does not look obviously
wrong the way a space-token rank does. The synthetic-data check is now load-bearing for
any future control on this cohort, not optional scaffolding.


## Findings from the 7 Sep permutation test: layer 27's sign is needle-content-dependent

> **WITHDRAWN 8 Sep.** The central claim — that layer 27's sign depends on needle
> content — does not survive a direct content swap holding construction and length
> fixed. Sơn's homemade negative result does not replicate beyond his 4 needles. See
> [8 Sep content swap](#findings-from-the-8-sep-content-swap-content-is-not-the-variable).
> The permutation methodology in this entry stands; the interpretation does not.

Sơn told Gautam in Slack: *"I've finished tracing the experiment like you asked — the
signal is still there... the failures are coming from the control itself, not the code."*
His notebook (`04-C2-debug.ipynb`) actually contains two conclusions that disagree with
each other and with that message. This entry resolves which one the data supports, and it
is neither "no signal" nor Sơn's Slack claim — it is a third thing.

**1. Sơn's own best-designed test already said C2 fails.** Cells 105–110 build a
multi-control baseline: each target's own readout probability against the mean of 4
unrelated single-token controls (`river`, `chair`, `window`, `garden`), matched by distance
and filler, t-tested across 21 conditions per layer. Result: layer 9 significantly
**negative** (0.921x [0.860, 0.985], p=0.020), layer 18 null (1.057x [0.942, 1.185],
p=0.329), layer 27 significantly **negative** (0.863x [0.768, 0.969], p=0.016). His own
written conclusion: *"the expected positive retention-specific effect is therefore not
reliably supported... C2 debugging is stopped here."*

**2. The Slack claim comes from a different, uncontrolled analysis two cells later.** A
single example — Paris, layer 27, rank 6,703 — plus a WRITE-vs-NOWRITE check, concluding
*"the signal does not disappear... memory retains content."* This does not control for
Paris being a generically favoured completion independent of what is stored — the exact
confound identified on 28 Aug (*"mango preferred regardless of what was actually
stored"*). In his own multi-control table, Paris specifically is **−0.204 log-units
(0.82x) at layer 27** — negative, the same needle and layer used as the positive example.

**3. A single permuted lens was ambiguous in the wrong direction.** Re-running his exact
168-forward-pass grid, decoding through both the real J-lens and one row-permuted lens
(the same C3-lens check applied to RULER): layer 27 real = 0.863x (replicates his number),
shuffled = **1.118x [1.070, 1.168], p<0.0001** — significant, tighter, and the opposite
sign. A genuine artifact should revert toward the null under permutation, the way RULER's
C3-lens does (real logP −113.4 → shuffled −152.9, worse, as expected). A permutation that
flips sign and gets *more* confident is not that pattern, and one arbitrary permutation
cannot distinguish a real structural artifact from an unlucky draw on n=21 conditions.

**4. Twenty independent permutations settle it: the negative effect is real, not an
artifact of decoding structure.** At layer 27, the real-lens fold (0.863x) is more extreme
than **all 20** independently-seeded permutations (null range [0.911, 1.105]); permutation
p = 0.048 — the best resolution 20 draws allow (1/21), since real ranks most extreme of 21
values. Layer 9 shows the identical pattern: real (0.921x) below the entire null range
[0.962, 1.025], same p = 0.048. Layer 18 stays null (real at the 90th percentile of the
permutation distribution, not extreme). If either negative effect were an artifact of
"any" linear readout rather than the fitted J-lens specifically, the real value would land
inside the permutation distribution, not consistently below all of it, at two independent
layers.

**5. Two independently-verified-real effects, opposite signs, same layer, same
checkpoint.** RULER's layer 27 (7 Sep control battery, above): 1.672x [1.446, 1.929],
p<0.0001, verified real via its own C3-lens collapse. Sơn's homemade-cohort layer 27:
0.863x [0.768, 0.969], p=0.016, now verified real via 20-permutation test. Neither is a
decoding artifact. The needle content differs completely: common English words and place
names (Sơn's set) versus 7-digit numeric sequences (RULER). **Layer 27's sign depends on
what kind of content is stored.** This is a real, open question, not a discrepancy to
average away or a tiebreak between cohorts.

**6. What this changes and does not change.** It does not change RULER's status as a
verified result — it stands. It does change what "layer 27 works" can mean: not a
content-general memory readout, but one whose direction flips with needle type. Whatever
mechanism produces a positive effect for digit sequences is actively suppressing
retrieval for common words at the same layer, in the same checkpoint. That is a more
specific and more interesting claim than either "AHN retains content" or "the control is
broken," and it is not yet explained by anything in the pre-registration.

**7. Precision caveat.** Twenty permutations cap the achievable p-value at 1/21 ≈ 0.048;
the test shows real is more extreme than every permutation drawn, not by how much. A
99-permutation repeat would resolve this to 1/100 if a tighter number is needed before
this goes in front of a reviewer — the qualitative conclusion (real ranks most extreme at
two independent layers) does not depend on it.

**8. Process note.** This is the fourth self-caught correction on C2 in three days (the
space-token target, the squared fold, the single-arbitrary-shuffle ambiguity, and now the
direction of this finding itself — an initial hope that the negative effect would turn out
to be an artifact was wrong). Each was caught by testing the statistic against synthetic
null and planted-effect data before trusting real output, and by refusing to accept a
result from a single run — one example, one pair, one permutation — as sufficient on its
own.


## Findings from the 8 Sep content swap: content is not the variable

Run 028. `04l_content_swap_rows.json`, `04m_content_swap_short_rows.json`,
`04n_content_swap_stats.json`; regenerate with `python content_swap_stats.py`.
**This withdraws the entry immediately above it.** Layer 27's sign is not
needle-content-dependent; that claim was made on a comparison that confounded content with
construction, and it does not survive a direct test.

**1. What Gautam asked, and why it was one experiment.** On 8 Sep: *"I'd do the
length-matched control first, since that directly tests whether the cohort/construction
difference is actually driving the result... I'd also prioritize understanding why the
layer-27 sign flips between digits and common-word needles."* RULER and the homemade cohort
differ in three tangled ways — construction, length, content — and only content had never
been varied on its own. Holding `build_niah_prompt` and length fixed and crossing content
(7 single-token words × 8 seven-digit strings) with eviction distance
(64/512/2048/4096/8192) tests both at once. Distance 8192 gives 8,064 + 128 + 8,192 =
16,384 tokens against RULER's 15,679, so the long end is length-matched by construction
rather than by post-hoc weighting.

**2. Word and digit needles behave the same.** Across 30 layer × distance × content cells,
the two content types track each other. Where both are significant they agree in sign and
magnitude (layer 18 at 4096: word 1.136×, digit 1.190×; at 8192: word 0.804×, digit
0.887×). There is no cell where words and digits point in opposite directions with both
intervals excluding 1. **The 7 Sep content-dependence claim is withdrawn.**

**3. Layer 27 is null in this construction — every distance, both contents, and pooled.**

| | pooled fold | 95% CI | p |
|---|---|---|---|
| layer 27, word (n=7) | 0.973× | [0.885, 1.065] | 0.59 |
| layer 27, digit (n=8) | 0.995× | [0.966, 1.021] | 0.76 |

**4. Sơn's homemade layer-27 result does not replicate.** His multi-control test
(`04-C2-debug.ipynb` cells 105–110) gave 0.863×, p=0.016, pooled across distances, on
**4 hand-picked needles** — Paris, Tokyo, banana, lantern. The same statistic, same
construction, same aggregation, with **7** word needles gives 0.973×, p=0.59. The effect
vanishes when the needle set stops being those four. This is the 4–5 Sep needle-category
lesson again: results computed on that original 4-needle sample keep failing to generalise
to the needle population, and the sample — not the mechanism — keeps turning out to be the
explanation.

**5. Layer 18 carries real but incoherent structure.** Five of thirty cells survive Holm at
0.05, and nine clear an uncorrected 0.05 against ~1.5 expected by chance, so something is
there. But the sign oscillates with distance — +1.430× at 64, 0.804× at 512, +1.190× at
4096, 0.804× at 8192 — which is not a monotone decay and not obviously a mechanism. Layer
18 also fails C3-lens (7 Sep entry), so its readout is a decoding artefact in the first
place. Recorded as unexplained; not a basis for any claim.

**6. Multiple comparisons are corrected here, not left to the reader.** Thirty cells, Holm
at 0.05, five survivors. Reporting the nine uncorrected hits would have overstated this
substantially, and at this grid size the uncorrected count is close to what noise alone
produces in the first few cells.

**7. What this leaves standing, and what it costs.** RULER's layer-27 positive result (7 Sep
control battery: 1.076× per digit, CI [1.054, 1.098], p<0.0001, C3-lens verified) is
untouched — different construction, 60 real examples, its own controls. What is now gone is
the homemade-side negative it was supposedly in conflict with. So the RULER-vs-homemade
difference is **not** length (matched here) and **not** content (crossed here); by
elimination it is construction, or the difference between 60 real RULER items and 7–8
synthetic needles. That is a narrower and more tractable question than "why does the sign
flip," and it is the one to put to Gautam.

**8. Process note — the correction that mattered most was to my own hypothesis.** The 7 Sep
entry proposed content-dependence and wrote it into the README, the tracker and the
diagnosis packet within the hour. It survived one day. The test that killed it was cheap
(138 + 207 forward passes, ~13 minutes of GPU) and was only run because Gautam asked for
the length-matched control first rather than accepting the rescope. Fifth correction in
four days on this control; the first four were bugs, this one was a hypothesis stated with
more confidence than one comparison could carry.

---

## Mentor decision after the 8 Sep checks

This is a decision record, not a new experimental finding. After reviewing the
length-matched and content-swap results, Gautam approved the following:

- RULER NIAH is the primary RQ2 cohort at the pre-registered n=60.
- DeltaNet and Mamba2 are unblocked and should start now.
- C2 remains a formal Table 4 **FAIL** against the pre-registered 10x threshold, while
  Methods reports the statistically significant 1.076x-per-digit effect (95% CI
  [1.054, 1.098], p<0.0001) as sub-threshold.
- Length, content, and construction checks belong in supporting/limitations analysis and
  do not block the next runs.
- The original Paris/Tokyo/banana/lantern four-needle issue must be stated explicitly so
  its non-generalizing effects are not overclaimed.
- RULER stays at n=60; no sample-size change is warranted without a specific reason.

Gautam also provided [BABILong](https://github.com/booydar/babilong),
[NoLiMa](https://github.com/adobe-research/NoLiMa),
[LongBench](https://github.com/THUDM/LongBench), and
[SCROLLS](https://github.com/tau-nlp/scrolls). They are catalogued with intended roles and
change controls in the [dataset register](DATASET_REGISTER_2026-09-08.md); none changes the
approved RULER n=60 run.


## Findings from the DeltaNet RULER control battery (8 Sep)

Run: `results/run_3b_dn/04i_ruler_controls_rows.json`, `04i_ruler_controls_stats.json`;
regenerate with `python ruler_controls.py --run-config run_3b_dn`. Same corrected
digit-sequence scoring as GDN run 026, the same RULER NIAH cohort (config 16384, n=60,
seed 20260820), and the same shared Qwen2.5-3B J-lens (`lens_validated=False`, unchanged
from run 026). 32 of 60 needles are evicted at every layer; every statistic below is
restricted to those.

**1. The layer structure matches GDN, and C4 now backs it independently.** C3-lens is
decisive again: at layers 9 and 18 the signal *survives* the row-permuted J-lens
(Δ +64.8 and +79.0 log-nats, CIs excluding zero on the wrong side), so those layers are
decoding artefacts, not memory. Layer 27's signal *collapses* under the permuted map
(Δ −86.2 log-nats [−89.7, −80.3]) — a real, layer-specific decoding. **C4
layer-permutation** (each layer's Jacobian rolled onto the next layer's index, run 9 Sep
for GDN and DN together) tells the same story from the other direction: routing L9/L18
state through the wrong layer's Jacobian leaves the readout intact (DN Δ +88.1 and +28.3,
GDN Δ +55.4 and +173.5, all CIs positive), while L27 degrades sharply (DN Δ −83.2
[−92.0, −77.7]; GDN Δ −43.8 [−58.0, −38.4]). L27 is the only readout in either cell that
is tied to its own layer's state. Same verdict as run 026: 9 and 18 are out, 27 is the
analysis layer.

**2. Layer 27 — C1 and C3-lens behave like GDN; C2 and C3-context do not.**

| control | DeltaNet | GDN run 026 |
|---|---|---|
| C1 | median mean-digit-rank 38,581 [31,751, 45,479] — below chance (75,968) | 49,776 [41,803, 54,797] — below chance |
| C2, per-digit effect | **0.969× [0.952, 0.988]**, permutation p = 0.0006 | 1.076× [1.054, 1.098], p < 0.0001 |
| C2, per-example effect | 0.805× [0.706, 0.917] — excludes 1.0 on the **low** side | 1.672× [1.446, 1.929] — excludes 1.0 on the high side |
| C2 vs pre-registered 10× bar | fails (and points the other way) | fails |
| C3-context | **order sensitive** — shuffling the context raises mean rank by 21,739 [8,352, 25,258] | no order sensitivity (Δ −1,004 [−7,946, +18,689]) |
| C3-lens | collapses, Δ −86.2 log-nats [−89.7, −80.3] | collapses, Δ −39.2 [−46.1, −34.4] |
| C4 layer-permutation | **degrades**, Δ −83.2 log-nats [−92.0, −77.7] — layer-specific | **degrades**, Δ −43.8 [−58.0, −38.4] — layer-specific |

**3. The C2 sign is opposite to GDN, and it is significant.** In GDN, at layer 27, the
stored needle reads out ~1.076× *more* probable than a matched cross-example distractor
(per digit). In DeltaNet the same measurement puts the stored needle ~0.969× *less*
probable — the exact digit string is *suppressed* relative to other examples' digit
strings. `effect_per_example` 0.805×, CI [0.706, 0.917], permutation p = 0.0006: a
directional effect, not noise, pointing the other way. `passes_preregistered_bar` is False
for both cells (the 10× bar was written for a single-token needle). `excludes_null` — the
script's one-sided `eff_lo > 1.0` test for a *retention-favouring* effect — is False for
DeltaNet because the effect sits below 1.0; that is a statement about direction, not an
absence of effect.

**4. C3-context diverges too.** DeltaNet layer 27 *is* order sensitive: shuffling the
context words worsens the digit rank by ~21,700 (CI excludes zero). GDN and the 29 Aug
homemade cohort both showed no clean order effect. Under the pre-registration's reading of
Table 4's C3 note, an order-sensitive readout is the content-memory outcome rather than
the recency one — but here it coincides with a *negative* C2, so "DeltaNet layer 27
encodes the digit string's position/order while suppressing its identity" is the shape to
investigate, not a clean retention result.

**5. C1 is where the three signals could still agree.** DeltaNet's layer-27 mean digit
rank (38,581) is further below chance than GDN's (49,776) — the correct answer's digits
are ranked roughly 2× better than a random token after eviction. Whatever layer 27 is
doing in DeltaNet, it retains *some* recoverable information about the answer; the C2
result says that information is not "this specific string is more likely than that one."

**6. Open for Gautam / Table 7.** The cross-cell RQ2 comparison now has to accommodate a
sign disagreement at the analysis layer between GDN (positive, p < 0.0001) and DeltaNet
(negative, p = 0.0006), both on the primary pre-registered cohort, both sub-threshold.
Mamba2 is pending on Modal (the H100 box is rootless and the Mamba2 fork's custom CUDA
kernels will not build there; Modal's CUDA-devel image compiles them, so the recurrent
scan runs on real kernels rather than the naive PyTorch fallback, and fallback-vs-kernel
fidelity is itself the open M2 question). The sign disagreement does not weaken the
instrument — L9/L18 artefact rejection is unanimous across C3-lens and C4, and L27
lens-collapse and layer-specificity hold in both cells — it is a substantive
architectural difference in what the layer-27 memory does.

**7. DeltaNet C4 status.** This closes the pre-registration's Table 4 C4 for two of the
three cells. Amendment 1 recorded that "the layer-permutation half has not yet been run";
it has now, for GDN (run 026 rows, reprocessed) and DeltaNet, via
`JacobianLens.permuted_layers()` wired into `measure_ruler_seq` and scored in
`ruler_controls.py`. The in-window / pre-eviction ceiling half of C4 is unchanged.


## Findings from the DeltaNet RQ1 rerun (nb03, 9 Sep)

`results/run_3b_dn/03_nowrite_reproduction.json`; `notebooks/03_nowrite_reproduction.ipynb`
with `RUN_CONFIG=configs/run_3b_dn.json`. LongBench-E HotpotQA, n=60, first-line scoring,
AHN vs NOWRITE. This is the DeltaNet RQ1 row and the first DeltaNet artefact carrying
per-example `boundary_js`.

**1. DeltaNet's AHN helps RQ1 more than the published GDN range.** First-line ΔF1
**+6.7 points**, bootstrap CI [+0.2, +13.7] (mean F1 0.406 with AHN vs 0.339 under
NOWRITE). Answer-change rate 0.367, CI [0.25, 0.50]. The published GatedDeltaNet
write-attrition study reported F1 shifts of 0.4–2.3 points, so the notebook's
`reproduction_ok` gate — `|ΔF1| ≤ 5.0` points and change-rate in [0.30, 0.50], both
GDN-calibrated — returns **False** on the magnitude term. The run itself is clean (60/60
rows, `boundary_js` present); the flag is a GDN yardstick applied to a different cell, not
a failed reproduction. Whether that 5-point bound should gate DN/M2 at all is a question
for Gautam.

**2. The benefit is not uniform across context length.**

| stratum | n | ΔF1 (first-line) | 95% CI |
|---|---|---|---|
| short | 20 | +0.095 | [+0.017, +0.195] |
| mid | 20 | **+0.160** | [+0.035, +0.310] |
| long | 20 | **−0.053** | [−0.156, 0.000] |

Mid- and short-context examples drive the aggregate gain; at long context DeltaNet's AHN
is flat to slightly negative. This is the behavioural-side counterpart to RQ2's finding
that L27 retains recoverable information but does not make the stored string more probable
than a distractor.

**3. Per-example `boundary_js`** is now on disk for DeltaNet (bootstrap mean 0.124,
CI [0.095, 0.154]), which is what Table 8 row 2 needs — the JS-vs-ΔF1 correlation, not the
JS magnitude. GDN's nb03 rerun (`363e996`) carries the same field; Mamba2's comes from
Modal.


## Findings from the Mamba2 RULER control battery (9 Sep)

Run: `results/run_3b_m2/04i_ruler_controls_rows.json`, `04i_ruler_controls_stats.json`;
regenerate with `python ruler_controls.py --run-config run_3b_m2`. Same corrected
digit-sequence scoring as GDN run 026 and DeltaNet, the same RULER NIAH cohort (config
16384, n=60, seed 20260820), the same shared Qwen2.5-3B J-lens (`lens_validated=False`).
32 of 60 needles are evicted at every layer; every statistic below is restricted to those.

**Executed on real Mamba2 kernels.** The shared H100 box is rootless and cannot build the
`yuweihao/mamba` fork's custom CUDA kernels, so an in-repo M2 run falls back to fla's
naive scan. This battery ran on Modal (`modal_m2.py`, a `pytorch/pytorch:2.5.1-cuda12.4
-cudnn9-devel` image with flash-attn, the Seerkfang FLA fork and the Mamba fork compiled
from source), where `is_fast_path_available` is True. The fallback-vs-kernel fidelity
question is therefore closed for these numbers; it stays open only for any future run done
on the box, so M2 re-runs go through Modal.

**1. L9 and L18 are rejected on two independent grounds, same as GDN and DeltaNet.**
C3-lens: the signal *survives* the row-permuted J-lens at both layers (Δ +104.1 log-nats
[+89.4, +111.2] at L9, +53.1 [+38.8, +63.9] at L18 — CIs on the wrong side of zero), so
those readouts are decoding artefacts. **C4 layer-permutation** agrees: routing L9/L18
state through the wrong layer's Jacobian barely moves the readout (Δ +57.2 [+50.2, +62.7]
and +36.7 [+32.8, +43.9]) rather than degrading it. Both diagnostics disqualify 9 and 18.

**2. L27 is the analysis layer by C4, but its C3-lens does not cleanly collapse — the
first cell where the two "is L27 real" checks disagree.**

| control | Mamba2 | DeltaNet | GDN run 026 |
|---|---|---|---|
| C1, mean digit rank | 76,951 [71,390, 92,924] — **spans chance** (75,968) | 38,581 [31,751, 45,479] — below chance | 49,776 [41,803, 54,797] — below chance |
| C2, per-digit effect | 1.013× — n/a | 0.969× [0.952, 0.988], p = 0.0006 | 1.076× [1.054, 1.098], p < 0.0001 |
| C2, per-example effect | **1.095× [0.886, 1.356], permutation p = 0.399 — null** | 0.805× [0.706, 0.917], p = 0.0006 (low side) | 1.672× [1.446, 1.929], p < 0.0001 (high side) |
| C2 vs pre-registered 10× bar | fails (no directional effect at all) | fails (points low) | fails (points high) |
| C3-context | no order sensitivity (Δ −9,121 [−22,347, +3,937]) | order sensitive (Δ +21,739 [+8,352, +25,258]) | no order sensitivity |
| C3-lens | **borderline — Δ +2.14 log-nats, CI [−2.85, +10.26] crosses zero**, neither a clean collapse nor a clean survival | collapses, Δ −86.2 [−89.7, −80.3] | collapses, Δ −39.2 [−46.1, −34.4] |
| C4 layer-permutation | **degrades**, Δ −116.3 log-nats [−125.4, −111.8] — layer-specific | degrades, Δ −83.2 [−92.0, −77.7] | degrades, Δ −43.8 [−58.0, −38.4] |

**3. Layer 27's C2 is null in both directions.** GDN reads the stored digit string out
~1.076× *more* probable than a matched cross-example distractor (per digit, p < 0.0001);
DeltaNet reads it ~0.969× *less* probable (p = 0.0006); Mamba2's per-example effect is
1.095× with CI [0.886, 1.356] and permutation p = 0.399 — it does not exclude 1.0 on
either side. Whatever L27 does in Mamba2, it does not make the exact stored string more or
less probable than another example's.

**4. Layer 27 barely retains recoverable answer information at all.** M2's L27 mean digit
rank is 76,951 against a chance rank of 75,968 — its bootstrap CI [71,390, 92,924] spans
chance. GDN (49,776) and DeltaNet (38,581) both sit clearly below chance, i.e. the correct
digits are ranked meaningfully better than random after eviction. In Mamba2 they are not.
This is the weakest layer-27 C1 of the three cells by a wide margin.

**5. C4 is the only control that says L27 is doing anything layer-specific.** The
layer-permutation delta (−116.3 log-nats, the sharpest of the three cells) is unambiguous
that the L27 readout is tied to L27's own state. But C1 is at chance, C2 is null, and
C3-lens does not collapse — so "layer-specific" here describes a decoding path that
carries almost no retention signal, rather than a working content readout.

**6. C3-context matches GDN, not DeltaNet.** Shuffling the context words does not change
the L27 digit rank (Δ −9,121, CI includes zero). DeltaNet was order sensitive; GDN and the
29 Aug homemade cohort were not. Two of the three cells show no order effect at the
analysis layer.

**7. What Table 7 / RQ2 now has to accommodate.** All three cells have the RULER control
battery on the primary pre-registered cohort, and the analysis layer tells a different
story in each: GDN — positive, significant C2 (p < 0.0001), C3-lens collapses; DeltaNet —
negative, significant C2 (p = 0.0006), order sensitive, C3-lens collapses; Mamba2 — null
C2, C1 at chance, C3-lens does not collapse, only C4 marks it layer-specific. The
instrument's artefact rejection is still unanimous (L9/L18 out in all three cells by both
C3-lens and C4), but the cross-cell RQ2 comparison at L27 is now a three-way divergence,
not a two-way sign flip. Mamba2 is the cell where the layer-27 memory shows the least
evidence of holding the needle's content.

**8. C4 status.** This completes the pre-registration's Table 4 C4 layer-permutation half
for all three cells (Amendment 1 recorded it as unrun). GDN and DeltaNet landed 8–9 Sep;
Mamba2 here, via the same `JacobianLens.permuted_layers()` wired into `measure_ruler_seq`
and scored in `ruler_controls.py`. The in-window / pre-eviction ceiling half of C4 is
unchanged.


## Findings from the Mamba2 RQ1 run (nb03, 9 Sep)

`results/run_3b_m2/03_nowrite_reproduction.json`; `notebooks/03_nowrite_reproduction.ipynb`
with `RUN_CONFIG=configs/run_3b_m2.json`, run on Modal alongside the RULER battery (same
real-kernel image). LongBench-E HotpotQA, n=60, first-line scoring, AHN vs NOWRITE. This
is the Mamba2 RQ1 row and carries per-example `boundary_js`.

**1. Answer-change rate lands squarely in the published band; F1 does not move.**
First-line answer-change rate **41.7%**, CI [30.0%, 53.3%] — inside the published
38–42% (GDN 33.3%, DeltaNet 36.7%). First-line ΔF1 **−0.31 points**, bootstrap CI
[−8.33, +7.46] (mean F1 0.336 with AHN vs 0.339 under NOWRITE). The notebook's
`reproduction_ok` gate returns **True** — change-rate in [0.30, 0.50] and |ΔF1| ≤ 5.0
points both hold — unlike DeltaNet, which tripped the magnitude term with +6.7 points.
**Week-6 milestone: PASS.**

**2. No F1 benefit at any context length.**

| stratum | n | ΔF1 (first-line) | 95% CI | change rate |
|---|---|---|---|---|
| short | 20 | +0.028 | [−0.133, +0.192] | 0.40 |
| mid | 20 | +0.010 | [−0.140, +0.150] | 0.40 |
| long | 20 | −0.047 | [−0.158, +0.025] | 0.45 |

Every stratum's ΔF1 CI spans zero; long is flat-to-slightly-negative, the same shape
DeltaNet showed. Mamba2's AHN changes roughly two answers in five without improving them.

**3. Per-example `boundary_js`** is on disk for Mamba2 (bootstrap mean 0.129,
CI [0.100, 0.160]), so Table 8 row 2 — the JS-vs-ΔF1 correlation — now has all three
cells (GDN `363e996`, DeltaNet `6296e5c`, Mamba2 here).

**4. Behavioural counterpart to the RQ2 null.** Mamba2 suppresses-writes-changes-answers
at the published rate but produces no measurable F1 shift, matching the RULER battery's
finding that its layer-27 memory shows no significant retention effect in either
direction. Of the three cells, Mamba2 is the one where AHN's compressed memory is hardest
to detect — behaviourally (ΔF1 ≈ 0) and in the readout (C2 null, C1 at chance).


## Findings from the 1000-context J-lens map-stability refit (9 Sep)

`results/run_3b_gdn/02_table3_jlens_validation_1000ctx.json`;
`notebooks/02-duplicate.ipynb` (the n=1000 variant of `02_jlens_fit_and_validate.ipynb`),
`jlens-venv`, on the shared H100 MIG box. Maps saved as
`jlens_qwen25_3b_1000ctx.pt` (corpus A) and `jlens_qwen25_3b_corpusB_1000ctx.pt`. This is
a robustness pass on the map-stability row only — it extends the 9–20 Aug 500-context
result (["Findings from the map-stability check and the RQ3 join"](#findings-from-the-map-stability-check-and-the-rq3-join))
and writes to a separate JSON so that entry's numbers are left intact.

**1. Map stability holds at n=1000, and tightens at every layer.** Two J-lenses fitted on
disjoint 1000-context wikitext corpora (skip 0 vs skip 20000, zero overlap asserted),
layers 9/18/27, `max_seq_len=256`, `skip_first=4`, top-10 token overlap on 30 held-out
contexts:

| layer | 500-context | 1000-context | Δ |
|---:|---:|---:|---:|
| 9 | 0.910 | **0.943** | +0.033 |
| 18 | 0.873 | **0.907** | +0.033 |
| 27 | 0.893 | **0.943** | +0.050 |
| min | 0.873 | **0.907** | |

All three layers clear the 0.80 bar with more margin than at n=500. Doubling the
averaging corpus moves the map closer to a fixed point — the two independent draws agree
more, not less — so undersampling is ruled out more firmly than the 500-context pass
already ruled it out. `map_stability.passed = true`.

**2. Convergence, not correctness — the readout checks do not move.** Checks 2 and 3 were
re-run on the n=1000 corpus-A map and land where they did at n=500:

| check | 500-context | 1000-context |
|---|---|---|
| known-fact recall, `rank_Paris` @ 9/18/27 | 805 / 59 / 8 | 671 / 68 / **7** |
| logit-lens agreement @ 9/18/27 | 0.00 / 0.00 / 0.05 | 0.00 / 0.00 / 0.05 |

Layer-27 `rank_Paris` is 7 vs 8 — noise, still not top-1. Agreement is byte-identical.
L27 top-5 is still `____`, `________`, `:**`. `TABLE_3_PASSED` stays `false` on checks 2
and 3, exactly as at n=500. More contexts made the map more self-consistent without
making it decode residuals to sensible tokens.

**3. What this does for the downstream negative.** The RULER control battery's reading —
that L9/L18 are decoding artefacts and L27 carries at best a modest, cell-dependent
retention signal — now rests on a map whose stability is confirmed at 2× the
pre-registered corpus size. A weak or backwards signal in that battery is a property of
what AHN retains (or of the cohort), not a fitting artefact of an under-converged lens.
This is the "stronger evidence for dropping to RQ1" the notebook's check-4 cell describes,
made stronger.

**4. Map cost for Table 10.** Corpus-B fresh fit: **5.88 GPU-h** at n=1000 (logged
`fit_b_gpu_hours`), against 1.25 GPU-h at n=500 — ~4.7× for 2× the contexts, well above
linear, most likely MIG-slice contention on the shared box (four users). Corpus-A shows
2.19 GPU-h but that run resumed from a partial checkpoint, so it understates the fresh
cost. Even at the higher figure a full two-corpus refit is ~12 GPU-h, still far under the
proposal's 40 h RQ2 abort threshold, but the superlinear scaling is worth re-measuring on
an uncontended slice before it feeds the 7B decision.

**5. Caveats.** `lens_validated=false` still travels on every downstream row — this pass
strengthens the stability row of Table 3, not the whole battery. Overlap is measured on
30 contexts at top-10, same as the 500-context check. Single backbone (Qwen2.5-3B), three
layers, one corpus source (wikitext-103-raw). The `.pt` and `.ckpt` maps are ~50 MB each
and live only on the box + HuggingFace until LFS/Hub storage is settled; the 1.4 KB
result JSON is the artefact of record.


## Findings from the r29 evicted-vs-in-window J-lens re-run (9 Sep)

`results/run_3b_gdn/04o_r29_evicted_vs_inwindow_jlens.json`;
`notebooks/04_niah_retention.ipynb`, the `# r29 --` cell (reuses the notebook's
`bundle`/`tok`/`probe`/`needles`/`measure`, writes its own file, does not touch
`04_retention_rows.json`). GDN 3B, homemade `build_niah_prompt`, eviction distance 1024,
8 single-token needles × 3 filler variants per condition (n=24 rows per layer×condition),
readout through the 1000-context backbone J-lens
(`jlens_qwen25_3b_1000ctx.pt`, `lens_validated=False`). This closes the caveat on the
[21 Aug C1 diagnosis](#findings-from-the-21-aug-c1-diagnosis) — that comparison used the
plain logit lens because no J-lens was on the box — by re-running it like-for-like.

**1. Median rank by condition, both readout bases (chance 75,968; lower is better).**

| layer | condition | `o_t` median | `d_resid` median (pre-registered) |
|---:|---|---:|---:|
| 9 | evicted | 87,917 | 90,910 |
| 9 | in-window | 126,262 | 117,796 |
| 18 | evicted | 114,277 | 103,707 |
| 18 | in-window | **36,676** | 92,530 |
| 27 | evicted | 68,437 | 113,971 |
| 27 | in-window | 68,875 | 94,006 |

**2. The J-lens rescues the layer-18 pre-eviction ceiling the logit lens was hiding.**
21 Aug (logit lens) read L18 in-window at 95,949 — barely better than its evicted 108,733,
which is why that diagnosis leaned toward "the prompt construction is wrong." Through the
J-lens, L18 in-window drops to **36,676** against evicted 114,277 (`o_t` basis): the
ceiling is now well above the floor, the direction C4 requires. That gap is a J-lens
effect — the plain logit lens flattened it. Consistent with the 8–204× J-lens-over-logit
advantage on isolated known-fact prompts (Finding 4, 19–20 Aug).

**3. Evicted content is still not read below chance anywhere, in either basis.** L9
87,917 / L18 114,277 / L27 68,437 (`o_t`); L9 90,910 / L18 103,707 / L27 113,971
(`d_resid`). L27's `o_t` 68,437 is modestly below chance, matching the homemade-cohort
L27 numbers from the 4–5 Sep sweep, but there is no in-window/evicted separation at L27
at all (68,437 vs 68,875) — the pre-eviction ceiling is not above the evicted floor. The
J-lens does not rescue evicted retention on this construction.

**4. Pre-registered basis: conclusion unchanged.** In `d_resid`, L18 evicted 103,707 vs
in-window 92,530 — an ~11k gap with the in-window ceiling itself near chance. This is the
2 Sep reading ("not 'the needle reads well until it is compressed'") holding up: the
ceiling-vs-floor separation that appears at L18 is in the `o_t` basis, not the
pre-registered one.

**5. Layers 9 and 27 behave as their prior records predict.** L9 in-window (126,262 `o_t`)
is *worse* than evicted (87,917) — ceiling below floor, C4 fails, the same degeneracy
signature flagged since the 18 Aug pilot, now confirmed through the fitted J-lens. L27
shows no order-of-magnitude gap either way on the homemade construction, consistent with
the 7 Sep finding that `build_niah_prompt` suppresses the L27 signal that RULER surfaces.

**6. Net.** The 21 Aug reading — the C1 failure points at NIAH prompt construction, not at
AHN's memory — survives the like-for-like re-run. The one thing that changes: the L18
in-window ceiling is ~2.6× better through the J-lens than the 21 Aug logit-lens number
(36,676 vs 95,949), so part of what looked like "the prompt places the needle where
nothing can read it" was the logit lens, not the prompt.

**7. Caveats.** `lens_validated=False` on every row — Table 3 checks 2 and 3 still fail.
One eviction distance (1024; the 21 Aug diagnosis was at ~515), one construction
(homemade, known to under-read L27 vs RULER), one cell (GDN), n=24 per layer×condition
after the `ahn_will_activate` / `needle_is_evicted` filters drop 2 of 10 needles. No C2 /
C3 analog — this is a C1/C4-shaped check only.


## Findings from the no-AHN floor run (r54, 9 Sep)

`results/run_3b_floor/06_no_ahn_floor.json` (+ `_nowindow.json` control);
`configs/run_3b_floor.json`, `no_ahn_floor.py`. Stock `Qwen/Qwen2.5-3B-Instruct`,
**no AHN merge**, sliding-window attention forced on at 8064 for every layer
(`use_sliding_window=True`, `sliding_window=8064`, `max_window_layers=0`), **no attention
sinks** (`num_attn_sinks=0`). Behavioural: greedy generation + task metric, not a lens
readout. RULER NIAH n=60 (`simonjegou/ruler` config 16384, 60/60 `niah_single_1`,
7-digit answers, substring match) and LongBench-E HotpotQA n=60 (same length-stratified
cohort construction and eligibility gate as notebook 03). Backbone is shared across the
GDN / DeltaNet / Mamba2 cells, so this one run is the Table 1 Primary floor for all three.
~7 GPU-minutes total.

**1. The floor.**

| cohort | n | metric | value |
|---|---:|---|---:|
| RULER NIAH, needle **evicted** (past the 8064 window) | 32 | substring acc | **0.000** |
| RULER NIAH, needle **evicted** | 32 | first-line F1 | 0.000 |
| RULER NIAH, needle in-window | 28 | substring acc | 0.393 |
| RULER NIAH, needle in-window | 28 | first-line F1 | 0.151 |
| LongBench-E HotpotQA | 60 | first-line F1 | **0.076** |
| LongBench-E HotpotQA | 60 | first-line EM | 0.033 |

(All F1/EM rows are first-line, matching the metric of record for the readout cohorts;
the full-generation RULER in-window F1 is 0.013.)

Chance on a 7-digit answer is effectively zero, so `evicted = 0.000` is a hard floor, not
a small number: with no AHN and the needle outside the attention window, the information
is simply not in the model's context. **Every evicted-needle retention result AHN
produces, at any layer, is therefore attributable to AHN** — there is no base-model
retrieval to subtract.

**2. The control rules out a broken load path.** Same script, `--no-window` (stock config,
full ~32K context), n=20: substring accuracy **1.000** on both the evicted-labelled and
in-window subsets, predictions exact (`pred='7700828.'`). Base Qwen2.5-3B is a perfect
NIAH retriever at 16K when it can see the whole context. The collapse to 0.000 / 0.393 is
the 8064 window doing exactly what the floor is meant to isolate, confirmed rather than
assumed.

**3. The window degrades retrieval even when the needle is nominally visible.** In-window
substring accuracy is 0.393, not ~1.0 — SWA disrupts the retrieval path for needles that
sit inside the last 8064 tokens, not only for evicted ones. So the sliding window is a
retrieval bottleneck in its own right, which is the gap AHN's memory pathway exists to
close. (The `_nowindow` control at 1.000 is the ceiling this 0.393 is measured against.)

**4. This removes the rows 35/36 blocker.** Rows 35/36 were held on the "NOWRITE-proxy vs
true module-removal" question — is zeroing the AHN write a valid stand-in for removing the
module? The floor answers it by running the real thing. Against notebook 03's GDN
**NOWRITE** first-line F1 (~0.339), the true no-AHN floor is **0.076** — a 0.26 F1 gap, so
NOWRITE-proxy is **not** equivalent to module removal. Part of that gap is the 128
attention sinks NOWRITE keeps and this floor drops (sinks sit at the sequence start, far
from any evicted needle, so they do not touch the RULER-evicted 0.000 result, but they do
help HotpotQA); isolating the sink contribution would need a floor-with-sinks run. Either
way, RQ1 and RQ2 now have a Table 1 Primary baseline that is a real configuration, not a
proxy, and the RQ tables can be reported against it without the caveat.

**5. Per-cell consequence.** RULER NIAH evicted floor = 0.000 for GDN, DeltaNet and
Mamba2 alike (shared backbone). The RULER control battery's L27 retention signals (GDN
positive C2, DeltaNet negative, Mamba2 null) are all measured above a base-model floor of
zero on the same cohort. HotpotQA first-line F1 floor = 0.076; the GDN nb03 AHN run
(~0.37, regenerated from the saved `_fl` rows) and DeltaNet (~0.41) both clear it
comfortably, Mamba2 (~0.34) by less but still well above.

**6. Caveats.** n=32 evicted / n=28 in-window — RULER-16384 places the needle at random
depth, so the evicted/in-window split (~53/47) is a property of the cohort, not a design
choice; the evicted n is smaller than 60. `num_attn_sinks=0` here vs 128 in every nb03 /
nb04 AHN run — deliberate (the floor is stock Qwen, which has no sink mechanism), but it
means the HotpotQA F1 comparison to NOWRITE is not sink-matched. One backbone, one scale
(3B), one RULER config (16384 ≈ 15.7K tokens), one QA dataset. `attn_impl=flash_attention_2`.
Substring match on the full 32-token generation, not first-token rank — a different (and
more lenient) metric than the readout cohorts use.


## Findings from the RULER retention curve on corrected scoring (Run 025 redone, 9 Sep)

`results/run_3b_gdn/04p_ruler_retention_curve.json`; `ruler_retention_curve.py`
(CPU-only, reprocesses `04i_ruler_controls_rows.json`). The 7 Sep "retention does not
decay across the range measured" point sat in the J-lens repeat section, whose rank
numbers (15,844 / 17,250) were then withdrawn as the rank of a space token by the
target-scoring bug fix. `04i` (Run 026) already carries per-example `eviction_distance`
and `mean_digit_rank` on the corrected digit-sequence scoring, so the curve re-runs with
no GPU: bin the 32 evicted examples per layer by how far past the compression boundary
the needle sat, bootstrap the median digit rank per bin, test rank-vs-distance with a
Spearman rho and a permutation p.

**1. Flat at every layer — the "no decay" reading survives the scoring fix.**

| layer | ρ (rank vs eviction distance) | perm p | median rank across bins | reading |
|---:|---:|---:|---|---|
| 9 | −0.015 | .93 | ~72k–90k | flat, at/near chance (75,968) — the degenerate layer |
| 18 | −0.217 | .22 | ~5k–14k | flat, but this is the C3-lens artefact layer (Run 026 pt 3), not retention |
| 27 | +0.066 | .72 | ~42k–62k | flat, below chance — the analysis layer |

Eviction distance spans 46 to 7,414 tokens past the boundary, four equal-count bins of 8.
No layer shows a monotone rank–distance relationship; every permutation p is far from
significance. Table 6's exponential fit failing (R² < 0 at all three layers) is therefore
a "nothing to fit" result, not a mis-specified model.

**2. Caveats.** n=32 evicted examples per layer, 8 per bin — thin, with wide bootstrap
CIs (L9 bin 1: [50,001, 94,793]). The distances are RULER's own random needle depths,
not a designed sweep. The window is 8,064 and RULER-16384 is ~15.7k tokens, so the
deepest evicted needle is only ~7.4k past the boundary — this says nothing about
eviction distances beyond that. `lens_validated=False` on every row.


## Findings from the RULER needle-placement check on corrected scoring (04q, 9 Sep)

`results/run_3b_gdn/04q_ruler_placement_check.json`; `ruler_placement_check.py`
(CPU-only, reprocesses `04i_ruler_controls_rows.json`). `ruler_needle_position.py` /
`04h` asked whether layer 27's below-chance RULER read is carried by needles that were
never evicted — an attention artefact rather than evidence about AHN. It ran on `04g`,
whose target was a leading-space token identical across all 60 examples (the Run 026
scoring bug). `04i` already carries `needle_pos` / `placement` / `needle_is_evicted` and
the corrected `mean_digit_rank` per row, so this re-does the split with no GPU: median
digit rank by placement verdict and by fractional needle depth (terciles), ordered
context, jlens readout, layer 27 the analysis layer.

**1. Layer 27's evicted signal is real, not an in-window artefact.** Evicted subset
alone (n=32): median digit rank **49,776**, 88% of examples beat chance (75,968),
bootstrap CI [42,273, 54,797] — excludes chance. In-window (n=28): median 37,619, 93%
beat chance. Both below chance; the below-chance read does not depend on the in-window
examples. The 49,776 matches Run 026's C1 figure exactly, cross-validating the two
scripts.

**2. The "evicted beats the pre-eviction ceiling" anomaly was a scoring artefact.**
`04h` on the buggy rows had evicted at median 17,250 *better* than in-window at 54,506 —
a supposedly-compressed read outperforming the directly-visible one, which was the
reason this check existed. Corrected, the order is the mechanically sensible one:
in-window (37,619) ≤ evicted (49,776). Nothing to explain.

**3. Needle depth does not matter at layer 27.** Median digit rank across
front / middle / back depth terciles: 48,202 / 43,809 / 42,709 (beat-chance 0.85 / 0.95
/ 0.90). Flat — consistent with the retention curve's finding of no eviction-distance
dependence.

**4. Layers 9 and 18 as expected.** L9 in-window median 113,176, 0% beating chance
(evicted 87,259, 31%) — the ceiling below the floor, the degeneracy signature. L18
evicted 6,254 / in-window 22,406, both 100% beating chance, but L18 is the C3-lens
artefact layer (Run 026 point 3): these low ranks are a decoding-path property, not
retention.

**5. Caveats.** n=32 evicted / n=28 in-window at one layer, one cell (GDN), one RULER
config. `in_sink_region` had too few examples to report. `mean_digit_rank` on the jlens
readout, `lens_validated=False`. `multi_occurrence_examples` is empty — no needle-string
ambiguity in this cohort.


## Findings from the Kashyap scoring-convention reconciliation (9 Sep)

`results/05_kashyap_reconciliation.json`; `kashyap_reconciliation.py` (CPU-only,
re-scores the saved generations in `results/run_3b_*/03_nowrite_reproduction.json`;
no GPU, no model). Two of our RQ1 numbers appeared to conflict with the concurrent
write-attrition submission (Kashyap 2026): mean F1 moves 0.4–2.3 points there, 38–42%
of answers change. Our DeltaNet ΔF1 was +6.7 points with a CI excluding zero, and our
GDN change rate was 33.3%. This entry tests whether either conflict is real. Neither is.

**1. Official LongBench-E scoring reads the full generation, and our metric of record
does not.** `scorer_e` in `eval/longbench/eval.py` truncates a prediction to its first
line **only** for `trec` / `triviaqa` / `samsum` / `lsht`. `hotpotqa` is not in that
list, so the official LongBench-E score for our cohort is computed on the whole
generation. Our primary metric since the 19–20 Aug correction has been first-line. The
two are not the same convention, and the difference is not small:

| cell | first-line (ours) | official LongBench-E | in Kashyap's 0.4–2.3 band |
|---|---:|---:|---|
| GatedDeltaNet | +3.34 pts | **−1.44** [−7.43, +3.99] | yes |
| DeltaNet | **+6.72** [+0.18, +13.65] | **−2.37** [−8.14, +2.70] | 0.07 pts outside |
| Mamba2 | −0.31 pts | **+2.63** [−2.72, +8.16] | 0.33 pts outside |

Re-scoring with the vendored official metric reproduces our `full_generation` column
exactly, and the non-E `scorer` path (its prefix-split chain never fires on these
generations) gives the same numbers again. So the existing full-generation column *is*
the official number; it was simply not the one being reported.

**2. The DeltaNet contradiction is withdrawn.** +6.7 points was a first-line artefact.
Under the convention Kashyap's band is stated in, DeltaNet is −2.37 points with a CI
spanning zero — the same "answers change, quality does not" shape he reports. No cell
contradicts his band once the conventions are matched; all three CIs span zero and all
three magnitudes land within 0.33 points of it.

**3. The scoring convention flips the sign of the RQ1 headline in all three cells.**
GDN +3.34 → −1.44, DeltaNet +6.72 → −2.37, Mamba2 −0.31 → +2.63. This is a larger
methodological exposure than the Kashyap comparison it was run to settle: Table 5's
headline direction is a function of a scoring choice, not of the data. Table 5 must
state which convention it reports and why, and the 19–20 Aug argument for first-line
(that it matches the published *direction* and change-rate band) has to be restated now
that the direction it matches is the opposite one.

**4. His 38–42% change rate is raw first-line string equality.** Sweeping every
plausible definition on the same saved generations:

| definition | GDN | DeltaNet | Mamba2 |
|---|---:|---:|---:|
| raw exact, full generation | 91.7% | 93.3% | 91.7% |
| **raw exact, first line** | **40.0%** | **40.0%** | 45.0% |
| normalised-EM, first line *(our headline)* | 33.3% | 36.7% | 41.7% |
| F1 changed, full generation | 53.3% | 46.7% | 50.0% |

Two of three cells land dead-centre in 38–42% under raw first-line equality. Our
headline 33.3% is the same comparison after `normalize_answer`, which folds away case,
punctuation and articles. The mild change-rate tension was a normalisation step.

**5. DeltaNet's ΔF1 significance was one example deep.** Of the three cells, only
DeltaNet's first-line CI excluded zero ([+0.43, +13.42] at n_boot=4000). Jackknifing:
**17 of 60 single-example deletions lose significance**, and dropping the single most
supportive example is enough (greedy k = 1). Leave-one-out means range +5.13 to +8.52
points. Even on its own metric that result could not have carried a claim against a
published band. GDN and Mamba2 were not significant to begin with, so there was nothing
to break.

**6. What this changes.** Both apparent conflicts with the concurrent submission are
withdrawn, and they had a single cause. What replaces them is a sharper internal
question — which scoring convention Table 5 reports — and a specific ask for Gautam:
which scorer produced his band, over which LongBench tasks, at what n, and whether
"removing all writes" keeps the 128 attention sinks (our no-AHN floor already shows
NOWRITE-proxy at F1 ~0.34 is not module removal at F1 0.076, so the two conditions may
not be the same contrast).

**7. Caveats.** One dataset (LongBench-E HotpotQA), n=60 per cell, one backbone, 3B
only. Kashyap's band is quoted from the proposal's related-work section, not read off
his artefacts; item 6's questions are exactly the ones that would let this comparison
be stated precisely rather than approximately. The normalisation order in
`ahn_interp.normalize_answer` differs subtly from the official one (articles before
punctuation rather than after); it made no difference on these generations, and the
official implementation is transcribed faithfully in `kashyap_reconciliation.py`.

**8. Process note.** This was reachable from data that had been on disk since 20 August.
The two numbers were compared against a published band for a month without checking
that they were measured the same way. The check cost no GPU and ran in seconds.
