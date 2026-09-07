# Diagnosis packet for Gautam — 7 Sep 2026

**Purpose.** Committed to the Meetings tab on 1 Sep — "Hannah sends the diagnosis packet
(C1 + C2 confound + corrected C3) ahead of the meeting; Gautam makes the call" — and never
sent. Everything it was meant to contain has since been superseded by stronger evidence.
This is that packet, six days late, built on what actually exists now rather than what
existed on 1 Sep.

**Two decisions are needed. Nothing below the fold is required reading to make them** —
the numbers you need are on this page; the reasoning behind them is in
[docs/FINDINGS.md](FINDINGS.md), linked at each claim for when you want it.

---

## Decision 1: candidate (b) has direct evidence now. Do we accept it and re-scope RQ2?

On 1 Sep the question was "C1, C2, and C3 all fail with a converged lens — is that (b) bad
prompt construction, or (c) AHN behaving like a recency mechanism, not a content store?"

That question is answered. **(b), at a small but verified magnitude.** Run 026, same
checkpoint, same instrument, RULER NIAH's real n=60 cohort instead of the homemade
`build_niah_prompt`:

| control | homemade cohort | RULER cohort | verdict |
|---|---|---|---|
| C1 (evicted rank vs. chance 75,968) | 111,694 — worse than chance | 49,776 — better than chance | cohort changes the outcome |
| C2 (cross-example, baseline-corrected) | 5.03x raw, withdrawn 3 Sep as pair-identity baseline | **1.076x per digit [1.054, 1.098], p<0.0001** | excludes the null; far under the 10x bar |
| C3-lens (row-permuted J-lens map) | not run before this | **collapses**, as it should | layer 27's signal is real decoding, not artefact |

Layers 9 and 18 are disqualified outright — their C3-lens signal *survives* the permuted
map, meaning whatever they show is a property of decoding, not memory. Layer 27 is the
only layer that passes every check.

**What this licenses:** *a small, statistically robust, memory-specific effect exists at
layer 27 on the primary pre-registered cohort, verified not to be a decoding artefact.*

**What it does not license:** "AHN retains the content." 1.076x per digit is an order of
magnitude short of the pre-registered 10x bar. This is not the 2 Sep claim revived — that
one used the wrong basis and the same C2 confound that's now controlled for here.

Full derivation: [7 Sep J-lens repeat](FINDINGS.md#findings-from-the-7-sep-j-lens-repeat-and-placement-check),
[7 Sep control battery](FINDINGS.md#findings-from-the-7-sep-target-scoring-bug-and-the-ruler-control-battery).

**The ask:** accept the homemade `build_niah_prompt` construction as the source of the
original C1/C2/C3 failure, and re-scope RQ2 around the RULER cohort going forward?

---

## Decision 2: how does a real-but-sub-bar effect get written up?

Expected Tables' C2 criterion is binary — pass at 10x, fail below it — and the
pre-registration gate says C1–C3 must pass before RQ2 populates. Layer 27's result doesn't
fit either box: not indistinguishable from noise (p<0.0001), not close to the bar either.

Calling it "C2: FAIL" reads identically to the 20 Aug result (median ratio 1.00, pure
noise) — that erases a real finding. Rounding it toward "pass" is the mistake this project
has already made twice (2 Sep, and run 025 four days ago). Neither default is right.

**The ask:** pick one —
1. Report as a stated pre-registration deviation, with the real number and why it's reported anyway.
2. Add a third category to Table 4 — "significant, sub-bar" — distinct from "fails, at chance."
3. Hold the 10x line as written; C2 fails, full stop, regardless of the p-value.

---

## What's been blocked on this since 1 Sep

- **DN and Mamba2 3B checkpoint merges** (`configs/run_3b_dn.json`, `run_3b_m2.json`) —
  correctly unrun; a second and third cell measured with a disputed instrument is wasted
  compute.
- **Table 7** (pairwise half-life ratios) and **the 7B decision** — both gated on the DN/M2
  runs above.
- **Methods' pre-registration-deviation writeup** (DIVERGENCE 6) — cannot be finalized
  until Decision 1 and 2 above are settled; the deviation being written up *is* these two
  decisions.
- **Related Works / Methods drafting** — needs the RQ2 framing fixed before it can commit
  to language about what AHN does or doesn't retain.

## What is resolved, no decision needed

- Undersampling (candidate a) — ruled out, map stability passes at all three layers.
- The 2 Sep layer-27 claim — withdrawn on its own terms (wrong basis, C2 confound), independent of anything here.
- RQ1's headline ΔF1 (+6.11 pts) — its own 95% CI spans zero at n=60; your published effect sits inside that interval, so "our effect is 3x yours" was a cohort-size artifact, not a real gap.
- The 4–5 Sep needle-category test (place names vs. common nouns) — real, Holm-significant, and a separate open item (does it change what needle set is of record) that doesn't block Decisions 1–2 above.

---

*Everything in this packet is reproducible from `results/run_3b_gdn/*.json` via
`ruler_cohort_stats.py`, `ruler_needle_position.py`, and `ruler_controls.py` at the repo
root. No number here was hand-transcribed from a notebook cell.*
