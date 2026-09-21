# Amendment 5 — BABILong construction-robustness pilot (DRAFT, retroactive)

**Status:** draft for Gautam's review. Not filed. Drafted 20 Sep 2026 from
`docs/BABILONG_PILOT_2026-09-16.md`, `results/babilong/07c_babilong_c1c2_full.json`, the git
history (2ceeed9 → 032aa5e, 16–18 Sep) and `docs/DATASET_REGISTER_2026-09-08.md`.
Items marked **[CONFIRM: Son]** are facts the repo does not record.

## Text for `Expected_Tables_and_Figures` (same form as Amendments 1–4)

**Amendment 5 — [date filed] (retroactive; pilot run 16–17 September 2026).** Filed under the
same discipline as Amendments 1–4: this document remains the pre-registration of record, no
placeholder has been replaced with a measured value, and no pass criterion, success threshold,
cut order or null-result plan is altered. This amendment records a construction-robustness
pilot on BABILong that was run **before** it was amended, contrary to the 8 September dataset
register, which required the examples, metric, sample size and stopping rule to be scoped in a
separate amendment before compute was spent. That amendment was not filed; this one is
therefore a disclosed deviation, not a pre-registration. The pilot is exploratory, is reported
in Limitations and supporting analysis only, and is not a primary or confirmatory cohort.
RULER NIAH n=60 remains the primary RQ2 cohort. (1) Cohort. BABILong (`RMT-team/babilong`),
task qa1, config 64k (the 16k and 32k configurations were inspected and rejected because their
contexts, about 14–15k and 30–31.6k Qwen tokens, do not reach the eviction regime). Of 100 qa1
examples, the 32 whose supporting fact lies more than 32,640 tokens from the end of the context
were frozen as the cohort before any GPU use; the frozen IDs are recorded in
`results/babilong/07_babilong_64k_evicted_cohort.json`. (2) Cell and configuration.
GatedDeltaNet 3B only, window 8064, 128 sinks, seed 20260820, layers 9/18/27, the 1000-context
J-lens map (`lens_validated=false`). (3) Metric and decision rule. The RULER controls were
re-implemented for multi-token answers as gold-answer rank (C1) and baseline-corrected
per-example effect (C2), reusing the pre-registered criteria unchanged: C1 passes if the median
rank is below chance (75,968); C2 passes at 10×. Method version
`babilong_c1c2_pointcapture_v3`. (4) Result. C1 fails at every layer (median rank 117,455 at
L9, 121,581 at L18, 138,655 at L27, all above chance). C2 fails at every layer: 0.824×
[0.751, 0.905] at L9, 1.421× [1.128, 1.776] at L18 (permutation p=0.0029), 0.962× [0.808,
1.142] at L27 (p=0.66). The layer-27 effect seen on RULER (1.076× per digit) is not reproduced
on this cohort. This shows that the J-lens C1/C2 signal does not generalise as a
construction-robust effect here; it does not show that AHN memory is empty, and it is
consistent with, not additional evidence for, the behavioural result that evicted RULER
accuracy is 0.000 in all three cells. (5) Limitations of the pilot: one cell (GDN), one task
(qa1), n=32, an unvalidated lens, and no pre-specified stopping rule.

## What was and was not pre-specified (for Gautam; not for the paper text)

| Item | Status |
|---|---|
| Amendment before compute (dataset register rule) | **Not filed.** Retroactive. |
| Sample size / stopping rule | **Not pre-specified.** n=32 is the count of qa1-64k examples passing the eviction screen (all 100 were screened). |
| Choice of 64k over 16k/32k | Made after inspecting **context lengths**, not outcomes. Doc states the cohort was frozen and scanned before GPU use. |
| Cohort freeze | Frozen in commit 01dd30e, before the C1/C2 result commit 0289b56. |
| Metric and criteria | Reused from the pre-registration (C1 below chance, C2 10×). Scorer adapted to multi-token answers, not pre-registered. |
| Method versions | Final is `pointcapture_v3`. **[CONFIRM: Son]** were v1/v2 run on this cohort and did they change after seeing results? Not recorded in the repo. If yes, disclose it here. |
| Screening threshold | Screened at 32,640 tokens but run at window 8064, so every "evicted" example is far past the run window (conservative). **[CONFIRM: Son]** why 32,640 was used. |
| Lens | `lens_validated=false`, same as every other RQ2 readout. |

## Compute for Table 10 (measured, from the run log in the result JSON)

32 examples, 4.6 min total (about 8 s each), **peak VRAM 12.7 GB**, on the box. It is the only
measured peak-VRAM figure in the project so far; add it to `build_table10.py`.

## Decision asked of Gautam

1. File this as Amendment 5 (retroactive), or leave the pilot unamended and mention it in
   Limitations only?
2. Confirm the pilot stays Limitations / supporting, never confirmatory.
