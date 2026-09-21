# Protocol alignment — RQ1/RQ3 vs the concurrent write-attrition study (DRAFT)

Tracker row 88. Drafted 20 Sep 2026 for Gautam's sign-off; Kashyap column filled in 21 Sep
from the manuscript. Purpose: every setting side by side, each marked **match / differs /
unknown**, so no difference between the two papers is left unexplained.

**Source for the Kashyap column (updated 21 Sep).** The 20 Sep draft had no artefacts from the
concurrent study and marked most of this column unknown. It is now read from the manuscript
Gautam shared — *What Does the Artificial Hippocampus Actually Do? A Mechanistic
Write-Attrition Study* (anonymous ACL submission 3599; cited here as Kashyap 2026). Section and
table references in the Kashyap column point into that manuscript. Where the manuscript is
silent, the cell still says **not stated**, and those rows are the remaining questions for
Gautam. We still hold no code, per-example outputs or cohort IDs from that study.

Ours is read from `notebooks/03_nowrite_reproduction.ipynb`, `configs/`, the saved
`results/run_3b_*/03_nowrite_reproduction.json` and `docs/FINDINGS.md`. The official AHN
LongBench harness (`eval/longbench/`) is kept as a third column for reference.

## Something to know before the table

**Status of the prompt-format mismatch (updated 21 Sep, closed).** Only GDN had the Qwen chat template (14 Sep, commit 16dd9c8); DeltaNet and Mamba2 were the 9 Sep raw-prompt runs, so Table 5 mixed formats. DN and M2 were re-run with the chat template on 21 Sep (commit 74aafde; raw-prompt backups kept as `03_nowrite_reproduction_pre_qwen_chat.json`). First-line ΔF1: GDN −4.10, DN −3.96, M2 −4.63 points, against +6.72 (DN) and −0.31 (M2) on the raw prompt. The DN +6.7 was a prompt-format artefact on top of the scoring artefact, and the 18 Sep cross-cell contrast (DN vs GDN +10.81, Holm p=0.032) does not survive matched formats (now +0.14, Holm p=1.0). Rows 4 and 10 below are updated to match; see the 21 Sep entry in `FINDINGS.md`.

**What the manuscript changes (21 Sep).** Four things the 20 Sep draft could not know:

1. **The 0.4–2.3 band has the same sign as ours.** It is Table 1b's "NOWRITE − full F1 at
   100%" on the balanced 60 (+0.4 / +1.0 / +2.3 for DN / GDN / Mamba2), i.e. AHN is *lower*
   than NOWRITE. Our AHN − NOWRITE of about −4 points points the same way; ours is about 2–10×
   larger. On all 96 cases the manuscript's full-write − NOWRITE is +2.4 / −0.8 / +3.6
   (Table 1c), and every interval spans zero.
2. **The cohorts are built differently.** Theirs is 20 shortest, 20 median and 20 longest of
   the 96 HotpotQA cases, chosen deterministically. Ours is a seeded random draw of 20 per
   length third from 8,192 < tokens ≤ 32,000. AHN's only resolved gains in the manuscript are
   in the longest-20 stratum (+7.1 / +10.8 / +10.6, Table 1c), so the length ranges of the two
   cohorts matter and should be compared directly.
3. **"Boundary JS" is a different quantity from ours** (row 11). Table 8 / Figure 8's
   "prior-work predictor" is therefore not the manuscript's predictor.
4. **Their NOWRITE and no-AHN agree on 81/81 RULER outputs**; ours disagree (run 032).
   Our floor config also sets sinks to 0 rather than 128 (row 14), which probably explains it.
   This needs checking before we cite either result.

## The table

| # | Setting | Ours | Official AHN harness (`eval/longbench`) | Kashyap 2026 (manuscript) | Status vs Kashyap |
|---|---|---|---|---|---|
| 1 | Dataset / version | `THUDM/LongBench`, `hotpotqa_e`, test split | `eval.sh` runs LongBench v1 (`zai-org/LongBench`); `pred.py --e` uses the -E list | LongBench-E HotpotQA (§2) | **match** (both -E); differs from `eval.sh` (v1) |
| 2 | Task list | `hotpotqa` only | `eval.sh`: dureader, hotpotqa, musique, narrativeqa, qmsum, triviaqa (v1). `pred.py --e`: 13 tasks incl. hotpotqa | HotpotQA only for the LongBench contrasts. BYPASS is run separately on LV-Eval FactRecall (n=30) and HotpotWikiQA (n=29) at 128K (Table 1a) | **match** — the 0.4–2.3 band is one task, not a multi-task mean |
| 3 | n and cohort | n=60, length-stratified 20/20/20 (short/mid/long thirds), eligible = 8,192 < tokens ≤ 32,000; realised 8,491–17,293 tokens | all examples | All 96 HotpotQA cases for full vs NOWRITE; **balanced 60 = 20 shortest, 20 median, 20 longest** of the 96 for attrition (§2) | **differs** — deterministic extremes-plus-median vs seeded draw within thirds; no 32k cap stated; their token range not stated |
| 4 | Prompt format | **All three cells: Qwen chat template** (DN and M2 re-run 21 Sep, commit 74aafde; raw-prompt runs kept as `_pre_qwen_chat.json`) | chat template applied when `"qwen2"` is in the model name (skipped otherwise) | not stated | **unknown** — our internal mismatch is closed |
| 5 | max_new_tokens | 32 | 32 (`dataset2maxlen`: hotpotqa) | not stated | **unknown**; matches official |
| 6 | Decoding | greedy (`do_sample=False`) | not checked in this pass | greedy — "greedy decoding gives each paired contrast one deterministic trajectory" (§2, Validation) | **match** |
| 7 | Truncation | none; our filter drops prompts > 32,000 tokens | keeps the last `max_length − max_gen` tokens when over `max_length` (`max_length` is a CLI arg) | 128K LV-Eval cohorts: prompts above 131,072 tokens middle-truncated (§2). LongBench-E: not stated | **unknown** for LongBench-E |
| 8 | Scorer | headline = **first-line** F1 (`ahn_interp.qa_f1_score`); full-generation stored | official `scorer_e` scores the **full generation** for hotpotqa (first-line only for trec/triviaqa/samsum/lsht) | "F1"; first-line vs full generation not stated | **unknown** — on the chat-template data it no longer matters: full-generation ΔF1 is a constant 0.48 pt lower in all three cells (one shared NOWRITE example), so no sign flips and the pairwise contrasts are identical (`results/05_table5_rq1_crosscell_official.json`). The raw-prompt runs did flip the sign in all three cells |
| 9 | Answer normalisation | `normalize_answer` (articles removed before punctuation) | official order differs slightly | not stated | **unknown**; differs from official (no effect seen on our generations) |
| 10 | "Answer changed" | 30.0 / 31.7 / 30.0% (GDN/DN/M2, chat template, n=60; identical under normalised EM, raw first-line equality and raw full-generation equality; 95% CIs include the band, e.g. GDN [18.3, 41.7]). Raw-prompt data gave 33.3 / 36.7 / 41.7% normalised and 40 / 40 / 45% raw first-line | n/a | **40.0 / 41.7 / 38.3%** (DN / GDN / Mamba2) at 100% attrition on the **balanced 60** (Table 1b). Changed = greedy output differs; normalisation not stated. Described as "output sensitivity rather than semantic equivalence" | **differs in cohort** (row 3); definition partly known. Point estimates below the band, CIs include it |
| 11 | Boundary JS | Jensen–Shannon (natural log, ε=1e-12) between AHN and NOWRITE softmax at the first generated position, full vocabulary, one value per example | n/a | JS between the attrited (rate r) and full-write (r=0) distributions, restricted to the **top-20 union plus a residual bucket**, **averaged over B boundaries** recorded **every 64 evictions** (Eq. 1). Log base not stated | **differs** — position (eviction boundaries vs first generated token), support (top-20 + bucket vs full vocab) and contrast (attrition level vs AHN/NOWRITE) all differ |
| 12 | F1 in ρ(JS, F1) | we log both per-example ΔF1 (ρ≈0) and F1_nowrite (ρ −0.29 to −0.36, significant) | n/a | ρ(JS, changed answer) = .41 / .34 / .35; ρ(JS, F1) = −.06 / −.09 / −.00 (Table 1b), glossed as "does not predict utility". Computed over the repeated 60-example cohort across attrition levels (Fig. 1 caption); ΔF1 vs F1 level not stated | **differs** — theirs pools attrition levels, ours is one value per example; F1 definition still **unknown** |
| 13 | NOWRITE | AHN-module output zeroed by a forward hook on all 36 AHN layers; sinks and window unchanged | n/a | "suppresses recurrent-memory updates after KV eviction" (§2). A no-AHN runner that skips writes and reads matches NOWRITE on **81/81** RULER outputs (12–24K, §2 Validation) | **differs in mechanism** (suppress writes vs zero the output), equivalent if the state starts at zero. Our run 032 has NOWRITE ≠ no-AHN (F1 ~0.34 vs 0.076) — see row 14 |
| 14 | Attention sinks | 128 (floor run: 0) | 128 via `eval.sh` arg | 128 (LongBench-E and LV-Eval, §2) | **match** for AHN runs. Our floor's 0 sinks is the likely cause of the row-13 NOWRITE ≠ no-AHN gap — **verify** |
| 15 | Sliding window | 8064 | `eval.sh` arg | 8,064 for LongBench-E; 32,640 for LV-Eval (§2) | **match** |
| 16 | Seed / cohort draw | 20260820; only affects which 60 examples are drawn | n/a | LongBench cohort is deterministic by length; 128K cohorts sampled "uniformly with a fixed seed" (§2) | **differs** (see row 3) |
| 17 | Checkpoints | `ByteDance-Seed/AHN-{GDN,DN,Mamba2}-for-Qwen-2.5-Instruct-3B`, merged into Qwen2.5-3B-Instruct | same release | released merged Qwen2.5-3B DN, GDN and Mamba2 checkpoints (§2, footnote to `bytedance-seed/AHN`). 7B checkpoints appear for internal-state traces only, with no generation (Fig. 2a) | **match** at 3B |
| 18 | CIs | percentile bootstrap, n_boot=4000 | n/a | paired bootstrap for score and length differences; length-stratified resampling for correlations (§2); n_boot not stated | **partial match** — we do not stratify the resampling of correlations |
| 19 | Precision / attention kernel | BF16, `flash_attention_2` (`ahn_interp.load_ahn_model` defaults) | — | BF16, FlashAttention-2 / FlexAttention path (§2) | **match** |
| 20 | Layers and positions recorded | readout at 3 sampled layers (9, 18, 27), final prompt position | — | recurrent and hidden statistics at **all 36 layers**, plus top-20 token probabilities, confidence and entropy, **every 64 evictions** on LongBench-E (every 512 tokens on LV-Eval) (§2) | **differs** — theirs is much denser in both layers and positions |
| 21 | Teacher-forced answer probability | not computed | — | P_ans(r): geometric-mean probability of the full-write greedy answer under attrition rate r (Eq. 2); rises as writes are removed (Table 1b) | **not in ours** — cheap to add and directly comparable |

**Tally (21 rows):** 7 match (1, 2, 6, 14, 15, 17, 19), 1 partial (18), 7 differ (3, 10, 11, 12, 13, 16, 20), 5 unknown (4, 5, 7, 8, 9), 1 not in ours (21). Rows 3, 11 and 12 change what Table 5 and Table 8 can claim about the comparison. Row 13/14 changes how the floor run can be cited.

## What to ask Gautam (in priority order)

1. Row 3: can we have the 96 case IDs, or the token range of each 20-case stratum, so we can
   rerun RQ1 on the exact balanced 60?
2. Row 11/12: is ρ(JS, F1) against ΔF1 or the F1 level, and is it pooled across attrition
   levels? Do you want Table 8 row 2 recomputed with your Eq. 1 definition on our cohort?
3. Row 8/10: first-line or full-generation F1, and is "changed" judged on normalised or raw
   text?
4. Row 4/5/7: was the Qwen chat template applied, what `max_new_tokens`, and what
   `max_length` / truncation on LongBench-E?

## Our own to-dos this exposes

- ~~Re-run DN and M2 nb03 with the chat template (row 58), then regenerate Tables 5 and 8.~~ **Done 21 Sep** (commits 74aafde, 6e82f34): DN and M2 nb03 re-run, Table 5 and Figure 3 regenerated, and 04b re-run for all three cells so Table 8 is rebuilt on the chat template.
- Save the ρ(JS, changed) / ρ(JS, F1) check to a results JSON on the chat-format data (row 80). *Partly covered:* Table 8 now saves ρ(boundary JS, ΔF1) on chat-template data (`results/05_table8_rq3_crosscell.json`); ρ(JS, answer changed) and ρ(JS, F1 level) are still not saved.
- If we report first-line anywhere, state that it is not the official convention (row 91).
- **New (21 Sep, from the manuscript):**
  - Rebuild the cohort as 20 shortest / median / longest of the 96 LongBench-E HotpotQA cases and rerun RQ1 on it (row 3). Report the longest-20 stratum separately, since that is where the manuscript's only resolved AHN gain sits.
  - Implement the manuscript's boundary JS (Eq. 1) alongside ours before calling Table 8 row 2 "prior work's predictor" (row 11).
  - Rerun the floor with 128 sinks and check NOWRITE against no-AHN output by output (rows 13–14).
  - Add P_ans (Eq. 2) to nb03 so the confidence shift can be compared directly (row 21).
  - Reword the related-work sentence in the proposal and FINDINGS: the 0.4–2.3 band is NOWRITE − full on the balanced 60, the same sign as our result (row 10 and point 1 above).
