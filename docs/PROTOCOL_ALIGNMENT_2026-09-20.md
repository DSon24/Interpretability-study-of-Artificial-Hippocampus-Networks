# Protocol alignment — RQ1/RQ3 vs the concurrent write-attrition study (DRAFT)

Tracker row 88. Drafted 20 Sep 2026 for Gautam's sign-off. Purpose: every setting side by
side, each marked **match / differs / unknown**, so no difference between the two papers is
left unexplained.

**What this table can and cannot say.** We hold no artefacts from the concurrent study
(Kashyap 2026, under review ACL). Everything known about it comes from the proposal's
related-work section: F1 shift 0.4–2.3 points, 38–42% of answers changed, ρ(JS, changed
answers) 0.34–0.41, ρ(JS, F1) −0.09 to 0.00, and window 8064 (confirmed 1 Sep). So the
comparison column is the **official AHN LongBench harness** in `eval/longbench/`, which we can
read, and the Kashyap column is **unknown** wherever Gautam has not told us. Filling in the
Kashyap column is the ask.

Ours is read from `notebooks/03_nowrite_reproduction.ipynb`, `configs/`, the saved
`results/run_3b_*/03_nowrite_reproduction.json` and `docs/FINDINGS.md`.

## Something to know before the table

**Status of the prompt-format mismatch (updated 21 Sep, closed).** Only GDN had the Qwen chat template (14 Sep, commit 16dd9c8); DeltaNet and Mamba2 were the 9 Sep raw-prompt runs, so Table 5 mixed formats. DN and M2 were re-run with the chat template on 21 Sep (commit 74aafde; raw-prompt backups kept as `03_nowrite_reproduction_pre_qwen_chat.json`). First-line ΔF1: GDN −4.10, DN −3.96, M2 −4.63 points, against +6.72 (DN) and −0.31 (M2) on the raw prompt. The DN +6.7 was a prompt-format artefact on top of the scoring artefact, and the 18 Sep cross-cell contrast (DN vs GDN +10.81, Holm p=0.032) does not survive matched formats (now +0.14, Holm p=1.0). Rows 4 and 10 below are updated to match; see the 21 Sep entry in `FINDINGS.md`.

## The table

| # | Setting | Ours | Official AHN harness (`eval/longbench`) | Concurrent study | Status |
|---|---|---|---|---|---|
| 1 | Dataset / version | `THUDM/LongBench`, `hotpotqa_e`, test split | `eval.sh` runs LongBench v1 (`zai-org/LongBench`); `pred.py --e` uses the -E list | unknown | **differs** from `eval.sh` (E vs v1); Kashyap unknown |
| 2 | Task list | `hotpotqa` only | `eval.sh`: dureader, hotpotqa, musique, narrativeqa, qmsum, triviaqa (v1). `pred.py --e`: 13 tasks incl. hotpotqa | unknown; is 0.4–2.3 pts a mean over several tasks? | **unknown**; a multi-task mean regresses toward zero vs one long-context task |
| 3 | n and cohort | n=60, length-stratified 20/20/20 (short/mid/long thirds), eligible = 8,192 < tokens ≤ 32,000; realised 8,491–17,293 tokens | all examples | reportedly 60 (nb03 comment); source not verified | **unknown** — verify |
| 4 | Prompt format | **All three cells: Qwen chat template** (DN and M2 re-run 21 Sep, commit 74aafde; raw-prompt runs kept as `_pre_qwen_chat.json`) | chat template applied when `"qwen2"` is in the model name (skipped otherwise) | unknown | **match** with the official harness for all three cells; our internal mismatch is closed; Kashyap unknown |
| 5 | max_new_tokens | 32 | 32 (`dataset2maxlen`: hotpotqa) | unknown | **match** (official) |
| 6 | Decoding | greedy (`do_sample=False`) | not checked in this pass | unknown | **unknown** |
| 7 | Truncation | none; our filter drops prompts > 32,000 tokens | keeps the last `max_length − max_gen` tokens when over `max_length` (`max_length` is a CLI arg) | unknown `max_length` | **unknown**; if his `max_length` < 17.3k some of our prompts would be cut in his harness |
| 8 | Scorer | headline = **first-line** F1 (`ahn_interp.qa_f1_score`); full-generation stored | official `scorer_e` scores the **full generation** for hotpotqa (first-line only for trec/triviaqa/samsum/lsht) | unknown | **differs** — but on the chat-template data it no longer changes the RQ1 verdict: full-generation ΔF1 is a constant 0.48 pt lower in all three cells (one shared NOWRITE example), so no sign flips and the pairwise contrasts are identical (`results/05_table5_rq1_crosscell_official.json`). The raw-prompt runs did flip the sign in all three cells |
| 9 | Answer normalisation | `normalize_answer` (articles removed before punctuation) | official order differs slightly | unknown | **differs** (no effect seen on our generations) |
| 10 | "Answer changed" | 30.0 / 31.7 / 30.0% (GDN/DN/M2, chat template; identical under normalised EM, raw first-line equality and raw full-generation equality; 95% CIs include the band, e.g. GDN [18.3, 41.7]). Raw-prompt data gave 33.3 / 36.7 / 41.7% normalised and 40 / 40 / 45% raw first-line | n/a | 38–42% | **unknown** — no definition reaches the band on the chat-template data, so the earlier inference that raw first-line equality explains it does not hold there; Kashyap's prompt format is unknown (row 4) |
| 11 | Boundary JS | Jensen–Shannon (natural log, ε=1e-12) between AHN and NOWRITE softmax at the first generated position, full vocabulary, one value per example | n/a | unknown: position, log base, which logits | **unknown** |
| 12 | F1 in ρ(JS, F1) | we log both per-example ΔF1 (ρ≈0) and F1_nowrite (ρ −0.29 to −0.36, significant) | n/a | "−0.09 to 0.00": ΔF1 or F1 level? | **unknown** — the two give different answers on our data |
| 13 | NOWRITE | AHN-module output zeroed by a forward hook on all 36 AHN layers; sinks and window unchanged | n/a | "removing all writes": zeroed output, frozen state, or no module? | **unknown**; run 032 shows NOWRITE ≠ no-AHN (F1 ~0.34 vs 0.076) |
| 14 | Attention sinks | 128 (floor run: 0) | 128 via `eval.sh` arg | unknown | **unknown** |
| 15 | Sliding window | 8064 | `eval.sh` arg | confirmed 8064 (1 Sep) | **match** (per Gautam) |
| 16 | Seed / cohort draw | 20260820; only affects which 60 examples are drawn | n/a | unknown; same 60 or a different sample? | **unknown**; a different draw adds sampling noise at n=60 |
| 17 | Checkpoints | `ByteDance-Seed/AHN-{GDN,DN,Mamba2}-for-Qwen-2.5-Instruct-3B`, merged into Qwen2.5-3B-Instruct | same release | unknown scale / cell | **unknown** |
| 18 | CIs | percentile bootstrap, n_boot=4000 | n/a | unknown | **unknown** |

**Tally (18 rows):** 3 match (4, 5, 15), 3 differ (1, 8, 9), 12 unknown (2, 3, 6, 7, 10–14, 16–18). Rows 2, 10, 11, 12 and 13 change what Table 5 and Table 8 report; row 8 no longer does on the chat-template data.

## What to ask Gautam (in priority order)

1. Row 8/10: which scorer and which change-rate definition produced 0.4–2.3 pts and 38–42%?
2. Row 2/3: which tasks and n?
3. Row 13/14: does "removing all writes" keep the 128 sinks, and is it zeroed output or no module?
4. Row 11/12: exact boundary-JS definition, and is "F1" ΔF1 or the F1 level?
5. Row 4/7: was the Qwen chat template applied, and what `max_length`?
6. Row 1: v1 or -E?
7. Row 16/17: same 60 examples, same checkpoint and scale?

## Our own to-dos this exposes

- ~~Re-run DN and M2 nb03 with the chat template (row 58), then regenerate Tables 5 and 8.~~ **Done 21 Sep** (commits 74aafde, 6e82f34): DN and M2 nb03 re-run, Table 5 and Figure 3 regenerated, and 04b re-run for all three cells so Table 8 is rebuilt on the chat template.
- Save the ρ(JS, changed) / ρ(JS, F1) check to a results JSON on the chat-format data (row 80). *Partly covered:* Table 8 now saves ρ(boundary JS, ΔF1) on chat-template data (`results/05_table8_rq3_crosscell.json`); ρ(JS, answer changed) and ρ(JS, F1 level) are still not saved.
- If we report first-line anywhere, state that it is not the official convention (row 91).
