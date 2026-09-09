# Planning documents

The two source documents for this project live outside the repository (Google Drive,
`02 Learning & Programs / Algoverse 7-11 Gautam DPHS`). They are the authority for what
this repo is trying to produce; this file is a pointer plus the parts that engineering
decisions depend on, so a notebook can be understood without opening them.

- **`Proposal_AHN_Compressive_Memory.docx`** — Draft 1, 16 Aug 2026. Research questions,
  methods, controls, compute budget, timeline, open questions.
- **`Expected_Tables_and_Figures_rev3.docx`** — the current pre-registration of record.
  Prepared 18 Aug 2026; Amendment 1 filed 25 Aug, Amendment 2 filed 3 Sep, Amendment 3
  filed 8 Sep, and Amendment 4 filed 9 Sep (the multi-family plan — see below). Ten tables
  (plus Table 1b) and eight figures with placeholder values, validation gates, and a
  per-RQ null-result plan. `_rev1` and the unsuffixed original are kept for the audit
  trail — read `_rev3`.

> **`PROPOSAL.pdf` in Drive is a render of this file and is now stale.** It predates
> Amendment 3 and still shows the pre-audit C3 wording. Re-render before circulating.

## Amendment 3 decision — 8 Sep 2026

Gautam approved the RULER re-scope at the pre-registered n=60 and cleared DeltaNet and
Mamba2 to start. C2 remains a formal Table 4 FAIL against the 10x bar, while Methods must
report the statistically significant sub-threshold effect (1.076x per digit, 95% CI
[1.054, 1.098], p<0.0001). Length/content/construction checks are supporting and
limitations analyses, not blockers. The original four-needle result is explicitly
non-generalizing. See the [decision record](DECISION_RECORD_2026-09-08.md) and the
[register of datasets Gautam provided](DATASET_REGISTER_2026-09-08.md).

## Deviations of record (audit, 3 Sep 2026)

The repo was audited against both documents on 3 Sep. Direction is on plan — RQ framing,
sequencing, settings of record, cut order and null-result discipline all match. Six
divergences and one artefact bug were recorded in Amendment 2 and the Execution Tracker;
the table below now reflects their status after Amendment 3.

| # | Deviation | Where it bites |
|---|---|---|
| 1 | **Readout basis.** Expected Tables §1 specifies the Δ-readout (residual stream, AHN − NOWRITE). The battery reports the `o_t` readout instead. Both are stored per row in `04_retention_rows.json` (`rank` vs `rank_c1_residual`). | The 2 Sep "layer 27 is a working instrument" conclusion holds only in the `o_t` basis. Evicted medians at L27: 68,132 (`o_t`) vs 109,793 (Δ-resid) against chance 75,968. Declare the basis of record. |
| 2 | **RULER was unused at audit; resolved 7–8 Sep.** Run 026 populated the pre-registered n=60 cohort and Amendment 3 makes it primary for the re-scoped RQ2. | The homemade cohort is supporting/limitations analysis. DN/Mamba2 now repeat the RULER protocol. |
| 3 | **One control remains unrun.** C3-lens was completed in the RULER battery and disqualified layers 9 and 18; C4's layer-permutation half remains unrun. | Complete C4 during the now-unblocked three-cell sequence; it no longer gates starting DN/M2. |
| 4 | **No-AHN floor unrun.** Table 1 marks it *Primary — floor*; zero references in the repo. NOWRITE ≠ no-AHN. | No retention claim yet has the baseline the proposal requires it to exceed. |
| 5 | **Missing artefacts.** Figure 2 (never-cut set) absent from `05_`; Table 10 has no file; Tables 1–2 have no artefact. | Figure 2 is explicitly protected by §8's cut order. |
| 6 | **Gate crossed; framing decided 8 Sep.** C2 is significant but below its 10x bar, and Table 6/`04b` were populated after the original binary gate failed. | Amendment 3 records Gautam's approved deviation: Table 4 says FAIL; Methods reports the 1.076x effect and rationale. |
| 🔴 | **Artefact bug.** `03_nowrite_reproduction.json`'s summary and `05_table5_rq1.json` still hold full-generation metrics; the first-line headline lives only in README prose, and recomputing it from the saved `_fl` fields does not reproduce the README. | Change rate 35.0% vs 33.3%; pooled ΔF1 +3.34 vs +6.11 pts. Short and mid match to the decimal — the **long stratum flips sign** (−2.61 vs +5.72). One of README or JSON is stale. |

## The artefact set (Expected Tables and Figures)

| # | Artefact | Produced by |
|---|---|---|
| Table 1 | Ablation grid — which conditions exist, which are primary | fill in before running — **no artefact yet** |
| Table 1b | Multi-family extension grid (Llama / Mistral) — **Amendment 2**; plan is the "Multi-family evaluation" section below (**Amendment 4**) | gated on the 3B baseline + a checkpoint-feasibility check; owner Devin → Hannah + Sơn |
| Table 2 | Evaluation cohorts and sizes | fill in before running — **no artefact yet** |
| **Table 3** | **J-lens sanity checks on the base model** | `02_jlens_fit_and_validate.ipynb` |
| **Table 4** | **Control battery C1–C4** | `04_niah_retention.ipynb` |
| Table 5 | RQ1 task effect, length-stratified | `05_analysis_and_figures.ipynb` |
| Table 6 | Retention summary per checkpoint and layer, with fit R² | `05_` |
| Table 7 | Pairwise half-life ratios with their own CIs | `05_` (needs ≥ 2 cells) |
| Table 8 | RQ3 headline correlation vs boundary JS | `05_` (needs the `04b` join) |
| Table 9 | Variance decomposition | `05_` |
| Table 10 | Compute accounting | fill in continuously |
| Figure 2 | The three primary controls | `04_` + `05_` |
| Figure 3 | RQ1 forest plot | `05_` |
| Figure 4 | Retention decay, log y | `05_` |
| Figure 5 | Target rank vs eviction distance | `05_` |
| Figure 6 | Layer profile | `05_` |
| Figure 7 | Readout entropy | `05_` |
| Figure 8 | RQ3 scatter, side by side with boundary JS | `05_` (needs `04b`) |

**Tables 3 and 4 and Figure 2 never get cut.** From §8 of the document: *"Validation is
not the part you drop when time is short — it is the part that determines whether
anything else in the paper means what we say it means."*

Cut order if Week 9 arrives consuming buffer: Table 9 → Figure 7 → all 7B rows →
LV-Eval → Figure 6.

## Validation gates, restated as code

`Table 3` (J-lens, base model — all pass/fail):

1. final-layer identity — KL < 0.01 against the model's own next-token distribution
2. logit-lens agreement — top-1 agreement ≥ 60% at mid layers
3. known-fact recall — "The capital of France is" ranks " Paris" top-1 from ~layer 20
4. map stability — two maps from disjoint 500-context corpora agree on top-10 ≥ 80%
5. map cost — wall-clock GPU-hours; gates the 7B decision

`Table 4` (controls — C1–C3 must pass before RQ2 or RQ3 is populated; C4 is diagnostic):

| ID | Control | Pass criterion | What failure means |
|---|---|---|---|
| C1 | zero-state (NOWRITE) | `P_mem(y*)` drops to chance | the lens reads the backbone, not the memory. **Fatal — stop** |
| C2 | distractor token | a near-but-absent token stays ≥ 1 order of magnitude below `y*` | readout reflects topic, not the stored item; RQ2 weakens to "semantic gist" |
| C3 | shuffled context | half-life drops substantially when word order is destroyed | the state encodes recency, not content. **This is the interesting negative result, and it is publishable** |
| C4 | layer permutation / pre-eviction baseline | layer L's map degrades on layer L′; in-window readout is the ceiling | the map is not layer-specific, or the placement is wrong |

> **The two documents disagree about C3, and only one version has been run.** The proposal
> docx defines C3 as the **shuffled lens** — decode through a permuted J-lens map, ruling
> out that the structure is an artefact of the decoding procedure. The Expected Tables doc
> redefined C3 as **shuffled context**, which is what the repo implements. C4's
> layer-permutation half is likewise unrun (Amendment 1 admits this). Both missing controls
> are CPU-only and both speak to the open C1 question — see Deviations 3 above.

## Metrics of record

| Metric | Definition |
|---|---|
| `Rank@t(y*)` | rank of the target in the J-lens readout of the AHN output at eviction distance *t* |
| `P_mem(y*|t)` | probability mass on the target in that readout |
| retention half-life | eviction distance at which `P_mem` falls to half its post-eviction peak |
| Δ-readout | AHN-output readout minus the readout at the same position under NOWRITE — **on the residual stream**, see the README's Findings |
| readout entropy | entropy of the J-lens token distribution |
| task score | F1 per example |
| `ρ(half-life, ΔF1)` | the RQ3 test, against `ρ(JS, F1) ≈ 0` from concurrent work |

## Null-result plans

Written down in advance so the analysis is not fitted to the picture afterwards.

- **RQ1 null** (most likely — prior work found F1 shifts of only 0.4–2.3 points): does not
  damage the paper. RQ1 exists to establish there is a behavioural effect worth
  explaining; its absence is itself the setup for RQ3.
- **RQ2 null**: either no cell-family difference (a clean negative result — three
  architecturally distinct cells converge on the same retention behaviour) or no
  detectable target signal after C1 passes (harder; pushes toward RQ1 + methodology).
- **RQ3 null**: the framing shifts from explanatory to descriptive. Still the first
  content-level description of AHN memory. Agree in advance with Gautam that this is an
  acceptable outcome rather than a failed project.

## Multi-family evaluation — implementation plan (Amendment 2 scope, Amendment 4 plan)

**Status:** planned, not started. Gated on Gate 0 and Gate 1 below.
**Owner:** Hannah Kim + Sơn Nguyễn — reassigned from Devin at the **31 Aug 2026** meeting.
Devin has been inactive since ~20 Aug and the plan was ~2 weeks overdue; the team agreed to
take the arm on itself (*"I don't think Devin will be working on it … we should just take
that on ourselves"*). Branch off **`main`** (`hannah_8.19.26` deprecated 2 Sep).
**Recorded in:** Table 1b and a Table 2 cohort row of `Expected_Tables_and_Figures_rev3.docx`,
both marked `[A2]`; **Amendment 4** of that document carries this plan.

### What it is

Gautam asked (raised ~20 Aug, confirmed 31 Aug) that one Qwen model be replaced with
comparable **Llama and Mistral** models, so the readout results describe *AHN compressive
memory* rather than *Qwen2.5 + AHN*. The axis is
**{Qwen2.5, Llama, Mistral} × {DeltaNet, GatedDeltaNet, Mamba2}** — nine cells — compared
at the **7B** tier, where each family has a comparable instruct base. The Qwen2.5-7B arm
already in Table 1 is the reference; the Llama and Mistral arms are **Secondary**.

### Why it is not straightforward (carried from Amendment 2)

1. **Confound.** The proposal's Limitations section states everything is Qwen2.5 *"because
   that is what has been released"*, and its strongest methodological argument is that the
   three cells share a backbone, corpus, recipe and author — so a difference between them is
   the cell. A second backbone family reintroduces exactly that confound. The multi-family
   result is therefore a **generality check on RQ2/RQ3**, reported as such — not a fourth
   cell in the primary comparison.
2. **No released checkpoint.** Every released AHN module is `*-for-Qwen-2.5-Instruct`
   (ByteDance-Seed HF collection: 3B / 7B / 14B, all three cells). There is no Llama or
   Mistral AHN checkpoint. Training one uses the upstream LLaMA-Factory recipe at ~32 GPUs
   × ~700 steps per family — outside both the training-free premise and the team's compute.

### Gate 0 — finish the Qwen baseline first (31 Aug meeting decision)

No multi-family compute until:

- C1–C4 verdicts are recorded on the 3B reference (or a dated deviation, as with Amendment 3);
- 3B retention curves + Table 7 pairwise ratios are populated for DN / GDN / M2;
- the seed run (seed 20260820) is complete and the Qwen-7B arm numbers are in hand.

> *"Validate contributions C1–C4 before beginning larger-scale or multi-family experiments …
> finish the existing 3B baseline before running the full model matrix."*

### Gate 1 — checkpoint feasibility (resolve before spending compute)

1. Search the HF Hub and the upstream repo / issues for a community AHN checkpoint on a
   Llama or Mistral base at ~7B with a documented training recipe. If one exists → use it,
   record provenance and its training data (the Gate 2 config-of-record match still applies).
2. If none exists, pick one **with Gautam**:
   - **Fallback A — scale axis instead of family axis.** Replace {Qwen, Llama, Mistral} with
     the released Qwen2.5 ladder **{3B, 7B, 14B} × {DN, GDN, M2}**. Same nine-cell shape,
     every checkpoint exists, no training, no new confound. Retitle Table 1b to "scale
     extension". Loses the cross-family test; keeps a scaling result.
   - **Fallback B — descope.** Drop Table 1b, keep Qwen-only, state single-family as an
     explicit Limitation (the proposal already concedes this).

### Gate 2 — protocol, only if Gate 1 yields real Llama/Mistral checkpoints

Config of record must match the Qwen arm **exactly** or the arm compares to nothing:
sliding window 8064, `num_attn_sinks` 128, seed 20260820, `use_ahn_router` off, same
cohorts (RULER NIAH n=60, LongBench-E HotpotQA n=60), same token-count strata, same
3-sampled-layers rule, same Δ-readout basis of record.

Per family: merge base + AHN weights (`merge_weights.py`); run `00_setup_and_config_audit`
and confirm `sliding_window` / `num_attn_sinks` / `use_ahn_router` against the Qwen audit;
**re-fit the J-lens map on that backbone** — a Qwen map does not transfer — and pass the
Table 3 sanity checks on that base model before any AHN readout; run `04` (C1–C4) per cell;
run the `04b` join for RQ3 on matching per-example keys. Extend Table 6 with `family` as a
column and Table 7 with within- and cross-family ratios.

### Presentation — decide after the 3B baseline numbers (left open on 31 Aug)

Default: one Table 6 with a `family` column (not nine tables), one retention-decay figure
with families as panels, Table 7 ratios within and across families. Split into per-family
tables only if the single table breaks the paper's space budget (4–5 tables short track,
~10–11 long).

### Compute

≈ 11–12 GPU-h per family (J-lens fit ~2, Table 3 ~2, `04` × 3 cells ~3, `04b` × 3 ~4.5), so
**≈ 22–24 GPU-h for both** — roughly doubling the project's measured budget. This is why
Gate 0 exists. Step detail in [`GPU_PLAN_2026-08-20.md`](GPU_PLAN_2026-08-20.md) →
"Multi-family extension".

### Null result

Half-lives and the RQ3 ρ within noise across families is a **clean positive for
generality** — the retention behaviour is a property of the cell, not the backbone — and
strengthens the shared-mechanism claim. A sharp difference means the RQ2 finding is
Qwen-specific and is reported as such.
