# Planning documents

The two source documents for this project live outside the repository (Google Drive,
`02 Learning & Programs / Algoverse 7-11 Gautam DPHS`). They are the authority for what
this repo is trying to produce; this file is a pointer plus the parts that engineering
decisions depend on, so a notebook can be understood without opening them.

- **`Proposal_AHN_Compressive_Memory.docx`** — Draft 1, 16 Aug 2026. Research questions,
  methods, controls, compute budget, timeline, open questions.
- **`Expected_Tables_and_Figures_rev2.docx`** — the current pre-registration of record.
  Prepared 18 Aug 2026; Amendment 1 filed 25 Aug, Amendment 2 filed 3 Sep. Ten tables
  (plus Table 1b) and eight figures with placeholder values, validation gates, and a
  per-RQ null-result plan. `_rev1` and the unsuffixed original are kept for the audit
  trail — read `_rev2`.

> **`PROPOSAL.pdf` in Drive is a render of this file and is now stale.** It predates
> Amendment 2 and still shows the pre-audit C3 wording. Re-render before circulating.

## Deviations of record (audit, 3 Sep 2026)

The repo was audited against both documents on 3 Sep. Direction is on plan — RQ framing,
sequencing, settings of record, cut order and null-result discipline all match. Six
divergences and one artefact bug are open; each is a task in the Execution Tracker under
goal **G2 Instrument validation** (or G3/G7 where noted), and all are recorded in
Amendment 2.

| # | Deviation | Where it bites |
|---|---|---|
| 1 | **Readout basis.** Expected Tables §1 specifies the Δ-readout (residual stream, AHN − NOWRITE). The battery reports the `o_t` readout instead. Both are stored per row in `04_retention_rows.json` (`rank` vs `rank_c1_residual`). | The 2 Sep "layer 27 is a working instrument" conclusion holds only in the `o_t` basis. Evicted medians at L27: 68,132 (`o_t`) vs 109,793 (Δ-resid) against chance 75,968. Declare the basis of record. |
| 2 | **RULER unused.** `ai.load_ruler()` ([`ahn_interp.py:784`](../ahn_interp.py)) has zero call sites; all NIAH work uses the homemade `build_niah_prompt`. | RULER is Table 2's *primary* RQ2 cohort and the reason ground truth is known by construction. Running it is the cheapest test of C1 candidate (b). |
| 3 | **Two controls unrun.** The proposal defines C3 as the shuffled *lens*; the Expected Tables doc redefined it as shuffled *context*, and only that version has been run. `JacobianLens.shuffled()` and `.permuted_layers()` exist with zero call sites. | The decoding-artefact control and C4's layer-permutation half both bear directly on the open C1 question. Both are CPU-only. |
| 4 | **No-AHN floor unrun.** Table 1 marks it *Primary — floor*; zero references in the repo. NOWRITE ≠ no-AHN. | No retention claim yet has the baseline the proposal requires it to exceed. |
| 5 | **Missing artefacts.** Figure 2 (never-cut set) absent from `05_`; Table 10 has no file; Tables 1–2 have no artefact. | Figure 2 is explicitly protected by §8's cut order. |
| 6 | **Gate crossed.** "C1–C3 must pass before RQ2 or RQ3 is populated" — they failed; Table 6 and `04b` were populated anyway, labelled provisional. | Report as a stated deviation in Methods, not as a caveat. |
| 🔴 | **Artefact bug.** `03_nowrite_reproduction.json`'s summary and `05_table5_rq1.json` still hold full-generation metrics; the first-line headline lives only in README prose, and recomputing it from the saved `_fl` fields does not reproduce the README. | Change rate 35.0% vs 33.3%; pooled ΔF1 +3.34 vs +6.11 pts. Short and mid match to the decimal — the **long stratum flips sign** (−2.61 vs +5.72). One of README or JSON is stale. |

## The artefact set (Expected Tables and Figures)

| # | Artefact | Produced by |
|---|---|---|
| Table 1 | Ablation grid — which conditions exist, which are primary | fill in before running — **no artefact yet** |
| Table 1b | Multi-family extension grid (Llama / Mistral) — **Amendment 2** | blocked on Devin's implementation plan |
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

## Scope change under review — multi-family evaluation (Amendment 2)

Gautam has asked that one Qwen model be replaced with comparable **Llama and Mistral**
models, to create a multi-family evaluation. Recorded as Table 1b and a Table 2 cohort row
in `Expected_Tables_and_Figures_rev2.docx`, both marked `[A2]`. **Nothing runs until the
implementation plan is written down and the team has reviewed it** — the point of the plan
is to confirm the models, settings, datasets and metrics match the Qwen arm's config of
record (window 8064, sinks 128, seed 20260820, same cohorts and strata).

Two things to settle first, because this cuts against the design as pre-registered:

1. The proposal's Limitations section states that everything is Qwen2.5 *"because that is
   what has been released"*, and its strongest methodological argument is that the three
   cells share a backbone, corpus, recipe and author — so a difference between them cannot
   be attributed to anything but the cell. A second backbone family reintroduces exactly
   the confound that argument removes.
2. The released AHN checkpoints are **Qwen-only**. A Llama or Mistral arm needs either new
   AHN training — which the proposal explicitly rules out as training-free — or a different
   comparison altogether.

Owner: Devin. Branch off **`main`**; `hannah_8.19.26` is deprecated as of 2 Sep.
