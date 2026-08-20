# Planning documents

The two source documents for this project live outside the repository (Google Drive,
`02 Learning & Programs / Algoverse 7-11 Gautam DPHS`). They are the authority for what
this repo is trying to produce; this file is a pointer plus the parts that engineering
decisions depend on, so a notebook can be understood without opening them.

- **`Proposal_AHN_Compressive_Memory.docx`** — Draft 1, 16 Aug 2026. Research questions,
  methods, controls, compute budget, timeline, open questions.
- **`Expected_Tables_and_Figures.docx`** — 18 Aug 2026. Ten tables and eight figures with
  placeholder values, validation gates, and a per-RQ null-result plan.

## The artefact set (Expected Tables and Figures)

| # | Artefact | Produced by |
|---|---|---|
| Table 1 | Ablation grid — which conditions exist, which are primary | fill in before running |
| Table 2 | Evaluation cohorts and sizes | fill in before running |
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
