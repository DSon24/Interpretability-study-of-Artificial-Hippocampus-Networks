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

Task ownership, deadlines and what is blocked live in the
**[Execution Tracker](https://docs.google.com/spreadsheets/d/1kbelJ5kGhtnIK80fDKlorGt633R152YX/edit?gid=1055888288#gid=1055888288)**
(Drive, team-only) -- three tabs: Execution Tracker, Experiment Log (one row per run), and
Meetings (decisions and action items). This README carries the technical state; the tracker
carries who is doing what by when. Keep the Experiment Log tab updated as runs land -- it is
the only place a run's config, seed and conclusion sit together.

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
| RULER NIAH n=60 — the *primary* RQ2 cohort | **done 7 Sep in the pre-registered instrument, and it reverses the C1 reading.** Layer 27 reads the *evicted* needle at median rank **17,250** of 151,936 (84.4% of examples beat chance) where the homemade prompts give 111,694 — worse than chance. The in-window confound was checked and runs backwards (evicted 17,250 vs in-window 54,506, permutation p=0.0021) | `04g_ruler_niah_stats.json`, `04h_ruler_needle_position.json` |
| C2 follow-up — is the distractor control measuring anything? | **done, and it reframes C2** — the raw ratio is confounded by pair identity; after baseline correction no layer shows a memory-specific effect | `notebooks/04-C2-debug.ipynb`, Sơn, 28–31 Aug |
| C3 follow-up — corrected shuffled-context rerun | **done** — needle held at a fixed token position; 96 matched rows. C3 does **not** give the expected order-sensitivity in either direction | `notebooks/04_niah_C3_analyze.ipynb`, Sơn, 29 Aug |
| Per-layer recomputation of the control battery | **done, no GPU** — layer 9's readout is degenerate and fails C4. The 2 Sep "layer 27 works" claim is **withdrawn**: wrong readout basis, and its C2 ratio vanishes under baseline correction | `04c_per_layer_controls.json`, `04d_c2_baseline_corrected.json`, 2–3 Sep |
| C1 rank correction + construction ladder + needle-category test | **done, reframes C1** — layers 18/27 show a real, Holm-significant needle-identity effect (12 place names vs. 12 common nouns); the pooled "C1 fails" verdict was an artifact of the original 4-needle sample, not evidence of no retention | `04c1_rank_baseline.json`, `06_construction_ladder.json`, `04f_needle_category_stats.json`, 4–5 Sep |
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

## Findings

The full findings log lives in **[docs/FINDINGS.md](docs/FINDINGS.md)** -- ten entries,
written as an append-only record so that later corrections sit visibly on top of what they
correct rather than quietly replacing it. **The 7 Sep J-lens repeat reverses the reading of
every entry before it** and is the current state of the C1 question.

| Entry | What it established |
|---|---|
| [19-20 Aug run](docs/FINDINGS.md#findings-from-the-1920-aug-run) | Instrumentation gates pass; the J-lens beats the logit lens 8-204x but fails Table 3 as written |
| [20 Aug NIAH retention](docs/FINDINGS.md#findings-from-the-20-aug-niah-retention-run) | C1, C2, C3 fail on the 210-config sweep; C4 passes. The result that blocks scaling |
| [Map stability + RQ3 join](docs/FINDINGS.md#findings-from-the-map-stability-check-and-the-rq3-join) | Top-10 overlap 0.91/0.87/0.89 rules out undersampling; RQ3 correlations do not survive Holm |
| [18 Aug pilot](docs/FINDINGS.md#findings-from-the-18-aug-pilot) | Quarantined DO NOT CITE; five bugs identified from it |
| [21 Aug C1 diagnosis](docs/FINDINGS.md#findings-from-the-21-aug-c1-diagnosis) | The memory channel is live, so C1's failure is not a dead hook -- points at prompt construction |
| [28-31 Aug C2 and C3](docs/FINDINGS.md#findings-from-the-2831-aug-c2-and-c3-investigation) | C2's raw ratio is pair-identity baseline, not memory; C3 gives no clean order-sensitivity |
| [2 Sep per-layer re-analysis](docs/FINDINGS.md#findings-from-the-2-sep-per-layer-re-analysis-corrected-3-sep) | CORRECTED 3 Sep -- the "layer 27 works" claim is withdrawn on two independent grounds |
| [4-5 Sep C1 rank correction](docs/FINDINGS.md#findings-from-the-45-sep-c1-rank-correction-construction-ladder-and-needle-category-test) | Baseline-corrected C1, the construction ladder, and the place-name vs common-noun split |
| [7 Sep RULER cohort](docs/FINDINGS.md#findings-from-the-7-sep-ruler-cohort-run) | The primary cohort, run at last -- logit-lens, superseded by the entry below |
| [7 Sep J-lens repeat + placement check](docs/FINDINGS.md#findings-from-the-7-sep-j-lens-repeat-and-placement-check) | **Layer 27 reads the *evicted* needle at rank 17,250 of 151,936 in the pre-registered instrument.** The in-window confound runs backwards (evicted beats in-window, p=0.0021). The C1 failure is a property of the homemade prompt construction, not of AHN |

## Repository layout

```
ahn_interp.py                 shared instrumentation — imported by every notebook
per_layer_controls.py         CPU-only: recompute Table 4 within layer, both readout bases
extract_c2_corrected.py       CPU-only: pull Son's baseline-corrected C2 out of the notebook
probe_construction.py         CPU-only: the NIAH construction ladder — Findings, 4–5 Sep
probe_prompt_format.py        CPU-only: prompt-format probe behind the same findings
configs/
  run_3b_gdn.json                    the run of record; run_3b_dn / run_3b_m2 exist, unrun
merged_ckpt/                  merged checkpoints (gitignored; rebuilt on every box)
notebooks/
  00_setup_and_config_audit.ipynb    load a checkpoint; record window / sinks / router
  01_instrumentation_gate.ipynb      prove the hook captures the memory's contribution
  02_jlens_fit_and_validate.ipynb    fit the J-lens; run Table 3's five checks
  02-duplicate.ipynb                 1000-context refit — intentional, see Housekeeping
  03_nowrite_reproduction.ipynb      Week-6 milestone: 38–42% changed answers
  04_niah_retention.ipynb            retention curves + controls C1–C4 (Tables 4, 6)
  04-C2-debug.ipynb                  C2 readout-bias investigation — Sơn, 28–31 Aug
  04_niah_C3_analyze.ipynb           corrected C3 shuffled-context rerun — Sơn, 29 Aug
  05_analysis_and_figures.ipynb      CPU only — Tables 5–9, Figures 3–8
  0.4b.ipynb                         RQ3 join — Sơn's build; see Findings
  AHN_clean.ipynb                    predates ahn_interp.py; hardcodes /workspace, unused
  pilot/                             Sơn's 18 Aug notebooks, kept for provenance
docs/
  FINDINGS.md                        the findings log — split out of this README, 6 Sep
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
    04c1_rank_baseline.json            per-pair rank baseline behind the C1 correction
    04d_c2_baseline_corrected.json     baseline-corrected C2 per layer — every effect vanishes
    04d1_c1_rank_baseline_corrected.json  baseline-corrected C1 — Findings, 4–5 Sep
    04e_needle_category_extended.json  place names vs common nouns vs person names
    04f_needle_category_stats.json     Holm-corrected stats for the category test
    04b_joined_retention_task.json     RQ3 join — no significant rank/ΔF1 correlation
    05_table5_rq1.json                 RQ1 by stratum, with bootstrap CIs
    05_table6_retention_summary.json   per-layer half-life fit — inadequate, see Findings
    05_table8_rq3.json                 RQ3 table; 05_table9_variance.json — variance decomposition
    06_construction_ladder.json        NIAH construction ladder; 06_prompt_format_probe.json
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

**Housekeeping — 6 Sep.** Checkpoint paths are no longer hardcoded anywhere: configs carry
`ckpt_name` and `ai.resolve_ckpt()` resolves it against `$AHN_CKPT_ROOT`. Three teammates'
`/home/jupyter-dphs-*` paths had accumulated across the notebooks, all dead after a box
reset. `notebooks/debugging/04_niah_C3_analyze.ipynb` was removed — byte-identical to the
copy at `notebooks/04_niah_C3_analyze.ipynb` apart from a stale `/workspace/...` path, with
all 42 output-bearing cells hashing the same. The findings log moved to
[docs/FINDINGS.md](docs/FINDINGS.md); this README had reached 1,085 lines.

## Setting up a fresh GPU box

The box resets roughly every 48 hours, so this is a from-scratch recipe rather than a
one-time note. Budget ~20 minutes, nearly all of it downloads. Walked end to end on
6 Sep 2026; every trap below cost real time, so none of them are hypothetical.

**We are unprivileged users on a shared JupyterHub box — no root, no `sudo`.** Every step
below stays inside your home directory: a venv, pip into that venv, a user-level Jupyter
kernel, and environment variables. Nothing here installs a system package or changes a
system setting, and nothing should. What is fixed and not ours to change: the NVIDIA
driver, the system CUDA toolkit, the MIG partitioning of the cards, and the JupyterHub
install itself. Three or four of us share the machine, so also treat GPU slices as shared —
check what is idle before taking one.

**1. Clone.**

```bash
git clone https://github.com/DSon24/Interpretability-study-of-Artificial-Hippocampus-Networks.git
cd Interpretability-study-of-Artificial-Hippocampus-Networks
```

Use the repository URL, not a `/tree/main` link copied from the browser — that is the
GitHub web path and git rejects it with `repository not found`.

A reset wipes `~/.gitconfig` along with everything else, so set your identity before you
commit anything — otherwise the first `git commit` fails with *"Author identity unknown"*:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Push credentials are gone too. Use a personal access token when git prompts, or
`gh auth login` if the CLI is there. Do not run `git config credential.helper store` — it
writes the token in plain text to a home directory on a machine four of us share.

**2. Main environment.**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval]"
pip install ipykernel && python -m ipykernel install --user --name ahn-venv --display-name "AHN (transformers 4.51)"
```

`wandb`, `accelerate`, `flash-linear-attention` (the Seerkfang fork — the PyPI package of
that name is a different library) and a `torch` pin are declared in `pyproject.toml` as of
6 Sep. They are import-time requirements: `qwen2_ahn` imports wandb and fla at module
scope, and transformers 4.51.0 only binds `init_empty_weights` when accelerate is present,
so without it `from_pretrained` dies with a bare `NameError`.

**Register the kernel under exactly that name.** Every notebook records the kernel it was
last opened with, so a differently-named kernel rewrites the file's metadata just by
opening it, and the next `git pull` aborts with *"local changes would be overwritten"*.
Fourteen notebooks had accumulated six different kernel names this way. The two of record
are `ahn-venv` for the main environment and `jlens-venv` for notebook 02.

**3. Check torch against the driver.**

```bash
nvidia-smi
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Want `True`. The pin exists because an unpinned resolve now picks torch 2.14 with CUDA 13
wheels, which refuse to initialise against this box's driver (570.148.08 = CUDA 12.8) with
*"The NVIDIA driver on your system is too old"*.

**Ignore that error's advice.** It tells you to update the driver; we are unprivileged
users on a shared JupyterHub box and cannot. The driver, the CUDA toolkit and the MIG
partitioning are all set by whoever administers the box. The only lever on our side is the
torch pin — so match torch to the driver, never the other way round. If an admin ever
upgrades the driver, re-pin rather than deleting the pin.

Related trap: installing `scipy`/`matplotlib` unpinned drags numpy from the project's
`1.26.4` up to 2.x. Pin them — `scipy==1.14.1`, `matplotlib==3.9.2` — if you need them.

**4. FlashAttention-2 — required for correctness, not speed.**

```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

Sliding-window attention is **not implemented for `sdpa`**. Load without FA2 and
transformers prints one warning, ignores `CFG["sliding_window"]`, and AHN never
activates — the run completes and every retention number in it is meaningless. Use the
prebuilt wheel matching torch/python/ABI; the PyPI sdist compiles against `nvcc`, which
this box does not have and which we cannot install without root. If no prebuilt wheel
matches a future torch/python combination, change the torch pin to one that has a wheel —
do not try to build from source here.

**5. Merge a checkpoint.**

```bash
python ./examples/scripts/utils/merge_weights.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --ahn-path  ByteDance-Seed/AHN-GDN-for-Qwen-2.5-Instruct-3B \
  --output-path ./merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN
```

The long *"newly initialized: `model.layers.*.ahn.fn.*`"* warning is expected — base
Qwen2.5 has no AHN parameters, so they are random until `load_ahn_overrides` overwrites
them. Confirm they landed:

```bash
python -c "
from safetensors import safe_open
import glob
ks=[]
for f in glob.glob('./merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN/*.safetensors'):
    with safe_open(f,'pt') as h: ks += [k for k in h.keys() if 'ahn' in k]
print(len(ks))"
```

Expect **288** (8 tensors x 36 layers). That proves presence, not values — notebook 01's
Gate B is what proves the AHN weights are real rather than random.

Nothing records this path. Configs carry `ckpt_name`, and `ai.resolve_ckpt("<dir name>")`
resolves it at runtime against `$AHN_CKPT_ROOT`, defaulting to `<repo>/merged_ckpt` — so
merging into the repo as above needs no environment variable at all. Only if the
checkpoints live elsewhere on the box:

```bash
export AHN_CKPT_ROOT=/path/to/merged_ckpt     # before starting Jupyter
```

A checkpoint that cannot be found raises a `FileNotFoundError` listing every location
tried and what the root actually contains, rather than the `HFValidationError` you get
when transformers reinterprets a dead path as a Hub repo id.

**6. Pick a MIG slice.**

The H100s are partitioned with MIG, so a process gets one 20 GB slice, not a card, and
`device_map="cuda:<idle_index>"` does not select it. UUIDs change on every box reset.

```bash
nvidia-smi -L     # list slices
nvidia-smi        # check which are actually idle — 3 other people share this box
export CUDA_VISIBLE_DEVICES=MIG-<uuid>
```

Pick an idle `1g.20gb`; a 3B job needs 6–16 GB. Inside the process the slice is always
`cuda:0`, so leave `device_map="cuda"` alone. Independent notebooks can run in parallel on
separate slices — wall clock, not GPU-hours, is the binding constraint.

For a **notebook**, the shell `export` does not reach the kernel — the Jupyter server
spawns it. Put this in the first cell, above any torch import, because CUDA reads the
variable once at init and later assignment is silently ignored:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-<uuid>"   # from nvidia-smi -L, changes on reset
```

**7. Smoke test before opening Jupyter.**

```bash
python -c "
import torch, ahn_interp as ai
print('ckpt root:', ai.ckpt_root())
b = ai.load_ahn_model(ai.resolve_ckpt('Qwen-2.5-Instruct-3B-AHN-GDN'), sliding_window=8064, num_attn_sinks=128)
print(torch.cuda.get_device_name(0), round(torch.cuda.memory_allocated()/1e9,2),'GB')
print(b.model.config._attn_implementation, b.model.config.sliding_window)"
```

Want ~6–7 GB, `flash_attention_2`, `8064`. Cheapest place to find a broken environment.

**8. Only if you are fitting a new J-lens (notebook 02).**

```bash
python3 -m venv ~/jlens-venv && source ~/jlens-venv/bin/activate
git clone https://github.com/anthropics/jacobian-lens.git && pip install -e jacobian-lens
pip install datasets accelerate ipykernel
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
python -m ipykernel install --user --name jlens-venv --display-name "jlens (transformers>=5)"
```

This env needs its own torch pin for the same driver reason. Usually you should **not**
refit: the corpus A and B lenses cost 1.92 + 1.25 GPU-h and live on the Hub. Pull them
instead, and note the `.pt` is a plain tensor dict that loads fine under the main
`transformers==4.51.0` env:

```bash
huggingface-cli login
huggingface-cli download gautam-dphs/ahn-interp-jlens-qwen25-3b --local-dir results/run_3b_gdn
```

The Hub stores corpus A as **`jlens_qwen25_3b_corpusA.pt`**, while the repo writes and
reads `jlens_qwen25_3b.pt` — the names do not match, so a plain download used to leave
every notebook silently on the logit lens. `ai.resolve_lens_path()` now accepts the Hub's
name, so the download above is sufficient; if no lens is found it raises with the
directory listing rather than falling back. Confirm after loading the model that cell 5
prints `readout: jlens`, not `logit_lens`.

That fallback is not cosmetic. On 7 Sep it swapped the readout mid-rerun, overwrote the
J-lens run of record in `04_retention_rows.json` with logit-lens rows, and flipped C3 from
fail to pass — a control verdict changing because of a filename. The logit-lens rerun is
kept as `04_*_logitlens_2026-09-07.json`.

Ignorable noise throughout: TensorFlow's cuFFT/cuDNN/cuBLAS *"already registered"* errors
(TF is dead weight from the `[train]` extra) and `df: ~/.triton/autotune: No such file`.

## Running an experiment

The box has a shell, so work directly in the clone — there is no need to upload anything.
Every notebook's bootstrap cell walks up the tree for `ahn_interp.py`, so running from
`notebooks/` inside the repo resolves it. Download the JSON at the end.

1. Open the notebook from `notebooks/` in the clone.
2. Edit the `CFG` cell — cell family, `sliding_window`, `num_attn_sinks`. The checkpoint
   is **not** hardcoded any more: `CFG["model_path"] = ai.resolve_ckpt("<dir name>")`
   resolves against `$AHN_CKPT_ROOT`, defaulting to `<repo>/merged_ckpt`. If the merged
   checkpoints live somewhere else on the box, export the root once before starting
   Jupyter (or set it in the first cell, before `import ahn_interp`):

   ```bash
   export AHN_CKPT_ROOT=/path/to/merged_ckpt
   ```

   A missing checkpoint now raises a `FileNotFoundError` naming the root it searched and
   what it found there, instead of an `HFValidationError` from transformers reinterpreting
   the dead path as a Hub repo id. The run configs in `configs/` carry `ckpt_name`, not an
   absolute path, for the same reason — the box's `/home/jupyter-dphs-*` changes on every
   reset. (`notebooks/AHN_clean.ipynb` is the one exception: it predates `ahn_interp` and
   still hardcodes `/workspace/...`.)
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

Building that environment, merging a checkpoint, and pinning a MIG slice are all in
**Setting up a fresh GPU box** above — the box resets every ~48 hours, so that is the
recipe you rerun, not a one-time note.

Three cells at 3B: swap `AHN-GDN` for `AHN-DN` and `AHN-Mamba2` in the merge command.
Mamba2 additionally needs the forked mamba
(`MAMBA_FORCE_BUILD=TRUE pip install "git+https://github.com/yuweihao/mamba.git"`).
Keep `merged_ckpt/` out of git (it already is).

## Experimental settings of record

| Setting | Value | Note |
|---|---|---|
| Backbone | Qwen2.5-3B-Instruct, then 7B | 14B dropped — ~28 GB BF16 exceeds the cards we get (24 GB when the call was made; 20 GB MIG slices on the current H100 box) |
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
   signal at layers 18 and 27). (b) prompt construction — **confirmed, and it is the
   answer at layer 27.** The 7 Sep J-lens repeat (Findings) reads the *evicted* RULER
   needle at median rank 17,250 [11,318, 30,936] against chance 75,968, with 84.4% of
   examples beating chance, while the homemade prompts read 111,694 [105,947, 121,525] —
   worse than chance — in the same instrument on the same day. The in-window confound was
   tested and runs backwards: evicted needles read *better* than visible ones (17,250 vs
   54,506, permutation p=0.0021), which is what the D-resid basis should do, since the
   memory is the only channel carrying an evicted needle. `build_niah_prompt` was hiding
   the signal. **This does not clear C2 or C3** — no analog of either exists on RULER, and
   C2 is what killed the previous layer-27 claim. (d) **layer
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
- **New evidence, 4–5 Sep — neither (a) nor (b) above is the answer.** Extending the C1
  needle set from 4 to 24 words (12 place names, 12 common nouns) finds a real,
  Holm-significant retention difference by needle identity at layers 18 and 27 (see
  "Findings from the 4–5 Sep C1 rank correction..." below) — place names read out far
  below chance, most common nouns don't. The pooled "C1 fails" verdict was averaging a
  working signal against a broken one across an arbitrary 4-word sample, not evidence
  the instrument or AHN's memory is broken. This needs a name and a decision from him:
  is needle-identity-dependent retention the finding to build RQ2 around, and if so, what
  needle set is now of record — the original 4, or a principled category design testing
  what Finding 4 only spot-checked?
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
