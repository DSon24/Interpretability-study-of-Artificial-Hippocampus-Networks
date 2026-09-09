# Roadmap

Repo-side roadmap. The team's Gantt / critical-path roadmap is a Word document in Drive
(Week 5 deliverable); this file is the version engineering tracks against, and it stays in
sync with the **Execution Tracker** (goals G1–G8) and the amendments in
[`PROPOSAL.md`](PROPOSAL.md).

**Target:** ARR October cycle → NAACL 2027, Industry track (main track as the alternative).
**Hard deadline:** 12 Oct 2026 submission. The binding constraint is the **data-freeze
date**, not the submission date — Gautam writes from delivered results and needs ≥ 1 week
before submission.

## Now

- **RQ2 core at 3B.** Table 4 C1–C4 verdicts per cell (DeltaNet, then Mamba2); Figure 2
  primary-controls panel. RULER NIAH n=60 is the primary RQ2 cohort (Amendment 3).
- **Table 10** compute accounting — fill continuously.
- **OpenReview** author-account activation — 14-business-day lead time, internal due 26 Sep.

## Next

- 3B retention curves + Table 7 pairwise half-life ratios for all three cells.
- `04b` RQ3 join on matching per-example keys; Table 8.
- **7B checkpoints, all three cells** — gated on the 3B comparison and compute accounting;
  first on the pre-agreed cut list if buffer is consumed.
- Decision only: BABILong construction-robustness pilot (no compute by default).

## Later — gated

- **Multi-family evaluation** (Amendment 2 scope / Amendment 4 plan).
  {Qwen2.5, Llama, Mistral} × {DeltaNet, GatedDeltaNet, Mamba2} at the 7B tier, as a
  generality check on RQ2/RQ3 — not extra cells in the primary comparison.
  Owner **Hannah + Sơn** (reassigned from Devin, 31 Aug).
  - **Gate 0:** Qwen 3B baseline + C1–C4 verdicts + seed run complete.
  - **Gate 1:** a real Llama/Mistral AHN checkpoint is confirmed to exist — else fall back
    to the released Qwen2.5 {3B, 7B, 14B} **scale axis**, else descope to a Limitation.
    Decided with Gautam. (Every released AHN module is Qwen2.5-only.)
  - **Gate 2:** config of record identical to the Qwen arm; each backbone gets its own
    J-lens map and its own Table 3 pass.
  - Full plan + gates: [`PROPOSAL.md`](PROPOSAL.md) → "Multi-family evaluation".
    Compute: [`GPU_PLAN_2026-08-20.md`](GPU_PLAN_2026-08-20.md) → "Multi-family extension"
    (≈ 22–24 GPU-h, ~2× the measured budget).

## Milestones

| Date | Milestone |
|---|---|
| 2026-09-10 | Table 4 C1–C4 verdicts, DeltaNet 3B |
| 2026-09-12 | Primary 3B RULER results (DeltaNet, Mamba2) in hand |
| 2026-09-12 | BABILong pilot — decision only |
| 2026-09-20 | ARR AI-assistance disclosure check; prose written in-house |
| 2026-09-26 | Author list frozen; OpenReview accounts live + ORCID linked |
| early Oct | Data freeze — Gautam writes from delivered results |
| 2026-10-12 | ARR October submission (hard) |
| 2026-12-18 | Meta-review released (meta score ≥ 3 needed) |

## Cut order (if Week 9 consumes buffer)

Table 9 → Figure 7 → all 7B rows → LV-Eval → Figure 6.
Tables 3 and 4 and Figure 2 are never cut.
Multi-family is not on this list because it is not on the primary path — it does not start
unless Gate 0 and Gate 1 clear.
