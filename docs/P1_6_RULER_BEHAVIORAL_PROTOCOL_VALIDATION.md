# P1-6 — RULER Behavioral Protocol Validation

## Status

**PASS WITH INDEPENDENT-REPRODUCTION CAVEAT**

No additional GPU experiment was required. P1-6 is covered by the completed
P0-4 behavioral diagnostic, with token-boundary details handled separately
under P0-3.

## Question

Does the RULER behavioral experiment use a frozen, correctly separated cohort,
paired conditions, direct behavioral metrics, and a verified GLOBAL NOWRITE
intervention?

## Evidence

Primary artifacts:

- `scripts/validate_p0_4_global_behavior.py`
- `results/validation/p0_4_global_behavior.json`
- `results/run_3b_gdn/06_ahn_on_ruler.json`
- `results/validation/p0_1_nowrite_scope.json`

## Protocol checks

- Frozen RULER `niah_single_1` cohort at configuration 16384.
- 60 paired examples.
- 32 evicted examples and 28 in-window controls.
- Identical examples and scoring settings across AHN ON and GLOBAL NOWRITE.
- Primary metric: exact answer.
- Secondary metrics: substring match, F1, and first-line F1.
- No J-Lens was used in the behavioral outcome.
- AHN activity was observed under AHN ON.
- All 36 AHN outputs were verified zero under GLOBAL NOWRITE.
- Raw paired rows and aggregate uncertainty were saved.

## Behavioral result

For the 32 evicted examples:

- AHN ON exact/substring retrieval: 0/32.
- GLOBAL NOWRITE exact/substring retrieval: 0/32.
- Paired exact-answer difference: 0.
- Paired bootstrap 95% interval: [0, 0].

For the 28 in-window controls:

- AHN ON substring retrieval: 28/28.
- GLOBAL NOWRITE substring retrieval: 28/28.

The in-window control confirms that the prompt and scorer were capable of
recovering answers when the needle remained accessible.

## Interpretation

Within this frozen run, removing all AHN outputs did not change answer-retrieval
performance on the evicted cohort. Both conditions were already at the
behavioral floor.

This is a direct output-level conclusion and does not depend on J-Lens.

## Limitation

The independent four-example rerun did not reproduce every saved output-level
detail. Generated wording was not byte-for-byte stable, and the validation
script records this as a reproduction warning rather than aborting or treating
wording differences as causal.

Therefore, P1-6 validates the recorded paired behavioral protocol and its
answer-performance conclusion, but it does not support causal interpretation
of raw generated-text differences.
