# BABILong construction-robustness pilot

Date: 2026-09-16

## Goal

Test whether the C1/C2 J-Lens retention pipeline behaves sensibly on BABILong rather than only on the RULER construction.

This is an exploratory construction-robustness check, not a replacement for the primary RULER RQ2 cohort.

## Initial 16k inspection

Dataset:
- `RMT-team/babilong`
- config: `16k`
- split/task: `qa1`
- 100 examples
- fields: `input`, `question`, `target`

Example targets included:
- bathroom
- kitchen
- bedroom

Observed context lengths were about 14k-15k Qwen tokens.

This is too short for the current AHN configuration because the recent window is approximately 32,640 tokens. Therefore the 16k configuration does not create the eviction regime needed for this retention test.

The answers are also not necessarily single-token under naive tokenization, so the RULER digit-specific C1/C2 scorer cannot be assumed to transfer unchanged.

## Decision

Next inspect BABILong `32k`.

For each example:
1. tokenize the full context with the Qwen tokenizer;
2. locate the actual supporting fact containing both the queried entity and target;
3. measure the support fact's distance from the end of the context;
4. classify it as genuinely evicted only when the distance exceeds the AHN recent window (`32640`);
5. if 32k provides too few genuinely evicted examples, move to BABILong `64k`.

Do not launch a full BABILong GPU benchmark yet.

First establish a small valid C1 cohort, then adapt C1 scoring to BABILong answers, and only after C1 works proceed to C2.
