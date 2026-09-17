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

## 64k cohort freeze

BABILong `32k` was rejected for the retention pilot because its contexts were only
30,104–31,598 Qwen tokens, below the AHN recent window of 32,640 tokens. In the
20-example inspection, 0/20 supporting facts were genuinely evicted.

BABILong `64k` was then inspected.

Initial 20-example check:
- context range: 58,653–63,361 Qwen tokens
- supporting fact found: 20/20
- genuinely evicted: 6/20
- evicted fraction: 0.30

The complete `qa1` 64k cohort was then scanned before GPU use.

Full cohort:
- total examples: 100
- genuinely evicted examples: 32
- usable fraction: 0.32
- AHN recent window: 32,640 tokens

Frozen candidate IDs:

`[0, 4, 6, 8, 11, 17, 21, 24, 29, 30, 31, 32, 37, 40, 41, 49, 50, 60, 66, 67, 68, 72, 73, 74, 77, 83, 84, 86, 88, 89, 90, 99]`

These 32 examples form the candidate cohort for the BABILong C1 pilot.

Next step: run a small C1 smoke test on 5 frozen examples before scaling to the
full cohort.
