# 06b — RULER answer-prefix punctuation-control ablation

**Status:** completed  
**Recorded:** 2026-09-19  
**Git commit:** `032aa5efef69ac3183d10c4ff159dce8439599ef`

## Question

Is the semantic `answer_prefix` necessary for the layer-27 RULER C2 effect?

## Controlled manipulation

The token suffix associated with the answer prefix was replaced in token space with the same number of ordinary `" ."` punctuation tokens.

This preserved:
- total sequence length
- all tokens before the manipulated suffix
- gold-answer tokens
- gold-answer positions

Because the tokenizer has a boundary-crossing token at the prefix boundary, this is a same-length token-suffix punctuation control, not a character-only replacement.

## Fixed setup

- Model: `Qwen-2.5-Instruct-3B-AHN-GDN`
- RULER: `config=16384`, `split=test`, `n=60`, `seed=20260820`
- Frozen evicted cohort: 32 examples
- Analysis layer: L27
- Probe layers: `[9, 18, 27]`
- Sliding window / sinks: `8064 / 128`
- Readout: Corpus-A J-Lens
- Unordered C2 pairs: 496
- Pairs dropped: 0
- Bootstrap: example-level paired, 2,000 replicates
- Randomization test: paired example-level label swap, 5,000 permutations

## Results

| Condition | L27 C2 effect | 95% CI |
|---|---:|---:|
| Original prefix | **1.910967×** | [1.176174, 3.232673] |
| Punctuation control | **1.713996×** | [1.052953, 2.814297] |

Controlled ratio (`control / original`): **0.896926×**

95% CI: **[0.556051, 1.460719]**

Paired randomization p-value: **0.689262**

## Interpretation

The same-length punctuation replacement did not eliminate the L27 RULER C2 effect.

Therefore, under this fixed setup, answer-prefix semantics are not necessary for the overall L27 effect.

This does not show that the prefix has zero influence. Digit position 6 weakened under the control condition.

## Historical-baseline caveat

The fresh original-prefix baseline here is **1.910967×**, while historical Run026 was approximately **1.076164×**.

Because the historical checkpoint/lens/runtime identity could not be fully verified, this is a new internally controlled replication/ablation, not an exact Run026 remeasurement.

## Provenance

- Model shard 1 SHA256: `639de2c369ed0b383d1b8f4f61db26a371f31be06ebc545d80a2af22e2abba57`
- Model shard 2 SHA256: `0406ceb594144db43fc6826dc285cdaab973c76c00487f32c99eb208dd090dd2`
- Corpus-A lens SHA256: `17af009149b23c0b6d288ea993e555c6bb33a57fae10e4fefe6b07ea31166b2a`
- Historical 04i rows SHA256: `356c84e7f31039c65cfcba683e1985dc2b0d268dc0d0e711050f08e42c300aa2`
