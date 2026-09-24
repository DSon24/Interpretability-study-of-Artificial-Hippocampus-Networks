# Progress Log — 2026-09-24

Branch: `validation`

This log records two execution-tracker tasks closed on 24 Sep 2026. These are validation / methodology closures, not new primary scientific results.

## Row 76 — Verify J-Lens norm convention against reference `jlens` package

**Status: DONE**

The installed Anthropic `jacobian-lens` reference implementation was inspected directly from the dedicated J-Lens environment.

Reference behavior:

1. `JacobianLens.transport()` maps a source-layer residual into the final-layer basis with `residual @ J_bar.T`.
2. `JacobianLens.apply()` transports first and then calls `model.unembed(residual)`.
3. `HFLensModel.unembed()` applies the model's final norm before the LM head.

The project implementation in `ahn_interp.py` matches the same order:

1. `JacobianLens.transport()` uses `vec @ J.T`.
2. `readout_logits()` applies `m.model.norm(...)`.
3. The normalized vector is decoded with `m.lm_head.weight`.

Therefore the suspected double-normalization / norm-convention mismatch is ruled out.

**Conclusion:** the project's J-Lens readout convention matches the Anthropic reference: transport -> final norm -> LM head. This does not change the existing Table 3 validation failures; it only rules out norm convention as their cause.

## Row 82 — Decide floor-with-sinks run vs limitation

**Status: DONE**

**Decision: option (b) — do not run an additional floor-with-sinks experiment. Record the mismatch as a limitation.**

The original stock-Qwen no-AHN floor uses:

- no AHN module
- `num_attn_sinks=0`
- sliding window 8064

The trained AHN / NOWRITE configuration uses:

- merged AHN-GDN checkpoint
- `num_attn_sinks=128`
- sliding window 8064

The later GLOBAL NOWRITE validation already provides the cleaner sink-matched causal intervention within the trained AHN checkpoint: all 36 AHN outputs are zeroed while the checkpoint, window, and 128 attention sinks remain fixed.

On the frozen RULER `niah_single_1` cohort:

- evicted examples: AHN ON = 0/32 substring retrieval
- evicted examples: GLOBAL NOWRITE = 0/32 substring retrieval
- in-window controls: AHN ON = 28/28 substring retrieval
- in-window controls: GLOBAL NOWRITE = 28/28 substring retrieval

Relevant artifacts:

- `results/validation/p0_4_global_behavior.json`
- `docs/P1_5_NOAHN_VS_NOWRITE_VALIDATION.md`
- `docs/P1_6_RULER_BEHAVIORAL_PROTOCOL_VALIDATION.md`

**Conclusion:** no extra GPU run is needed for row 82. GLOBAL NOWRITE is the causal within-checkpoint control; the stock no-AHN result remains a separate architectural floor. The sink mismatch matters for direct no-AHN-vs-NOWRITE comparisons (especially HotpotQA) and should be reported as a limitation rather than treated as an equivalent intervention.

## Tracker state after this update

- Row 76: DONE — J-Lens norm convention verified against reference implementation.
- Row 82: DONE — no additional floor-with-sinks run; limitation recorded.
