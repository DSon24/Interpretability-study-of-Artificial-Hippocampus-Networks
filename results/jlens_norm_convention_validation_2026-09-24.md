## J-Lens norm convention validation — 24 Sep 2026

**Row 76 — J-Lens norm convention: DONE.** The installed Anthropic
`jacobian-lens` reference implementation was checked directly. Its
`JacobianLens.transport()` maps residuals with `residual @ J_bar.T`;
`JacobianLens.apply()` then calls `model.unembed()`, and
`HFLensModel.unembed()` applies the model's final norm before the LM head.
Our `ahn_interp.py` follows the same convention:
`vec @ J.T -> model.model.norm(...) -> lm_head`. The suspected
double-normalization / norm-convention mismatch is therefore ruled out.
This does **not** change the existing Table 3 failures; it rules out norm
convention as their cause.
