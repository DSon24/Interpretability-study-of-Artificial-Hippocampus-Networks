# P2-1 — Checkpoint Identity and Provenance

## Status

**COMPLETED WITH UPSTREAM-PROVENANCE CAVEAT**

## Local checkpoint identity

The evaluated model is the local merged checkpoint:

`merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN`

Its two weight shards reproduce the hashes previously recorded by P0-4:

- `model-00001-of-00002.safetensors`:
  `639de2c369ed0b383d1b8f4f61db26a371f31be06ebc545d80a2af22e2abba57`
- `model-00002-of-00002.safetensors`:
  `0406ceb594144db43fc6826dc285cdaab973c76c00487f32c99eb208dd090dd2`

The safetensor index contains 722 tensor entries and references exactly these
two shards.

## Architecture verification

- Transformer layers: `36`
- AHN layers observed at runtime: `36`
- AHN implementation: `GatedDeltaNet`
- Hidden size: `2048`
- Attention heads: `16`
- Runtime dtype: `bfloat16`

## Runtime configuration

The checkpoint's embedded `config.json` contains:

- `sliding_window = 256`
- `use_sliding_window = false`
- `max_window_layers = 70`

The experiments do not use those embedded defaults. The runtime experiment
configuration explicitly applies:

- `sliding_window = 8064`
- `use_sliding_window = true`
- `num_attn_sinks = 128`
- `attn_impl = flash_attention_2`

The existing runtime configuration audit confirms that these overrides were
active during the recorded experiments.

Therefore, loading the checkpoint without applying
`configs/run_3b_gdn.json` does not reproduce the experimental attention
configuration.

## Provenance limitation

The merged checkpoint's `config.json` does not contain:

- `_name_or_path`
- `_commit_hash`

Consequently, the exact local checkpoint is fully frozen and reproducible by
hash, but its original upstream base/merge revision cannot be reconstructed
from the checkpoint metadata alone.

This is a provenance limitation, not evidence that different checkpoint
weights were used across the validated experiments.

## Conclusion

The validated experiments consistently used the same locally hashed
AHN-GDN checkpoint. Reproduction must use both:

1. The recorded checkpoint files identified by SHA256.
2. The runtime settings in `configs/run_3b_gdn.json`.

The paper or final artifact should not claim complete upstream merge
provenance unless the missing base-model and merge revision are recovered
from an external source.

## Reproducibility

- Script: `scripts/validate_p2_1_checkpoint_identity.py`
- Result: `results/validation/p2_1_checkpoint_identity.json`
- GPU required: no
