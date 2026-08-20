# Pilot results — 18 Aug 2026 (SUPERSEDED)

Produced by `notebooks/pilot/son_pilot_*_2026-08-18.ipynb`. Kept for provenance only.
**Do not cite, plot, or carry these numbers forward.**

Three reasons, all documented in the root README under "Findings from the 18 Aug pilot":

1. The decoded vector was `ahn_attn_output`, which lives in the concatenated-head space
   *before* `self_attn.o_proj` — not the residual stream. Dimensions coincide at 3B
   (16 heads x 128 = 2048), so the decode ran and returned noise.
2. Needles sat at roughly token 5, inside the `num_attn_sinks` prefix, where they are
   never compressed and stay losslessly visible to attention.
3. The final RMSNorm was not applied before the unembedding.

Two diagnostics in the data itself say the same thing:

- Every rank in `multi_needle_delta.json` and `retention_decay.json` falls between 7k
  and 149k on a 151,936-token vocabulary — indistinguishable from chance.
- In `retention_decay.json` the pre-eviction baseline (Paris, 110712) is *worse* than
  most evicted conditions (~95000). A ceiling below the floor means the measurement is
  not measuring retention.

`layer_profile.json` is the least affected of the three: it reports per-layer L1 norms of
the AHN output, which are basis-independent up to the `o_proj` rotation, so the shape of
the profile is roughly informative even though the absolute values are not the residual
-stream contribution.

Replacement: `notebooks/04_niah_retention.ipynb` -> `results/run_*/04_retention_rows.json`.
