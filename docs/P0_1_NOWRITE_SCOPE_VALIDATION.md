# P0-1 — NOWRITE Scope

## Why P0-1 happened

`AHNProbe.run()` uses `layers=` for both capturing and disabling AHN layers.

So `nowrite=True, layers=[9, 18, 27]` disables only layers 9, 18, and 27 — not all 36 AHN layers.

## Why it matters

- Local NOWRITE = selected layers OFF
- Global NOWRITE = all 36 AHN layers OFF

If we claim AHN memory is completely removed, we need global NOWRITE.

## Validation

Compare:

1. AHN ON
2. 3 layers OFF
3. 36 layers OFF

If 3-layer and 36-layer results differ, they are different controls.
