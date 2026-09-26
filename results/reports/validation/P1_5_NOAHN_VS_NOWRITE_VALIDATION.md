# P1-5 — No-AHN Versus GLOBAL NOWRITE

## Status

**COMPLETED**

## Question

Do the true no-AHN baseline and runtime GLOBAL NOWRITE represent the same
control?

## Cohort validation

The three behavioral artifacts use the same:

- 60 RULER `niah_single_1` examples
- Seed `20260820`
- Sliding window `8064`
- 32 evicted and 28 in-window examples
- Example indices, token lengths, needle positions and gold answers
- Behavioral scoring fields

## Configuration difference

### No-AHN

- Model: stock `Qwen/Qwen2.5-3B-Instruct`
- AHN cell: none
- Attention sinks: `0`
- Sliding window: `8064`

### GLOBAL NOWRITE

- Model: merged `Qwen-2.5-Instruct-3B-AHN-GDN`
- AHN cell: `GatedDeltaNet`
- Attention sinks: `128`
- Sliding window: `8064`
- Intervention: all 36 AHN outputs zeroed during inference

The checkpoint, model cell and attention-sink configuration therefore differ.

## Behavioral comparison

### Evicted examples

- GLOBAL NOWRITE substring accuracy: `0/32 = 0.0`
- No-AHN substring accuracy: `0/32 = 0.0`
- Paired difference: `0.0`
- 95% bootstrap CI: `[0.0, 0.0]`
- Exact McNemar p-value: `1.0`

Both controls agree that evicted answer retrieval is at floor.

### In-window examples

- GLOBAL NOWRITE substring accuracy: `28/28 = 1.0`
- No-AHN substring accuracy: `11/28 = 0.392857`
- Paired difference: `0.607143`
- 95% bootstrap CI: `[0.428571, 0.785714]`
- Discordant examples: `17` favor GLOBAL NOWRITE, `0` favor no-AHN
- Exact McNemar p-value: `1.52587890625e-05`

The strong in-window difference demonstrates that these controls are not
behaviorally equivalent.

## Interpretation

GLOBAL NOWRITE estimates an acute intervention within the trained AHN-GDN
model: the checkpoint and attention configuration remain fixed while all AHN
outputs are suppressed.

The no-AHN result is a between-model architectural reference using stock Qwen
without AHN modules or attention sinks.

Therefore:

- Use GLOBAL NOWRITE for causal statements about removing AHN output from the
  trained AHN model.
- Use no-AHN as a separate architectural performance floor.
- Do not describe no-AHN and GLOBAL NOWRITE as equivalent interventions.
- Their agreement at `0/32` supports the evicted retrieval-floor conclusion,
  but does not make the two controls interchangeable.

## Reproducibility

- Script: `scripts/validate_p1_5_noahn_vs_nowrite.py`
- Result: `results/validation/p1_5_noahn_vs_nowrite.json`
- Bootstrap replicates: `20,000`
- GPU required: no
