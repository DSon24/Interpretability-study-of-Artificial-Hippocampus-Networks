## GLOBAL NOWRITE behavioral validation — 24 Sep 2026

**Row 82 — floor-with-sinks decision: DONE.** We will **not** run an additional
stock-Qwen floor-with-sinks experiment. The original no-AHN floor uses
`num_attn_sinks=0`, while the AHN/NOWRITE configuration uses 128 sinks, so
direct no-AHN-vs-NOWRITE comparisons (especially HotpotQA) retain that
configuration mismatch as a limitation. The later GLOBAL NOWRITE validation
already supplies the cleaner within-checkpoint, sink-matched causal control:
the merged AHN-GDN checkpoint, sliding window, and 128 sinks are held fixed
while all 36 AHN outputs are zeroed. On the frozen RULER evicted cohort,
AHN ON and GLOBAL NOWRITE both retrieve 0/32 answers; on the 28 in-window
controls both retrieve 28/28 by substring match. The no-AHN result is therefore
kept as a separate architectural floor rather than treated as an equivalent
intervention. No additional GPU run is required for this tracker item.
