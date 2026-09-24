# P1-3 — Corrected C2 Statistical Validation

## Status

**COMPLETED**

The corrected candidate-conditioned C2 statistic was audited using the
32 frozen RULER examples as the independent statistical units.

## Source artifact

- `results/run_3b_gdn/06c_ruler_cross_term_context_conditioning.json`
- Corrected candidate-conditioned C2 point estimate: `6.316858044094561`
- Frozen independent examples: `32`
- Overlapping unordered example pairs: `496`
- Pairs dropped: `0`

## Dependence problem

The 496 C2 pair values are not 496 independent observations. Each of the
32 examples participates in 31 different pairs. Treating every pair as
independent would constitute pseudo-replication and underestimate
uncertainty.

The validation therefore resamples complete example identities rather than
individual pair rows.

## Validated results

- Corrected log effect: `1.8432219396195104`
- Corrected multiplicative effect: `6.316858044094561`
- Example-level bootstrap 95% CI on log scale:
  `[-0.7793080838216889, 4.523419714941366]`
- Example-level bootstrap 95% CI on effect scale:
  `[0.45872330801728384, 92.1501997253343]`
- Jackknife log-effect standard error: `1.4427047973861047`
- Candidate-identity permutation, one-sided p-value:
  `0.0899491005089949`
- Candidate-identity permutation, two-sided p-value:
  `0.1783782162178378`

## Interpretation

The corrected C2 point estimate is large, but its example-level uncertainty
is also very large. The 95% confidence interval contains the null value of
`1.0`, and neither permutation p-value is below `0.05`.

Therefore, this cohort does not provide conventional statistically
significant evidence for a positive corrected C2 effect. The appropriate
claim is:

> The corrected candidate-conditioned analysis produced a 6.32x point
> estimate, but the effect was not statistically distinguishable from the
> null across the 32 frozen examples.

This does not invalidate the corrected computation. It limits the strength
of the statistical conclusion that can be drawn from this cohort.

## Reproducibility

- Script: `scripts/validate_p1_3_c2_statistics.py`
- Result: `results/validation/p1_3_c2_statistics.json`
- Bootstrap replicates: `20,000`
- Permutation replicates: `100,000`
- Seed: `20260924`
- GPU required: no
