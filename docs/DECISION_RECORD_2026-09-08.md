# Mentor decision record — 8 Sep 2026

Gautam confirmed the following after reviewing the length-matched and content-swap
checks. This record closes the two decisions posed in the 7 Sep diagnosis packet and is
the operational authority for the next runs.

## Approved decisions

1. **Re-scope RQ2 to the RULER cohort.** RULER NIAH remains the primary cohort at the
   pre-registered **n=60**. Do not increase the cohort without a specific reason and a
   further amendment.
2. **Start DeltaNet and Mamba2.** Merge the two 3B implementations and run the same RULER
   n=60 protocol used for GatedDeltaNet. The length/content/construction question is no
   longer a gate.
3. **Report C2 as significant but sub-threshold.** Table 4 remains a formal **FAIL** against
   the pre-registered 10x criterion. Methods must also report the measured effect:
   **1.076x per digit, 95% CI [1.054, 1.098], permutation p<0.0001**.
4. **Keep the construction checks as supporting/limitations analysis.** At matched length,
   the homemade construction is null at layer 27 for both word needles (0.973x, p=0.59)
   and seven-digit needles (0.995x, p=0.76). This rules out length and content as the
   explanation for the RULER/homemade difference; it does not establish which part of the
   construction causes it.
5. **Document the four-needle limitation.** Results from the original
   Paris/Tokyo/banana/lantern set repeatedly failed to generalize. No population-level
   claim may rest on that set alone.
6. **Keep the original sample size.** RULER n=60 is a pre-registered cohort size, not a
   10% subsample.

## Immediate execution order

1. Merge DeltaNet and Mamba2 at 3B.
2. Run both cells on RULER NIAH n=60 on the restored H100 environment.
3. Complete the no-AHN floor, boundary Jensen–Shannon logging, and C4 layer-permutation
   control.
4. Populate Table 7 from the three-cell results, then make the 7B decision.

The H100 environment is available again. It is an unprivileged shared environment:
**no root or sudo**; installs and kernels must remain user-local.

## New dataset resources

Gautam also provided BABILong, NoLiMa, LongBench, and SCROLLS. Their roles and change
controls are recorded in [the dataset register](DATASET_REGISTER_2026-09-08.md). None of
them changes the approved RULER n=60 primary run.

