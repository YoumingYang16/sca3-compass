# Numerical issue ledger — preserve historical evidence

1. D001(A0) strict raw-reference ranking versus log-cutoff comparisons: if x<y
but floating logx=logy, a strict log comparison may omit an exceedance that the
raw map includes. Reviewer supplied the analytic construction; no evidence yet
that an actual D001 family triggered it. D001 remains development, not certified.
Do not overwrite frozen results. This affects a numerical certification claim,
not the ideal-arithmetic interval-sweep proof.

2. D002(A0.1) forward interpolation and inverse cutoff can round in opposing
directions; allclose tests are not an outward-rounding proof. Retired as current
main candidate, kept as development evidence. No general finite-precision
guarantee is made for its small Power figures.

3. Active A0.2+ uses no inverse interpolation or nuisance sweep. Scores are
same paired maximum with conservative >= tail ranks. Original shape/SVD/
calibration floating arithmetic remains non-interval-certified, as in frozenR4.
Conservative exceptions are recorded, not dropped. This distinction must appear
in any eventual G2 decision; ideal finite-sample theorem is not a bit-level proof.

4. Ancillary sampler first two-tangent envelope could be inefficient or lose
curvature resolution on a very broad conditional plateau. BEFORE any full-family
conditional experiment it was replaced by three tangents and expanding slope
brackets. The plateau test compares with independent quadrature. No MCMC,
conditional-mean recentering mistake or silent sampler approximation is allowed.

5. A0.4 first selector test failed on exact representation of4/36 versus1-32/36.
Default count-rule receipt now preserves the former expression; variance-rule
receipts use1-a. Test fixed, no prior experiment affected. This is a numerical
representation fix, not an algorithmic/statistical contribution.
