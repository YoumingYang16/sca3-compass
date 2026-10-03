# Numerical, randomization and input scope

Theorems describe exact real arithmetic and independent continuous random
draws. Implementations use NumPy float64 and deterministic recorded PRNG seeds.
This is not an interval-certified implementation over arbitrary ill-conditioned
matrices. Passing tests or observed FDR bounds does not remove this distinction.

- Coverage-quantile denominator for F4,2 is explicitly certified by an exact
  integer binomial polynomial; finite SciPy estimates only propose candidates.
- The shape envelope is a proven global spectral inequality, not a grid or
  local optimizer pretending to attain the nuisance supremum.
- The predictive bridge uses integer rank counting including ties and +1;
  no estimated tiny tail is passed off as an exact probability. Resolution is
  1/(B_MC+1). All reference draws are retained, never filtered for favourable
  values; separate old confidence-envelope draws had a different purpose.
- Positive-statistic clipping is essential. Increasing a variance does NOT
  give a conservative upper-tail p for a negative statistic; such p is set1.
- Matrix inverses/eigenvalues and special-function tails in the guarded
  reference remain floating point. Highly ill-conditioned or degenerate
  inputs require explicit failure, not pseudoinverse substitution or ridge
  that would silently violate affine equivariance.
- Invalid input dimensions/types/budgets raise. Predictive numerical failures
  return whole-family zero evidence/no discoveries with an error receipt;
  this outcome stays in experiment denominators. Source tests inject a failure.
- Standard pattern-fit nonconvergence sets gamma0 through the existing
  learning-only helper. It is recorded separately from numerical test failure.
- Fresh auxiliary randomness is part of the method. Seeds are fixed before
  observing outcomes and logged. Repeating an analysis to cherry-pick its
  reference bank invalidates the stated randomized guarantee. Replaying the
  exact saved seed for reproducibility is not another confirmation.

Three specified saved DEV cases were replayed exactly (normal,t3,mixed-sign),
including identical rank p and continuous evidence. Their calibration estimates
were recomputed at80 decimal digits; relative differences <=3.14e-15
(numerical-audit.json). This audits calibration, NOT all matrix eigensolvers,
near-boundary projections or all possible inputs; no universal bit certificate.

Grid helper now enforces exactly represented dyadic support. It upward rounds
the exact rational denominator of represented heights and downward rounds each
normalized table height. That certifies the table budget for this representation.
Final gamma mixtures, upstream linear algebra and PRNG law still have the stated
floating qualifications. Tests with arbitrary off-grid .005 must fail, not be
rounded onto the support. Data/model assumptions remain the larger applicability
limitation; no numerical test can establish those from a bare tensor.
